import joblib
import json
import os
import uuid
import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score
from scipy.stats import ks_2samp, wasserstein_distance, entropy

from src.data import generate_fraud_data, get_random_fraud_rate

# Constants
TRAIN_DATA_PATH = 'data/raw/creditcard.csv'
SCALER_PATH = 'artifacts/scaler.pkl'
MODEL_PATH = 'artifacts/best_xgboost_model.json'
MODEL_METRICS_PATH = 'artifacts/test_metrics.json'
THRESHOLD = 0.77

# For streamlit
TOP_FEATURES = ['V14', 'V4', 'V7']

def get_train_data(SEED=42):
    """
    Loads the creditcard dataset from TRAIN_DATA_PATH and performs a three-way split:
    Train/Val/Test. Returns X_train, the training data.
    
    Parameters
    ----------
    SEED : int, optional
        The random seed for the data split. Defaults to 42.
    
    Returns
    -------
    X_train : pandas.DataFrame
        The training data.
    """
    df = pd.read_csv(TRAIN_DATA_PATH).drop(columns=['Time'])
    
    y = df[['Class']]
    X = df.drop(columns=['Class'])
    del df
    # Three-way split: Train/Val/Test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=SEED, stratify=y_temp  # 0.25 of 0.8 = 0.2 overall
    )   
    return X_train


def predict(X): 
    """
    Predicts fraud labels for the given input features using a pre-trained XGBoost model.

    Parameters
    ----------
    X : pandas.DataFrame
        The input features for prediction.

    Returns
    -------
    numpy.ndarray
        An array of predicted labels where 1 indicates fraud and 0 indicates non-fraud.
    """
    # load scaler & model
    scaler = joblib.load(SCALER_PATH) 
    model_xgb = xgb.XGBClassifier() 
    model_xgb.load_model(MODEL_PATH) 
    # transform & predict
    X = scaler.transform(X)
    y_pred = np.where(model_xgb.predict_proba(X)[:,1] > THRESHOLD, 1, 0) 
    return y_pred 


def calculate_drift_metrics(X_test, X_live, bins=10):
    """
    Calculate KS, JSD, Wasserstein, and PSI between two datasets.

    Parameters
    ----------
    X_test : array-like
        Reference dataset (e.g., test set feature column).
    X_live : array-like
        Live dataset to compare against reference.
    bins : int
        Number of bins for PSI & JSD histograms.

    Returns
    -------
    dict
        {
            "ks_stat": float,
            "ks_pvalue": float,
            "jsd": float,
            "wasserstein": float,
            "psi": float
        }
    """
    # Drop NaNs
    X_test = pd.Series(X_test).dropna().values
    X_live = pd.Series(X_live).dropna().values

    # KS Test
    ks_stat, ks_pvalue = ks_2samp(X_test, X_live)

    # Jensen-Shannon Divergence (bounded between 0 and 1)
    hist_range = (min(X_test.min(), X_live.min()), max(X_test.max(), X_live.max()))
    p_hist, bin_edges = np.histogram(X_test, bins=bins, range=hist_range, density=True)
    q_hist, _ = np.histogram(X_live, bins=bins, range=hist_range, density=True)
    # Normalize to probabilities
    p_prob = p_hist / (p_hist.sum() + 1e-12)
    q_prob = q_hist / (q_hist.sum() + 1e-12)
    m = 0.5 * (p_prob + q_prob)
    jsd = 0.5 * (entropy(p_prob, m) + entropy(q_prob, m))
    jsd = np.sqrt(jsd)  # Ensure bounded [0, 1]

    # Wasserstein Distance
    wass = wasserstein_distance(X_test, X_live)

    # Population Stability Index (PSI)
    # Avoid zero counts with small epsilon
    psi_vals = (p_prob - q_prob) * np.log((p_prob + 1e-12) / (q_prob + 1e-12))
    psi = np.sum(psi_vals)
    output =  {
        "ks_stat": ks_stat,
        "ks_pvalue": ks_pvalue,
        "jsd": jsd,
        "wasserstein": wass,
        "psi": psi,
        "bins": bin_edges.tolist(),      # convert numpy arrays to lists for JSON friendliness
        "test_hist": p_prob.tolist(),
        "live_hist": q_prob.tolist()
    }
    return {k: float(v) if isinstance(v, (np.floating, float)) else v for k, v in output.items()}


def calculate_accuracy_metrics(y_true, y_pred, test_json_path):
    """
    Compare online classification metrics to gold (test set) metrics.

    Parameters
    ----------
    y_true : array-like
        Ground truth labels (binary 0/1).
    y_pred : array-like
        Model predictions (binary 0/1).
    gold_json_path : str
        Path to JSON file containing gold metrics.

    Returns
    -------
    dict
        {
            "online": {...},
            "offline_test": {...}
        }
    """
    # --- Load gold metrics ---
    with open(test_json_path, "r") as f:
        offline_test_metrics = json.load(f)
    # Keep only precision & recall
    offline_test_metrics = {
        k: float(v) for k, v in offline_test_metrics['1'].items() if k in ("precision", "recall")
    }

    # --- Compute online metrics ---
    y_true = pd.Series(y_true).dropna() 
    y_pred = pd.Series(y_pred).dropna() 
    y_true, y_pred = y_true.align(y_pred, join="inner") 

    precision = precision_score(y_true, y_pred, zero_division=0) 
    recall = recall_score(y_true, y_pred, zero_division=0) 
    total_samples = len(y_true) 
    total_true = int((y_true == 1).sum()) 
    total_pred_true = int((y_pred == 1).sum()) 

    online_metrics = {
        "precision": float(precision), 
        "recall": float(recall), 
        "total_samples": total_samples, 
        "total_true": total_true, 
        "total_pred_true": total_pred_true 
    }

    return {
        "online": online_metrics,
        "offline_test": offline_test_metrics
    }


def single_run():
    """
    Single run of the data generation and inference pipeline.

    This function is supposed to be called repeatedly to simulate a stream of data. It
    generates a unique run ID, generates new "live" data, checks for feature drift w.r.t.
    the training data, and documents online metrics.

    This function does not return anything. Instead, it appends results to the following
    global variables:
    - LIVE_DATA: a list of dictionaries, each containing a run ID and the corresponding
      live data as a Pandas DataFrame
    - LIVE_FEATURE_DRIFT: a list of dictionaries, each containing a run ID and the
      corresponding feature drift metrics as a dictionary with feature names as keys
    - LIVE_METRICS: a list of dictionaries, each containing a run ID and the
      corresponding online metrics as a dictionary
    """
    # generate run ID
    run_id = uuid.uuid4()
    
    # get train data for distribution comparison
    X_train = get_train_data()
    
    # generate "live" data
    fraud_rate = get_random_fraud_rate()
    df = generate_fraud_data(n_samples=None, fraud_rate=fraud_rate) 
    st.session_state['live_data'].append({run_id: df})
    
    # split for inference
    X = df.drop(columns=['Fraud'])
    y = df['Fraud']
    del df
    
    # check drift
    drift= {run_id: {}}
    for feature in TOP_FEATURES:
        drift[run_id][feature] = calculate_drift_metrics(X_train[feature], X[feature])
    st.session_state['live_feature_drift'].append(drift)
    
    # document online metrics
    st.session_state['live_metrics'].append({run_id: calculate_accuracy_metrics(y, predict(X), MODEL_METRICS_PATH)})

    
  
def full_run(limit = 14):
    """
    Repeatedly call `single_run` to generate a sequence of runs.

    Parameters
    ----------
    limit : int
        The number of runs to perform. Defaults to 14.

    Returns
    -------
    None
    """
    for i in range(limit):
        single_run()
    
    
if __name__ == "__main__":
    full_run()