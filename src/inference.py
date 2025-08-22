"""
Model inference and monitoring for fraud prediction.

This module provides functions for making predictions with the trained fraud
detection model, calculating feature drift metrics, and monitoring model
performance in production-like scenarios.
"""

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

from src.constants import SEED, THRESHOLD, TOP_FEATURES, TRAIN_DATA_PATH, SCALER_PATH, MODEL_PATH, MODEL_METRICS_PATH
from src.data import generate_fraud_data, get_random_fraud_rate


@st.cache_data
def load_training_data():
    """Load training data for distribution comparison with caching.
    
    Loads the credit card dataset from TRAIN_DATA_PATH and performs a three-way
    split to extract the training portion for use as a reference distribution
    in drift detection. Results are cached to avoid reprocessing.
    
    Returns:
        pandas.DataFrame: Training features (X_train) for distribution comparison.
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


@st.cache_resource
def load_model_and_scaler():
    """Load and cache the pre-trained model and scaler.
    
    Loads the XGBoost model and StandardScaler from disk and caches them
    as resources to avoid reloading on every prediction call.
    
    Returns:
        tuple: (model_xgb, scaler) - The loaded model and scaler.
    """
    scaler = joblib.load(SCALER_PATH)
    model_xgb = xgb.XGBClassifier()
    model_xgb.load_model(MODEL_PATH)
    return model_xgb, scaler


def predict(X): 
    """Predict fraud labels for input features.
    
    Uses cached model and scaler to make predictions efficiently.
    
    Args:
        X (pandas.DataFrame): Input features for prediction.
    
    Returns:
        numpy.ndarray: Array of predicted labels where 1 indicates fraud
            and 0 indicates non-fraud.
    """
    # Use cached model and scaler
    model_xgb, scaler = load_model_and_scaler()
    
    # transform & predict
    X = scaler.transform(X)
    y_pred = np.where(model_xgb.predict_proba(X)[:,1] > THRESHOLD, 1, 0) 
    return y_pred 


@st.cache_data
def calculate_drift_metrics(X_test, X_live, bins=20):
    """Calculate distribution drift metrics between reference and live datasets.
    
    Computes multiple statistical measures to quantify the difference between
    feature distributions: Kolmogorov-Smirnov test, Jensen-Shannon Divergence,
    Wasserstein distance, and Population Stability Index. Results are cached
    to avoid recomputing for identical data.
    
    Args:
        X_test (array-like): Reference dataset (e.g., test set feature column).
        X_live (array-like): Live dataset to compare against reference.
        bins (int): Number of bins for histogram-based metrics (PSI & JSD).
            Defaults to 20.
    
    Returns:
        dict: Dictionary containing drift metrics:
            - ks_stat (float): Kolmogorov-Smirnov statistic
            - ks_pvalue (float): KS test p-value
            - jsd (float): Jensen-Shannon Divergence (bounded [0, 1])
            - wasserstein (float): Wasserstein distance
            - psi (float): Population Stability Index
            - bins (list): Histogram bin edges
            - test_hist (list): Reference dataset histogram
            - live_hist (list): Live dataset histogram
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


@st.cache_data
def calculate_accuracy_metrics(y_true, y_pred, test_json_path):
    """Compare online classification metrics to offline test metrics with caching.
    
    Loads gold standard metrics from the test set and compares them to
    current online performance metrics, focusing on precision and recall.
    Results are cached to avoid reprocessing identical data.
    
    Args:
        y_true (array-like): Ground truth labels (binary 0/1).
        y_pred (array-like): Model predictions (binary 0/1).
        test_json_path (str): Path to JSON file containing offline test metrics.
    
    Returns:
        dict: Dictionary containing:
            - online (dict): Current online performance metrics
            - offline_test (dict): Gold standard offline test metrics
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
    """Execute a single monitoring run.
    
    Performs one complete cycle of the monitoring pipeline: generates new
    synthetic data, checks for feature drift against training data, makes
    predictions, and documents online metrics. Results are stored in
    Streamlit session state for visualization.
    
    This function simulates a production monitoring scenario where new data
    arrives in batches and model performance is continuously tracked.
    
    Returns:
        None: Results are stored in st.session_state variables.
    """
    # generate run ID
    run_id = uuid.uuid4()
    
    # get train data for distribution comparison (now cached)
    X_train = load_training_data()
    
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

    
  
def full_run(limit=14):
    """Execute multiple monitoring runs.
    
    Repeatedly calls single_run to generate a sequence of monitoring cycles,
    simulating continuous model monitoring over time.
    
    Args:
        limit (int): Number of runs to perform. Defaults to 14.
    
    Returns:
        None: Results are stored in st.session_state variables.
    """
    for i in range(limit):
        single_run()
    
    
if __name__ == "__main__":
    full_run()