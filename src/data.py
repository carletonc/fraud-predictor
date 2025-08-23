"""
Data generation and management utilities for fraud prediction.

This module provides functions for downloading credit card fraud data from Kaggle,
generating synthetic fraud data based on statistical distributions, and
managing data loading operations.
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from src.constants import DATA_DIR, TRAIN_DATA_PATH, FRAUD_STATS, NON_FRAUD_STATS 



def download_creditcard_data():
    """Download the credit card fraud dataset from Kaggle using .env credentials.
    
    Downloads the credit card fraud dataset from Kaggle and saves it to the
    data directory. Requires Kaggle API credentials to be set in a .env file.
    
    Returns:
        None: Downloads data to DATA_DIR directory.
    """
    # Load environment variables from .env file
    load_dotenv()
    # Authenticate and download
    import kaggle
    kaggle.api.authenticate()
    os.makedirs(DATA_DIR, exist_ok=True)
    
    kaggle.api.dataset_download_files(
        'mlg-ulb/creditcardfraud', 
        path=DATA_DIR, 
        unzip=True,
    )
    return 


def get_random_fraud_rate(base_rate=0.001727, scale=10000):
    """Generate a random fraud rate drawn from a beta distribution.
    
    The distribution is parameterized by the base rate (the mean of the distribution)
    and a scale parameter that controls the variance of the distribution.
    
    Args:
        base_rate (float): The mean of the distribution. Defaults to 0.001727.
        scale (float): The scale parameter of the distribution. Higher scale results
            in lower variance. Defaults to 10000.
    
    Returns:
        float: A random fraud rate drawn from the beta distribution.
    """
    # scale controls the variance (higher scale = lower variance)
    alpha = base_rate * scale
    beta = (1 - base_rate) * scale
    return np.random.beta(alpha, beta)


#@st.cache_data(show_spinner=False)
def generate_fraud_data(n_samples, fraud_rate=None):
    """Generate synthetic fraud detection data based on provided dataset statistics.
    
    Generates synthetic data that mimics the statistical properties of the
    original credit card fraud dataset, including feature distributions for
    both legitimate and fraudulent transactions. Results are cached to avoid
    regenerating identical data.
    
    Args:
        n_samples (int): Number of samples to generate. If None, generates a
            random number between 100,000 and 200,000.
        fraud_rate (float): Proportion of fraudulent transactions. 
            If None, uses a random rate from get_random_fraud_rate().
    
    Returns:
        pandas.DataFrame: Generated data with V1-V28 features, Amount, and Fraud
            label column. The data is shuffled and includes both legitimate (0)
            and fraudulent (1) transactions.
    """
    # If fraud_rate is None, generate a random one (this won't be cached)
    if fraud_rate is None:
        fraud_rate = get_random_fraud_rate()
    
    # If n_samples is None, randomly generate a sample size
    if n_samples is None:
        n_samples = np.random.randint(100000, 200001)  # Random number
    
    # Calculate number of fraud vs legitimate samples
    n_fraud = int(n_samples * fraud_rate)
    n_legit = n_samples - n_fraud
    
    # Pre-allocate arrays for better memory efficiency
    all_features = list(NON_FRAUD_STATS.keys())  # Assuming same features in both stats
    n_features = len(all_features)
    
    # Generate all data at once using vectorized operations
    data = np.empty((n_samples, n_features + 1))  # +1 for Fraud column
    
    # Generate legitimate transactions
    if n_legit > 0:
        for i, feature in enumerate(all_features):
            data[:n_legit, i] = np.random.normal(
                NON_FRAUD_STATS[feature]['mean'], 
                NON_FRAUD_STATS[feature]['std'],
                size=n_legit
            )
        data[:n_legit, -1] = 0  # Fraud labels
    
    # Generate fraudulent transactions
    if n_fraud > 0:
        for i, feature in enumerate(all_features):
            data[n_legit:, i] = np.random.normal(
                FRAUD_STATS[feature]['mean'], 
                FRAUD_STATS[feature]['std'],
                size=n_fraud
            )
        data[n_legit:, -1] = 1  # Fraud labels
    
    # Create DataFrame
    data = pd.DataFrame(data, columns=all_features + ['Fraud'])
    
    # Ensure Amount is non-negative
    data['Amount'] = np.maximum(data['Amount'], 0)
    
    # Shuffle the data
    #data = data.sample(frac=1)
    return data



if __name__ == "__main__":
    # Generate 1000 samples
    download_creditcard_data()