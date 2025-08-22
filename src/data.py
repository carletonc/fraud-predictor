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


@st.cache_data(show_spinner=False)
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
    
    # Generate data
    data = []
    
    # Generate legitimate transactions
    for i in range(n_legit):
        row = {}
        for feature in NON_FRAUD_STATS:
            row[feature] = np.random.normal(
                NON_FRAUD_STATS[feature]['mean'], 
                NON_FRAUD_STATS[feature]['std']
            )
        row['Fraud'] = 0 # synthetic feedback mechanism
        data.append(row)
    
    # Generate fraudulent transactions
    for i in range(n_fraud):
        row = {}
        for feature in FRAUD_STATS:
            row[feature] = np.random.normal(
                FRAUD_STATS[feature]['mean'], 
                FRAUD_STATS[feature]['std']
            )
        row['Fraud'] = 1 # synthetic feedback mechanism
        data.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Ensure Amount is non-negative
    df['Amount'] = np.maximum(df['Amount'], 0)
    
    # Shuffle the data
    #df = df.sample(frac=1).reset_index(drop=True)
    
    # Prepare features and labels
    #X = df.drop(columns=['Fraud'])
    #y = df['Fraud']
    return df



if __name__ == "__main__":
    # Generate 1000 samples
    download_creditcard_data()