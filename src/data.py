import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from src.constants import DATA_DIR, TRAIN_DATA_PATH, FRAUD_STATS, NON_FRAUD_STATS 



def download_creditcard_data():
    """Download the credit card fraud dataset from Kaggle using .env credentials."""
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
    """
    Returns a random fraud rate drawn from a beta distribution.
    
    The distribution is parameterized by the base rate (the mean of the distribution)
    and a scale parameter that controls the variance of the distribution.
    
    Parameters:
    base_rate (float): The mean of the distribution (default is 0.001727)
    scale (float): The scale parameter of the distribution (default is 10000)
    """
    # scale controls the variance (higher scale = lower variance)
    alpha = base_rate * scale
    beta = (1 - base_rate) * scale
    return np.random.beta(alpha, beta)


def generate_fraud_data(n_samples, fraud_rate=get_random_fraud_rate()):
    """
    Generate synthetic fraud detection data based on the provided dataset statistics.
    
    Parameters:
    n_samples (int): Number of samples to generate
    fraud_rate (float): Proportion of fraudulent transactions (default: 0.001727)
    
    Returns:
    pandas.DataFrame: Generated data with V1-V28 features and Amount
    """
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