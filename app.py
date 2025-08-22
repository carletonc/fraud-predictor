"""
Main Streamlit application for fraud detection monitoring.

This module serves as the entry point for the fraud detection dashboard,
orchestrating data download, model training, and the monitoring interface.
"""

import streamlit as st
#from streamlit_profiler import Profiler
import os

from src.constants import TRAIN_DATA_PATH, MODEL_PATH
from src.data import download_creditcard_data
from src.train import train_and_serialize
from src.inference import single_run, full_run
from src.ui import header_buttons, plot_feature_drift, plot_online_metrics


# Configure page
st.set_page_config(
    page_title="Fraud Detection Dashboard", 
    page_icon="🎯", 
    layout="wide", 
    initial_sidebar_state="collapsed" 
)

#def iniitalize_session_state():
for val in ['live_data', 'live_metrics', 'live_feature_drift']:
    if val not in st.session_state:
        st.session_state[val] = []


def main():
    """Main application entry point.
    
    Initializes the fraud detection dashboard, ensures required data and
    model artifacts are available, and displays the monitoring interface.
    Handles data download and model training if needed.
    
    Returns:
        None: Updates Streamlit interface.
    """
    st.title("🎯 Fraud Detection Dashboard")
    header_buttons()
    
    LIMIT = 14
    
    if not os.path.exists(TRAIN_DATA_PATH):
        with st.spinner("Credit card dataset not found. Downloading from Kaggle..."):
            download_creditcard_data()
            
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Model artifacts not found. Training and serializing..."):
            train_and_serialize()
    
    if len(st.session_state['live_data']) < LIMIT:
        with st.spinner("Generating online data..."):
            full_run(LIMIT)
    
    plot_online_metrics(limit=LIMIT)
    plot_feature_drift(limit=LIMIT)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        #st.error(f"App crashed: {e}")
        raise e
        import traceback
        #st.text(traceback.format_exc())
        print(traceback.format_exc())
    
    #with Profiler():
    # put your app code here
    #    main()