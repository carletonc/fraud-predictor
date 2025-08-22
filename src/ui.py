"""
Streamlit user interface for fraud prediction monitoring.

This module provides the web-based interface for monitoring fraud detection
model performance, including real-time metrics visualization, feature drift
analysis, and interactive data generation controls.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from src.constants import ABOUT, TOP_FEATURES
from src.inference import single_run


def header_buttons():
    """Display header section with app description and control buttons.
    
    Shows the application description and provides buttons for generating
    new data batches and clearing all stored data from the session state.
    
    Returns:
        None: Updates Streamlit interface elements.
    """
    st.write(ABOUT)
    
    col1, col2, col3, col4 = st.columns([1,1,1, 1]) 

    with col1:
        if st.button("Generate New Batch"):
            single_run()
            #st.rerun() 

    with col2:
        if st.button("Clear All Data"):
            # Clear session state or any data here
            for key in st.session_state.keys():
                 del st.session_state[key]
            st.rerun()
            
    with col3:
        #st.session_state['history_len'] = st.number_input(
        #    'History Length', 
        #    min_value=7, 
        #    max_value=28, 
        #    value=14,
            #width='stretch'
       #    )
        pass


@st.cache_data
def process_metrics_data(metrics_data, limit):
    """Process and cache metrics data for plotting.
    
    Converts the nested metrics data structure into a pandas DataFrame
    suitable for plotting. Results are cached to avoid reprocessing
    identical data.
    
    Args:
        metrics_data (list): List of metrics dictionaries from session state.
        limit (int): Number of most recent batches to process.
    
    Returns:
        pandas.DataFrame: Processed metrics data for plotting.
    """
    return pd.DataFrame(
        [pd.json_normalize(metrics, sep="_").iloc[0]
         for d in metrics_data[-limit:]
         for metrics in d.values()],
        index=[i for i in range(min(len(metrics_data), limit))]
    )


@st.cache_data
def process_drift_data(drift_data, limit):
    """Process and cache drift data for plotting.
    
    Converts the nested drift data structure into a pandas DataFrame
    suitable for plotting. Results are cached to avoid reprocessing
    identical data.
    
    Args:
        drift_data (list): List of drift dictionaries from session state.
        limit (int): Number of most recent batches to process.
    
    Returns:
        pandas.DataFrame: Processed drift data for plotting.
    """
    return pd.DataFrame(
        [pd.json_normalize(metrics, sep="_").iloc[0] 
         for d in drift_data[-limit:] 
         for metrics in d.values()],
        index=[i for i in range(min(len(drift_data), limit))]
    )


def plot_online_metrics(limit=14):
    """Plot online model performance metrics over time.
    
    Creates three interactive charts showing precision, recall, and sample
    statistics over multiple monitoring batches. Compares online performance
    to offline test set benchmarks.
    
    Args:
        limit (int): Number of most recent batches to display. Defaults to 14.
    
    Returns:
        None: Displays Plotly charts in Streamlit interface.
    """
    st.markdown("---")
    st.subheader("Model Monitoring")
    
    # Use cached data processing
    df = process_metrics_data(st.session_state['live_metrics'], limit)
    
    # Define colors for clarity and consistency
    precision_online_color = '#1f77b4'  # blue
    precision_offline_color = '#aec7e8'  # same blue, dashed
    recall_online_color = '#ff7f0e'  # orange
    recall_offline_color = '#ffbb78'  # same orange, dashed
    fraud_color = 'red'
    samples_color = 'lightgray'

    col1, col2, col3 = st.columns(3)

    # Precision
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df.index, 
            y=df['online_precision'],
            mode='lines',
            name='Online Precision',
            line=dict(color=precision_online_color, dash='solid')
        ))
        fig.add_trace(go.Scatter(
            x=df.index, 
            y=df['offline_test_precision'],
            mode='lines',
            name='Offline Precision',
            line=dict(color=precision_offline_color, dash='dash')
        ))
        fig.update_layout(
            title="Precision Over Time",
            xaxis_title="Batch",
            yaxis_title="Precision",
            yaxis=dict(range=[0, 1]),
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        st.plotly_chart(fig, use_container_width=True)

    # Recall
    with col2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df.index, 
            y=df['online_recall'],
            mode='lines',
            name='Online Recall',
            line=dict(color=recall_online_color, dash='solid')
        ))
        fig.add_trace(go.Scatter(
            x=df.index, 
            y=df['offline_test_recall'],
            mode='lines',
            name='Offline Recall',
            line=dict(color=recall_offline_color, dash='dash')
        ))
        fig.update_layout(
            title="Recall Over Time",
            xaxis_title="Batch",
            yaxis_title="Recall",
            yaxis=dict(range=[0, 1]),
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        st.plotly_chart(fig, use_container_width=True)

    # Samples
    with col3:
        fig = go.Figure()
        # Bar chart for total samples
        fig.add_trace(go.Bar(
            x=df.index,
            y=df["online_total_samples"],
            name="Total Samples",
            marker_color=samples_color,
            yaxis="y1"
        ))
        # Line chart for fraud detected
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df["online_total_true"],
            name="Fraud Detected",
            mode='lines+markers',
            marker_color=fraud_color,
            yaxis="y2"
        ))

        fig.update_layout(
            title="Samples & Fraud Detected Over Time",
            xaxis=dict(title="Batch"),
            yaxis=dict(
                title="Total Samples",
                showgrid=False,
                zeroline=False,
                side='left',
                range=[0, max(df["online_total_samples"]) * 1.1],
                showline=True,
            ),
            yaxis2=dict(
                title="Fraud Instances",
                overlaying='y',
                side='right',
                range=[0, max(df["online_total_true"]) * 1.1],
                showline=True,
            ),
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            bargap=0.2,
        )
        st.plotly_chart(fig, use_container_width=True)
        

def plot_feature_drift(limit=14):
    """Plot feature drift analysis over time.
    
    Creates interactive visualizations showing how feature distributions
    change over monitoring batches, including Jensen-Shannon Divergence
    trends and histogram overlays comparing offline vs online distributions.
    
    Args:
        limit (int): Number of most recent batches to display. Defaults to 14.
    
    Returns:
        None: Displays Plotly charts in Streamlit interface.
    """
    st.markdown("---")
    st.subheader("Feature Drift")

    # Use cached data processing
    df = process_drift_data(st.session_state['live_feature_drift'], limit)

    cols = st.columns(len(TOP_FEATURES))
    latest_idx = df.index[-1]
    
    def get_step_points(bin_edges, counts):
        """Convert histogram data to step plot format.
        
        Args:
            bin_edges (list): Histogram bin edge values.
            counts (list): Histogram count values.
        
        Returns:
            tuple: (x_coordinates, y_coordinates) for step plot.
        """
        x = []
        y = []
        for i in range(len(counts)):
            x.extend([bin_edges[i], bin_edges[i+1]])
            y.extend([counts[i], counts[i]])
        return x, y

    for i, feature in enumerate(TOP_FEATURES):
        col = cols[i]

        jsd_col = f"{feature}_jsd"
        bins_col = f"{feature}_bins"
        live_hist_col = f"{feature}_live_hist"
        test_hist_col = f"{feature}_test_hist"

        with col:
            # Line chart for JSD drift over batches
            fig_line = go.Figure()
            if jsd_col in df.columns:
                fig_line.add_trace(go.Scatter(
                    x=df.index,
                    y=df[jsd_col],
                    mode='lines+markers',
                    name=f"{feature} JSD Drift"
                ))
                fig_line.update_layout(
                    title=f"`{feature}` Drift (JSD)",
                    xaxis_title="Batch",
                    yaxis_title="JSD",
                    yaxis=dict(range=[0, 0.5]),
                    height=300
                )
                st.plotly_chart(fig_line, use_container_width=True)
            else:
                st.warning(f"Column '{jsd_col}' not found in data.")

            # Histogram overlay for latest batch distributions
            if all(c in df.columns for c in [bins_col, live_hist_col, test_hist_col]):
                bins = np.array(df[bins_col].iloc[-1])
                live_hist = np.array(df[live_hist_col].iloc[-1])
                test_hist = np.array(df[test_hist_col].iloc[-1])

                x_test, y_test = get_step_points(bins, test_hist)
                x_live, y_live = get_step_points(bins, live_hist)

                fig_hist = go.Figure()

                fig_hist.add_trace(go.Scatter(
                    x=x_test,
                    y=y_test,
                    mode='lines',
                    name='Offline',
                    line=dict(color='lightblue'),
                    fill='tozeroy',
                    fillcolor='rgba(173, 216, 230, 0.5)',  # semi-transparent lightblue
                ))
                fig_hist.add_trace(go.Scatter(
                    x=x_live,
                    y=y_live,
                    mode='lines',
                    name='Online',
                    line=dict(color='orange'),
                    fill='tozeroy',
                    fillcolor='rgba(255, 165, 0, 0.5)',  # semi-transparent orange
                ))
                fig_hist.update_layout(
                    title=f"`{feature}` Latest Batch Distribution",
                    xaxis_title="Distribution",
                    yaxis_title="Probability Density",
                    height=300,
                )
                st.plotly_chart(fig_hist, use_container_width=True)
            else:
                st.info(f"Histogram data columns for '{feature}' not found.")