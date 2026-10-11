"""
Streamlit web interface for time series analysis.

This module provides a web-based interface for exploring time series
data, running forecasts, and visualizing results.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml
from plotly.subplots import make_subplots

from src.app import TimeSeriesAnalysisApp
from src.data_utils import DataGenerator, DataLoader
from src.models import AnomalyDetector, LSTMForecaster, ProphetForecaster
from src.visualization import InteractiveVisualizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Time Series Analysis Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #1f77b4;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #c3e6cb;
    }
</style>
""", unsafe_allow_html=True)


def load_config() -> dict:
    """Load configuration from YAML file."""
    try:
        with open("config/config.yaml", 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        st.error(f"Error loading configuration: {e}")
        return {}


def initialize_app() -> Optional[TimeSeriesAnalysisApp]:
    """Initialize the time series analysis app."""
    try:
        app = TimeSeriesAnalysisApp()
        return app
    except Exception as e:
        st.error(f"Error initializing app: {e}")
        return None


def display_data_overview(data: pd.DataFrame) -> None:
    """Display data overview."""
    st.subheader("📊 Data Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Data Points", len(data))
    
    with col2:
        st.metric("Date Range", f"{data['ds'].min().strftime('%Y-%m-%d')} to {data['ds'].max().strftime('%Y-%m-%d')}")
    
    with col3:
        st.metric("Mean Value", f"{data['y'].mean():.2f}")
    
    with col4:
        st.metric("Std Deviation", f"{data['y'].std():.2f}")
    
    # Display data table
    st.subheader("📋 Data Preview")
    st.dataframe(data.head(10), use_container_width=True)
    
    # Display basic statistics
    st.subheader("📈 Data Statistics")
    st.dataframe(data['y'].describe(), use_container_width=True)


def display_forecast_results(forecasts: Dict, train_data: pd.DataFrame) -> None:
    """Display forecast results."""
    st.subheader("🔮 Forecast Results")
    
    if not forecasts:
        st.warning("No forecasts available. Please run the analysis first.")
        return
    
    # Model selection
    selected_model = st.selectbox("Select Model to View", list(forecasts.keys()))
    
    if selected_model:
        forecast = forecasts[selected_model]
        
        # Create interactive plot
        interactive_viz = InteractiveVisualizer()
        fig = interactive_viz.create_interactive_forecast(
            train_data,
            forecast,
            title=f"{selected_model} Forecast"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display forecast table
        st.subheader(f"📊 {selected_model} Forecast Data")
        st.dataframe(forecast, use_container_width=True)


def display_model_comparison(forecasts: Dict, metrics: Dict, train_data: pd.DataFrame) -> None:
    """Display model comparison."""
    st.subheader("⚖️ Model Comparison")
    
    if len(forecasts) < 2:
        st.warning("Need at least 2 models for comparison.")
        return
    
    # Create comparison plot
    interactive_viz = InteractiveVisualizer()
    fig = interactive_viz.create_model_comparison_dashboard(
        train_data,
        forecasts,
        metrics,
        title="Model Comparison Dashboard"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display metrics comparison
    if metrics:
        st.subheader("📊 Performance Metrics")
        metrics_df = pd.DataFrame(metrics).T
        st.dataframe(metrics_df, use_container_width=True)
        
        # Highlight best model
        best_model = min(metrics.keys(), key=lambda x: metrics[x]['RMSE'])
        st.success(f"🏆 Best performing model: **{best_model}** (lowest RMSE)")


def display_anomaly_results(anomaly_results: Dict, data: pd.DataFrame) -> None:
    """Display anomaly detection results."""
    st.subheader("🚨 Anomaly Detection Results")
    
    if not anomaly_results:
        st.warning("No anomaly detection results available.")
        return
    
    # Method selection
    selected_method = st.selectbox("Select Detection Method", list(anomaly_results.keys()))
    
    if selected_method:
        results = anomaly_results[selected_method]
        anomaly_indices = results['indices']
        
        # Display anomaly statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Anomalies", len(anomaly_indices))
        
        with col2:
            anomaly_rate = len(anomaly_indices) / len(data) * 100
            st.metric("Anomaly Rate", f"{anomaly_rate:.2f}%")
        
        with col3:
            if anomaly_indices:
                anomaly_values = data.iloc[anomaly_indices]['y']
                st.metric("Avg Anomaly Value", f"{anomaly_values.mean():.2f}")
        
        # Create anomaly plot
        fig = go.Figure()
        
        # Plot normal data
        normal_data = data.drop(index=anomaly_indices)
        fig.add_trace(go.Scatter(
            x=normal_data['ds'],
            y=normal_data['y'],
            mode='lines',
            name='Normal',
            line=dict(color='blue', width=1)
        ))
        
        # Plot anomalies
        if anomaly_indices:
            anomaly_data = data.iloc[anomaly_indices]
            fig.add_trace(go.Scatter(
                x=anomaly_data['ds'],
                y=anomaly_data['y'],
                mode='markers',
                name='Anomalies',
                marker=dict(color='red', size=8)
            ))
        
        fig.update_layout(
            title=f"Anomaly Detection - {selected_method.title()}",
            xaxis_title="Date",
            yaxis_title="Value",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display anomaly details
        if anomaly_indices:
            st.subheader("🔍 Anomaly Details")
            anomaly_data = data.iloc[anomaly_indices]
            st.dataframe(anomaly_data, use_container_width=True)


def main():
    """Main Streamlit application."""
    # Header
    st.markdown('<h1 class="main-header">📈 Time Series Analysis Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("⚙️ Configuration")
    
    # Load configuration
    config = load_config()
    if not config:
        st.error("Failed to load configuration. Please check config/config.yaml")
        return
    
    # Data source selection
    st.sidebar.subheader("📊 Data Source")
    data_source = st.sidebar.selectbox(
        "Select Data Source",
        ["synthetic", "yfinance"],
        index=0
    )
    
    # Model selection
    st.sidebar.subheader("🤖 Models")
    enable_prophet = st.sidebar.checkbox("Prophet", value=True)
    enable_lstm = st.sidebar.checkbox("LSTM", value=True)
    
    # Analysis parameters
    st.sidebar.subheader("📈 Analysis Parameters")
    forecast_periods = st.sidebar.slider("Forecast Periods", 10, 100, 60)
    test_size = st.sidebar.slider("Test Size", 0.1, 0.5, 0.2)
    
    # Anomaly detection
    st.sidebar.subheader("🚨 Anomaly Detection")
    enable_anomaly_detection = st.sidebar.checkbox("Enable Anomaly Detection", value=True)
    anomaly_contamination = st.sidebar.slider("Anomaly Contamination", 0.01, 0.2, 0.1)
    
    # Run analysis button
    if st.sidebar.button("🚀 Run Analysis", type="primary"):
        with st.spinner("Running analysis..."):
            try:
                # Initialize app
                app = initialize_app()
                if not app:
                    return
                
                # Update config based on sidebar selections
                app.config['data']['source'] = data_source
                app.config['data']['forecast_periods'] = forecast_periods
                app.config['data']['test_size'] = test_size
                app.config['models']['prophet']['enabled'] = enable_prophet
                app.config['models']['lstm']['enabled'] = enable_lstm
                app.config['anomaly_detection']['enabled'] = enable_anomaly_detection
                app.config['anomaly_detection']['contamination'] = anomaly_contamination
                
                # Run analysis
                app.run_complete_analysis()
                
                # Store results in session state
                st.session_state['app'] = app
                st.session_state['analysis_complete'] = True
                
                st.success("✅ Analysis completed successfully!")
                
            except Exception as e:
                st.error(f"❌ Error during analysis: {e}")
                logger.error(f"Analysis error: {e}")
    
    # Display results if analysis is complete
    if st.session_state.get('analysis_complete', False):
        app = st.session_state['app']
        
        # Create tabs for different views
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Data Overview", "🔮 Forecasts", "⚖️ Model Comparison", 
            "🚨 Anomalies", "📋 Summary"
        ])
        
        with tab1:
            display_data_overview(app.data)
        
        with tab2:
            display_forecast_results(app.forecasts, app.train_data)
        
        with tab3:
            display_model_comparison(app.forecasts, app.metrics, app.train_data)
        
        with tab4:
            if hasattr(app, 'anomaly_results'):
                display_anomaly_results(app.anomaly_results, app.data)
            else:
                st.info("No anomaly detection results available.")
        
        with tab5:
            summary = app.get_summary()
            
            st.subheader("📋 Analysis Summary")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Data Points", summary['data_points'])
                st.metric("Training Points", summary['train_points'])
                st.metric("Test Points", summary['test_points'])
            
            with col2:
                st.metric("Forecast Periods", summary['forecast_periods'])
                st.metric("Models Trained", len(summary['models_trained']))
                if 'best_model' in summary:
                    st.metric("Best Model", summary['best_model'])
            
            # Display detailed metrics
            if 'metrics' in summary:
                st.subheader("📊 Model Performance")
                metrics_df = pd.DataFrame(summary['metrics']).T
                st.dataframe(metrics_df, use_container_width=True)
            
            # Display anomaly summary
            if 'anomalies_detected' in summary:
                st.subheader("🚨 Anomaly Detection Summary")
                for method, count in summary['anomalies_detected'].items():
                    st.write(f"**{method.title()}**: {count} anomalies detected")
    
    else:
        # Welcome message
        st.markdown("""
        ## Welcome to the Time Series Analysis Dashboard! 🎉
        
        This dashboard provides a comprehensive interface for time series analysis, including:
        
        - **Data Loading**: Support for synthetic data generation and real-world data sources
        - **Multiple Models**: Prophet, LSTM, and other forecasting models
        - **Anomaly Detection**: Identify unusual patterns in your data
        - **Interactive Visualizations**: Explore your data and results
        - **Model Comparison**: Compare different forecasting approaches
        
        ### Getting Started
        
        1. Configure your analysis parameters in the sidebar
        2. Select your data source and models
        3. Click "Run Analysis" to start the process
        4. Explore the results in the different tabs
        
        ### Features
        
        - **Prophet**: Facebook's robust forecasting tool for business time series
        - **LSTM**: Deep learning approach for complex patterns
        - **Anomaly Detection**: Identify outliers and unusual events
        - **Interactive Plots**: Zoom, pan, and explore your data
        - **Performance Metrics**: Compare model accuracy
        
        Ready to get started? Configure your settings in the sidebar and run your first analysis!
        """)


if __name__ == "__main__":
    main()
