"""
Visualization utilities for time series analysis.

This module provides functions to create comprehensive plots for
time series data, forecasts, and anomaly detection results.
"""

import logging
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)


class TimeSeriesVisualizer:
    """Create visualizations for time series analysis."""
    
    def __init__(self, style: str = "seaborn-v0_8", figure_size: Tuple[int, int] = (12, 8)):
        """Initialize the visualizer.
        
        Args:
            style: Matplotlib style to use
            figure_size: Default figure size (width, height)
        """
        self.style = style
        self.figure_size = figure_size
        plt.style.use(style)
    
    def plot_time_series(
        self,
        data: pd.DataFrame,
        title: str = "Time Series Data",
        xlabel: str = "Date",
        ylabel: str = "Value",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot basic time series data.
        
        Args:
            data: Time series data with columns ['ds', 'y']
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        ax.plot(data['ds'], data['y'], linewidth=1.5, alpha=0.8)
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        return fig
    
    def plot_forecast(
        self,
        historical_data: pd.DataFrame,
        forecast_data: pd.DataFrame,
        title: str = "Time Series Forecast",
        xlabel: str = "Date",
        ylabel: str = "Value",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot historical data with forecast.
        
        Args:
            historical_data: Historical data with columns ['ds', 'y']
            forecast_data: Forecast data with columns ['ds', 'yhat', 'yhat_lower', 'yhat_upper']
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        # Plot historical data
        ax.plot(historical_data['ds'], historical_data['y'], 
                label='Historical', linewidth=2, color='blue', alpha=0.8)
        
        # Plot forecast
        ax.plot(forecast_data['ds'], forecast_data['yhat'], 
                label='Forecast', linewidth=2, color='red', alpha=0.8)
        
        # Plot confidence intervals
        ax.fill_between(forecast_data['ds'], 
                       forecast_data['yhat_lower'], 
                       forecast_data['yhat_upper'],
                       alpha=0.3, color='red', label='Confidence Interval')
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Forecast plot saved to {save_path}")
        
        return fig
    
    def plot_anomalies(
        self,
        data: pd.DataFrame,
        anomaly_indices: List[int],
        title: str = "Anomaly Detection",
        xlabel: str = "Date",
        ylabel: str = "Value",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot time series with highlighted anomalies.
        
        Args:
            data: Time series data with columns ['ds', 'y']
            anomaly_indices: List of anomaly indices
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        # Plot normal data
        normal_data = data.drop(index=anomaly_indices)
        ax.plot(normal_data['ds'], normal_data['y'], 
                label='Normal', linewidth=1.5, color='blue', alpha=0.7)
        
        # Plot anomalies
        if anomaly_indices:
            anomaly_data = data.iloc[anomaly_indices]
            ax.scatter(anomaly_data['ds'], anomaly_data['y'], 
                      label='Anomalies', color='red', s=50, alpha=0.8, zorder=5)
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Anomaly plot saved to {save_path}")
        
        return fig
    
    def plot_model_comparison(
        self,
        historical_data: pd.DataFrame,
        forecasts: dict,
        title: str = "Model Comparison",
        xlabel: str = "Date",
        ylabel: str = "Value",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot comparison of multiple forecasting models.
        
        Args:
            historical_data: Historical data with columns ['ds', 'y']
            forecasts: Dictionary of model forecasts {model_name: forecast_data}
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        # Plot historical data
        ax.plot(historical_data['ds'], historical_data['y'], 
                label='Historical', linewidth=2, color='black', alpha=0.8)
        
        # Plot forecasts
        colors = ['red', 'green', 'blue', 'orange', 'purple']
        for i, (model_name, forecast_data) in enumerate(forecasts.items()):
            color = colors[i % len(colors)]
            ax.plot(forecast_data['ds'], forecast_data['yhat'], 
                    label=f'{model_name} Forecast', linewidth=2, 
                    color=color, alpha=0.8)
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Model comparison plot saved to {save_path}")
        
        return fig
    
    def plot_decomposition(
        self,
        data: pd.DataFrame,
        trend: Optional[np.ndarray] = None,
        seasonal: Optional[np.ndarray] = None,
        residual: Optional[np.ndarray] = None,
        title: str = "Time Series Decomposition",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot time series decomposition.
        
        Args:
            data: Time series data with columns ['ds', 'y']
            trend: Trend component
            seasonal: Seasonal component
            residual: Residual component
            title: Plot title
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(4, 1, figsize=(self.figure_size[0], self.figure_size[1] * 1.5))
        
        # Original data
        axes[0].plot(data['ds'], data['y'], linewidth=1.5, color='blue')
        axes[0].set_title('Original', fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # Trend
        if trend is not None:
            axes[1].plot(data['ds'], trend, linewidth=1.5, color='green')
            axes[1].set_title('Trend', fontweight='bold')
            axes[1].grid(True, alpha=0.3)
        
        # Seasonal
        if seasonal is not None:
            axes[2].plot(data['ds'], seasonal, linewidth=1.5, color='orange')
            axes[2].set_title('Seasonal', fontweight='bold')
            axes[2].grid(True, alpha=0.3)
        
        # Residual
        if residual is not None:
            axes[3].plot(data['ds'], residual, linewidth=1.5, color='red')
            axes[3].set_title('Residual', fontweight='bold')
            axes[3].grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Decomposition plot saved to {save_path}")
        
        return fig


class InteractiveVisualizer:
    """Create interactive visualizations using Plotly."""
    
    def __init__(self):
        """Initialize the interactive visualizer."""
        pass
    
    def create_interactive_forecast(
        self,
        historical_data: pd.DataFrame,
        forecast_data: pd.DataFrame,
        title: str = "Interactive Time Series Forecast"
    ) -> go.Figure:
        """Create interactive forecast plot.
        
        Args:
            historical_data: Historical data with columns ['ds', 'y']
            forecast_data: Forecast data with columns ['ds', 'yhat', 'yhat_lower', 'yhat_upper']
            title: Plot title
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        # Add historical data
        fig.add_trace(go.Scatter(
            x=historical_data['ds'],
            y=historical_data['y'],
            mode='lines',
            name='Historical',
            line=dict(color='blue', width=2)
        ))
        
        # Add forecast
        fig.add_trace(go.Scatter(
            x=forecast_data['ds'],
            y=forecast_data['yhat'],
            mode='lines',
            name='Forecast',
            line=dict(color='red', width=2)
        ))
        
        # Add confidence interval
        fig.add_trace(go.Scatter(
            x=forecast_data['ds'],
            y=forecast_data['yhat_upper'],
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        fig.add_trace(go.Scatter(
            x=forecast_data['ds'],
            y=forecast_data['yhat_lower'],
            mode='lines',
            fill='tonexty',
            fillcolor='rgba(255,0,0,0.2)',
            line=dict(width=0),
            name='Confidence Interval',
            hoverinfo='skip'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Date',
            yaxis_title='Value',
            hovermode='x unified',
            template='plotly_white'
        )
        
        return fig
    
    def create_model_comparison_dashboard(
        self,
        historical_data: pd.DataFrame,
        forecasts: dict,
        metrics: dict,
        title: str = "Model Comparison Dashboard"
    ) -> go.Figure:
        """Create interactive model comparison dashboard.
        
        Args:
            historical_data: Historical data
            forecasts: Dictionary of model forecasts
            metrics: Dictionary of model metrics
            title: Dashboard title
            
        Returns:
            Plotly figure with subplots
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Forecasts', 'Metrics Comparison', 'Error Analysis', 'Model Performance'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Plot forecasts
        fig.add_trace(go.Scatter(
            x=historical_data['ds'],
            y=historical_data['y'],
            mode='lines',
            name='Historical',
            line=dict(color='black', width=2)
        ), row=1, col=1)
        
        colors = ['red', 'green', 'blue', 'orange', 'purple']
        for i, (model_name, forecast_data) in enumerate(forecasts.items()):
            color = colors[i % len(colors)]
            fig.add_trace(go.Scatter(
                x=forecast_data['ds'],
                y=forecast_data['yhat'],
                mode='lines',
                name=f'{model_name} Forecast',
                line=dict(color=color, width=2)
            ), row=1, col=1)
        
        # Plot metrics
        model_names = list(metrics.keys())
        metric_names = list(next(iter(metrics.values())).keys())
        
        for metric in metric_names:
            values = [metrics[model][metric] for model in model_names]
            fig.add_trace(go.Bar(
                x=model_names,
                y=values,
                name=metric,
                showlegend=False
            ), row=1, col=2)
        
        fig.update_layout(
            title=title,
            height=800,
            template='plotly_white'
        )
        
        return fig
