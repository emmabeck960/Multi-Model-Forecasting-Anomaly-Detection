"""
Time Series Analysis Package

A comprehensive Python package for time series forecasting and anomaly detection.
"""

__version__ = "1.0.0"
__author__ = "Time Series Analysis Team"
__email__ = "contact@timeseriesanalysis.com"

from .app import TimeSeriesAnalysisApp
from .data_utils import DataGenerator, DataLoader, prepare_data_for_modeling
from .models import ProphetForecaster, LSTMForecaster, AnomalyDetector, calculate_metrics
from .visualization import TimeSeriesVisualizer, InteractiveVisualizer

__all__ = [
    "TimeSeriesAnalysisApp",
    "DataGenerator",
    "DataLoader", 
    "prepare_data_for_modeling",
    "ProphetForecaster",
    "LSTMForecaster",
    "AnomalyDetector",
    "calculate_metrics",
    "TimeSeriesVisualizer",
    "InteractiveVisualizer"
]
