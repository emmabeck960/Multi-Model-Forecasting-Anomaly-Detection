"""
Data generation and loading utilities for time series analysis.

This module provides functions to generate synthetic time series data
and load real-world datasets for forecasting and anomaly detection.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    yf = None

logger = logging.getLogger(__name__)


class DataGenerator:
    """Generate synthetic time series data with various patterns."""
    
    def __init__(self, random_state: int = 42):
        """Initialize the data generator.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        np.random.seed(random_state)
    
    def generate_trend_data(
        self,
        start_date: str = "2020-01-01",
        end_date: str = "2023-01-01",
        trend_slope: float = 0.1,
        noise_level: float = 0.1,
        frequency: str = "D"
    ) -> pd.DataFrame:
        """Generate time series data with linear trend.
        
        Args:
            start_date: Start date for the time series
            end_date: End date for the time series
            trend_slope: Slope of the linear trend
            noise_level: Standard deviation of noise
            frequency: Frequency of data points
            
        Returns:
            DataFrame with columns ['ds', 'y']
        """
        date_range = pd.date_range(start=start_date, end=end_date, freq=frequency)
        n_points = len(date_range)
        
        # Generate trend
        trend = np.linspace(0, trend_slope * n_points, n_points)
        
        # Add noise
        noise = np.random.normal(0, noise_level, n_points)
        
        # Combine trend and noise
        values = trend + noise
        
        return pd.DataFrame({
            'ds': date_range,
            'y': values
        })
    
    def generate_seasonal_data(
        self,
        start_date: str = "2020-01-01",
        end_date: str = "2023-01-01",
        amplitude: float = 1.0,
        period: int = 365,
        noise_level: float = 0.1,
        frequency: str = "D"
    ) -> pd.DataFrame:
        """Generate time series data with seasonal patterns.
        
        Args:
            start_date: Start date for the time series
            end_date: End date for the time series
            amplitude: Amplitude of seasonal pattern
            period: Period of seasonal pattern in days
            noise_level: Standard deviation of noise
            frequency: Frequency of data points
            
        Returns:
            DataFrame with columns ['ds', 'y']
        """
        date_range = pd.date_range(start=start_date, end=end_date, freq=frequency)
        n_points = len(date_range)
        
        # Generate seasonal pattern
        t = np.arange(n_points)
        seasonal = amplitude * np.sin(2 * np.pi * t / period)
        
        # Add noise
        noise = np.random.normal(0, noise_level, n_points)
        
        # Combine seasonal pattern and noise
        values = seasonal + noise
        
        return pd.DataFrame({
            'ds': date_range,
            'y': values
        })
    
    def generate_complex_data(
        self,
        start_date: str = "2020-01-01",
        end_date: str = "2023-01-01",
        trend_slope: float = 0.05,
        seasonal_amplitude: float = 2.0,
        seasonal_period: int = 365,
        weekly_amplitude: float = 0.5,
        weekly_period: int = 7,
        noise_level: float = 0.2,
        frequency: str = "D"
    ) -> pd.DataFrame:
        """Generate complex time series with trend, seasonality, and noise.
        
        Args:
            start_date: Start date for the time series
            end_date: End date for the time series
            trend_slope: Slope of the linear trend
            seasonal_amplitude: Amplitude of yearly seasonal pattern
            seasonal_period: Period of yearly seasonal pattern
            weekly_amplitude: Amplitude of weekly seasonal pattern
            weekly_period: Period of weekly seasonal pattern
            noise_level: Standard deviation of noise
            frequency: Frequency of data points
            
        Returns:
            DataFrame with columns ['ds', 'y']
        """
        date_range = pd.date_range(start=start_date, end=end_date, freq=frequency)
        n_points = len(date_range)
        
        # Generate trend
        t = np.arange(n_points)
        trend = trend_slope * t
        
        # Generate yearly seasonality
        yearly_seasonal = seasonal_amplitude * np.sin(2 * np.pi * t / seasonal_period)
        
        # Generate weekly seasonality
        weekly_seasonal = weekly_amplitude * np.sin(2 * np.pi * t / weekly_period)
        
        # Add noise
        noise = np.random.normal(0, noise_level, n_points)
        
        # Combine all components
        values = trend + yearly_seasonal + weekly_seasonal + noise
        
        return pd.DataFrame({
            'ds': date_range,
            'y': values
        })
    
    def add_anomalies(
        self,
        data: pd.DataFrame,
        anomaly_rate: float = 0.05,
        anomaly_magnitude: float = 3.0
    ) -> Tuple[pd.DataFrame, List[int]]:
        """Add anomalies to time series data.
        
        Args:
            data: Input time series data
            anomaly_rate: Proportion of points to make anomalous
            anomaly_magnitude: Magnitude of anomalies (in standard deviations)
            
        Returns:
            Tuple of (data_with_anomalies, anomaly_indices)
        """
        data_copy = data.copy()
        n_points = len(data_copy)
        n_anomalies = int(n_points * anomaly_rate)
        
        # Randomly select points to make anomalous
        anomaly_indices = np.random.choice(
            n_points, size=n_anomalies, replace=False
        )
        
        # Calculate anomaly magnitude based on data standard deviation
        std_dev = data_copy['y'].std()
        anomaly_shift = anomaly_magnitude * std_dev
        
        # Add anomalies (randomly positive or negative)
        for idx in anomaly_indices:
            direction = np.random.choice([-1, 1])
            data_copy.loc[idx, 'y'] += direction * anomaly_shift
        
        return data_copy, list(anomaly_indices)


class DataLoader:
    """Load real-world time series datasets."""
    
    @staticmethod
    def load_yfinance_data(
        symbol: str,
        start_date: str,
        end_date: str,
        column: str = "Close"
    ) -> pd.DataFrame:
        """Load stock data from Yahoo Finance.
        
        Args:
            symbol: Stock symbol (e.g., 'GOOG', 'AAPL')
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            column: Column to use for forecasting
            
        Returns:
            DataFrame with columns ['ds', 'y']
        """
        if not YFINANCE_AVAILABLE:
            raise ImportError("yfinance is not installed. Please install it with: pip install yfinance")
        
        try:
            logger.info(f"Loading {symbol} data from {start_date} to {end_date}")
            df = yf.download(symbol, start=start_date, end=end_date)
            
            if df.empty:
                raise ValueError(f"No data found for symbol {symbol}")
            
            # Reset index and select columns
            data = df.reset_index()[["Date", column]]
            data.columns = ["ds", "y"]
            
            # Remove any NaN values
            data = data.dropna()
            
            logger.info(f"Loaded {len(data)} data points")
            return data
            
        except Exception as e:
            logger.error(f"Error loading yfinance data: {e}")
            raise
    
    @staticmethod
    def load_csv_data(file_path: str, date_column: str, value_column: str) -> pd.DataFrame:
        """Load time series data from CSV file.
        
        Args:
            file_path: Path to CSV file
            date_column: Name of date column
            value_column: Name of value column
            
        Returns:
            DataFrame with columns ['ds', 'y']
        """
        try:
            logger.info(f"Loading data from {file_path}")
            df = pd.read_csv(file_path)
            
            # Select and rename columns
            data = df[[date_column, value_column]].copy()
            data.columns = ["ds", "y"]
            
            # Convert date column to datetime
            data['ds'] = pd.to_datetime(data['ds'])
            
            # Remove any NaN values
            data = data.dropna()
            
            logger.info(f"Loaded {len(data)} data points")
            return data
            
        except Exception as e:
            logger.error(f"Error loading CSV data: {e}")
            raise


def prepare_data_for_modeling(
    data: pd.DataFrame,
    test_size: float = 0.2,
    scale: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[StandardScaler]]:
    """Prepare data for modeling by splitting into train/test sets.
    
    Args:
        data: Input time series data
        test_size: Proportion of data to use for testing
        scale: Whether to scale the data
        
    Returns:
        Tuple of (train_data, test_data, scaler)
    """
    # Sort by date
    data = data.sort_values('ds').reset_index(drop=True)
    
    # Calculate split point
    split_idx = int(len(data) * (1 - test_size))
    
    # Split data
    train_data = data.iloc[:split_idx].copy()
    test_data = data.iloc[split_idx:].copy()
    
    scaler = None
    if scale:
        scaler = StandardScaler()
        train_data['y'] = scaler.fit_transform(train_data[['y']]).flatten()
        test_data['y'] = scaler.transform(test_data[['y']]).flatten()
    
    logger.info(f"Split data: {len(train_data)} train, {len(test_data)} test")
    
    return train_data, test_data, scaler
