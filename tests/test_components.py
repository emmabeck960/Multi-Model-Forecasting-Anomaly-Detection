"""
Unit tests for time series analysis components.

This module contains tests for data utilities, models, and visualization components.
"""

import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.data_utils import DataGenerator, DataLoader, prepare_data_for_modeling
from src.models import ProphetForecaster, LSTMForecaster, AnomalyDetector, calculate_metrics
from src.app import TimeSeriesAnalysisApp


class TestDataGenerator(unittest.TestCase):
    """Test cases for DataGenerator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.generator = DataGenerator(random_state=42)
    
    def test_generate_trend_data(self):
        """Test trend data generation."""
        data = self.generator.generate_trend_data(
            start_date="2020-01-01",
            end_date="2020-01-10",
            trend_slope=1.0,
            noise_level=0.1
        )
        
        self.assertEqual(len(data), 10)
        self.assertIn('ds', data.columns)
        self.assertIn('y', data.columns)
        self.assertTrue(data['ds'].is_monotonic_increasing)
    
    def test_generate_seasonal_data(self):
        """Test seasonal data generation."""
        data = self.generator.generate_seasonal_data(
            start_date="2020-01-01",
            end_date="2020-01-10",
            amplitude=2.0,
            period=7
        )
        
        self.assertEqual(len(data), 10)
        self.assertIn('ds', data.columns)
        self.assertIn('y', data.columns)
    
    def test_generate_complex_data(self):
        """Test complex data generation."""
        data = self.generator.generate_complex_data(
            start_date="2020-01-01",
            end_date="2020-01-10"
        )
        
        self.assertEqual(len(data), 10)
        self.assertIn('ds', data.columns)
        self.assertIn('y', data.columns)
    
    def test_add_anomalies(self):
        """Test anomaly addition."""
        data = self.generator.generate_trend_data(
            start_date="2020-01-01",
            end_date="2020-01-10"
        )
        
        data_with_anomalies, anomaly_indices = self.generator.add_anomalies(
            data, anomaly_rate=0.2, anomaly_magnitude=2.0
        )
        
        self.assertEqual(len(anomaly_indices), 2)  # 20% of 10 points
        self.assertEqual(len(data_with_anomalies), len(data))


class TestDataLoader(unittest.TestCase):
    """Test cases for DataLoader class."""
    
    @patch('src.data_utils.yf.download')
    def test_load_yfinance_data(self, mock_download):
        """Test loading data from Yahoo Finance."""
        # Mock the yfinance response
        mock_df = pd.DataFrame({
            'Date': pd.date_range('2020-01-01', periods=5),
            'Close': [100, 101, 102, 103, 104]
        })
        mock_df.set_index('Date', inplace=True)
        mock_download.return_value = mock_df
        
        data = DataLoader.load_yfinance_data(
            symbol="TEST",
            start_date="2020-01-01",
            end_date="2020-01-05"
        )
        
        self.assertEqual(len(data), 5)
        self.assertIn('ds', data.columns)
        self.assertIn('y', data.columns)
        mock_download.assert_called_once_with("TEST", start="2020-01-01", end="2020-01-05")


class TestProphetForecaster(unittest.TestCase):
    """Test cases for ProphetForecaster class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.forecaster = ProphetForecaster()
        self.data = pd.DataFrame({
            'ds': pd.date_range('2020-01-01', periods=100),
            'y': np.random.randn(100).cumsum()
        })
    
    def test_initialization(self):
        """Test forecaster initialization."""
        self.assertEqual(self.forecaster.name, "Prophet")
        self.assertFalse(self.forecaster.is_fitted)
    
    @patch('src.models.Prophet')
    def test_fit(self, mock_prophet):
        """Test model fitting."""
        mock_model = MagicMock()
        mock_prophet.return_value = mock_model
        
        self.forecaster.fit(self.data)
        
        self.assertTrue(self.forecaster.is_fitted)
        mock_model.fit.assert_called_once_with(self.data)
    
    def test_predict_without_fit(self):
        """Test prediction without fitting."""
        with self.assertRaises(ValueError):
            self.forecaster.predict(10)


class TestLSTMForecaster(unittest.TestCase):
    """Test cases for LSTMForecaster class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.forecaster = LSTMForecaster(
            sequence_length=10,
            epochs=1,  # Reduced for testing
            batch_size=32
        )
        self.data = pd.DataFrame({
            'ds': pd.date_range('2020-01-01', periods=50),
            'y': np.random.randn(50).cumsum()
        })
    
    def test_initialization(self):
        """Test forecaster initialization."""
        self.assertEqual(self.forecaster.name, "LSTM")
        self.assertFalse(self.forecaster.is_fitted)
    
    def test_create_sequences(self):
        """Test sequence creation."""
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        X, y = self.forecaster._create_sequences(data)
        
        self.assertEqual(len(X), len(data) - self.forecaster.sequence_length)
        self.assertEqual(len(y), len(data) - self.forecaster.sequence_length)
        self.assertEqual(X.shape[1], self.forecaster.sequence_length)


class TestAnomalyDetector(unittest.TestCase):
    """Test cases for AnomalyDetector class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.detector = AnomalyDetector()
        self.data = pd.DataFrame({
            'ds': pd.date_range('2020-01-01', periods=100),
            'y': np.random.randn(100)
        })
    
    def test_initialization(self):
        """Test detector initialization."""
        self.assertEqual(self.detector.method, "isolation_forest")
        self.assertEqual(self.detector.contamination, 0.1)
        self.assertFalse(self.detector.is_fitted)
    
    def test_detect_without_fit(self):
        """Test detection without fitting."""
        with self.assertRaises(ValueError):
            self.detector.detect(self.data)


class TestCalculateMetrics(unittest.TestCase):
    """Test cases for metrics calculation."""
    
    def test_calculate_metrics(self):
        """Test metrics calculation."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 1.9, 3.1, 3.9, 5.1])
        
        metrics = calculate_metrics(y_true, y_pred)
        
        self.assertIn('MAE', metrics)
        self.assertIn('MSE', metrics)
        self.assertIn('RMSE', metrics)
        self.assertIn('MAPE', metrics)
        
        # All metrics should be positive
        for metric, value in metrics.items():
            self.assertGreaterEqual(value, 0)


class TestPrepareDataForModeling(unittest.TestCase):
    """Test cases for data preparation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.data = pd.DataFrame({
            'ds': pd.date_range('2020-01-01', periods=100),
            'y': np.random.randn(100)
        })
    
    def test_prepare_data_for_modeling(self):
        """Test data preparation."""
        train_data, test_data, scaler = prepare_data_for_modeling(
            self.data, test_size=0.2, scale=False
        )
        
        self.assertEqual(len(train_data), 80)  # 80% of 100
        self.assertEqual(len(test_data), 20)   # 20% of 100
        self.assertIsNone(scaler)
    
    def test_prepare_data_with_scaling(self):
        """Test data preparation with scaling."""
        train_data, test_data, scaler = prepare_data_for_modeling(
            self.data, test_size=0.2, scale=True
        )
        
        self.assertEqual(len(train_data), 80)
        self.assertEqual(len(test_data), 20)
        self.assertIsNotNone(scaler)


class TestTimeSeriesAnalysisApp(unittest.TestCase):
    """Test cases for TimeSeriesAnalysisApp class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a minimal config for testing
        self.config = {
            'data': {
                'source': 'synthetic',
                'start_date': '2020-01-01',
                'end_date': '2020-01-10',
                'forecast_periods': 5,
                'test_size': 0.2
            },
            'models': {
                'prophet': {'enabled': False},
                'lstm': {'enabled': False}
            },
            'anomaly_detection': {'enabled': False},
            'visualization': {'style': 'default', 'figure_size': [10, 6]},
            'logging': {'level': 'INFO', 'format': '%(message)s', 'file': 'test.log'},
            'output': {'save_models': False, 'save_predictions': True, 'results_dir': 'test_results'}
        }
    
    @patch('src.app.yaml.safe_load')
    @patch('builtins.open')
    def test_load_config(self, mock_open, mock_yaml_load):
        """Test configuration loading."""
        mock_yaml_load.return_value = self.config
        
        app = TimeSeriesAnalysisApp.__new__(TimeSeriesAnalysisApp)
        app.config_path = "test_config.yaml"
        
        config = app._load_config()
        
        self.assertEqual(config, self.config)


if __name__ == '__main__':
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestDataGenerator,
        TestDataLoader,
        TestProphetForecaster,
        TestLSTMForecaster,
        TestAnomalyDetector,
        TestCalculateMetrics,
        TestPrepareDataForModeling,
        TestTimeSeriesAnalysisApp
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)
