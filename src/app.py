"""
Main application for time series analysis and forecasting.

This module provides the main interface for running time series analysis,
including data loading, model training, forecasting, and visualization.
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .data_utils import DataGenerator, DataLoader, prepare_data_for_modeling
from .models import AnomalyDetector, LSTMForecaster, ProphetForecaster, calculate_metrics
from .visualization import InteractiveVisualizer, TimeSeriesVisualizer

logger = logging.getLogger(__name__)


class TimeSeriesAnalysisApp:
    """Main application class for time series analysis."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the application.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.data = None
        self.models = {}
        self.forecasts = {}
        self.metrics = {}
        self.anomaly_detector = None
        self.visualizer = TimeSeriesVisualizer(
            style=self.config['visualization']['style'],
            figure_size=tuple(self.config['visualization']['figure_size'])
        )
        self.interactive_visualizer = InteractiveVisualizer()
        
        # Setup logging
        self._setup_logging()
        
        # Create output directories
        self._create_directories()
    
    def _load_config(self) -> dict:
        """Load configuration from YAML file.
        
        Returns:
            Configuration dictionary
        """
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
            logger.info(f"Configuration loaded from {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise
    
    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        log_config = self.config['logging']
        
        # Create logs directory if it doesn't exist
        os.makedirs(os.path.dirname(log_config['file']), exist_ok=True)
        
        logging.basicConfig(
            level=getattr(logging, log_config['level']),
            format=log_config['format'],
            handlers=[
                logging.FileHandler(log_config['file']),
                logging.StreamHandler()
            ]
        )
    
    def _create_directories(self) -> None:
        """Create necessary output directories."""
        directories = ['logs', 'results', 'models', 'data']
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def load_data(self) -> pd.DataFrame:
        """Load or generate time series data.
        
        Returns:
            Time series data
        """
        data_config = self.config['data']
        
        if data_config['source'] == 'synthetic':
            logger.info("Generating synthetic time series data")
            generator = DataGenerator()
            self.data = generator.generate_complex_data(
                start_date=data_config['start_date'],
                end_date=data_config['end_date']
            )
            
            # Add anomalies if enabled
            if self.config['anomaly_detection']['enabled']:
                self.data, self.anomaly_indices = generator.add_anomalies(
                    self.data,
                    anomaly_rate=0.05,
                    anomaly_magnitude=3.0
                )
                logger.info(f"Added {len(self.anomaly_indices)} anomalies")
        
        elif data_config['source'] == 'yfinance':
            logger.info(f"Loading {data_config['symbol']} data from Yahoo Finance")
            self.data = DataLoader.load_yfinance_data(
                symbol=data_config['symbol'],
                start_date=data_config['start_date'],
                end_date=data_config['end_date']
            )
        
        elif data_config['source'] == 'file':
            # This would be implemented to load from CSV files
            raise NotImplementedError("File loading not implemented yet")
        
        logger.info(f"Loaded {len(self.data)} data points")
        return self.data
    
    def train_models(self) -> None:
        """Train all enabled forecasting models."""
        logger.info("Training forecasting models")
        
        # Prepare data
        train_data, test_data, scaler = prepare_data_for_modeling(
            self.data,
            test_size=self.config['data']['test_size']
        )
        
        self.train_data = train_data
        self.test_data = test_data
        self.scaler = scaler
        
        # Train Prophet model
        if self.config['models']['prophet']['enabled']:
            logger.info("Training Prophet model")
            prophet_config = self.config['models']['prophet']
            prophet_model = ProphetForecaster(
                yearly_seasonality=prophet_config['yearly_seasonality'],
                weekly_seasonality=prophet_config['weekly_seasonality'],
                daily_seasonality=prophet_config['daily_seasonality'],
                seasonality_mode=prophet_config['seasonality_mode'],
                changepoint_prior_scale=prophet_config['changepoint_prior_scale']
            )
            prophet_model.fit(train_data)
            self.models['Prophet'] = prophet_model
        
        # Train LSTM model
        if self.config['models']['lstm']['enabled']:
            logger.info("Training LSTM model")
            lstm_config = self.config['models']['lstm']
            lstm_model = LSTMForecaster(
                sequence_length=lstm_config['sequence_length'],
                hidden_size=lstm_config['hidden_size'],
                num_layers=lstm_config['num_layers'],
                dropout=lstm_config['dropout'],
                epochs=lstm_config['epochs'],
                batch_size=lstm_config['batch_size'],
                learning_rate=lstm_config['learning_rate']
            )
            lstm_model.fit(train_data)
            self.models['LSTM'] = lstm_model
        
        logger.info("All models trained successfully")
    
    def make_forecasts(self) -> None:
        """Make forecasts using all trained models."""
        logger.info("Making forecasts")
        
        forecast_periods = self.config['data']['forecast_periods']
        
        for model_name, model in self.models.items():
            logger.info(f"Making forecast with {model_name}")
            forecast = model.predict(forecast_periods)
            self.forecasts[model_name] = forecast
        
        logger.info("All forecasts completed")
    
    def detect_anomalies(self) -> None:
        """Detect anomalies in the data."""
        if not self.config['anomaly_detection']['enabled']:
            logger.info("Anomaly detection disabled")
            return
        
        logger.info("Detecting anomalies")
        
        methods = self.config['anomaly_detection']['methods']
        contamination = self.config['anomaly_detection']['contamination']
        
        for method in methods:
            detector = AnomalyDetector(method=method, contamination=contamination)
            detector.fit(self.data)
            
            data_with_anomalies, anomaly_indices = detector.detect(self.data)
            
            logger.info(f"Detected {len(anomaly_indices)} anomalies using {method}")
            
            # Store results
            if not hasattr(self, 'anomaly_results'):
                self.anomaly_results = {}
            self.anomaly_results[method] = {
                'data': data_with_anomalies,
                'indices': anomaly_indices
            }
    
    def evaluate_models(self) -> None:
        """Evaluate model performance on test data."""
        logger.info("Evaluating model performance")
        
        for model_name, model in self.models.items():
            # Make predictions on test data
            test_forecast = model.predict(len(self.test_data))
            
            # Calculate metrics
            metrics = calculate_metrics(
                self.test_data['y'].values,
                test_forecast['yhat'].values
            )
            
            self.metrics[model_name] = metrics
            
            logger.info(f"{model_name} metrics: {metrics}")
    
    def create_visualizations(self) -> None:
        """Create all visualizations."""
        logger.info("Creating visualizations")
        
        # Create output directory for plots
        plots_dir = Path("results/plots")
        plots_dir.mkdir(exist_ok=True)
        
        # Plot original time series
        self.visualizer.plot_time_series(
            self.data,
            title="Original Time Series Data",
            save_path=str(plots_dir / "original_data.png")
        )
        
        # Plot forecasts
        for model_name, forecast in self.forecasts.items():
            self.visualizer.plot_forecast(
                self.train_data,
                forecast,
                title=f"{model_name} Forecast",
                save_path=str(plots_dir / f"{model_name.lower()}_forecast.png")
            )
        
        # Plot model comparison
        if len(self.forecasts) > 1:
            self.visualizer.plot_model_comparison(
                self.train_data,
                self.forecasts,
                title="Model Comparison",
                save_path=str(plots_dir / "model_comparison.png")
            )
        
        # Plot anomalies
        if hasattr(self, 'anomaly_results'):
            for method, results in self.anomaly_results.items():
                self.visualizer.plot_anomalies(
                    self.data,
                    results['indices'],
                    title=f"Anomaly Detection - {method.title()}",
                    save_path=str(plots_dir / f"anomalies_{method}.png")
                )
        
        logger.info("All visualizations created")
    
    def save_results(self) -> None:
        """Save all results to files."""
        logger.info("Saving results")
        
        results_dir = Path("results")
        
        # Save forecasts
        for model_name, forecast in self.forecasts.items():
            forecast.to_csv(
                results_dir / f"{model_name.lower()}_forecast.csv",
                index=False
            )
        
        # Save metrics
        if self.metrics:
            metrics_df = pd.DataFrame(self.metrics).T
            metrics_df.to_csv(results_dir / "model_metrics.csv")
        
        # Save models
        if self.config['output']['save_models']:
            models_dir = Path("models")
            for model_name, model in self.models.items():
                model.save_model(str(models_dir / f"{model_name.lower()}_model.pkl"))
        
        logger.info("Results saved successfully")
    
    def run_complete_analysis(self) -> None:
        """Run the complete time series analysis pipeline."""
        logger.info("Starting complete time series analysis")
        
        try:
            # Load data
            self.load_data()
            
            # Train models
            self.train_models()
            
            # Make forecasts
            self.make_forecasts()
            
            # Detect anomalies
            self.detect_anomalies()
            
            # Evaluate models
            self.evaluate_models()
            
            # Create visualizations
            self.create_visualizations()
            
            # Save results
            self.save_results()
            
            logger.info("Complete analysis finished successfully")
            
        except Exception as e:
            logger.error(f"Error during analysis: {e}")
            raise
    
    def get_summary(self) -> Dict:
        """Get a summary of the analysis results.
        
        Returns:
            Dictionary with analysis summary
        """
        summary = {
            'data_points': len(self.data),
            'train_points': len(self.train_data),
            'test_points': len(self.test_data),
            'models_trained': list(self.models.keys()),
            'forecast_periods': self.config['data']['forecast_periods']
        }
        
        if self.metrics:
            summary['best_model'] = min(
                self.metrics.keys(),
                key=lambda x: self.metrics[x]['RMSE']
            )
            summary['metrics'] = self.metrics
        
        if hasattr(self, 'anomaly_results'):
            summary['anomalies_detected'] = {
                method: len(results['indices'])
                for method, results in self.anomaly_results.items()
            }
        
        return summary
