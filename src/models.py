"""
Time series forecasting models.

This module provides implementations of various forecasting models including
Prophet, ARIMA, LSTM, and anomaly detection methods.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from prophet import Prophet
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


class BaseForecaster(ABC):
    """Abstract base class for time series forecasters."""
    
    def __init__(self, name: str):
        """Initialize the forecaster.
        
        Args:
            name: Name of the forecaster
        """
        self.name = name
        self.model = None
        self.is_fitted = False
    
    @abstractmethod
    def fit(self, data: pd.DataFrame) -> None:
        """Fit the model to the data.
        
        Args:
            data: Training data with columns ['ds', 'y']
        """
        pass
    
    @abstractmethod
    def predict(self, periods: int) -> pd.DataFrame:
        """Make predictions for future periods.
        
        Args:
            periods: Number of periods to forecast
            
        Returns:
            DataFrame with predictions
        """
        pass
    
    def save_model(self, filepath: str) -> None:
        """Save the trained model.
        
        Args:
            filepath: Path to save the model
        """
        if self.is_fitted:
            joblib.dump(self.model, filepath)
            logger.info(f"Model saved to {filepath}")
        else:
            logger.warning("Model not fitted, cannot save")
    
    def load_model(self, filepath: str) -> None:
        """Load a trained model.
        
        Args:
            filepath: Path to the saved model
        """
        self.model = joblib.load(filepath)
        self.is_fitted = True
        logger.info(f"Model loaded from {filepath}")


class ProphetForecaster(BaseForecaster):
    """Prophet-based time series forecaster."""
    
    def __init__(
        self,
        yearly_seasonality: bool = True,
        weekly_seasonality: bool = True,
        daily_seasonality: bool = False,
        seasonality_mode: str = "additive",
        changepoint_prior_scale: float = 0.05
    ):
        """Initialize Prophet forecaster.
        
        Args:
            yearly_seasonality: Whether to include yearly seasonality
            weekly_seasonality: Whether to include weekly seasonality
            daily_seasonality: Whether to include daily seasonality
            seasonality_mode: Seasonality mode ('additive' or 'multiplicative')
            changepoint_prior_scale: Changepoint prior scale
        """
        super().__init__("Prophet")
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.seasonality_mode = seasonality_mode
        self.changepoint_prior_scale = changepoint_prior_scale
    
    def fit(self, data: pd.DataFrame) -> None:
        """Fit Prophet model to the data.
        
        Args:
            data: Training data with columns ['ds', 'y']
        """
        logger.info("Fitting Prophet model")
        
        self.model = Prophet(
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
            seasonality_mode=self.seasonality_mode,
            changepoint_prior_scale=self.changepoint_prior_scale
        )
        
        self.model.fit(data)
        self.is_fitted = True
        logger.info("Prophet model fitted successfully")
    
    def predict(self, periods: int) -> pd.DataFrame:
        """Make predictions using Prophet.
        
        Args:
            periods: Number of periods to forecast
            
        Returns:
            DataFrame with predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        logger.info(f"Making Prophet predictions for {periods} periods")
        
        # Create future dataframe
        future = self.model.make_future_dataframe(periods=periods)
        
        # Make predictions
        forecast = self.model.predict(future)
        
        # Return only future predictions
        future_forecast = forecast.tail(periods)[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
        future_forecast.columns = ['ds', 'yhat', 'yhat_lower', 'yhat_upper']
        
        return future_forecast


class LSTMForecaster(BaseForecaster):
    """LSTM-based time series forecaster."""
    
    def __init__(
        self,
        sequence_length: int = 60,
        hidden_size: int = 50,
        num_layers: int = 2,
        dropout: float = 0.2,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.001
    ):
        """Initialize LSTM forecaster.
        
        Args:
            sequence_length: Length of input sequences
            hidden_size: Size of LSTM hidden state
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer
        """
        super().__init__("LSTM")
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.scaler = StandardScaler()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def _create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training.
        
        Args:
            data: Input time series data
            
        Returns:
            Tuple of (X, y) sequences
        """
        X, y = [], []
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:(i + self.sequence_length)])
            y.append(data[i + self.sequence_length])
        return np.array(X), np.array(y)
    
    def fit(self, data: pd.DataFrame) -> None:
        """Fit LSTM model to the data.
        
        Args:
            data: Training data with columns ['ds', 'y']
        """
        logger.info("Fitting LSTM model")
        
        # Prepare data
        values = data['y'].values.reshape(-1, 1)
        scaled_values = self.scaler.fit_transform(values).flatten()
        
        # Create sequences
        X, y = self._create_sequences(scaled_values)
        
        if len(X) == 0:
            raise ValueError("Not enough data to create sequences")
        
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X).unsqueeze(-1)
        y_tensor = torch.FloatTensor(y)
        
        # Create data loader
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Initialize model
        self.model = LSTMModel(
            input_size=1,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(self.device)
        
        # Define loss and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Training loop
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 20 == 0:
                logger.info(f"Epoch [{epoch+1}/{self.epochs}], Loss: {total_loss/len(dataloader):.4f}")
        
        self.is_fitted = True
        logger.info("LSTM model fitted successfully")
    
    def predict(self, periods: int) -> pd.DataFrame:
        """Make predictions using LSTM.
        
        Args:
            periods: Number of periods to forecast
            
        Returns:
            DataFrame with predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        logger.info(f"Making LSTM predictions for {periods} periods")
        
        self.model.eval()
        predictions = []
        
        # Get last sequence for prediction
        with torch.no_grad():
            # Use the last sequence_length points for initial prediction
            last_sequence = torch.FloatTensor(self.scaler.transform(
                self.last_values[-self.sequence_length:].reshape(-1, 1)
            )).unsqueeze(0).unsqueeze(-1).to(self.device)
            
            for _ in range(periods):
                pred = self.model(last_sequence)
                predictions.append(pred.item())
                
                # Update sequence for next prediction
                last_sequence = torch.cat([
                    last_sequence[:, 1:, :],
                    pred.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
                ], dim=1)
        
        # Inverse transform predictions
        predictions = self.scaler.inverse_transform(
            np.array(predictions).reshape(-1, 1)
        ).flatten()
        
        # Create future dates
        last_date = self.last_date
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=periods,
            freq='D'
        )
        
        return pd.DataFrame({
            'ds': future_dates,
            'yhat': predictions,
            'yhat_lower': predictions * 0.95,  # Simple confidence interval
            'yhat_upper': predictions * 1.05
        })


class LSTMModel(nn.Module):
    """LSTM neural network model for time series forecasting."""
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
        """Initialize LSTM model.
        
        Args:
            input_size: Size of input features
            hidden_size: Size of LSTM hidden state
            num_layers: Number of LSTM layers
            dropout: Dropout rate
        """
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        """Forward pass through the LSTM model.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


class AnomalyDetector:
    """Anomaly detection for time series data."""
    
    def __init__(self, method: str = "isolation_forest", contamination: float = 0.1):
        """Initialize anomaly detector.
        
        Args:
            method: Detection method ('isolation_forest' or 'statistical')
            contamination: Expected proportion of anomalies
        """
        self.method = method
        self.contamination = contamination
        self.model = None
        self.is_fitted = False
    
    def fit(self, data: pd.DataFrame) -> None:
        """Fit anomaly detection model.
        
        Args:
            data: Training data with columns ['ds', 'y']
        """
        logger.info(f"Fitting {self.method} anomaly detector")
        
        if self.method == "isolation_forest":
            self.model = IsolationForest(
                contamination=self.contamination,
                random_state=42
            )
            self.model.fit(data[['y']])
        
        elif self.method == "statistical":
            # Simple statistical method based on z-scores
            self.mean = data['y'].mean()
            self.std = data['y'].std()
            self.threshold = 3.0  # 3 standard deviations
        
        self.is_fitted = True
        logger.info("Anomaly detector fitted successfully")
    
    def detect(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, List[int]]:
        """Detect anomalies in the data.
        
        Args:
            data: Data to analyze for anomalies
            
        Returns:
            Tuple of (data_with_anomaly_flags, anomaly_indices)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before detection")
        
        data_copy = data.copy()
        
        if self.method == "isolation_forest":
            predictions = self.model.predict(data[['y']])
            anomaly_indices = np.where(predictions == -1)[0].tolist()
            data_copy['is_anomaly'] = predictions == -1
        
        elif self.method == "statistical":
            z_scores = np.abs((data['y'] - self.mean) / self.std)
            anomaly_indices = np.where(z_scores > self.threshold)[0].tolist()
            data_copy['is_anomaly'] = z_scores > self.threshold
        
        logger.info(f"Detected {len(anomaly_indices)} anomalies")
        return data_copy, anomaly_indices


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate forecasting metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Dictionary of metrics
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'MAPE': mape
    }
