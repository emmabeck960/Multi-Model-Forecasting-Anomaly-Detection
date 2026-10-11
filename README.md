# Prophet for Time Series Forecasting

A comprehensive Python project for time series forecasting and anomaly detection using modern machine learning techniques.

## Features

- **Multiple Forecasting Models**: Prophet, LSTM, ARIMA, and more
- **Anomaly Detection**: Isolation Forest and statistical methods
- **Interactive Web Interface**: Streamlit dashboard for exploration
- **Command Line Interface**: CLI for batch processing
- **Comprehensive Visualizations**: Static and interactive plots
- **Synthetic Data Generation**: Realistic time series for testing
- **Real Data Integration**: Yahoo Finance and CSV support
- **Model Comparison**: Performance metrics and visual comparisons
- **Configuration Management**: YAML-based configuration
- **Logging and Monitoring**: Comprehensive logging system

## Installation

### Prerequisites

- Python 3.10 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/kryptologyst/Prophet-for-Time-Series-Forecasting.git
cd Prophet-for-Time-Series-Forecasting
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Verify installation:
```bash
python -c "import prophet, torch, streamlit; print('Installation successful')"
```

## Quick Start

### Web Interface (Recommended)

Launch the Streamlit dashboard:
```bash
streamlit run streamlit_app.py
```

Open your browser to `http://localhost:8501` and:
1. Configure your analysis parameters in the sidebar
2. Select data source and models
3. Click "Run Analysis"
4. Explore results in the interactive tabs

### Command Line Interface

Run analysis with default settings:
```bash
python cli.py
```

Run with custom configuration:
```bash
python cli.py --config custom_config.yaml --verbose
```

Skip certain components:
```bash
python cli.py --skip-anomaly-detection --skip-visualization
```

### Programmatic Usage

```python
from src.app import TimeSeriesAnalysisApp

# Initialize app
app = TimeSeriesAnalysisApp("config/config.yaml")

# Run complete analysis
app.run_complete_analysis()

# Get results
summary = app.get_summary()
print(f"Best model: {summary['best_model']}")
```

## Configuration

The project uses YAML configuration files. Key settings in `config/config.yaml`:

### Data Settings
```yaml
data:
  source: "synthetic"  # Options: synthetic, yfinance, file
  symbol: "GOOG"       # For Yahoo Finance
  start_date: "2020-01-01"
  end_date: "2023-01-01"
  forecast_periods: 60
  test_size: 0.2
```

### Model Settings
```yaml
models:
  prophet:
    enabled: true
    yearly_seasonality: true
    weekly_seasonality: true
    daily_seasonality: false
  
  lstm:
    enabled: true
    sequence_length: 60
    hidden_size: 50
    num_layers: 2
    epochs: 100
```

### Anomaly Detection
```yaml
anomaly_detection:
  enabled: true
  methods: ["isolation_forest", "statistical"]
  contamination: 0.1
```

## Project Structure

```
time-series-analysis/
├── src/                    # Source code
│   ├── __init__.py
│   ├── app.py             # Main application
│   ├── data_utils.py      # Data generation and loading
│   ├── models.py          # Forecasting models
│   └── visualization.py   # Plotting utilities
├── config/                # Configuration files
│   └── config.yaml
├── tests/                 # Unit tests
│   └── test_components.py
├── notebooks/             # Jupyter notebooks
├── data/                  # Data storage
├── models/                # Saved models
├── results/               # Output results
├── logs/                  # Log files
├── streamlit_app.py       # Web interface
├── cli.py                 # Command line interface
├── requirements.txt       # Dependencies
├── .gitignore            # Git ignore rules
└── README.md             # This file
```

## Models

### Prophet
Facebook's robust forecasting tool, ideal for business time series with:
- Automatic trend detection
- Seasonal pattern recognition
- Holiday effects handling
- Missing data tolerance

### LSTM
Deep learning approach using Long Short-Term Memory networks:
- Handles complex non-linear patterns
- Captures long-term dependencies
- Suitable for multivariate time series

### ARIMA
Classical statistical method for:
- Stationary time series
- Trend and seasonal decomposition
- Interpretable results

## Anomaly Detection

### Isolation Forest
- Unsupervised anomaly detection
- Effective for high-dimensional data
- Robust to outliers

### Statistical Methods
- Z-score based detection
- Simple and interpretable
- Good for univariate time series

## Data Sources

### Synthetic Data
- Configurable trends and seasonality
- Controlled noise levels
- Anomaly injection capabilities
- Perfect for testing and development

### Yahoo Finance
- Real stock market data
- Multiple symbols supported
- Historical data access
- Automatic data cleaning

### CSV Files
- Custom time series data
- Flexible date formats
- Data validation and cleaning

## Visualization

### Static Plots (Matplotlib/Seaborn)
- Time series plots
- Forecast visualizations
- Model comparisons
- Anomaly detection results
- Decomposition plots

### Interactive Plots (Plotly)
- Zoomable and pannable charts
- Hover information
- Model comparison dashboards
- Real-time data exploration

## Performance Metrics

The project calculates comprehensive metrics:
- **MAE**: Mean Absolute Error
- **MSE**: Mean Squared Error
- **RMSE**: Root Mean Squared Error
- **MAPE**: Mean Absolute Percentage Error

## Testing

Run the test suite:
```bash
python -m pytest tests/ -v
```

Run specific test categories:
```bash
python -m pytest tests/test_components.py::TestDataGenerator -v
```

## Logging

The project includes comprehensive logging:
- Configurable log levels
- File and console output
- Structured log messages
- Error tracking and debugging

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## Dependencies

### Core Libraries
- pandas >= 2.0.0
- numpy >= 1.24.0
- scipy >= 1.10.0

### Time Series
- prophet >= 1.1.0
- statsmodels >= 0.14.0
- pmdarima >= 2.0.0
- tslearn >= 0.6.0
- darts >= 0.24.0
- sktime >= 0.21.0

### Machine Learning
- scikit-learn >= 1.3.0
- torch >= 2.0.0
- transformers >= 4.30.0

### Visualization
- matplotlib >= 3.7.0
- seaborn >= 0.12.0
- plotly >= 5.15.0

### Web Interface
- streamlit >= 1.25.0

### Data Sources
- yfinance >= 0.2.0

## Troubleshooting

### Common Issues

1. **Prophet Installation Issues**
   ```bash
   pip install prophet --no-cache-dir
   ```

2. **PyTorch Installation**
   ```bash
   pip install torch torchvision torchaudio
   ```

3. **Memory Issues with LSTM**
   - Reduce batch size in config
   - Decrease sequence length
   - Use smaller hidden size

4. **Streamlit Port Issues**
   ```bash
   streamlit run streamlit_app.py --server.port 8502
   ```

### Performance Optimization

- Use GPU for LSTM training (automatic detection)
- Adjust batch sizes for your hardware
- Enable model caching for repeated runs
- Use data scaling for better convergence

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Facebook Prophet team for the excellent forecasting library
- PyTorch team for deep learning capabilities
- Streamlit team for the web interface framework
- The open-source Python data science community

## Support

For questions, issues, or contributions:
- Create an issue on GitHub
- Check the documentation
- Review the test cases for usage examples
- Consult the configuration examples

---

**Happy Forecasting!** 📈
# Prophet-for-Time-Series-Forecasting
