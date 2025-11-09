"""
Demand Forecasting System - Collins

A Python-based demand forecasting system using object-oriented programming.
Includes Prophet, ARIMA, and ML regression models with a manager class
to coordinate predictions.

Main Components:
- BaseModel: Abstract base class for all forecasting models
- ProphetModel: Facebook Prophet implementation
- ARIMAModel: ARIMA time series model
- MLRegressionModel: Machine learning regression models
- ForecastManager: Coordinates multiple models and predictions
- Data loading, evaluation, and export utilities

Example:
    >>> from data_loader import generate_sample_data
    >>> from prophet_model import ProphetModel
    >>> from forecast_manager import ForecastManager
    >>> 
    >>> data = generate_sample_data(periods=36)
    >>> model = ProphetModel()
    >>> model.fit(data, 'demand', 'date')
    >>> predictions = model.predict(12)
"""

__version__ = "1.0.0"
__author__ = "UDG 2025 Project"

# Import main classes for easier access
from base_model import BaseModel
from prophet_model import ProphetModel
from arima_model import ARIMAModel
from ml_regression_model import MLRegressionModel
from forecast_manager import ForecastManager

__all__ = [
    'BaseModel',
    'ProphetModel',
    'ARIMAModel',
    'MLRegressionModel',
    'ForecastManager',
]
