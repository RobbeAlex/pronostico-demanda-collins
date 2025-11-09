"""
ARIMA model implementation for demand forecasting.
"""
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from base_model import BaseModel


class ARIMAModel(BaseModel):
    """
    ARIMA model implementation for time series forecasting.
    
    Uses statsmodels ARIMA for AutoRegressive Integrated Moving Average modeling.
    """
    
    def __init__(self, name: str = "ARIMA", order: tuple = (1, 1, 1)):
        """
        Initialize ARIMA model.
        
        Args:
            name: Name identifier for the model
            order: ARIMA order (p, d, q) tuple
        """
        super().__init__(name)
        self.order = order
        self.date_column = None
        self.target_column = None
        self.last_date = None
        self.freq = None
        
    def fit(self, data: pd.DataFrame, target_column: str, date_column: str) -> None:
        """
        Train the ARIMA model on historical data.
        
        Args:
            data: DataFrame with historical data
            target_column: Name of the column containing the target variable
            date_column: Name of the column containing dates
        """
        # Prepare data
        data_sorted = data.sort_values(by=date_column).copy()
        data_sorted[date_column] = pd.to_datetime(data_sorted[date_column])
        
        # Set date as index
        time_series = data_sorted.set_index(date_column)[target_column]
        
        # Infer frequency
        self.freq = pd.infer_freq(time_series.index)
        if self.freq is None:
            # Default to monthly if can't infer
            self.freq = 'ME'
        
        # Fit ARIMA model
        self.model = ARIMA(time_series, order=self.order)
        self.model = self.model.fit()
        
        self.is_fitted = True
        self.date_column = date_column
        self.target_column = target_column
        self.last_date = data_sorted[date_column].max()
        
    def predict(self, periods: int) -> pd.DataFrame:
        """
        Generate predictions for future periods.
        
        Args:
            periods: Number of periods to forecast
            
        Returns:
            DataFrame with predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Get forecast
        forecast_result = self.model.forecast(steps=periods)
        
        # Create future dates
        future_dates = pd.date_range(
            start=self.last_date + pd.DateOffset(months=1),
            periods=periods,
            freq='ME'
        )
        
        # Get confidence intervals
        forecast_df = self.model.get_forecast(steps=periods)
        conf_int = forecast_df.conf_int()
        
        # Create predictions DataFrame
        predictions = pd.DataFrame({
            'date': future_dates,
            'prediction': forecast_result.values,
            'lower_bound': conf_int.iloc[:, 0].values,
            'upper_bound': conf_int.iloc[:, 1].values
        })
        
        return predictions.reset_index(drop=True)
    
    def get_confidence_intervals(self, predictions: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate confidence intervals for predictions.
        
        Args:
            predictions: DataFrame with predictions
            
        Returns:
            DataFrame with confidence intervals
        """
        if 'lower_bound' in predictions.columns and 'upper_bound' in predictions.columns:
            return predictions[['date', 'lower_bound', 'upper_bound']]
        else:
            raise ValueError("Predictions DataFrame must contain lower_bound and upper_bound columns")
