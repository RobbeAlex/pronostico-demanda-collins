"""
Prophet model implementation for demand forecasting.
"""
import pandas as pd
from prophet import Prophet
from base_model import BaseModel


class ProphetModel(BaseModel):
    """
    Prophet model implementation for time series forecasting.
    
    Uses Facebook's Prophet library for forecasting with automatic
    trend and seasonality detection.
    """
    
    def __init__(self, name: str = "Prophet", **prophet_params):
        """
        Initialize Prophet model.
        
        Args:
            name: Name identifier for the model
            **prophet_params: Additional parameters for Prophet model
        """
        super().__init__(name)
        self.prophet_params = prophet_params
        self.date_column = None
        self.target_column = None
        
    def fit(self, data: pd.DataFrame, target_column: str, date_column: str) -> None:
        """
        Train the Prophet model on historical data.
        
        Args:
            data: DataFrame with historical data
            target_column: Name of the column containing the target variable
            date_column: Name of the column containing dates
        """
        # Prophet requires columns named 'ds' and 'y'
        prophet_data = pd.DataFrame({
            'ds': pd.to_datetime(data[date_column]),
            'y': data[target_column]
        })
        
        # Initialize and fit Prophet model
        self.model = Prophet(**self.prophet_params)
        self.model.fit(prophet_data)
        
        self.is_fitted = True
        self.date_column = date_column
        self.target_column = target_column
        
    def predict(self, periods: int) -> pd.DataFrame:
        """
        Generate predictions for future periods.
        
        Args:
            periods: Number of periods to forecast
            
        Returns:
            DataFrame with predictions including date, yhat, yhat_lower, yhat_upper
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Create future dataframe
        future = self.model.make_future_dataframe(periods=periods, freq='ME')
        
        # Make predictions
        forecast = self.model.predict(future)
        
        # Return only future predictions
        predictions = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods)
        predictions = predictions.rename(columns={
            'ds': 'date',
            'yhat': 'prediction',
            'yhat_lower': 'lower_bound',
            'yhat_upper': 'upper_bound'
        })
        
        return predictions.reset_index(drop=True)
    
    def get_confidence_intervals(self, predictions: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate confidence intervals for predictions.
        
        Args:
            predictions: DataFrame with predictions
            
        Returns:
            DataFrame with confidence intervals (already included in Prophet predictions)
        """
        if 'lower_bound' in predictions.columns and 'upper_bound' in predictions.columns:
            return predictions[['date', 'lower_bound', 'upper_bound']]
        else:
            raise ValueError("Predictions DataFrame must contain lower_bound and upper_bound columns")
