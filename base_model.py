"""
Base class for demand forecasting models.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np


class BaseModel(ABC):
    """
    Abstract base class for demand forecasting models.
    
    All forecasting models must inherit from this class and implement
    the fit, predict, and get_confidence_intervals methods.
    """
    
    def __init__(self, name: str):
        """
        Initialize the base model.
        
        Args:
            name: Name identifier for the model
        """
        self.name = name
        self.is_fitted = False
        self.model = None
        
    @abstractmethod
    def fit(self, data: pd.DataFrame, target_column: str, date_column: str) -> None:
        """
        Train the forecasting model on historical data.
        
        Args:
            data: DataFrame with historical data
            target_column: Name of the column containing the target variable
            date_column: Name of the column containing dates
        """
        pass
    
    @abstractmethod
    def predict(self, periods: int) -> pd.DataFrame:
        """
        Generate predictions for future periods.
        
        Args:
            periods: Number of periods to forecast
            
        Returns:
            DataFrame with predictions
        """
        pass
    
    @abstractmethod
    def get_confidence_intervals(self, predictions: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate confidence intervals for predictions.
        
        Args:
            predictions: DataFrame with predictions
            
        Returns:
            DataFrame with confidence intervals (lower and upper bounds)
        """
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model.
        
        Returns:
            Dictionary with model information
        """
        return {
            'name': self.name,
            'is_fitted': self.is_fitted,
            'model_type': self.__class__.__name__
        }
