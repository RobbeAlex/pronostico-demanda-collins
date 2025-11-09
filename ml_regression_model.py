"""
Machine Learning Regression model implementation for demand forecasting.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from base_model import BaseModel


class MLRegressionModel(BaseModel):
    """
    Machine Learning Regression model for time series forecasting.
    
    Uses feature engineering to convert time series into supervised learning problem
    and applies ML regression algorithms.
    """
    
    def __init__(self, name: str = "MLRegression", model_type: str = "random_forest", **model_params):
        """
        Initialize ML Regression model.
        
        Args:
            name: Name identifier for the model
            model_type: Type of ML model ('random_forest' or 'linear_regression')
            **model_params: Additional parameters for the ML model
        """
        super().__init__(name)
        self.model_type = model_type
        self.model_params = model_params
        self.date_column = None
        self.target_column = None
        self.last_date = None
        self.last_values = None
        
    def _create_features(self, data: pd.DataFrame, date_column: str) -> pd.DataFrame:
        """
        Create time-based features from dates.
        
        Args:
            data: DataFrame with date column
            date_column: Name of the date column
            
        Returns:
            DataFrame with engineered features
        """
        df = data.copy()
        df[date_column] = pd.to_datetime(df[date_column])
        
        # Extract time-based features
        df['year'] = df[date_column].dt.year
        df['month'] = df[date_column].dt.month
        df['quarter'] = df[date_column].dt.quarter
        df['day_of_year'] = df[date_column].dt.dayofyear
        
        # Create lag features (previous periods)
        for lag in [1, 2, 3, 6, 12]:
            df[f'lag_{lag}'] = df[self.target_column].shift(lag)
        
        # Rolling statistics
        df['rolling_mean_3'] = df[self.target_column].rolling(window=3, min_periods=1).mean()
        df['rolling_mean_6'] = df[self.target_column].rolling(window=6, min_periods=1).mean()
        df['rolling_std_3'] = df[self.target_column].rolling(window=3, min_periods=1).std()
        
        return df
    
    def fit(self, data: pd.DataFrame, target_column: str, date_column: str) -> None:
        """
        Train the ML Regression model on historical data.
        
        Args:
            data: DataFrame with historical data
            target_column: Name of the column containing the target variable
            date_column: Name of the column containing dates
        """
        self.target_column = target_column
        self.date_column = date_column
        
        # Create features
        df_features = self._create_features(data, date_column)
        
        # Drop rows with NaN (due to lag features)
        df_features = df_features.dropna()
        
        # Store last values for future predictions
        self.last_date = pd.to_datetime(data[date_column]).max()
        self.last_values = data.tail(12).copy()  # Keep last 12 periods for lag features
        
        # Prepare features and target
        feature_cols = ['year', 'month', 'quarter', 'day_of_year',
                       'lag_1', 'lag_2', 'lag_3', 'lag_6', 'lag_12',
                       'rolling_mean_3', 'rolling_mean_6', 'rolling_std_3']
        
        X = df_features[feature_cols]
        y = df_features[target_column]
        
        # Initialize and fit the model
        if self.model_type == "random_forest":
            self.model = RandomForestRegressor(n_estimators=100, random_state=42, **self.model_params)
        elif self.model_type == "linear_regression":
            self.model = LinearRegression(**self.model_params)
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")
        
        self.model.fit(X, y)
        self.is_fitted = True
        
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
        
        predictions = []
        current_data = self.last_values.copy()
        
        for i in range(periods):
            # Generate next date
            next_date = self.last_date + pd.DateOffset(months=i+1)
            
            # Create features for next period
            temp_df = pd.DataFrame({
                self.date_column: [next_date],
                self.target_column: [0]  # Placeholder
            })
            
            # Combine with historical data for lag features
            combined = pd.concat([current_data, temp_df], ignore_index=True)
            df_features = self._create_features(combined, self.date_column)
            
            # Get the last row (our prediction period)
            last_row = df_features.iloc[-1]
            
            # Prepare features
            feature_cols = ['year', 'month', 'quarter', 'day_of_year',
                           'lag_1', 'lag_2', 'lag_3', 'lag_6', 'lag_12',
                           'rolling_mean_3', 'rolling_mean_6', 'rolling_std_3']
            
            X_pred = last_row[feature_cols].to_frame().T
            
            # Make prediction
            pred = self.model.predict(X_pred)[0]
            
            predictions.append({
                'date': next_date,
                'prediction': pred
            })
            
            # Update current_data with prediction for next iteration
            new_row = pd.DataFrame({
                self.date_column: [next_date],
                self.target_column: [pred]
            })
            current_data = pd.concat([current_data, new_row], ignore_index=True)
            current_data = current_data.tail(12)  # Keep only last 12 periods
        
        # Create predictions DataFrame
        predictions_df = pd.DataFrame(predictions)
        
        # Add confidence intervals (simple estimation based on historical variance)
        std_error = np.std(self.last_values[self.target_column]) * 0.5
        predictions_df['lower_bound'] = predictions_df['prediction'] - 1.96 * std_error
        predictions_df['upper_bound'] = predictions_df['prediction'] + 1.96 * std_error
        
        return predictions_df
    
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
