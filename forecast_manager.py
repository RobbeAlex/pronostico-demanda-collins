"""
Manager class to coordinate demand forecasting predictions.
"""
from typing import List, Dict, Any
import pandas as pd
from base_model import BaseModel


class ForecastManager:
    """
    Manager class to coordinate multiple forecasting models and aggregate predictions.
    
    This class allows running multiple models, comparing their results, and
    generating ensemble predictions.
    """
    
    def __init__(self):
        """Initialize the forecast manager."""
        self.models: Dict[str, BaseModel] = {}
        self.predictions: Dict[str, pd.DataFrame] = {}
        
    def add_model(self, model: BaseModel) -> None:
        """
        Add a forecasting model to the manager.
        
        Args:
            model: Instance of a forecasting model (subclass of BaseModel)
        """
        if not isinstance(model, BaseModel):
            raise ValueError("Model must be an instance of BaseModel")
        
        self.models[model.name] = model
        
    def remove_model(self, model_name: str) -> None:
        """
        Remove a model from the manager.
        
        Args:
            model_name: Name of the model to remove
        """
        if model_name in self.models:
            del self.models[model_name]
        if model_name in self.predictions:
            del self.predictions[model_name]
            
    def fit_all(self, data: pd.DataFrame, target_column: str, date_column: str) -> None:
        """
        Fit all registered models on the provided data.
        
        Args:
            data: DataFrame with historical data
            target_column: Name of the column containing the target variable
            date_column: Name of the column containing dates
        """
        for model_name, model in self.models.items():
            print(f"Fitting model: {model_name}")
            try:
                model.fit(data, target_column, date_column)
                print(f"✓ {model_name} fitted successfully")
            except Exception as e:
                print(f"✗ Error fitting {model_name}: {str(e)}")
                
    def predict_all(self, periods: int) -> Dict[str, pd.DataFrame]:
        """
        Generate predictions from all fitted models.
        
        Args:
            periods: Number of periods to forecast
            
        Returns:
            Dictionary mapping model names to their predictions
        """
        self.predictions = {}
        
        for model_name, model in self.models.items():
            if not model.is_fitted:
                print(f"⚠ Skipping {model_name}: model not fitted")
                continue
                
            print(f"Generating predictions for: {model_name}")
            try:
                predictions = model.predict(periods)
                self.predictions[model_name] = predictions
                print(f"✓ {model_name} predictions generated")
            except Exception as e:
                print(f"✗ Error predicting with {model_name}: {str(e)}")
                
        return self.predictions
    
    def get_ensemble_predictions(self, method: str = "mean") -> pd.DataFrame:
        """
        Create ensemble predictions by combining multiple model outputs.
        
        Args:
            method: Aggregation method ('mean', 'median', or 'weighted')
            
        Returns:
            DataFrame with ensemble predictions
        """
        if not self.predictions:
            raise ValueError("No predictions available. Run predict_all() first.")
        
        # Collect all predictions
        all_preds = []
        for model_name, pred_df in self.predictions.items():
            temp_df = pred_df.copy()
            temp_df['model'] = model_name
            all_preds.append(temp_df)
        
        combined = pd.concat(all_preds, ignore_index=True)
        
        # Aggregate by date
        if method == "mean":
            ensemble = combined.groupby('date').agg({
                'prediction': 'mean',
                'lower_bound': 'mean',
                'upper_bound': 'mean'
            }).reset_index()
        elif method == "median":
            ensemble = combined.groupby('date').agg({
                'prediction': 'median',
                'lower_bound': 'min',
                'upper_bound': 'max'
            }).reset_index()
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
        
        return ensemble
    
    def get_model_comparison(self) -> pd.DataFrame:
        """
        Get a comparison of all model predictions.
        
        Returns:
            DataFrame with predictions from all models side by side
        """
        if not self.predictions:
            raise ValueError("No predictions available. Run predict_all() first.")
        
        # Start with the first model's dates
        first_model = list(self.predictions.keys())[0]
        comparison = pd.DataFrame({'date': self.predictions[first_model]['date']})
        
        # Add predictions from each model
        for model_name, pred_df in self.predictions.items():
            comparison[f'{model_name}_prediction'] = pred_df['prediction'].values
            comparison[f'{model_name}_lower'] = pred_df['lower_bound'].values
            comparison[f'{model_name}_upper'] = pred_df['upper_bound'].values
        
        return comparison
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the forecast manager state.
        
        Returns:
            Dictionary with summary information
        """
        return {
            'total_models': len(self.models),
            'fitted_models': sum(1 for m in self.models.values() if m.is_fitted),
            'models_with_predictions': len(self.predictions),
            'model_names': list(self.models.keys())
        }
