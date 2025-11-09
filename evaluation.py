"""
Evaluation metrics for demand forecasting models.
"""
import pandas as pd
import numpy as np
from typing import Dict, List


def mean_absolute_error(actual: pd.Series, predicted: pd.Series) -> float:
    """
    Calculate Mean Absolute Error (MAE).
    
    Args:
        actual: Actual values
        predicted: Predicted values
        
    Returns:
        MAE value
    """
    return np.mean(np.abs(actual - predicted))


def mean_squared_error(actual: pd.Series, predicted: pd.Series) -> float:
    """
    Calculate Mean Squared Error (MSE).
    
    Args:
        actual: Actual values
        predicted: Predicted values
        
    Returns:
        MSE value
    """
    return np.mean((actual - predicted) ** 2)


def root_mean_squared_error(actual: pd.Series, predicted: pd.Series) -> float:
    """
    Calculate Root Mean Squared Error (RMSE).
    
    Args:
        actual: Actual values
        predicted: Predicted values
        
    Returns:
        RMSE value
    """
    return np.sqrt(mean_squared_error(actual, predicted))


def mean_absolute_percentage_error(actual: pd.Series, predicted: pd.Series) -> float:
    """
    Calculate Mean Absolute Percentage Error (MAPE).
    
    Args:
        actual: Actual values
        predicted: Predicted values
        
    Returns:
        MAPE value (as percentage)
    """
    # Avoid division by zero
    mask = actual != 0
    return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100


def symmetric_mean_absolute_percentage_error(actual: pd.Series, predicted: pd.Series) -> float:
    """
    Calculate Symmetric Mean Absolute Percentage Error (sMAPE).
    
    Args:
        actual: Actual values
        predicted: Predicted values
        
    Returns:
        sMAPE value (as percentage)
    """
    denominator = (np.abs(actual) + np.abs(predicted)) / 2
    # Avoid division by zero
    mask = denominator != 0
    return np.mean(np.abs(actual[mask] - predicted[mask]) / denominator[mask]) * 100


def r_squared(actual: pd.Series, predicted: pd.Series) -> float:
    """
    Calculate R-squared (coefficient of determination).
    
    Args:
        actual: Actual values
        predicted: Predicted values
        
    Returns:
        R-squared value
    """
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    
    if ss_tot == 0:
        return 0.0
    
    return 1 - (ss_res / ss_tot)


def evaluate_predictions(actual: pd.Series, predicted: pd.Series) -> Dict[str, float]:
    """
    Calculate all evaluation metrics for predictions.
    
    Args:
        actual: Actual values
        predicted: Predicted values
        
    Returns:
        Dictionary with all metrics
    """
    metrics = {
        'MAE': mean_absolute_error(actual, predicted),
        'MSE': mean_squared_error(actual, predicted),
        'RMSE': root_mean_squared_error(actual, predicted),
        'MAPE': mean_absolute_percentage_error(actual, predicted),
        'sMAPE': symmetric_mean_absolute_percentage_error(actual, predicted),
        'R2': r_squared(actual, predicted)
    }
    
    return metrics


def compare_models(actual: pd.Series, predictions_dict: Dict[str, pd.Series]) -> pd.DataFrame:
    """
    Compare multiple models using evaluation metrics.
    
    Args:
        actual: Actual values
        predictions_dict: Dictionary mapping model names to their predictions
        
    Returns:
        DataFrame with metrics for each model
    """
    results = []
    
    for model_name, predicted in predictions_dict.items():
        metrics = evaluate_predictions(actual, predicted)
        metrics['Model'] = model_name
        results.append(metrics)
    
    df_results = pd.DataFrame(results)
    
    # Reorder columns to have Model first
    cols = ['Model'] + [col for col in df_results.columns if col != 'Model']
    df_results = df_results[cols]
    
    return df_results


def calculate_prediction_intervals_coverage(actual: pd.Series, 
                                            lower_bound: pd.Series,
                                            upper_bound: pd.Series) -> float:
    """
    Calculate the percentage of actual values within prediction intervals.
    
    Args:
        actual: Actual values
        lower_bound: Lower bound of prediction intervals
        upper_bound: Upper bound of prediction intervals
        
    Returns:
        Coverage percentage (0-100)
    """
    within_interval = (actual >= lower_bound) & (actual <= upper_bound)
    coverage = np.mean(within_interval) * 100
    
    return coverage


def calculate_bias(actual: pd.Series, predicted: pd.Series) -> float:
    """
    Calculate prediction bias (mean error).
    
    Args:
        actual: Actual values
        predicted: Predicted values
        
    Returns:
        Bias value (positive = overestimation, negative = underestimation)
    """
    return np.mean(predicted - actual)
