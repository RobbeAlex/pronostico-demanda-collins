"""
Basic tests for the demand forecasting system.

These tests verify that the main components work correctly.
"""

import pandas as pd
import numpy as np
from data_loader import generate_sample_data, validate_data
from prophet_model import ProphetModel
from arima_model import ARIMAModel
from ml_regression_model import MLRegressionModel
from forecast_manager import ForecastManager
from evaluation import evaluate_predictions, mean_absolute_error


def test_data_generation():
    """Test sample data generation."""
    print("Testing data generation...")
    data = generate_sample_data(periods=24)
    assert len(data) == 24, "Data should have 24 periods"
    assert 'date' in data.columns, "Data should have date column"
    assert 'demand' in data.columns, "Data should have demand column"
    print("✓ Data generation test passed")


def test_data_validation():
    """Test data validation."""
    print("\nTesting data validation...")
    data = generate_sample_data(periods=24)
    try:
        validate_data(data, required_columns=['date', 'demand'])
        print("✓ Data validation test passed")
    except ValueError as e:
        raise AssertionError(f"Data validation failed: {e}")


def test_prophet_model():
    """Test Prophet model."""
    print("\nTesting Prophet model...")
    data = generate_sample_data(periods=36)
    model = ProphetModel(name="TestProphet")
    
    # Fit model
    model.fit(data, target_column='demand', date_column='date')
    assert model.is_fitted, "Model should be fitted"
    
    # Generate predictions
    predictions = model.predict(periods=6)
    assert len(predictions) == 6, "Should generate 6 predictions"
    assert 'prediction' in predictions.columns, "Should have prediction column"
    assert 'lower_bound' in predictions.columns, "Should have lower_bound column"
    assert 'upper_bound' in predictions.columns, "Should have upper_bound column"
    
    print("✓ Prophet model test passed")


def test_arima_model():
    """Test ARIMA model."""
    print("\nTesting ARIMA model...")
    data = generate_sample_data(periods=36)
    model = ARIMAModel(name="TestARIMA", order=(1, 1, 1))
    
    # Fit model
    model.fit(data, target_column='demand', date_column='date')
    assert model.is_fitted, "Model should be fitted"
    
    # Generate predictions
    predictions = model.predict(periods=6)
    assert len(predictions) == 6, "Should generate 6 predictions"
    assert 'prediction' in predictions.columns, "Should have prediction column"
    
    print("✓ ARIMA model test passed")


def test_ml_regression_model():
    """Test ML Regression model."""
    print("\nTesting ML Regression model...")
    data = generate_sample_data(periods=36)
    model = MLRegressionModel(name="TestRF", model_type="random_forest")
    
    # Fit model
    model.fit(data, target_column='demand', date_column='date')
    assert model.is_fitted, "Model should be fitted"
    
    # Generate predictions
    predictions = model.predict(periods=6)
    assert len(predictions) == 6, "Should generate 6 predictions"
    assert 'prediction' in predictions.columns, "Should have prediction column"
    
    print("✓ ML Regression model test passed")


def test_forecast_manager():
    """Test ForecastManager."""
    print("\nTesting ForecastManager...")
    data = generate_sample_data(periods=36)
    
    # Create manager
    manager = ForecastManager()
    
    # Add models
    manager.add_model(ProphetModel(name="Prophet"))
    manager.add_model(ARIMAModel(name="ARIMA"))
    manager.add_model(MLRegressionModel(name="RF"))
    
    assert len(manager.models) == 3, "Manager should have 3 models"
    
    # Fit all models
    manager.fit_all(data, target_column='demand', date_column='date')
    
    # Generate predictions
    predictions = manager.predict_all(periods=6)
    assert len(predictions) == 3, "Should have predictions from 3 models"
    
    # Get ensemble
    ensemble = manager.get_ensemble_predictions(method="mean")
    assert len(ensemble) == 6, "Ensemble should have 6 predictions"
    
    # Get comparison
    comparison = manager.get_model_comparison()
    assert len(comparison) == 6, "Comparison should have 6 rows"
    
    print("✓ ForecastManager test passed")


def test_evaluation_metrics():
    """Test evaluation metrics."""
    print("\nTesting evaluation metrics...")
    actual = pd.Series([100, 110, 120, 130, 140])
    predicted = pd.Series([102, 108, 122, 128, 142])
    
    # Test MAE
    mae = mean_absolute_error(actual, predicted)
    assert mae > 0, "MAE should be positive"
    
    # Test comprehensive evaluation
    metrics = evaluate_predictions(actual, predicted)
    assert 'MAE' in metrics, "Should have MAE metric"
    assert 'RMSE' in metrics, "Should have RMSE metric"
    assert 'R2' in metrics, "Should have R2 metric"
    
    print("✓ Evaluation metrics test passed")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Running Demand Forecasting System Tests")
    print("=" * 60)
    
    try:
        test_data_generation()
        test_data_validation()
        test_prophet_model()
        test_arima_model()
        test_ml_regression_model()
        test_forecast_manager()
        test_evaluation_metrics()
        
        print("\n" + "=" * 60)
        print("✓ All tests passed successfully!")
        print("=" * 60)
        return True
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"✗ Tests failed: {str(e)}")
        print("=" * 60)
        raise


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    run_all_tests()
