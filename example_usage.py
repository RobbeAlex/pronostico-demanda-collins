"""
Example usage of the demand forecasting system.

This script demonstrates how to:
1. Load or generate sample data
2. Initialize forecasting models (Prophet, ARIMA, ML Regression)
3. Use the ForecastManager to coordinate predictions
4. Evaluate model performance
5. Export results and visualizations
"""

import pandas as pd
from data_loader import generate_sample_data, validate_data
from prophet_model import ProphetModel
from arima_model import ARIMAModel
from ml_regression_model import MLRegressionModel
from forecast_manager import ForecastManager
from evaluation import evaluate_predictions, compare_models
from exporter import create_summary_report


def main():
    """Main function to demonstrate the forecasting system."""
    
    print("=" * 60)
    print("Demand Forecasting System - Example Usage")
    print("=" * 60)
    
    # Step 1: Generate sample data
    print("\n1. Generating sample demand data...")
    data = generate_sample_data(periods=36, trend=10, seasonality_amplitude=50, noise_level=10)
    print(f"   Generated {len(data)} periods of data")
    print(f"   Date range: {data['date'].min()} to {data['date'].max()}")
    print(f"   Sample data:\n{data.head()}")
    
    # Step 2: Validate data
    print("\n2. Validating data...")
    try:
        validate_data(data, required_columns=['date', 'demand'])
        print("   ✓ Data validation passed")
    except ValueError as e:
        print(f"   ✗ Data validation failed: {e}")
        return
    
    # Step 3: Initialize forecasting models
    print("\n3. Initializing forecasting models...")
    
    # Prophet model
    prophet_model = ProphetModel(
        name="Prophet",
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False
    )
    print("   ✓ Prophet model initialized")
    
    # ARIMA model
    arima_model = ARIMAModel(
        name="ARIMA",
        order=(1, 1, 1)
    )
    print("   ✓ ARIMA model initialized")
    
    # ML Regression model (Random Forest)
    ml_model = MLRegressionModel(
        name="RandomForest",
        model_type="random_forest"
    )
    print("   ✓ ML Regression model initialized")
    
    # Step 4: Set up ForecastManager
    print("\n4. Setting up Forecast Manager...")
    manager = ForecastManager()
    manager.add_model(prophet_model)
    manager.add_model(arima_model)
    manager.add_model(ml_model)
    print(f"   ✓ Added {len(manager.models)} models to manager")
    
    # Step 5: Fit all models
    print("\n5. Training all models...")
    manager.fit_all(data, target_column='demand', date_column='date')
    
    # Step 6: Generate predictions
    print("\n6. Generating predictions for next 12 months...")
    predictions = manager.predict_all(periods=12)
    
    # Display predictions summary
    print("\n   Predictions summary:")
    for model_name, pred_df in predictions.items():
        print(f"\n   {model_name}:")
        print(f"   {pred_df.head()}")
    
    # Step 7: Get ensemble predictions
    print("\n7. Creating ensemble predictions...")
    ensemble = manager.get_ensemble_predictions(method="mean")
    print(f"   Ensemble predictions:\n{ensemble.head()}")
    
    # Step 8: Compare models
    print("\n8. Comparing model predictions...")
    comparison = manager.get_model_comparison()
    print(f"\n   Model comparison:\n{comparison.head()}")
    
    # Step 9: Export results
    print("\n9. Exporting results...")
    try:
        # Add ensemble to predictions dict
        predictions_with_ensemble = predictions.copy()
        predictions_with_ensemble['Ensemble'] = ensemble
        
        # Create summary report
        create_summary_report(
            historical_data=data,
            predictions_dict=predictions_with_ensemble,
            metrics_df=pd.DataFrame(),  # Would include actual metrics if we had test data
            output_dir='output',
            date_column='date',
            target_column='demand'
        )
    except Exception as e:
        print(f"   ⚠ Export skipped (may need display): {e}")
    
    # Step 10: Display summary
    print("\n10. System Summary:")
    summary = manager.get_summary()
    for key, value in summary.items():
        print(f"    {key}: {value}")
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)
    print("\nNext steps:")
    print("- Load your own data using data_loader.load_csv_data()")
    print("- Tune model parameters for better performance")
    print("- Split data for training/validation")
    print("- Use evaluation metrics to compare models")
    print("- Export predictions and visualizations")


if __name__ == "__main__":
    main()
