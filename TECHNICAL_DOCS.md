# Demand Forecasting System - Technical Documentation

## Project Overview

This is a comprehensive demand forecasting system built using Python and Object-Oriented Programming principles. The system is designed to generate monthly demand forecasts for products and clients using multiple forecasting models.

## Architecture

### Design Pattern: Strategy Pattern with Template Method

The system uses the **Strategy Pattern** to allow interchangeable forecasting algorithms and the **Template Method** pattern through the abstract base class to define the common interface.

```
BaseModel (Abstract)
    ├── ProphetModel
    ├── ARIMAModel
    └── MLRegressionModel

ForecastManager (Coordinator)
    └── manages multiple BaseModel instances
```

### Core Components

#### 1. BaseModel (base_model.py)
**Purpose**: Abstract base class that defines the interface for all forecasting models.

**Key Methods**:
- `fit(data, target_column, date_column)`: Train the model
- `predict(periods)`: Generate future predictions
- `get_confidence_intervals(predictions)`: Calculate uncertainty bounds
- `get_model_info()`: Return model metadata

**Design Principle**: Liskov Substitution Principle - any subclass can be used wherever BaseModel is expected.

#### 2. ProphetModel (prophet_model.py)
**Purpose**: Facebook Prophet implementation for time series with trend and seasonality.

**Features**:
- Automatic trend detection
- Multiple seasonality patterns (yearly, weekly, daily)
- Built-in confidence intervals
- Holiday effects support

**Best Use Cases**:
- Data with strong seasonal patterns
- Multiple years of historical data
- Missing values or outliers present

#### 3. ARIMAModel (arima_model.py)
**Purpose**: AutoRegressive Integrated Moving Average statistical model.

**Features**:
- Configurable (p, d, q) parameters
- Statistical confidence intervals
- Good for stationary time series

**Best Use Cases**:
- Stationary or nearly stationary data
- Linear relationships
- When interpretability is important

#### 4. MLRegressionModel (ml_regression_model.py)
**Purpose**: Machine learning approach with feature engineering.

**Features**:
- Automatic feature engineering (lags, rolling statistics)
- Support for Random Forest and Linear Regression
- Time-based features (year, month, quarter)

**Best Use Cases**:
- Complex non-linear patterns
- Multiple external features available
- Large datasets

#### 5. ForecastManager (forecast_manager.py)
**Purpose**: Coordinate multiple models and generate ensemble predictions.

**Features**:
- Manage multiple models simultaneously
- Batch training and prediction
- Ensemble methods (mean, median)
- Model comparison utilities

**Key Methods**:
- `add_model(model)`: Register a model
- `fit_all(data, ...)`: Train all models
- `predict_all(periods)`: Generate predictions from all models
- `get_ensemble_predictions(method)`: Combine model predictions
- `get_model_comparison()`: Compare model outputs

### Utility Modules

#### data_loader.py
Functions for data operations:
- `load_csv_data()`: Load from CSV files
- `load_excel_data()`: Load from Excel files
- `generate_sample_data()`: Create synthetic data for testing
- `filter_by_product_client()`: Filter data
- `validate_data()`: Ensure data quality
- `aggregate_by_period()`: Time-based aggregation

#### evaluation.py
Comprehensive metrics for model evaluation:
- **MAE**: Mean Absolute Error
- **MSE**: Mean Squared Error  
- **RMSE**: Root Mean Squared Error
- **MAPE**: Mean Absolute Percentage Error
- **sMAPE**: Symmetric MAPE
- **R²**: Coefficient of Determination
- **Bias**: Systematic over/under prediction
- **Coverage**: Prediction interval accuracy

#### exporter.py
Results export functionality:
- CSV export
- Excel export with multiple sheets
- Comparison charts
- Forecast visualizations
- Metrics tables
- Complete summary reports

## Usage Patterns

### Basic Usage

```python
from data_loader import load_csv_data
from prophet_model import ProphetModel

# Load data
data = load_csv_data('sales.csv')

# Create and train model
model = ProphetModel(name="Prophet")
model.fit(data, target_column='demand', date_column='date')

# Generate predictions
predictions = model.predict(periods=12)
```

### Multi-Model Comparison

```python
from forecast_manager import ForecastManager
from prophet_model import ProphetModel
from arima_model import ARIMAModel
from ml_regression_model import MLRegressionModel

# Setup manager
manager = ForecastManager()
manager.add_model(ProphetModel(name="Prophet"))
manager.add_model(ARIMAModel(name="ARIMA", order=(1,1,1)))
manager.add_model(MLRegressionModel(name="RF"))

# Train all models
manager.fit_all(data, 'demand', 'date')

# Get predictions
predictions = manager.predict_all(12)
ensemble = manager.get_ensemble_predictions(method="mean")
```

### Evaluation

```python
from evaluation import evaluate_predictions, compare_models

# Split data
train = data[:-12]
test = data[-12:]

# Train and predict
manager.fit_all(train, 'demand', 'date')
predictions = manager.predict_all(12)

# Evaluate
metrics = evaluate_predictions(test['demand'], predictions['Prophet']['prediction'])
comparison = compare_models(test['demand'], {
    'Prophet': predictions['Prophet']['prediction'],
    'ARIMA': predictions['ARIMA']['prediction']
})
```

## Data Requirements

### Input Data Format

The system expects pandas DataFrames with at minimum:
- **Date column**: DateTime format, regular frequency (monthly recommended)
- **Target column**: Numeric values (demand, sales, etc.)
- **Optional**: product_id, client_id for filtering

### Example Data Structure

```
| date       | demand | product_id | client_id  |
|------------|--------|------------|------------|
| 2021-01-31 | 1000   | PROD_001   | CLIENT_001 |
| 2021-02-28 | 1050   | PROD_001   | CLIENT_001 |
| 2021-03-31 | 1100   | PROD_001   | CLIENT_001 |
```

### Data Quality Requirements

1. **No missing dates**: Use forward-fill or interpolation
2. **Consistent frequency**: Monthly is default, but configurable
3. **Sufficient history**: Minimum 24 periods recommended
4. **Non-negative values**: Demand should be >= 0

## Performance Considerations

### Computational Complexity

- **Prophet**: O(n) for fitting, fast predictions
- **ARIMA**: O(n²) to O(n³) for parameter estimation
- **ML Regression**: O(n * features * trees) for Random Forest

### Scalability

- **Single product/client**: All models perform well
- **Multiple products/clients**: Use ForecastManager with filtering
- **Large datasets (>10K rows)**: ML Regression preferred
- **Real-time predictions**: Use pre-fitted models

### Memory Usage

- **Prophet**: Moderate (stores full time series)
- **ARIMA**: Low (stores coefficients only)
- **ML Regression**: Moderate to High (stores trees)

## Extension Points

### Adding New Models

1. Create class inheriting from `BaseModel`
2. Implement required methods: `fit()`, `predict()`, `get_confidence_intervals()`
3. Register with ForecastManager

Example:
```python
from base_model import BaseModel

class CustomModel(BaseModel):
    def fit(self, data, target_column, date_column):
        # Implementation
        pass
    
    def predict(self, periods):
        # Implementation
        pass
    
    def get_confidence_intervals(self, predictions):
        # Implementation
        pass
```

### Customizing Features

For ML models, extend feature engineering in `_create_features()`:
```python
# Add custom features
df['day_of_week'] = df[date_column].dt.dayofweek
df['is_weekend'] = df['day_of_week'].isin([5, 6])
```

## Best Practices

### 1. Data Preparation
- Clean data before feeding to models
- Handle outliers appropriately
- Ensure consistent time intervals

### 2. Model Selection
- Use Prophet for seasonal data
- Use ARIMA for stationary data
- Use ML for complex patterns
- Use ensemble for robust predictions

### 3. Validation
- Always use train/test split
- Calculate multiple metrics
- Check prediction intervals coverage
- Visualize results

### 4. Production Deployment
- Save fitted models using pickle/joblib
- Version models and data
- Monitor prediction accuracy
- Retrain periodically

## Testing

Run the test suite:
```bash
python test_forecasting_system.py
```

Tests cover:
- Data generation and validation
- Each model individually
- ForecastManager functionality
- Evaluation metrics

## Dependencies

Core libraries:
- pandas: Data manipulation
- numpy: Numerical operations
- scikit-learn: ML models
- prophet: Prophet model
- statsmodels: ARIMA model
- matplotlib/seaborn: Visualization
- openpyxl: Excel export

## Version Information

- **Version**: 1.0.0
- **Python**: 3.8+
- **License**: UDG 2025 Project

## Troubleshooting

### Common Issues

1. **Prophet installation fails**
   - Install dependencies: `pip install pystan`
   - On Windows: Install C++ build tools

2. **Memory errors with large datasets**
   - Use data aggregation
   - Filter by product/client
   - Use sampling for validation

3. **Poor predictions**
   - Check data quality
   - Try different model parameters
   - Use ensemble predictions
   - Validate assumptions (stationarity for ARIMA)

## Future Enhancements

Potential improvements:
- [ ] Support for hourly/daily frequencies
- [ ] Automated hyperparameter tuning
- [ ] External regressor support
- [ ] Real-time streaming predictions
- [ ] Web API interface
- [ ] Model persistence and versioning
- [ ] Advanced ensemble methods (weighted, stacking)
- [ ] Automated model selection
