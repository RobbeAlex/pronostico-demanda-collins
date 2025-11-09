"""
Data loading utilities for demand forecasting system.
"""
import pandas as pd
from typing import Optional, List


def load_csv_data(file_path: str, date_column: str = 'date', parse_dates: bool = True) -> pd.DataFrame:
    """
    Load data from a CSV file.
    
    Args:
        file_path: Path to the CSV file
        date_column: Name of the date column
        parse_dates: Whether to parse dates automatically
        
    Returns:
        DataFrame with loaded data
    """
    if parse_dates:
        df = pd.read_csv(file_path, parse_dates=[date_column])
    else:
        df = pd.read_csv(file_path)
    
    return df


def load_excel_data(file_path: str, sheet_name: str = 0, date_column: str = 'date') -> pd.DataFrame:
    """
    Load data from an Excel file.
    
    Args:
        file_path: Path to the Excel file
        sheet_name: Sheet name or index to load
        date_column: Name of the date column
        
    Returns:
        DataFrame with loaded data
    """
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    
    if date_column in df.columns:
        df[date_column] = pd.to_datetime(df[date_column])
    
    return df


def generate_sample_data(periods: int = 36, 
                         trend: float = 10.0, 
                         seasonality_amplitude: float = 50.0,
                         noise_level: float = 10.0,
                         start_date: str = '2021-01-01') -> pd.DataFrame:
    """
    Generate sample demand data for testing.
    
    Args:
        periods: Number of time periods to generate
        trend: Trend component strength
        seasonality_amplitude: Amplitude of seasonal pattern
        noise_level: Level of random noise
        start_date: Start date for the time series
        
    Returns:
        DataFrame with sample data
    """
    import numpy as np
    
    # Generate dates
    dates = pd.date_range(start=start_date, periods=periods, freq='M')
    
    # Generate trend component
    trend_component = np.arange(periods) * trend
    
    # Generate seasonal component (yearly cycle)
    seasonality = seasonality_amplitude * np.sin(2 * np.pi * np.arange(periods) / 12)
    
    # Generate random noise
    np.random.seed(42)
    noise = np.random.normal(0, noise_level, periods)
    
    # Combine components
    base_value = 1000
    demand = base_value + trend_component + seasonality + noise
    
    # Ensure non-negative values
    demand = np.maximum(demand, 0)
    
    # Create DataFrame
    df = pd.DataFrame({
        'date': dates,
        'demand': demand,
        'product_id': 'PROD_001',
        'client_id': 'CLIENT_001'
    })
    
    return df


def filter_by_product_client(data: pd.DataFrame, 
                             product_id: Optional[str] = None,
                             client_id: Optional[str] = None) -> pd.DataFrame:
    """
    Filter data by product and/or client.
    
    Args:
        data: DataFrame with demand data
        product_id: Product ID to filter by (optional)
        client_id: Client ID to filter by (optional)
        
    Returns:
        Filtered DataFrame
    """
    filtered_data = data.copy()
    
    if product_id is not None and 'product_id' in filtered_data.columns:
        filtered_data = filtered_data[filtered_data['product_id'] == product_id]
    
    if client_id is not None and 'client_id' in filtered_data.columns:
        filtered_data = filtered_data[filtered_data['client_id'] == client_id]
    
    return filtered_data


def validate_data(data: pd.DataFrame, 
                 required_columns: List[str],
                 date_column: str = 'date') -> bool:
    """
    Validate that the data has required columns and proper format.
    
    Args:
        data: DataFrame to validate
        required_columns: List of required column names
        date_column: Name of the date column
        
    Returns:
        True if data is valid, raises ValueError otherwise
    """
    # Check if required columns exist
    missing_columns = set(required_columns) - set(data.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Check if date column can be converted to datetime
    try:
        pd.to_datetime(data[date_column])
    except Exception as e:
        raise ValueError(f"Cannot convert {date_column} to datetime: {str(e)}")
    
    # Check for null values in required columns
    null_counts = data[required_columns].isnull().sum()
    if null_counts.any():
        raise ValueError(f"Found null values in required columns: {null_counts[null_counts > 0].to_dict()}")
    
    return True


def aggregate_by_period(data: pd.DataFrame,
                       date_column: str,
                       value_column: str,
                       freq: str = 'M',
                       agg_func: str = 'sum') -> pd.DataFrame:
    """
    Aggregate data by time period.
    
    Args:
        data: DataFrame with time series data
        date_column: Name of the date column
        value_column: Name of the value column to aggregate
        freq: Frequency for aggregation ('D', 'W', 'M', 'Q', 'Y')
        agg_func: Aggregation function ('sum', 'mean', 'median', 'max', 'min')
        
    Returns:
        Aggregated DataFrame
    """
    df = data.copy()
    df[date_column] = pd.to_datetime(df[date_column])
    df = df.set_index(date_column)
    
    if agg_func == 'sum':
        aggregated = df[value_column].resample(freq).sum()
    elif agg_func == 'mean':
        aggregated = df[value_column].resample(freq).mean()
    elif agg_func == 'median':
        aggregated = df[value_column].resample(freq).median()
    elif agg_func == 'max':
        aggregated = df[value_column].resample(freq).max()
    elif agg_func == 'min':
        aggregated = df[value_column].resample(freq).min()
    else:
        raise ValueError(f"Unknown aggregation function: {agg_func}")
    
    result = aggregated.reset_index()
    result.columns = [date_column, value_column]
    
    return result
