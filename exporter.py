"""
Export utilities for forecasting results.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Optional
import os


def export_to_csv(predictions: pd.DataFrame, file_path: str) -> None:
    """
    Export predictions to a CSV file.
    
    Args:
        predictions: DataFrame with predictions
        file_path: Path to save the CSV file
    """
    predictions.to_csv(file_path, index=False)
    print(f"✓ Predictions exported to: {file_path}")


def export_to_excel(predictions_dict: Dict[str, pd.DataFrame], file_path: str) -> None:
    """
    Export multiple prediction sets to an Excel file with separate sheets.
    
    Args:
        predictions_dict: Dictionary mapping sheet names to DataFrames
        file_path: Path to save the Excel file
    """
    with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
        for sheet_name, df in predictions_dict.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    
    print(f"✓ Predictions exported to: {file_path}")


def export_comparison_chart(predictions_dict: Dict[str, pd.DataFrame],
                           output_path: str,
                           title: str = "Model Predictions Comparison",
                           figsize: tuple = (12, 6)) -> None:
    """
    Create and export a comparison chart of multiple model predictions.
    
    Args:
        predictions_dict: Dictionary mapping model names to prediction DataFrames
        output_path: Path to save the chart
        title: Chart title
        figsize: Figure size (width, height)
    """
    plt.figure(figsize=figsize)
    
    for model_name, pred_df in predictions_dict.items():
        plt.plot(pred_df['date'], pred_df['prediction'], marker='o', label=model_name)
        
        # Plot confidence intervals if available
        if 'lower_bound' in pred_df.columns and 'upper_bound' in pred_df.columns:
            plt.fill_between(pred_df['date'], 
                           pred_df['lower_bound'], 
                           pred_df['upper_bound'], 
                           alpha=0.2)
    
    plt.xlabel('Date')
    plt.ylabel('Demand')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Comparison chart saved to: {output_path}")


def export_forecast_with_history(historical_data: pd.DataFrame,
                                 predictions: pd.DataFrame,
                                 date_column: str,
                                 target_column: str,
                                 output_path: str,
                                 title: str = "Demand Forecast",
                                 figsize: tuple = (14, 7)) -> None:
    """
    Create and export a chart showing historical data and forecasts.
    
    Args:
        historical_data: DataFrame with historical data
        predictions: DataFrame with predictions
        date_column: Name of the date column
        target_column: Name of the target column
        output_path: Path to save the chart
        title: Chart title
        figsize: Figure size (width, height)
    """
    plt.figure(figsize=figsize)
    
    # Plot historical data
    plt.plot(historical_data[date_column], 
            historical_data[target_column], 
            marker='o', 
            label='Historical', 
            color='blue',
            linewidth=2)
    
    # Plot predictions
    plt.plot(predictions['date'], 
            predictions['prediction'], 
            marker='s', 
            label='Forecast', 
            color='red',
            linewidth=2,
            linestyle='--')
    
    # Plot confidence intervals
    if 'lower_bound' in predictions.columns and 'upper_bound' in predictions.columns:
        plt.fill_between(predictions['date'], 
                        predictions['lower_bound'], 
                        predictions['upper_bound'], 
                        alpha=0.3,
                        color='red',
                        label='Confidence Interval')
    
    plt.xlabel('Date')
    plt.ylabel('Demand')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Forecast chart saved to: {output_path}")


def export_metrics_table(metrics_df: pd.DataFrame, 
                        output_path: str,
                        title: str = "Model Performance Metrics") -> None:
    """
    Create and export a table visualization of model metrics.
    
    Args:
        metrics_df: DataFrame with model metrics
        output_path: Path to save the chart
        title: Chart title
    """
    fig, ax = plt.subplots(figsize=(10, len(metrics_df) * 0.5 + 1))
    ax.axis('tight')
    ax.axis('off')
    
    # Create table
    table = ax.table(cellText=metrics_df.values,
                    colLabels=metrics_df.columns,
                    cellLoc='center',
                    loc='center',
                    colWidths=[0.15] * len(metrics_df.columns))
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style header
    for i in range(len(metrics_df.columns)):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(metrics_df) + 1):
        for j in range(len(metrics_df.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    plt.title(title, fontsize=14, weight='bold', pad=20)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Metrics table saved to: {output_path}")


def create_summary_report(historical_data: pd.DataFrame,
                         predictions_dict: Dict[str, pd.DataFrame],
                         metrics_df: pd.DataFrame,
                         output_dir: str,
                         date_column: str = 'date',
                         target_column: str = 'demand') -> None:
    """
    Create a complete summary report with all visualizations and data exports.
    
    Args:
        historical_data: DataFrame with historical data
        predictions_dict: Dictionary of model predictions
        metrics_df: DataFrame with model metrics
        output_dir: Directory to save all outputs
        date_column: Name of the date column
        target_column: Name of the target column
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Export predictions to Excel
    export_to_excel(predictions_dict, os.path.join(output_dir, 'predictions.xlsx'))
    
    # Export comparison chart
    export_comparison_chart(predictions_dict, 
                           os.path.join(output_dir, 'comparison_chart.png'))
    
    # Export forecast charts for each model
    for model_name, predictions in predictions_dict.items():
        export_forecast_with_history(historical_data,
                                     predictions,
                                     date_column,
                                     target_column,
                                     os.path.join(output_dir, f'forecast_{model_name}.png'),
                                     title=f'{model_name} Forecast')
    
    # Export metrics
    metrics_df.to_csv(os.path.join(output_dir, 'metrics.csv'), index=False)
    export_metrics_table(metrics_df, os.path.join(output_dir, 'metrics_table.png'))
    
    print(f"\n✓ Complete summary report created in: {output_dir}")
