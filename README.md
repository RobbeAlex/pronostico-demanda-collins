# Pronostico Demanda Collins

Sistema en Python con Programación Orientada a Objetos para generar pronósticos mensuales de demanda por producto y cliente. Incluye modelos como Prophet, ARIMA y regresión, con intervalos de confianza y visualización. Proyecto integrador UDG 2025.

## 📋 Características

- **Arquitectura Orientada a Objetos**: Diseño modular con clases base y especializadas
- **Múltiples Modelos de Pronóstico**:
  - **Prophet**: Modelo de Facebook para series temporales con estacionalidad
  - **ARIMA**: Modelo AutoRegresivo Integrado de Media Móvil
  - **ML Regression**: Modelos de Machine Learning (Random Forest, Regresión Lineal)
- **Gestión Centralizada**: Clase ForecastManager para coordinar múltiples modelos
- **Evaluación Completa**: Métricas MAE, MSE, RMSE, MAPE, sMAPE, R²
- **Intervalos de Confianza**: Estimación de incertidumbre en predicciones
- **Exportación de Resultados**: CSV, Excel, y visualizaciones gráficas

## 🚀 Instalación

### Requisitos previos
- Python 3.8 o superior
- pip (gestor de paquetes de Python)

### Pasos de instalación

1. Clonar el repositorio:
```bash
git clone https://github.com/RobbeAlex/pronostico-demanda-collins.git
cd pronostico-demanda-collins
```

2. Instalar dependencias:
```bash
pip install -r requirements.txt
```

## 📖 Uso

### Ejemplo Básico

```python
from data_loader import generate_sample_data
from prophet_model import ProphetModel
from arima_model import ARIMAModel
from ml_regression_model import MLRegressionModel
from forecast_manager import ForecastManager

# 1. Cargar o generar datos
data = generate_sample_data(periods=36)

# 2. Inicializar modelos
prophet = ProphetModel(name="Prophet")
arima = ARIMAModel(name="ARIMA", order=(1, 1, 1))
ml_model = MLRegressionModel(name="RandomForest")

# 3. Configurar el gestor
manager = ForecastManager()
manager.add_model(prophet)
manager.add_model(arima)
manager.add_model(ml_model)

# 4. Entrenar modelos
manager.fit_all(data, target_column='demand', date_column='date')

# 5. Generar predicciones
predictions = manager.predict_all(periods=12)

# 6. Obtener predicciones ensemble
ensemble = manager.get_ensemble_predictions(method="mean")
```

### Ejecutar Ejemplo Completo

```bash
python example_usage.py
```

## 🏗️ Estructura del Proyecto

```
pronostico-demanda-collins/
├── base_model.py           # Clase base abstracta para modelos
├── prophet_model.py        # Implementación del modelo Prophet
├── arima_model.py          # Implementación del modelo ARIMA
├── ml_regression_model.py  # Implementación de modelos ML
├── forecast_manager.py     # Gestor de múltiples modelos
├── data_loader.py          # Utilidades para carga de datos
├── evaluation.py           # Métricas de evaluación
├── exporter.py             # Exportación de resultados
├── example_usage.py        # Ejemplo de uso completo
├── requirements.txt        # Dependencias del proyecto
└── README.md              # Documentación
```

## 🔧 Componentes Principales

### BaseModel (Clase Abstracta)
Clase base que define la interfaz común para todos los modelos:
- `fit()`: Entrenar el modelo
- `predict()`: Generar predicciones
- `get_confidence_intervals()`: Obtener intervalos de confianza

### ProphetModel
Implementación del modelo Prophet de Facebook:
- Detección automática de tendencias
- Manejo de estacionalidad múltiple
- Intervalos de confianza incluidos

### ARIMAModel
Modelo ARIMA tradicional:
- Configurable con parámetros (p, d, q)
- Adecuado para series temporales estacionarias
- Intervalos de confianza estadísticos

### MLRegressionModel
Modelos de Machine Learning para pronósticos:
- Random Forest (predeterminado)
- Regresión Lineal
- Ingeniería de características automática (lags, rolling stats)
- Intervalos de confianza estimados

### ForecastManager
Gestión centralizada de múltiples modelos:
- Entrenamiento de todos los modelos
- Generación de predicciones paralelas
- Comparación de modelos
- Predicciones ensemble (promedio, mediana)

## 📊 Utilidades

### Carga de Datos (data_loader.py)
- `load_csv_data()`: Cargar datos desde CSV
- `load_excel_data()`: Cargar datos desde Excel
- `generate_sample_data()`: Generar datos de prueba
- `filter_by_product_client()`: Filtrar por producto/cliente
- `validate_data()`: Validar formato de datos
- `aggregate_by_period()`: Agregar datos por período

### Evaluación (evaluation.py)
Métricas disponibles:
- MAE (Mean Absolute Error)
- MSE (Mean Squared Error)
- RMSE (Root Mean Squared Error)
- MAPE (Mean Absolute Percentage Error)
- sMAPE (Symmetric MAPE)
- R² (Coefficient of Determination)
- Bias (sesgo de predicción)
- Coverage (cobertura de intervalos)

### Exportación (exporter.py)
- `export_to_csv()`: Exportar a CSV
- `export_to_excel()`: Exportar a Excel
- `export_comparison_chart()`: Gráfico de comparación
- `export_forecast_with_history()`: Gráfico con histórico
- `export_metrics_table()`: Tabla de métricas
- `create_summary_report()`: Reporte completo

## 💡 Ejemplos de Uso Avanzado

### Cargar Datos Propios
```python
from data_loader import load_csv_data, validate_data

# Cargar datos
data = load_csv_data('mis_datos.csv', date_column='fecha')

# Validar
validate_data(data, required_columns=['fecha', 'demanda'])
```

### Configurar Modelos Específicos
```python
# Prophet con parámetros personalizados
prophet = ProphetModel(
    name="Prophet_Custom",
    yearly_seasonality=True,
    weekly_seasonality=False,
    changepoint_prior_scale=0.05
)

# ARIMA con orden específico
arima = ARIMAModel(name="ARIMA_211", order=(2, 1, 1))

# Random Forest optimizado
ml_model = MLRegressionModel(
    name="RF_Optimized",
    model_type="random_forest",
    n_estimators=200,
    max_depth=10
)
```

### Evaluación con Datos de Prueba
```python
from evaluation import evaluate_predictions, compare_models

# Dividir datos
train_data = data[:-12]  # Todos menos últimos 12 meses
test_data = data[-12:]   # Últimos 12 meses

# Entrenar
manager.fit_all(train_data, 'demand', 'date')

# Predecir
predictions = manager.predict_all(12)

# Evaluar cada modelo
for model_name, pred_df in predictions.items():
    metrics = evaluate_predictions(
        test_data['demand'], 
        pred_df['prediction']
    )
    print(f"{model_name}: {metrics}")
```

### Generar Reporte Completo
```python
from exporter import create_summary_report
from evaluation import compare_models

# Obtener predicciones de todos los modelos
predictions = manager.predict_all(12)

# Crear DataFrame de métricas (si tienes datos de prueba)
predictions_series = {
    name: df['prediction'] 
    for name, df in predictions.items()
}
metrics_df = compare_models(test_data['demand'], predictions_series)

# Generar reporte
create_summary_report(
    historical_data=train_data,
    predictions_dict=predictions,
    metrics_df=metrics_df,
    output_dir='resultados_pronostico',
    date_column='date',
    target_column='demand'
)
```

## 📦 Dependencias

- **pandas**: Manipulación de datos
- **numpy**: Operaciones numéricas
- **scikit-learn**: Modelos de Machine Learning
- **prophet**: Modelo Prophet de Facebook
- **statsmodels**: Modelos ARIMA
- **matplotlib**: Visualizaciones
- **seaborn**: Visualizaciones avanzadas

## 🤝 Contribución

Este es un proyecto académico para UDG 2025. Para sugerencias o mejoras, por favor abrir un issue en el repositorio.

## 📝 Licencia

Proyecto Integrador UDG 2025

## 👥 Autores

Proyecto desarrollado para el curso integrador de la Universidad de Guadalajara 2025.
