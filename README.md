# 📊 Sistema de Pronóstico de Demanda Farmacéutica – Grupo Collins

## 🧠 Descripción del Proyecto

Este sistema automatiza el análisis de series de tiempo para generar pronósticos mensuales por producto y cliente, incluyendo intervalos de confianza. Está diseñado para apoyar la toma de decisiones estratégicas en Grupo Collins, una empresa farmacéutica mexicana.

## 🧪 Justificación de la POO

El sistema se basa en Programación Orientada a Objetos para representar entidades como modelos de pronóstico, productos, clientes y reportes. Esto permite escalabilidad, modularidad y reutilización de código.

## 🧱 Estructura de Clases

- `ForecastModel`: superclase base
- Subclases:
  - `ProphetModel`
  - `ARIMAModel`
  - `MLRegressionModel`
  - `XGBoostModel`
  - `EnsembleModel`
- `DemandForecasterSystem`: clase gestora del flujo
- `DataSetLoader`, `ReportGenerator`, `Visualizer`: clases auxiliares

## 🔁 Polimorfismo

Todos los modelos implementan los métodos `fit()`, `predict()` y `evaluate()` de forma distinta. El sistema los trata de forma uniforme:

```python
for model in lista_modelos:
    model.fit(datos)
    resultado = model.predict(horizonte)
