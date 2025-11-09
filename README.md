## 📊 Sistema de Pronóstico de Demanda Farmacéutica – Grupo Collins

Este proyecto desarrolla un sistema capaz de generar pronósticos mensuales de demanda para productos farmacéuticos por cliente. Utiliza modelos de series de tiempo y Machine Learning para apoyar decisiones estratégicas en **Grupo Collins**, como planeación de producción y abastecimiento.

---

## 🎯 Objetivos del Proyecto

### ✅ Objetivo General
Construir un sistema en Python que pronostique la demanda de medicamentos utilizando Programación Orientada a Objetos (POO), integrando distintos modelos de predicción y generando reportes automatizados.

### 🎯 Objetivos Específicos
- Implementar un diseño modular basado en POO para garantizar escalabilidad y mantenibilidad.
- Comparar diferentes modelos de pronóstico (estadísticos y ML).
- Automatizar el proceso de carga de datos, entrenamiento, evaluación y generación de pronósticos.
- Generar visualizaciones y reportes que faciliten la toma de decisiones.

---

## 🧠 ¿Por qué POO?

El sistema está construido con Programación Orientada a Objetos porque:

| Ventaja POO | Impacto en el Proyecto |
|-------------|-------------------------|
| Modularidad | Cada modelo es una clase independiente |
| Reutilización | Se pueden agregar nuevos modelos sin reescribir el sistema |
| Polimorfismo | Todos los modelos comparten la misma estructura (`fit()`, `predict()`, `evaluate()`) |
| Escalabilidad | Permite integrar nuevos algoritmos sin modificar el flujo principal |

---
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

## 🧪 Modelos de Pronóstico Integrados

El sistema incluye una arquitectura flexible con diferentes tipos de modelos:

| Tipo | Modelo |
|------|---------|
| Estadístico | ARIMA |
| Clásico ML | Regresión Lineal |
| Ensamble | EnsembleModel (promedia el resultado de varios modelos) |

> 🧩 **Todos heredan de `ForecastModel`**, lo que permite tratarlos de forma uniforme.

Ejemplo del polimorfismo aplicado:

```python
for model in modelos:
    model.fit(data)
    pred = model.predict(horizon=12)
    score = model.evaluate(test_data)

```
## 🧱 Arquitectura del Proyecto
```bash
src/
│
├── models/                # Modelos de pronóstico
│   ├── forecast_model.py
│   ├── arima_model.py
│   ├── ml_regression_model.py
│   └── ensemble_model.py
│
├── core/
│   ├── demand_forecaster_system.py
│   ├── dataset_loader.py
│   ├── evaluation_result.py
│   ├── report_generator.py
│   └── visualizer.py
│
└── main.py                # Punto de entrada del sistema
```
## 🧬 Diagrama UML

## 🔤 Versión ASCII
```ASCII
+------------------+
|  ForecastModel   | (abstract)
+------------------+
| - name           |
+------------------+
| + fit(data)      |
| + predict(h)     |
| + evaluate(test) |
+--------+---------+
         ^
         |
   -----------------------------
   |             |             |
+----------+  +--------------------+   +------------------+
|ARIMAModel|  |MLRegressionModel   |   |EnsembleModel      |
+----------+  +--------------------+   +------------------+
| + fit()  |  | + fit()            |   | + fit()           |
| + pred() |  | + pred()           |   | + pred()          |
+----------+  +--------------------+   +------------------+

+------------------------+
| EvaluationResult       |
+------------------------+
| - model_name           |
| - rmse                 |
| - mae                  |
| - mape                 |
+------------------------+
| + summary()            |
+------------------------+

+-----------------------------+
| DemandForecasterSystem      |
+-----------------------------+
| - models                    |
| - results                   |
+-----------------------------+
| + add_model(model)          |
| + load_data(path)           |
| + run_all(horizon)          |
| + generate_report()         |
+-----------------------------+
```
## 🚀 Cómo Ejecutar el Proyecto
1️⃣ Clonar repositorio
```bash
git clone https://github.com/RobbeAlex/pronostico-demanda-collins.git
cd pronostico-demanda-collins
```
2️⃣ Instalar dependencias
```bash
pip install -r requirements.txt
```
3️⃣ Ejecutar
```bash
python src/main.py
```
