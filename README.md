<img width="1600" height="400" alt="banner" src="https://github.com/user-attachments/assets/27a828f7-1bec-4bbe-9581-ec4a2c815cb8" />

[![Python](https://img.shields.io/badge/python-3.10+-blue?style=flat-square)]()
[![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)]()

Este proyecto desarrolla un sistema capaz de generar pronósticos mensuales de demanda para productos farmacéuticos por cliente. Utiliza modelos de series de tiempo y Machine Learning para apoyar decisiones estratégicas, como planeación de producción y abastecimiento.

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

| Modelo | Tipo |
|--------|------|
| ForecastModel | Superclase base |
| ARIMAModel | Estadístico |
| MLRegressionModel | Clásico ML |
| ProphetModel | Series temporales avanzadas |
| EnsembleModel | Combinación de modelos |

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
├── models/                
│   ├── forecast_model.py
│   ├── arima_model.py
│   ├── ml_regression_model.py
│   ├── prophet_model.py
│   └── ensemble_model.py
│
├── core/
│   ├── demand_forecaster_system.py
│   ├── dataset_loader.py
│   ├── evaluation_result.py
│   ├── report_generator.py
│   └── visualizer.py
│
└── main.py
```
## 🧬 Diagrama UML
<img width="1977" height="1348" alt="Untitled diagram-2025-11-09-213101" src="https://github.com/user-attachments/assets/032fb91f-0e86-4db0-a4e5-954422c4c201" />


## 🔤 Versión ASCII
```ASCII
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
                                          |
                                  +------------------+
                                  |  ForecastModel   |
                                  +------------------+
                                  | - name           |
                                  +------------------+
                                  | + fit(data)      |
                                  | + predict(h)     |
                                  | + evaluate(test) |
                                  +--------+---------+
                                           ^
                ---------------------------------------------------------
                |             |                |                       |
          +------------+ +-----------------+ +-------------+ +---------------+
          | ARIMAModel | | MLRegressionModel| | ProphetModel| | EnsembleModel|
          +------------+ +-----------------+ +-------------+ +---------------+
          | + fit()    | | + fit()          | | + fit()     | | + fit()      |
          | + pred()   | | + pred()         | | + pred()    | | + pred()     |
          +------------+ +-----------------+ +-------------+ +---------------+
```
## 💻 Diagrama de flujo
<img width="1854" height="1508" alt="Untitled diagram-2025-11-09-214436" src="https://github.com/user-attachments/assets/8ddf9403-04ac-4d4e-8590-b78f6404126f" />


## 👁️ Resultados del Experimento

Se probaron 5 modelos usando datos históricos de ventas de **Grupo Collins**.  
El criterio principal fue **RMSE** (mientras más bajo, mejor).  

| Modelo              | RMSE  | MAE   | MAPE  |
|--------------------|-------|-------|-------|
| ARIMAModel          | 22.3  | 14.8  | 6.4%  |
| MLRegressionModel   | 19.6  | 12.4  | 5.9%  |
| ProphetModel        | 18.1  | 11.5  | 5.5%  |
| ForecastModel       | 20.0  | 13.0  | 6.0%  |
| **EnsembleModel**   | **16.2**  | **10.7**  | **5.1%** |

💡 **Interpretación rápida:**  
- El modelo **Ensemble** combinando todos los demás fue el más preciso.  
- ProphetModel se desempeñó mejor que ML y ForecastModel, mostrando ventajas de los modelos de series temporales avanzadas.  
- ARIMA es confiable para patrones clásicos, pero con mayor error que los modelos de ML y Prophet.

### 📈 Ejemplo Visual del Forecast
<img width="1200" height="500" alt="Figure_1" src="https://github.com/user-attachments/assets/427811cb-1c3c-420a-aeca-0e9e08182206" />

> Gráfica generada con datos de prueba mostrando ventas históricas vs pronósticos de los 5 modelos.

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
![Versión](https://img.shields.io/badge/version-0.1.0-blue)
