## [0.1.1] - 2025-11-09
### Añadido
- Banner en  `README.md`
- Diagrama de Flujo en `README.md`
- Elementos graficos de resultados en `README.md`

## [0.1.0] - 2025-11-09
### Añadido
- Estructura modular inicial del sistema en `src/`
- Implementación de modelos: ARIMA, Prophet, ML y Ensemble
- Clase base `ForecastModel` con interfaz estándar (`fit`, `predict`, `evaluate`)
- Sistema orquestador `DemandForecasterSystem` en `core.py`
- Evaluación con métricas MAE, RMSE y MAPE
- README completo con arquitectura, diagramas UML y justificación técnica
- Diagrama ASCII y UML en `docs/`
- Archivo `requirements.txt` con dependencias clave
- Ejemplo funcional en `example_run.py`
- Pruebas unitarias básicas en `tests/test_core.py`
- Configuración inicial de CI con GitHub Actions (`python-ci.yml`)
- Licencia MIT en `LICENSE`
- Documentación técnica con Sphinx (`docs/`)

### Pendiente
- Integración de cobertura de pruebas con Codecov
- Validación cruzada y ajuste automático de hiperparámetros
- Interfaz CLI o GUI para usuarios no técnicos
- Publicación de release estable y documentación HTML
