class DemandForecasterSystem:
    """
    Sistema central de pronóstico de demanda.

    Orquesta la carga de datos, entrenamiento de modelos, generación de predicciones
    y evaluación de desempeño. Integra múltiples modelos bajo una arquitectura modular.

    Métodos:
        load_data() -> pd.Series
            Carga y preprocesa la serie temporal de demanda.

        run_pipeline() -> dict
            Ejecuta el flujo completo de pronóstico:
            - Entrena cada modelo
            - Genera predicciones
            - Evalúa desempeño
            - Devuelve métricas por modelo
    """
