class ForecastModel:
    """
    Clase base abstracta para modelos de pronóstico de demanda.

    Define la interfaz estándar que deben implementar todos los modelos concretos:
    - fit: entrenamiento del modelo con datos históricos
    - predict: generación de pronósticos futuros
    - evaluate: cálculo de métricas de desempeño

    Métodos:
        fit(data: pd.Series) -> None
            Entrena el modelo con la serie temporal proporcionada.

        predict(steps: int) -> pd.Series
            Genera predicciones para los próximos 'steps' periodos.

        evaluate(true_values: pd.Series, predicted_values: pd.Series) -> dict
            Calcula métricas de error entre valores reales y predichos.
    """

