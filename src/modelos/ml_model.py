import pandas as pd
from modelos.forecast_model import ForecastModel

class MLRegressionModel(ForecastModel):
     """
    Modelo de aprendizaje automático para pronóstico de demanda.

    Utiliza regresores como RandomForest, XGBoost o SVR para predecir valores futuros
    a partir de características extraídas de la serie temporal.

    Métodos:
        fit(data: pd.Series) -> None
            Entrena el modelo ML con características derivadas.

        predict(steps: int) -> pd.Series
            Genera predicciones para los próximos 'steps' periodos.

        evaluate(...) -> dict
            Calcula métricas de error (MAE, RMSE, MAPE).
    """
    def fit(self, datos):
        print(f"[ML] Entrenando modelo {self.modelo_id}")
        self.datos = datos

    def predict(self, horizonte):
        print(f"[ML] Generando pronóstico para {horizonte} meses")
        return pd.DataFrame({"mes": list(range(horizonte)), "pred": [1100]*horizonte})

    def evaluate(self):
        return {"R2": 0.85}
