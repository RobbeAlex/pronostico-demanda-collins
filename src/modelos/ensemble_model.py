import pandas as pd
from modelos.forecast_model import ForecastModel

class EnsembleModel(ForecastModel):
    """
    Modelo de ensamble que combina múltiples modelos base para mejorar el pronóstico.

    Puede usar promedio ponderado, stacking o votación para integrar predicciones
    de modelos como ARIMA, Prophet y ML.

    Métodos:
        fit(data: pd.Series) -> None
            Entrena todos los modelos base y ajusta el mecanismo de combinación.

        predict(steps: int) -> pd.Series
            Genera predicciones combinadas para los próximos 'steps' periodos.

        evaluate(...) -> dict
            Calcula métricas de error agregadas del ensamble.
    """
    def __init__(self, modelo_id, submodelos):
        super().__init__(modelo_id)
        self.submodelos = submodelos  # Lista de instancias de ForecastModel

    def fit(self, datos):
        print(f"[Ensemble] Entrenando modelo {self.modelo_id} con {len(self.submodelos)} submodelos")
        for modelo in self.submodelos:
            modelo.fit(datos)

    def predict(self, horizonte):
        print(f"[Ensemble] Generando pronóstico combinado para {horizonte} meses")
        predicciones = [modelo.predict(horizonte)["pred"] for modelo in self.submodelos]
        promedio = sum(predicciones) / len(predicciones)
        return pd.DataFrame({"mes": list(range(horizonte)), "pred": promedio})

    def evaluate(self):
        evaluaciones = [modelo.evaluate() for modelo in self.submodelos]
        return {"subevaluaciones": evaluaciones}
