import pandas as pd
from modelos.forecast_model import ForecastModel

class MLRegressionModel(ForecastModel):
    def fit(self, datos):
        print(f"[ML] Entrenando modelo {self.modelo_id}")
        self.datos = datos

    def predict(self, horizonte):
        print(f"[ML] Generando pronóstico para {horizonte} meses")
        return pd.DataFrame({"mes": list(range(horizonte)), "pred": [1100]*horizonte})

    def evaluate(self):
        return {"R2": 0.85}
