import pandas as pd
from modelos.forecast_model import ForecastModel

class ARIMAModel(ForecastModel):
    def fit(self, datos):
        print(f"[ARIMA] Entrenando modelo {self.modelo_id}")
        self.datos = datos

    def predict(self, horizonte):
        print(f"[ARIMA] Generando pronóstico para {horizonte} meses")
        return pd.DataFrame({"mes": list(range(horizonte)), "pred": [900]*horizonte})

    def evaluate(self):
        return {"MAE": 9876}
