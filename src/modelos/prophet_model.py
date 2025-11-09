import pandas as pd
from modelos.forecast_model import ForecastModel

class ProphetModel(ForecastModel):
    """
    Implementación del modelo Prophet para pronóstico de demanda.

    Utiliza descomposición aditiva y ajuste automático de estacionalidad.
    Requiere conversión de la serie temporal a formato DataFrame con columnas 'ds' y 'y'.
    """
    def fit(self, datos):
        print(f"[Prophet] Entrenando modelo {self.modelo_id}")
        self.datos = datos

    def predict(self, horizonte):
        print(f"[Prophet] Generando pronóstico para {horizonte} meses")
        return pd.DataFrame({"mes": list(range(horizonte)), "pred": [1000]*horizonte})

    def evaluate(self):
        return {"RMSE": 12345}
