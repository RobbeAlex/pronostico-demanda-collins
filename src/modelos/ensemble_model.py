import pandas as pd
from modelos.forecast_model import ForecastModel

class EnsembleModel(ForecastModel):
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
