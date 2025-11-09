import pandas as pd

class ForecastModel:
    def __init__(self, modelo_id):
        self.modelo_id = modelo_id
        self.datos = None

    def fit(self, datos):
        print(f"[{self.modelo_id}] Método 'fit' no implementado.")

    def predict(self, horizonte):
        print(f"[{self.modelo_id}] Método 'predict' no implementado.")
        return pd.DataFrame()

    def evaluate(self):
        print(f"[{self.modelo_id}] Método 'evaluate' no implementado.")
        return {}
