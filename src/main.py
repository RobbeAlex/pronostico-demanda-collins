from gestor import DemandForecasterSystem
from modelos.prophet_model import ProphetModel

if __name__ == "__main__":
    sistema = DemandForecasterSystem()
    sistema.agregar_modelo(ProphetModel("modelo_prophet"))
    resultados = sistema.generar_pronosticos(horizonte=6)
    for modelo_id, df in resultados.items():
        print(f"\nPronóstico generado por {modelo_id}:")
        print(df)
