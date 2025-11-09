from gestor import DemandForecasterSystem
from modelos.ml_model import MLRegressionModel

def test_sistema_integrado():
    sistema = DemandForecasterSystem()
    sistema.agregar_modelo(MLRegressionModel("ml_test"))
    resultados = sistema.generar_pronosticos(horizonte=2)
    assert "ml_test" in resultados
