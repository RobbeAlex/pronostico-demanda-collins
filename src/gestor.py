class DemandForecasterSystem:
    def __init__(self):
        self.modelos = []

    def agregar_modelo(self, modelo):
        self.modelos.append(modelo)

    def generar_pronosticos(self, horizonte):
        resultados = {}
        for modelo in self.modelos:
            modelo.fit("datos simulados")
            pred = modelo.predict(horizonte)
            resultados[modelo.modelo_id] = pred
        return resultados

    def exportar_resultados(self):
        print("Exportando resultados...")
