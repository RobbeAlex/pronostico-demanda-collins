from src.core import DemandForecasterSystem

if __name__ == "__main__":
    system = DemandForecasterSystem()
    results = system.run_pipeline()
    print("Resultados del pronóstico:")
    for model, metrics in results.items():
        print(f"{model}: {metrics}")
