```mermaid
flowchart TD
classDiagram
    %% Superclase base
    class ForecastModel {
        - modelo_id: str
        - datos: DataFrame
        + fit(datos): void
        + predict(horizonte): DataFrame
        + evaluate(): dict
    }

    %% Subclases
    class ProphetModel {
        + fit(datos): void
        + predict(horizonte): DataFrame
        + evaluate(): dict
    }

    class ARIMAModel {
        + fit(datos): void
        + predict(horizonte): DataFrame
        + evaluate(): dict
    }

    class MLRegressionModel {
        + fit(datos): void
        + predict(horizonte): DataFrame
        + evaluate(): dict
    }

    class EnsembleModel {
        - submodelos: list
        + fit(datos): void
        + predict(horizonte): DataFrame
        + evaluate(): dict
    }

    %% Clase gestora
    class DemandForecasterSystem {
        - modelos: list
        + agregar_modelo(modelo): void
        + generar_pronosticos(horizonte): dict
        + exportar_resultados(): void
    }

    %% Utilidades
    class DataSetLoader {
        + cargar_datos(ruta): DataFrame
    }

    class Visualizer {
        + graficar_pronostico(df, modelo_id): void
    }

    class ReportGenerator {
        + exportar_csv(resultados, ruta): void
    }

    %% Relaciones
    ForecastModel <|-- ProphetModel
    ForecastModel <|-- ARIMAModel
    ForecastModel <|-- MLRegressionModel
    ForecastModel <|-- EnsembleModel

    DemandForecasterSystem --> ForecastModel
    DemandForecasterSystem --> DataSetLoader
    DemandForecasterSystem --> Visualizer
    DemandForecasterSystem --> ReportGenerator
