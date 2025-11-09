```mermaid
flowchart TD
    A([Inicio])
    B([Cargar datos históricos desde CSV])
    C([Instanciar DemandForecasterSystem])
    D([Agregar modelos de pronóstico])
    E([Entrenar modelos con datos])
    F([Generar pronósticos por modelo])
    G([Evaluar desempeño de cada modelo])
    H([Graficar resultados])
    I([Exportar pronósticos a CSV])
    J([Fin])

    subgraph Modelos
        D1[[ProphetModel]]
        D2[[ARIMAModel]]
        D3[[MLRegressionModel]]
        D4[[EnsembleModel]]
    end

    A --> B --> C --> D
    D --> D1 & D2 & D3 & D4
    D4 --> E --> F --> G --> H --> I --> J
