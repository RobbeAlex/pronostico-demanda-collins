def exportar_csv(resultados, ruta):
    import pandas as pd
    df_final = pd.concat(resultados.values(), keys=resultados.keys())
    df_final.to_csv(ruta)
    print(f"Pronósticos exportados a {ruta}")
