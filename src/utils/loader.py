import pandas as pd

def cargar_datos(ruta):
    print(f"Cargando datos desde {ruta}")
    return pd.read_csv(ruta)
