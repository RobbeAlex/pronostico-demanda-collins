import matplotlib.pyplot as plt

def graficar_pronostico(df, modelo_id):
    plt.plot(df["mes"], df["pred"], label=modelo_id)
    plt.title(f"Pronóstico – {modelo_id}")
    plt.xlabel("Mes")
    plt.ylabel("Unidades")
    plt.legend()
    plt.savefig(f"output/graficas/{modelo_id}.png")
    plt.close()
