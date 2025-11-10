# 1. Crear tabla pivote (matriz ancha)
df_wide = df_modelar_agg.pivot_table(
    index=['Producto - Descripción', 'Cliente - Descripción'],
    columns='Fecha',
    values='Pedido_Piezas' # Usar la columna correcta
).fillna(0)
print(f"Matriz para PCA: {df_wide.shape[0]} filas (combinaciones) x {df_wide.shape[1]} columnas (fechas)")

# 2. Verificar que tenemos suficientes datos
if df_wide.shape[0] < 2:
    print("ERROR: Necesitas al menos 2 combinaciones para hacer PCA")
else:

    # 3. Escalar los datos
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(df_wide)

    # 4. Aplicar PCA
    pca_model = PCA(n_components=2)
    principal_components = pca_model.fit_transform(data_scaled)

    # 5. Crear DataFrame con componentes principales
    df_pca = pd.DataFrame(
    data=principal_components,
    columns=['PC1', 'PC2'],
    index=df_wide.index
).reset_index()
print(f" PCA completado. Varianza explicada: PC1={pca_model.explained_variance_ratio_[0]:.1%}, PC2={pca_model.explained_variance_ratio_[1]:.1%}")

# 6. Graficar
plt.figure(figsize=(14, 10))

# Crear el scatter plot
scatter = sns.scatterplot(
    data=df_pca,
    x='PC1',
    y='PC2',
    hue='Cliente - Descripción',
    style='Producto - Descripción',
    s=200,
    alpha=0.7
)
plt.title('Clusters de Comportamiento de Demanda (PCA)', fontsize=16,fontweight='bold')
plt.xlabel(f'Componente Principal 1 ({pca_model.explained_variance_ratio_[0]:.1%} de varianza)', fontsize=12)
plt.ylabel(f'Componente Principal 2 ({pca_model.explained_variance_ratio_[1]:.1%} de varianza)', fontsize=12)

# Ajustar leyenda
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.,fontsize=9)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 7. Mostrar algunas estadísticas
print("\n--- Resumen de Componentes Principales ---")
print(df_pca.groupby('Cliente - Descripción')[['PC1', 'PC2']].describe())
