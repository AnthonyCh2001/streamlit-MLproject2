import pandas as pd

# Cargar el archivo CSV
csv_path = './posters_pca_umap_clusters_modified.csv'  # Ajusta la ruta a tu archivo CSV
df = pd.read_csv(csv_path)
cluster = df[df['filename'] == int("1")]['cluster'].values
print(cluster)
