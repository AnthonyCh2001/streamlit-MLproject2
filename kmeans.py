import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import umap  


# funcion de clustering kmeans++

def kmeans_plus_plus(X, n_clusters=5, max_iter=100, tol=1e-4):
    n_samples, n_features = X.shape
    np.random.seed(42)

    # Se inicializa el primer centroide aleatoriamente
    centroids = np.zeros((n_clusters, n_features))
    centroids[0] = X[np.random.choice(n_samples)]

    # Se seleccionan los siguientes centroides de manera probabilística
    for i in range(1, n_clusters):
        # Calcular la distancia de cada punto al centroide más cercano
        dist_squared = np.min(np.linalg.norm(X[:, np.newaxis] - centroids[:i], axis=2)**2, axis=1)
        prob = dist_squared / np.sum(dist_squared)
        centroids[i] = X[np.random.choice(n_samples, p=prob)]

    # Se hace iteraciones hasta la convergencia o máximo de iteraciones
    labels = np.zeros(n_samples)
    for iteration in range(max_iter):
        # Se asigna cada punto al centroide más cercano
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        new_labels = np.argmin(distances, axis=1)

        # Se Verifica si las etiquetas cambiaron
        if np.all(new_labels == labels):
            break
        labels = new_labels

        # Se actualizan los centroides
        for i in range(n_clusters):
            centroids[i] = X[labels == i].mean(axis=0)

    return labels, centroids


# Se guardan los datos y se excluyen algunas columnas antes de usar normalizacion +pca + umap
df = pd.read_csv("caracteristicas_posters_combinado.csv")
filenames = df['filename'].str.replace('.jpg', '', regex=False)  
columnas_excluir = ['filename', '(no genres listed)','Action','Adventure','Animation','Children',
    'Comedy','Crime','Documentary','Drama','Fantasy','Film-Noir','Horror','IMAX','Musical',
    'Mystery','Romance','Sci-Fi','Thriller','War','Western']

columnas_visuales = [col for col in df.columns if col not in columnas_excluir]
X = df[columnas_visuales].to_numpy()


# Se normalizan los datos y se aplica pca junto con un print de la varianza explicada y varianza acumulada para observar cuanta informacion nos proporciona

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA()
pca.fit(X_scaled)
explained_variance = pca.explained_variance_ratio_
print(f"Varianza explicada por cada componente: {explained_variance}")
print(f"Varianza acumulada: {np.cumsum(explained_variance)}")
# Se aplica PCA conservando 95% de varianza
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)
print(f"Dimensiones después de PCA (95% varianza): {X_pca.shape[1]}")
# Se aplica UMAP sobre los datos reducidos por PCA
umap_reducer = umap.UMAP(n_components=2, random_state=42)
X_umap = umap_reducer.fit_transform(X_pca)
# Se aplica clustering con K-means++ sobre los datos reducidos por UMAP
labels, centroids = kmeans_plus_plus(X_umap, n_clusters=40)
# Se calculan las métricas de Silhouette y Calinski-Harabasz
silhouette_avg = silhouette_score(X_umap, labels)
calinski_avg = calinski_harabasz_score(X_umap, labels)
print(f"Silhouette Score: {silhouette_avg:.3f}")
print(f"Calinski-Harabasz Score: {calinski_avg:.3f}")

# Se guarda el resultado final para usarlo en el recomendador de peliculas (en caso se escoja este método para ello)
df_resultado = pd.DataFrame({
    'filename': filenames,
    'cluster': labels,
    'UMAP1': X_umap[:, 0],  # Componente 1 de UMAP
    'UMAP2': X_umap[:, 1]   # Componente 2 de UMAP
})

df_resultado.to_csv("posters_pca_umap_clusters.csv", index=False)
print("CSV generado: posters_pca_umap_clusters.csv")

# Se grafican los clusters
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_umap[:, 0], X_umap[:, 1], c=labels, cmap='tab20', s=10) 
plt.title("Clusters visualizados con UMAP (tras PCA 95%)")
plt.xlabel("UMAP1")
plt.ylabel("UMAP2")
plt.legend(*scatter.legend_elements(), title="Cluster")
plt.tight_layout()
plt.show()
