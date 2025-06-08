import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score
import umap
from scipy.spatial import KDTree
import heapq
import warnings
warnings.filterwarnings("ignore")


# Función para clusterización con OPTICS

def optics_from_scratch(X, eps=0.03, min_samples=20):
    n = len(X)  # Número de puntos en el dataset
    reachability = np.full(n, np.inf)  # Distancia de alcanzabilidad, que por defecto se debe colocar en infinito
    processed = np.zeros(n, dtype=bool)  # Marca para los puntos que ya han sido procesados
    core_dists = np.full(n, np.inf)  # Distancia núcleo de cada punto
    order = []  # Orden en que se procesan los puntos 

    # Crear estructura KDTree para búsqueda eficiente de vecinos
    tree = KDTree(X)

    # Calcular la distancia núcleo, osea el core distance, de cada punto
    for i in range(n):
        dists, _ = tree.query(X[i], k=min_samples + 1)  # +1 porque incluye al propio punto
        core_dists[i] = dists[-1]  # Última distancia del vecindario es la core distance

    # Función para actualizar la distancia de alcanzabilidad de los vecinos
    def update(p_idx, neighbors):
        for n_idx in neighbors:
            if not processed[n_idx]:
                # Distancia de alcanzabilidad = máx entre core distance y distancia euclidiana
                new_reach_dist = max(core_dists[p_idx], np.linalg.norm(X[p_idx] - X[n_idx]))
                # Guarda la menor distancia posible
                if reachability[n_idx] == np.inf:
                    reachability[n_idx] = new_reach_dist
                else:
                    reachability[n_idx] = min(reachability[n_idx], new_reach_dist)

    # Recorrer todos los puntos del dataset
    for i in range(n):
        if processed[i]:
            continue  # Si ya fue procesado, se salta

        # Buscar vecinos dentro del radio eps
        neighbors = tree.query_ball_point(X[i], eps)

        # Agregar el punto al orden de recorrido
        order.append(i)
        processed[i] = True  # Marcar como procesado

        # Si no tiene suficientes vecinos, no es núcleo → no se expande
        if core_dists[i] == np.inf:
            continue

        # Lista de semillas 
        seeds = []
        update(i, neighbors)  # Se actualizan las distancias de alcanzabilidad de sus vecinos

        for idx in neighbors:
            if not processed[idx]:
                seeds.append((reachability[idx], idx))  # Se agrega al heap

        heapq.heapify(seeds)  # Crea la cola de prioridad

        # Expansión hacia los vecinos más cercanos
        while seeds:
            _, current = heapq.heappop(seeds)
            if processed[current]:
                continue

            neighbors_c = tree.query_ball_point(X[current], eps)
            order.append(current)
            processed[current] = True

            if core_dists[current] != np.inf:
                update(current, neighbors_c)
                for idx in neighbors_c:
                    if not processed[idx]:
                        heapq.heappush(seeds, (reachability[idx], idx))

    # Se realiza el etiquetado de clusteres de acuerdo al umbral

    labels = -np.ones(n, dtype=int)  # -1: sin clúster asignado
    cluster_id = 0
    # Umbral para identificar límites de clúster (percentil 50 = mediana)
    threshold = np.percentile(reachability[np.isfinite(reachability)], 85)

    for idx in order:
        # Si la distancia de alcanzabilidad es alta y el punto es núcleo, empieza nuevo clúster
        if reachability[idx] > threshold:
            if core_dists[idx] <= eps:
                cluster_id += 1
        # Se asigna el clúster actual al punto
        labels[idx] = cluster_id

    # Retorna las etiquetas, distancias de alcanzabilidad y orden de recorrido
    return labels, reachability, order


# Se cargan los datos y se excluyen columnas

df = pd.read_csv("caracteristicas_posters_combinado.csv")
filenames = df['filename'].str.replace('.jpg', '', regex=False)
columnas_excluir = ['filename', '(no genres listed)', 'Action','Adventure','Animation','Children',
    'Comedy','Crime','Documentary','Drama','Fantasy','Film-Noir','Horror','IMAX','Musical',
    'Mystery','Romance','Sci-Fi','Thriller','War','Western']
columnas_visuales = [col for col in df.columns if col not in columnas_excluir]
X = df[columnas_visuales].to_numpy()


# Normalizar data + PCA + UMAP

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=0.95, random_state=42)
X_pca = pca.fit_transform(X_scaled)
print(f"Dimensiones después de PCA (95% varianza): {X_pca.shape[1]}")
umap_reducer = umap.UMAP(n_components=2, n_neighbors=10, min_dist=0.1, random_state=42)
X_umap = umap_reducer.fit_transform(X_pca)


# Clusterizado con OPTICS

labels, reachability, order = optics_from_scratch(X_umap, eps=0.0299, min_samples=8)
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
print(f"Clusters detectados (excluyendo outliers): {n_clusters}")

# Métricas

mask_valid = labels != -1
if np.sum(mask_valid) > 1 and n_clusters > 1:
    calinski_avg = calinski_harabasz_score(X_umap[mask_valid], labels[mask_valid])
    davies_bouldin = davies_bouldin_score(X_umap[mask_valid], labels[mask_valid])
    print(f"Calinski-Harabasz Score: {calinski_avg:.3f}")
    print(f"Davies-Bouldin Index: {davies_bouldin:.3f}")
else:
    print("No se pueden calcular métricas: solo un cluster válido o muy pocos puntos.")


# Se guarda el resultado final para usarlo en el recomendador de peliculas (en caso se escoja este método para ello)
df_resultado = pd.DataFrame({
    'filename': filenames,
    'cluster': labels,
    'UMAP1': X_umap[:, 0],
    'UMAP2': X_umap[:, 1]
})
ruta_salida = "posters_pca_umap_OPTICS_MANUAL.csv"
df_resultado.to_csv(ruta_salida, index=False)
print(f"📁 CSV generado: {ruta_salida}")


# Visualizar los clusters

plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_umap[:, 0], X_umap[:, 1], c=labels, cmap='tab20', s=5)
plt.title("Clusters visualizados con UMAP (tras PCA 95%) - OPTICS Manual")
plt.xlabel("UMAP1")
plt.ylabel("UMAP2")
plt.legend(*scatter.legend_elements(), title="Cluster", loc="best", bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.show()
