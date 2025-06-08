import pandas as pd
import numpy as np

# === 1. Cargar y limpiar movies_test.csv ===
ruta_movies = "C:/Users/antho/OneDrive/Escritorio/streamlit/movies_test.csv"

data = []
with open(ruta_movies, 'r', encoding='utf-8') as f:
    lines = f.readlines()

for line in lines:
    line = line.strip().rstrip(';')
    if not line or line.startswith("movieId"):
        continue

    if line.startswith('"'):
        line = line.strip('"')
        movie_id, title = line.split(',', 1)
        title = title.replace('""', '"').strip('"')
    else:
        movie_id, title = line.split(',', 1)

    data.append([movie_id.strip(), title.strip()])

df_movies = pd.DataFrame(data, columns=['movieId', 'title'])

# === 2. Cargar coordenadas UMAP + clusters ===
df_umap = pd.read_csv("C:/Users/antho/OneDrive/Escritorio/streamlit/posters_pca_umap_clusters_2.csv")

df_movies['movieId'] = df_movies['movieId'].astype(str).str.strip()
df_umap['filename'] = df_umap['filename'].astype(str).str.strip()

# Verificar movieIDs faltantes
missing_ids = set(df_movies['movieId']) - set(df_umap['filename'])
if missing_ids:
    print(f"⚠️ movieIDs no encontrados en df_umap: {len(missing_ids)}")

# Renombrar columna y unir
df_umap = df_umap.rename(columns={'filename': 'movieId'})
df = pd.merge(df_umap, df_movies, on='movieId')

# === 3. Generar recomendaciones por proximidad UMAP en mismo cluster ===
recommendations = []

for _, row in df.iterrows():
    query_id = row['movieId']
    cluster = row['cluster']
    coords = np.array([row['UMAP1'], row['UMAP2']])

    same_cluster = df[(df['cluster'] == cluster) & (df['movieId'] != query_id)].copy()

    same_cluster['distance'] = np.linalg.norm(same_cluster[['UMAP1', 'UMAP2']].values - coords, axis=1)

    top_10 = same_cluster.drop_duplicates(subset='movieId').nsmallest(10, 'distance')

    for pos, rec in enumerate(top_10.itertuples(), 1):
        recommendations.append({
            'ID': f"{query_id}_{pos}",
            'query_movie_id': query_id,
            'recommended_movie_id': rec.movieId,
            'position': pos
        })

df_result = pd.DataFrame(recommendations)

# === 4. Completar con recomendaciones aleatorias (de cualquier cluster) ===
movie_ids_con_recs = set(df_result['query_movie_id'])
todos_los_ids = set(df_movies['movieId'])
faltantes = todos_los_ids - movie_ids_con_recs
print(f"🔍 Películas sin recomendaciones: {len(faltantes)}")

# Usar cualquier película con coordenadas UMAP
peliculas_base = df.copy()

for query_id in faltantes:
    recomendaciones_fake = peliculas_base.sample(n=10, replace=True, random_state=int(query_id) % 1000)
    for pos, rec in enumerate(recomendaciones_fake.itertuples(), 1):
        recommendations.append({
            'ID': f"{query_id}_{pos}",
            'query_movie_id': query_id,
            'recommended_movie_id': rec.movieId,
            'position': pos
        })

# === 5. Verificación final y guardado ===
df_result = pd.DataFrame(recommendations)
actual = len(df_result)
esperado = len(df_movies) * 10

assert actual == esperado, f"❌ ERROR: Se esperaban {esperado} filas pero se generaron {actual}."
print(f"✅ Total de filas generadas: {actual} (esperado: {esperado})")

df_result.to_csv("formatted_movie_recommendations.csv", index=False)
