import numpy as np
import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv
import os
from scipy.spatial import distance

# Cargar las variables de entorno desde el archivo .env
load_dotenv()

# Obtener la clave de API de TMDb desde las variables de entorno
API_KEY = os.getenv('TMDB_API_KEY')

# Verificar que la API_KEY está cargada correctamente
if API_KEY is None:
    st.error("La clave de API de TMDb no está configurada correctamente.")
    raise ValueError("API Key no encontrada. Por favor, configura el archivo .env correctamente.")


# Función para obtener el poster desde TMDb

def obtener_poster(tmdb_id):
    """Obtener el poster de la película desde TMDb usando el tmdb_id"""
    url = f"https://api.themoviedb.org/3/movie/{tmdb_id}?api_key={API_KEY}"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        poster_path = data.get('poster_path')

        if poster_path:
            # Generar la URL completa del poster
            poster_url = f"https://image.tmdb.org/t/p/w500{poster_path}"
            return poster_url
    return None
# Función para calcular los 5 vecinos más cercanos
def obtener_vecinos_UMAP(df, image_id, n=5):
    """Calcular los 5 vecinos más cercanos en el espacio UMAP"""
    
    # Obtener las componentes UMAP de la película seleccionada
    movie_data = df[df['filename'] == image_id]
    umap_selected = movie_data[['UMAP1', 'UMAP2']].values[0]  # Componentes UMAP de la película seleccionada
    
    # Filtrar las películas del mismo cluster
    cluster = movie_data['cluster'].values[0]
    cluster_data = df[df['cluster'] == cluster]
    
    # Calcular la distancia euclidiana entre la película seleccionada y todas las demás en el mismo cluster
    distances = []
    for index, row in cluster_data.iterrows():
        umap_other = row[['UMAP1', 'UMAP2']].values
        dist = distance.euclidean(umap_selected, umap_other)  # Calcular la distancia euclidiana
        distances.append((row['filename'], dist))  # Guardar el nombre de la película y la distancia

    # Ordenar las distancias y obtener los 5 más cercanos, excluyendo la película seleccionada
    distances.sort(key=lambda x: x[1])
    closest_movies = [movie for movie, _ in distances if movie != image_id][:n]  # Excluir la película seleccionada

    return closest_movies


# Mostrar el poster y recomendaciones por cluster
def mostrar_poster_y_recomendaciones(df, links_df, image_id, n=5):
    """Mostrar el poster de la imagen e ir a las recomendaciones del mismo cluster"""
    
    # Convertir image_id a entero
    image_id = int(image_id)  # Convertimos a string para compararlo con los filenames

    # Verificar si el image_id existe en el DataFrame df
    if image_id not in df['filename'].values:
        st.write(f"Imagen con ID '{image_id}' no encontrada.")
        return

    # Buscar el cluster de la imagen proporcionada por el usuario
    cluster = df[df['filename'] == image_id]['cluster'].values[0]
    st.subheader(f"Poster de la imagen '{image_id}' (Cluster {cluster})")

    # Buscar el tmdbId correspondiente en links.csv usando movieId
    movie_id = int(image_id)  # Usar image_id como movieId (convertido a int)
    tmdb_id = links_df[links_df['movieId'] == movie_id]['tmdbId'].values[0]

    # Obtener el poster de la película desde TMDb
    poster_url = obtener_poster(tmdb_id)
    
    if poster_url:
        st.image(poster_url, caption=f"Poster de película con ID {image_id}", use_container_width=True)
    else:
        st.write("No se pudo obtener el poster de la película.")

    # Obtener los 5 vecinos más cercanos en el mismo cluster
    st.subheader(f"Recomendaciones para el Cluster {cluster}")
    recomendaciones = obtener_vecinos_UMAP(df, image_id, n)

    # Mostrar las recomendaciones
    col1, col2, col3, col4, col5 = st.columns(5)  # Crear 5 columnas para mostrar imágenes
    for i, f in enumerate(recomendaciones):
        # Usar directamente f como movieId
        movie_id_recomendacion = f  # Aquí, el filename es el mismo que movieId
        
        # Buscar el tmdbId de la recomendación en links.csv
        tmdb_id_recomendacion = links_df[links_df['movieId'] == int(movie_id_recomendacion)]['tmdbId'].values[0]
        poster_url_recomendacion = obtener_poster(tmdb_id_recomendacion)
        
        if poster_url_recomendacion:
            column = [col1, col2, col3, col4, col5][i]
            column.image(poster_url_recomendacion, caption=f, use_container_width=True)
        else:
            st.write(f"Imagen no encontrada para {f}")
    st.markdown("---")

# Main

# Cargar los CSV generados con K-Means++ y links.csv
csv_path = "./posters_pca_umap_clusters.csv"  # CSV con los clusters en el mismo nivel que app.py
links_csv_path = "./links.csv"  # CSV de links en el mismo nivel que app.py

# Cargar los CSVs
df = pd.read_csv(csv_path)
links_df = pd.read_csv(links_csv_path)

# Entrada de ID de imagen
image_id = st.text_input("Ingresa el ID de la imagen:")  # Ingresar el ID de la película

# Si el usuario ingresa un ID, mostrar el poster y las recomendaciones
if image_id:
    mostrar_poster_y_recomendaciones(df, links_df, image_id, n=5)
