import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.cluster import KMeans
from skimage.filters import sobel
from skimage.measure import shannon_entropy
from skimage.feature import hog

# Rutas
image_dir = ""
movies_csv = "movies.csv"

# Leer CSV con géneros múltiples por película
movies_df = pd.read_csv(movies_csv)
movies_df['filename'] = movies_df['movieId'].astype(str) + '.jpg'

# Expandir géneros en columnas (One Hot multi-label)
genre_dummies = movies_df['genres'].str.get_dummies(sep='|')
movies_df = pd.concat([movies_df[['filename']], genre_dummies], axis=1)
genre_dict = movies_df.set_index('filename').to_dict(orient='index')

# Clasificador Haar para rostros
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Saliency
def compute_saliency_opencv(img_gray):
    img_gray = cv2.resize(img_gray, (224, 224))
    img_float = img_gray.astype('float32')
    spectrum = np.fft.fft2(img_float)
    log_amplitude = np.log1p(np.abs(spectrum))
    phase = np.angle(spectrum)
    avg_log_amp = cv2.blur(log_amplitude, (3, 3))
    spectral_residual = log_amplitude - avg_log_amp
    saliency = np.abs(np.fft.ifft2(np.exp(spectral_residual + 1j * phase))) ** 2
    saliency = cv2.GaussianBlur(saliency, (9, 9), 2.5)
    saliency = cv2.normalize(saliency, None, 0, 255, cv2.NORM_MINMAX)
    return saliency

# Extracción de características
def extract_features(image_path):
    features = {}
    img = cv2.imread(image_path)
    if img is None:
        return None

    # Convertir formatos
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_resized = cv2.resize(img_rgb, (224, 224))
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)

    # Colores dominantes (KMeans)
    pixels = img_resized.reshape(-1, 3)
    kmeans = KMeans(n_clusters=3, random_state=42).fit(pixels)
    dominant_colors = np.mean(kmeans.cluster_centers_, axis=0)
    for i, val in enumerate(dominant_colors):
        features[f"dominant_color_avg_{['R','G','B'][i]}"] = val

    # Histograma HSV
    hist = cv2.calcHist([img_hsv], [0, 1, 2], None, [3, 3, 3], [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    for i, h in enumerate(hist):
        features[f"hsv_hist_{i}"] = h

    # Saliency
    saliency_map = compute_saliency_opencv(img_gray)
    features['saliency_mean'] = np.mean(saliency_map)
    features['saliency_std'] = np.std(saliency_map)

    # Bordes (Sobel)
    edge_map = sobel(img_gray / 255.0)
    features['edge_density'] = np.mean(edge_map)

    # Entropía
    features['entropy'] = shannon_entropy(img_gray)

    # Rostros
    faces = face_cascade.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=5)
    features['face_count'] = len(faces)
    features['avg_face_area'] = np.mean([w * h for (x, y, w, h) in faces]) if len(faces) > 0 else 0

    # HOG descriptor
    hog_features = hog(img_gray, pixels_per_cell=(16, 16), cells_per_block=(2, 2), feature_vector=True)
    for i in range(min(10, len(hog_features))):  # solo primeros 10 para no crecer demasiado
        features[f'hog_{i}'] = hog_features[i]

    # Momentos de Hu
    moments = cv2.moments(img_gray)
    hu_moments = cv2.HuMoments(moments).flatten()
    for i, val in enumerate(hu_moments):
        features[f"hu_moment_{i}"] = -np.sign(val) * np.log10(abs(val) + 1e-10)  # log scale

    return features

# Procesamiento de imágenes
data = []

for filename in tqdm(os.listdir(image_dir)):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')) and filename in genre_dict:
        path = os.path.join(image_dir, filename)
        feats = extract_features(path)
        if feats:
            feats['filename'] = filename  
            feats.update(genre_dict[filename])  
            data.append(feats)

# Guardar CSV final (con filename)
df = pd.DataFrame(data)
df.to_csv("caracteristicas_posters.csv", index=False)
print("Archivo 'caracteristicas_posters.csv' generado con one-hot encoding para múltiples géneros y columna filename.")