import numpy as np
import os
import requests
import zipfile as zipfile
import pandas as pd
import matplotlib.pyplot as plt


# Definir una semilla para reproducibilidad del código 
def initialize_seed():
    print("Definir una semilla para que el código sea reproducible")
    seed = 42
    np.random.seed(seed)
    return np.random.rand()

# Descarga y extracción del .csv
def download_and_extract(url, zip_path, extract_to):
    if not os.path.exists(zip_path):
        print("Descargando el archivo...")
        response = requests.get(url)
        with open(zip_path, 'wb') as file:
            file.write(response.content)
    if not os.path.exists(extract_to):
        print("Extrayendo el archivo...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(".")

# Cargar el archivo en pandas
def load_data(file_path):
    print("\nCargando el archivo CSV...")
    return pd.read_csv(file_path)

# Comprobar si los valores están dentro del rango
def check_rating_range(df):
    min_rating = df['rating'].min()
    max_rating = df['rating'].max()
    print(f"Rango de puntuaciones: mínimo={min_rating}, máximo={max_rating}")
    if min_rating < 0.5 or max_rating > 5:
        print("Advertencia: Las puntuaciones están fuera del rango esperado [0.5, 5].")
    else:
        print("Las puntuaciones están dentro del rango esperado.")

# Exploración inicial del DataFrame
def explore_data(df):
    num_users = df['userId'].nunique()
    num_movies = df['movieId'].nunique()
    num_ratings = len(df)
    
    print(f"\nNúmero de usuarios únicos: {num_users}")
    print(f"Número de películas únicas: {num_movies}")
    print(f"Número total de puntuaciones: {num_ratings}")

# Comprobar valores vacíos y duplicados
def check_na_and_duplicates(df):
    print("\nValores nulos por columna:")
    print(df.isna().sum())

    duplicates = df.duplicated().sum()
    print(f"\nNúmero de filas duplicadas: {duplicates}")

# Eliminar los películas con menos de 10 puntuaciones.
def remove_movies_with_low_ratings(df, rating_limit):
    print("\nEliminando películas con menos de 10 puntuaciones...")
    movie_counts = df['movieId'].value_counts()
    valid_movies = movie_counts[movie_counts >= rating_limit].index
    filtered_df = df[df['movieId'].isin(valid_movies)]
    print(f"películas restantes: {filtered_df['movieId'].nunique()}")
    return filtered_df

# Eliminar los usuarios con menos de 10 puntuaciones.
def remove_users_with_low_ratings(df, rating_limit):
    print("\nEliminando usuarios con menos de 10 puntuaciones...")
    user_counts = df['userId'].value_counts()
    valid_users = user_counts[user_counts >= rating_limit].index
    filtered_df = df[df['userId'].isin(valid_users)]
    print(f"Usuarios restantes: {filtered_df['userId'].nunique()}")
    print(f"películas restantes: {filtered_df['movieId'].nunique()}")
    print(f"Puntuaciones restantes: {len(filtered_df)}")
    return filtered_df

# Histograma del número de puntuaciones por usuario
def plot_user_histogram(df):
    print("\nGenerando histograma del número de puntuaciones por usuario...")
    user_ratings = df['userId'].value_counts()
    counts, bins = np.histogram(user_ratings, bins=30)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    plt.figure(figsize=(12, 6))
    plt.bar(bin_centers, counts, width=np.diff(bins), align='center', color='skyblue', edgecolor='black')
    plt.title('Número de puntuaciones por usuario')
    plt.xlabel('Número de puntuaciones')
    plt.ylabel('Cantidad de usuarios')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Añadir etiquetas
    for x, y in zip(bin_centers, counts):
        plt.text(x, y, str(int(y)), ha='center', va='bottom', fontsize=8)

    plt.show()

# Histograma del número de puntuaciones por película
def plot_movie_histogram(df):
    print("\nGenerando histograma del número de puntuaciones por película...")
    movie_ratings = df['movieId'].value_counts()
    counts, bins = np.histogram(movie_ratings, bins=30)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    plt.figure(figsize=(12, 6))
    plt.bar(bin_centers, counts, width=np.diff(bins), align='center', color='salmon', edgecolor='black')
    plt.title('Número de puntuaciones por película')
    plt.xlabel('Número de puntuaciones')
    plt.ylabel('Cantidad de películas')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Añadir etiquetas
    for x, y in zip(bin_centers, counts):
        plt.text(x, y, str(int(y)), ha='center', va='bottom', fontsize=8)

    plt.show()

# Histograma de la media de puntuaciones por usuario
def plot_user_rating_mean_histogram(df):
    print("\nGenerando histograma de la media de puntuaciones por usuario...")
    user_means = df.groupby('userId')['rating'].mean()
    plt.figure(figsize=(12, 6))
    plt.hist(user_means, bins=30, color='skyblue', edgecolor='black')
    plt.title('Media de puntuaciones por usuario')
    plt.xlabel('Media de puntuaciones')
    plt.ylabel('Cantidad de usuarios')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

# Histograma de la media de puntuaciones por película
def plot_movie_rating_mean_histogram(df):
    print("\nGenerando histograma de la media de puntuaciones por película...")
    movie_means = df.groupby('movieId')['rating'].mean()
    plt.figure(figsize=(12, 6))
    plt.hist(movie_means, bins=30, color='salmon', edgecolor='black')
    plt.title('Media de puntuaciones por película')
    plt.xlabel('Media de puntuaciones')
    plt.ylabel('Cantidad de películas')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

# Diagrama de barras
def plot_rating_distribution(df):
    print("\nGenerando diagrama de barras de la distribución de las puntuaciones...")
    rating_counts = df['rating'].value_counts().sort_index()

    plt.figure(figsize=(10, 6))
    plt.bar(rating_counts.index, rating_counts.values, color='teal', edgecolor='black', width=0.4)
    plt.title('Distribución de las puntuaciones')
    plt.xlabel('Puntuaciones')
    plt.ylabel('Frecuencia')
    plt.xticks(rating_counts.index)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    for x, y in zip(rating_counts.index, rating_counts.values):
        plt.text(x, y + 0.5, str(y), ha='center', va='bottom', fontsize=10)

    plt.show()


if __name__ == "__main__":
    URL = "http://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
    ZIP_PATH = "ml-latest-small.zip"
    EXTRACT_TO = "ml-latest-small"
    file_path = os.path.join(EXTRACT_TO, "ratings.csv")

    # Paso 1
    randon_number = initialize_seed()
    print(randon_number)

    # Paso 2
    download_and_extract(URL, ZIP_PATH, EXTRACT_TO)
    df = load_data(file_path)

    check_rating_range(df)
    explore_data(df)
    check_na_and_duplicates(df)

    # Paso 3
    rating_limit = 10
    df = remove_movies_with_low_ratings(df, rating_limit)
    df = remove_users_with_low_ratings(df, rating_limit)

    # Paso 4
    plot_user_histogram(df)
    plot_movie_histogram(df)

    # Paso 5
    plot_user_rating_mean_histogram(df)
    plot_movie_rating_mean_histogram(df)

    # Paso 6
    plot_rating_distribution(df)


