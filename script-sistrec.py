import numpy as np
import os
import requests
import zipfile as zipfile
import pandas as pd


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
    num_products = df['movieId'].nunique()
    num_ratings = len(df)
    
    print(f"\nNúmero de usuarios únicos: {num_users}")
    print(f"Número de películas únicas: {num_products}")
    print(f"Número total de puntuaciones: {num_ratings}")

# Comprobar valores vacíos y duplicados
def check_na_and_duplicates(df):
    print("\nValores nulos por columna:")
    print(df.isna().sum())

    duplicates = df.duplicated().sum()
    print(f"\nNúmero de filas duplicadas: {duplicates}")


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
