import numpy as np
import random
import os
import requests
import zipfile as zipfile
import pandas as pd
import matplotlib.pyplot as plt
from surprise import Dataset, Reader, KNNWithZScore, accuracy, NormalPredictor, SVD
from surprise.model_selection import GridSearchCV
import copy
from collections import defaultdict


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


# Crear el conjunto de datos de Surprise a partir del DataFrame
def create_surprise_dataset(df):
    print("\nCreando el conjunto de datos de Surprise...")
    reader = Reader(rating_scale=(0.5, 5))
    data = Dataset.load_from_df(df[['userId', 'movieId', 'rating']], reader)
    return data

# Definición de los folds
def set_my_folds(dataset, nfolds = 5, shuffle = True):
    
    raw_ratings = dataset.raw_ratings
    if (shuffle): raw_ratings = random.sample(raw_ratings, len(raw_ratings))

    chunk_size = int(1/nfolds * len(raw_ratings))
    thresholds = [chunk_size * x for x in range(0,nfolds)]
    
    print("set_my_folds> len(raw_ratings): %d" % len(raw_ratings))    
    
    folds = []
    
    for th in thresholds:                
        test_raw_ratings = raw_ratings[th: th + chunk_size]
        train_raw_ratings = raw_ratings[:th] + raw_ratings[th + chunk_size:]
    
        print("set_my_folds> threshold: %d, len(train_raw_ratings): %d, len(test_raw_ratings): %d" % (th, len(train_raw_ratings), len(test_raw_ratings)))
        
        folds.append((train_raw_ratings,test_raw_ratings))
       
    return folds

# Función que entrena y evalúa el modelo KNNWithZScore con GridSearchCV en cada fold
def evaluate_knn_with_gridsearch(folds, knn_param_grid):

    results = []

    for i, (train_ratings, test_ratings) in enumerate(folds):
        print(f'Fold: {i}')
        
        knn_gs = GridSearchCV(KNNWithZScore, knn_param_grid, measures=["mae"], cv=3, n_jobs=-1)
        
        train_dataset = copy.deepcopy(surprise_data)
        train_dataset.raw_ratings = train_ratings
        knn_gs.fit(train_dataset)
        
        best_mae = knn_gs.best_score["mae"]
        best_params = knn_gs.best_params["mae"]
        print(f'Grid search > mae={best_mae:.3f}, cfg={best_params}')
        
        results.append({'fold': i, 'mae': best_mae, 'params': best_params})
        
        knn_algo = knn_gs.best_estimator["mae"]
        
        knn_algo.fit(train_dataset.build_full_trainset())
        
        test_dataset = copy.deepcopy(surprise_data)
        test_dataset.raw_ratings = test_ratings
        test_set = test_dataset.construct_testset(raw_testset=test_ratings)
        
        knn_predictions = knn_algo.test(test_set)
        
        test_mae = accuracy.mae(knn_predictions, verbose=True)
        
        results[-1]['test_mae'] = test_mae

    df_results = pd.DataFrame(results)
    
    print("\nResultados finales:")
    print(df_results)


# Función que evalúa los tres algoritmos: NormalPredictor, KNNWithZScore, y SVD usando los folds existentes
def evaluate_algorithms_comparison(surprise_data, folds, knn_param_grid):
    results = []

    for i, (train_ratings, test_ratings) in enumerate(folds):
        print(f'Fold: {i}')
        
        trainset = surprise_data.construct_trainset(train_ratings)
        testset = surprise_data.construct_testset(test_ratings)

        # Parámetros por defecto del NormalPredictor
        np_model = NormalPredictor()
        np_model.fit(trainset)
        np_predictions = np_model.test(testset)
        np_mae = accuracy.mae(np_predictions, verbose=False)
        print(f'Fold {i} > NormalPredictor > MAE: {np_mae:.3f}')

        # Parámetros definidos con el Grid Search (knn_param_grid)
        knn_gs = GridSearchCV(KNNWithZScore, knn_param_grid, measures=["mae"], cv=3, n_jobs=-1)
        
        knn_gs.fit(surprise_data)  

        best_mae_knn = knn_gs.best_score["mae"]
        best_params_knn = knn_gs.best_params["mae"]
        print(f'Fold {i} > Grid search > KNNWithZScore > Best MAE: {best_mae_knn:.3f}, Params: {best_params_knn}')

        knn_algo = knn_gs.best_estimator["mae"]
        knn_algo.fit(trainset)
        knn_predictions = knn_algo.test(testset)
        knn_mae = accuracy.mae(knn_predictions, verbose=False)
        print(f'Fold {i} > KNNWithZScore > MAE: {knn_mae:.3f}')

        # Parámetros establecido a n_factor = 25
        svd_algo = SVD(n_factors=25)
        svd_algo.fit(trainset)
        svd_predictions = svd_algo.test(testset)
        svd_mae = accuracy.mae(svd_predictions, verbose=False)
        print(f'Fold {i} > SVD > n_factors=25 > MAE: {svd_mae:.3f}')

        results.append({
            'fold': i,
            'NormalPredictor_MAE': np_mae,
            'KNNWithZScore_MAE': knn_mae,
            'SVD_MAE': svd_mae
        })
    
    results_df = pd.DataFrame(results)
    print("\nResultados finales:")
    print(results_df)

    # Calcular el MAE promedio de cada algoritmo
    avg_mae_np = results_df['NormalPredictor_MAE'].mean()
    avg_mae_knn = results_df['KNNWithZScore_MAE'].mean()
    avg_mae_svd = results_df['SVD_MAE'].mean()

    print(f"\nMAE Promedio:\nNormalPredictor: {avg_mae_np:.3f}\nKNNWithZScore: {avg_mae_knn:.3f}\nSVD: {avg_mae_svd:.3f}")
    
    return results_df, avg_mae_np, avg_mae_knn, avg_mae_svd


# Función auxiliar 
def precision_recall_at_k(predictions, k=10, threshold=4):

    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = dict()
    recalls = dict()
    for uid, user_ratings in user_est_true.items():

        user_ratings.sort(key=lambda x: x[0], reverse=True)

        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])

        n_rel_and_rec_k = sum(
            ((true_r >= threshold) and (est >= threshold))
            for (est, true_r) in user_ratings[:k]
        )

        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0

        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0

    return precisions, recalls

# Mostrar una única gráfica con el rendimiento de los 3 algoritmos
def plot_precision_recall_curve_all(predictions_dict, k_values, threshold):
    plt.figure(figsize=(10, 7))

    for algo_name, predictions in predictions_dict.items():
        avg_precisions = []
        avg_recalls = []

        for k in k_values:
            precisions, recalls = precision_recall_at_k(predictions, k, threshold)

            avg_precision = sum(prec for prec in precisions.values()) / len(precisions)
            avg_recall = sum(rec for rec in recalls.values()) / len(recalls)

            avg_precisions.append(avg_precision)
            avg_recalls.append(avg_recall)

        plt.plot(avg_recalls, avg_precisions, marker='o', label=algo_name)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve at k')
    plt.legend()
    plt.grid()
    plt.show()

# Evalúa diferentes algoritmos y muestra una única gráfica de Precision-Recall
def evaluate_algorithms_and_plot(surprise_data, folds, k_values=[1, 2, 5, 10], threshold=4):
    predictions_dict = {}

    for i, (train_ratings, test_ratings) in enumerate(folds):
        print(f'Evaluando en Fold {i + 1}...')

        # Crear los conjuntos de datos para Surprise
        trainset = surprise_data.construct_trainset(train_ratings)
        testset = surprise_data.construct_testset(test_ratings)

        # Evaluar NormalPredictor
        np_model = NormalPredictor()
        np_model.fit(trainset)
        predictions_np = np_model.test(testset)
        predictions_dict['NormalPredictor'] = predictions_dict.get('NormalPredictor', []) + predictions_np

        # Evaluar KNNWithZScore
        knn_model = KNNWithZScore(sim_options={'name': 'cosine', 'user_based': False})
        knn_model.fit(trainset)
        predictions_knn = knn_model.test(testset)
        predictions_dict['KNNWithZScore'] = predictions_dict.get('KNNWithZScore', []) + predictions_knn

        # Evaluar SVD
        svd_model = SVD()
        svd_model.fit(trainset)
        predictions_svd = svd_model.test(testset)
        predictions_dict['SVD'] = predictions_dict.get('SVD', []) + predictions_svd

    plot_precision_recall_curve_all(predictions_dict, k_values, threshold)


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

    # # Paso 4
    # plot_user_histogram(df)
    plot_movie_histogram(df)

    # Paso 5
    plot_user_rating_mean_histogram(df)
    plot_movie_rating_mean_histogram(df)

    # Paso 6
    plot_rating_distribution(df)

    # Paso 7
    surprise_data = create_surprise_dataset(df)
    kf = set_my_folds(surprise_data, nfolds=5)

    # Parámetros para el GridSearchCV de SVD
    knn_param_grid = {
        'k': [25, 50, 75],
        'min_k': [1, 3, 5],
        'sim_options': {'name': ['pearson'], 'user_based': [False]}
    }
    # Paso 8
    # evaluate_knn_with_gridsearch(kf, knn_param_grid)

    # Definir los mejores parámetros de KNNWithZScore obtenidos en el GridSearchCV
    knn_best_params = {'k': [75], 'min_k': [3]}

    # Paso 9
    results_df, avg_mae_np, avg_mae_knn, avg_mae_svd = evaluate_algorithms_comparison(surprise_data, kf, knn_best_params)
    
    # Paso 10
    evaluate_algorithms_and_plot(surprise_data, kf)

