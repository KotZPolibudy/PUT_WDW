import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import minimize
import math
import io


def load_matrix_from_file(file_path):
    """Wczytuje macierz odległości z pliku tekstowego."""
    df = pd.read_csv(file_path, sep='\s+', header=None)

    # Nadajemy etykiety P0, P1, P2...
    labels = [f"P{i}" for i in range(len(df))]
    df.index = labels
    df.columns = labels
    return df


def validate_distance_matrix(df):
    matrix = df.values
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Macierz musi być kwadratowa!")
    if not np.allclose(matrix, matrix.T):
        raise ValueError("Macierz musi być symetryczna!")
    if not np.allclose(np.diag(matrix), 0):
        np.fill_diagonal(matrix, 0)
    return True


def stress_function(flat_coords, target_dist_mtx, n_points, target_dims):
    coords = flat_coords.reshape((n_points, target_dims))
    current_dist_mtx = squareform(pdist(coords))
    diff = (current_dist_mtx - target_dist_mtx) ** 2
    return np.sum(diff) / 2


def optimize_from_matrix(dist_df, target_dims=2):
    validate_distance_matrix(dist_df)
    n_points = len(dist_df)
    target_dist_mtx = dist_df.values
    labels = dist_df.index.tolist()

    # Dla dużych wartości w macierzy (np. 100-200), warto przeskalować zgadnięcie początkowe
    max_dist = np.max(target_dist_mtx)
    initial_guess = np.random.rand(n_points * target_dims) * max_dist

    res = minimize(
        stress_function,
        initial_guess,
        args=(target_dist_mtx, n_points, target_dims),
        method='L-BFGS-B',
        tol=1e-9
    )

    optimized_coords = res.x.reshape((n_points, target_dims))
    return optimized_coords, labels, res.fun


def print_comparison_stats(original_df, optimized_coords):
    calc_dist = squareform(pdist(optimized_coords))
    orig_dist = original_df.values
    error = np.abs(orig_dist - calc_dist)

    print("\n--- ANALIZA DOPASOWANIA (BŁĄD) ---")
    print(f"Średni błąd bezwzględny: {np.mean(error):.6f}")
    print(f"Maksymalny błąd: {np.max(error):.6f}")
    print("-" * 34)


def print_vertices_info(coords, labels):
    print("\n--- ZOPTYMALIZOWANE WSPÓŁRZĘDNE ---")
    df_coords = pd.DataFrame(coords, index=labels, columns=[f"Oś_{i + 1}" for i in range(coords.shape[1])])
    print(df_coords.round(4))
    print("-" * 34)


def display_plot(coords, labels, stress):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=coords[:, 0], y=coords[:, 1],
        mode='markers+text',
        text=labels,
        textposition="top center",
        marker=dict(size=10, color='royalblue', line=dict(width=1, color='DarkSlateGrey'))
    ))

    fig.update_layout(
        title=f"Rekonstrukcja układu (Stress: {stress:.4e})",
        yaxis=dict(scaleanchor="x", scaleratio=1, title="Oś Y"),
        xaxis=dict(title="Oś X"),
        template="plotly_white",
        width=800, height=800
    )
    fig.show()


try:
    # PODMIEŃ ŚCIEŻKĘ DO PLIKU
    custom_dist_df = load_matrix_from_file('clock--D.txt')

    # Obliczenia
    opt_coords, labels, final_stress = optimize_from_matrix(custom_dist_df, target_dims=2)

    # Raporty i wizualizacja
    print_vertices_info(opt_coords, labels)
    print_comparison_stats(custom_dist_df, opt_coords)
    display_plot(opt_coords, labels, final_stress)

except FileNotFoundError:
    print(
        "Błąd: Nie znaleziono pliku '.txt'. Upewnij się, że plik znajduje się w tym samym folderze co skrypt.")