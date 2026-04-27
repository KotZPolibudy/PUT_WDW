import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import minimize
import math


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
    """Funkcja celu (Stress)."""
    coords = flat_coords.reshape((n_points, target_dims))
    current_dist_mtx = squareform(pdist(coords))
    diff = (current_dist_mtx - target_dist_mtx) ** 2
    return np.sum(diff) / 2


def optimize_from_matrix(dist_df, target_dims=2):
    validate_distance_matrix(dist_df)
    n_points = len(dist_df)
    target_dist_mtx = dist_df.values
    labels = dist_df.index.tolist()

    initial_guess = np.random.rand(n_points * target_dims)
    res = minimize(
        stress_function,
        initial_guess,
        args=(target_dist_mtx, n_points, target_dims),
        method='L-BFGS-B'
    )

    optimized_coords = res.x.reshape((n_points, target_dims))
    return optimized_coords, labels, res.fun


def print_vertices_info(coords, labels):
    print("\n--- ZOPTYMALIZOWANE WSPÓŁRZĘDNE WIERZCHOŁKÓW ---")
    df_coords = pd.DataFrame(coords, index=labels, columns=[f"Oś_{i+1}" for i in range(coords.shape[1])])
    print(df_coords)
    print("-" * 50)


def display_plot(coords, labels, stress):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=coords[:, 0], y=coords[:, 1],
        mode='markers+text',
        text=labels,
        textposition="top center",
        marker=dict(size=15, color='royalblue', line=dict(width=2, color='DarkSlateGrey'))
    ))
    fig.update_layout(
        title=f"Rekonstrukcja układu (Stress: {stress:.6f})",
        xaxis=dict(title="X", showgrid=True, zeroline=True),
        yaxis=dict(title="Y", showgrid=True, zeroline=True),
        template="plotly_white"
    )
    fig.show()


data = {
    "Punkt_A": [0.0, 1.0, math.sqrt(2), 1.0],
    "Punkt_B": [1.0, 0.0, 1.0, math.sqrt(2)],
    "Punkt_C": [math.sqrt(2), 1.0, 0.0, 1.0],
    "Punkt_D": [1.0, math.sqrt(2), 1.0, 0.0]
}
custom_dist_df = pd.DataFrame(data, index=data.keys())

opt_coords, labels, final_stress = optimize_from_matrix(custom_dist_df, target_dims=2)

# Do zakomentowania w razie co
print_vertices_info(opt_coords, labels)  # Wypisanie współrzędnych w konsoli
display_plot(opt_coords, labels, final_stress)  # Wyświetlenie wykresu