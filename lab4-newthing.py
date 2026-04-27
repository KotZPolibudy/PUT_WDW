import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import minimize


def get_distance_matrix(points):
    """Tworzy macierz odległości."""
    return squareform(pdist(points))


def stress_function(flat_coords, target_dist_mtx, n_points, target_dims):
    """
    Funkcja celu (Stress): suma kwadratów różnic między odległościami.
    """
    # Przekształcamy płaski wektor z powrotem w macierz (N x D)
    coords = flat_coords.reshape((n_points, target_dims))

    # Obliczamy macierz odległości dla aktualnych współrzędnych
    current_dist_mtx = get_distance_matrix(coords)

    # Obliczamy różnicę (interesuje nas tylko górny trójkąt, by nie liczyć dwa razy)
    diff = (current_dist_mtx - target_dist_mtx) ** 2
    return np.sum(diff) / 2


def optimize_point_positions(target_dist_df, target_dims=2):
    """
    Używa solvera nieliniowego do znalezienia optymalnych pozycji punktów.
    """
    n_points = len(target_dist_df)
    target_dist_mtx = target_dist_df.values

    # Początkowe zgadnięcie (losowe pozycje w 2D)
    initial_guess = np.random.rand(n_points * target_dims)

    # Optymalizacja (używamy algorytmu L-BFGS-B lub SLSQP)
    res = minimize(
        stress_function,
        initial_guess,
        args=(target_dist_mtx, n_points, target_dims),
        method='L-BFGS-B'
    )

    # Przekształcenie wyniku do czytelnego formatu
    optimized_coords = res.x.reshape((n_points, target_dims))
    return optimized_coords, res.fun


# --- Scenariusz ---
N_POINTS = 4
ORIGINAL_DIM = 3
TARGET_DIM = 2

# 1. Tworzymy "ideał" w 3D i jego macierz odległości
points_3d = np.random.rand(N_POINTS, ORIGINAL_DIM)
dist_df = pd.DataFrame(get_distance_matrix(points_3d))

# 2. Optymalizacja: szukamy układu w 2D, który ma te same odległości
optimized_2d, final_stress = optimize_point_positions(dist_df, target_dims=TARGET_DIM)

# 3. Porównanie macierzy (Weryfikacja)
optimized_dist_mtx = get_distance_matrix(optimized_2d)

print(f"Finalny Stress (błąd dopasowania): {final_stress:.6f}")
print("\nOryginalna macierz (fragment):\n", dist_df.iloc[:4, :4])
print("\nNowa macierz w 2D (fragment):\n", pd.DataFrame(optimized_dist_mtx).iloc[:4, :4])

# 4. Wizualizacja wyniku 2D
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=optimized_2d[:, 0], y=optimized_2d[:, 1],
    mode='markers+text',
    text=[f"P{i}" for i in range(N_POINTS)],
    textposition="top center",
    marker=dict(size=12, color='red')
))
fig.update_layout(title=f"Zoptymalizowane punkty w 2D (Stress: {final_stress:.4f})",
                  xaxis_title="Oś X", yaxis_title="Oś Y")
fig.show()