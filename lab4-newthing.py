import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.spatial import distance_matrix


def generate_point_cloud(n_points=10, dimensions=3):
    return np.random.rand(n_points, dimensions)


def get_distance_matrix(points):
    return distance_matrix(points, points)


def visualize_points(points):
    dims = points.shape[1]

    if dims == 2:
        fig = px.scatter(x=points[:, 0], y=points[:, 1],
                         title="Chmura punktów 2D",
                         labels={'x': 'Oś X', 'y': 'Oś Y'})
    elif dims == 3:
        fig = go.Figure(data=[go.Scatter3d(
            x=points[:, 0], y=points[:, 1], z=points[:, 2],
            mode='markers',
            marker=dict(size=5, color=points[:, 2], colorscale='Viridis', opacity=0.8)
        )])
        fig.update_layout(title="Chmura punktów 3D",
                          scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'))
    else:
        print(f"Wizualizacja dostępna tylko dla 2 lub 3 wymiarów (obecnie: {dims}).")
        return

    fig.show()


# --- Parametry i uruchomienie ---
N = 10  # Liczba punktów
D = 3  # Liczba wymiarów (zmień na 2, aby zobaczyć wykres płaski)

# 1. Generowanie punktów
cloud = generate_point_cloud(n_points=N, dimensions=D)

# 2. Obliczanie macierzy odległości
dist_mtx = get_distance_matrix(cloud)

# 3. Wyświetlenie wyników
print(f"Wygenerowano {N} punktów w przestrzeni {D}D.")
print("\nMacierz odległości (pierwsze 5x5):")
print(np.round(dist_mtx[:5, :5], 3))

# 4. Wizualizacja
visualize_points(cloud)