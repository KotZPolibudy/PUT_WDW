import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import minimize
import argparse
import sys
import os


def calculate_frobenius_norm(matrix_a, matrix_b):
    return np.linalg.norm(matrix_a - matrix_b, ord='fro')


def stress_function(flat_coords, target_dist_mtx, n_points, target_dims):
    """Funkcja celu"""
    coords = flat_coords.reshape((n_points, target_dims))
    current_dist_mtx = squareform(pdist(coords))
    diff = (current_dist_mtx - target_dist_mtx) ** 2
    return np.sum(diff) / 2


def load_data(file_path):
    if not os.path.exists(file_path):
        print(f"Błąd: Plik '{file_path}' nie istnieje.")
        sys.exit(1)

    try:
        with open(file_path, 'r') as f:
            first_line = f.readline().split()
            if not first_line:
                raise ValueError("Plik jest pusty.")

            mode = first_line[0].upper()
            column_headers = first_line[1:]

        # Wczytanie reszty danych (bez pierwszej linii, pierwsza kolumna jako index)
        df = pd.read_csv(file_path, sep='\s+', skiprows=1, header=None, index_col=0)
        df.columns = column_headers

        if mode == 'X':
            print(f"Tryb X: Wykryto macierz obiekt/atrybut ({df.shape[0]} obiektów).")
            # Konwersja X -> D
            dist_matrix = squareform(pdist(df.values))
            dist_df = pd.DataFrame(dist_matrix, index=df.index, columns=df.index)
            return dist_df

        elif mode == 'D':
            print(f"Tryb D: Wykryto macierz odległości ({df.shape[0]} obiektów).")
            return df

        else:
            print("Błąd: Pierwszy znak w pliku musi być 'X' lub 'D'.")
            sys.exit(1)

    except Exception as e:
        print(f"Błąd podczas wczytywania pliku: {e}")
        sys.exit(1)


def run_optimization(dist_df, target_dims=2):
    n_points = len(dist_df)
    target_dist_mtx = dist_df.values

    # Inicjalizacja (przeskalowana do średniej odległości)
    scale = np.mean(target_dist_mtx)
    initial_guess = np.random.rand(n_points * target_dims) * scale

    res = minimize(
        stress_function,
        initial_guess,
        args=(target_dist_mtx, n_points, target_dims),
        method='L-BFGS-B'
    )

    optimized_coords = res.x.reshape((n_points, target_dims))

    # Obliczenie jakości działania (Norma Frobeniusa)
    current_dist_mtx = squareform(pdist(optimized_coords))
    f_norm = calculate_frobenius_norm(target_dist_mtx, current_dist_mtx)

    return optimized_coords, dist_df.index.tolist(), f_norm


def visualize(coords, labels, f_norm):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=coords[:, 0], y=coords[:, 1],
        mode='markers+text',
        text=labels,
        textposition="top center",
        marker=dict(size=12, color='royalblue', line=dict(width=2, color='white'))
    ))

    fig.update_layout(
        title=f"Wizualizacja MDS (Norma Frobeniusa: {f_norm:.6f})",
        xaxis=dict(title="Oś X"),
        yaxis=dict(title="Oś Y", scaleanchor="x", scaleratio=1),
        template="plotly_white",
        width=800, height=800
    )
    fig.show()


def main():
    parser = argparse.ArgumentParser(
        description="Program do wizualizacji MDS na podstawie macierzy X (atrybuty) lub D (odległości)."
    )
    parser.add_argument(
        "input_file",
        help="Ścieżka do tekstowego pliku wejściowego."
    )
    parser.add_argument(
        "--dims",
        type=int,
        default=2,
        help="Liczba wymiarów docelowych (domyślnie 2)."
    )

    # Jeśli nie podano argumentów, wyświetl pomoc
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()

    dist_df = load_data(args.input_file)
    print("optymalizacja...")
    coords, labels, quality_score = run_optimization(dist_df, args.dims)
    print("\n--- WYNIKI ---")
    print(f"Jakość działania (Norma Frobeniusa): {quality_score:.6f}")
    print("\nZoptymalizowane współrzędne:")
    output_df = pd.DataFrame(coords, index=labels, columns=[f"Oś_{i + 1}" for i in range(args.dims)])
    print(output_df)

    # 4. Wizualizacja (2D)
    if args.dims == 2:
        visualize(coords, labels, quality_score)
    else:
        print("\nWizualizacja pominięta (dostępna tylko dla 2 wymiarów).")


if __name__ == "__main__":
    main()