import math
import matplotlib.pyplot as plt
import random


def plot_barycentric_multi(points_list, vertices=None):
    if vertices is None:
        vertices = [[0, 0], [1, 0], [0.5, math.sqrt(3) / 2]]

    A, B, C = vertices

    # Listy na przeliczone współrzędne
    all_x = []
    all_y = []

    # 1. Przeliczamy punkty (bez rysowania w pętli)
    for p_coords in points_list:
        total = sum(p_coords)
        if total == 0: continue

        w1, w2, w3 = [p / total for p in p_coords]
        px = w1 * A[0] + w2 * B[0] + w3 * C[0]
        py = w1 * A[1] + w2 * B[1] + w3 * C[1]

        all_x.append(px)
        all_y.append(py)

    # 2. Tworzenie wykresu
    plt.figure(figsize=(8, 8))

    # Rysowanie szkieletu trójkąta
    tri_x = [A[0], B[0], C[0], A[0]]
    tri_y = [A[1], B[1], C[1], A[1]]
    plt.plot(tri_x, tri_y, 'k-', linewidth=1.5, alpha=0.7)

    plt.scatter(all_x, all_y, color='royalblue', alpha=0.6, edgecolors='none')

    plt.axis('equal')
    plt.grid(True, linestyle=':', alpha=0.3)
    plt.title(f"Rozkład barycentryczny ({len(all_x)} punktów)", fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_barycentric_colored(points_list, vertices=None):
    if vertices is None:
        vertices = [[0, 0], [1, 0], [0.5, math.sqrt(3) / 2]]

    A, B, C = vertices
    all_x, all_y, all_w3 = [], [], []

    for p_coords in points_list:
        total = sum(p_coords)
        if total == 0: continue

        w1, w2, w3 = [p / total for p in p_coords]
        px = w1 * A[0] + w2 * B[0] + w3 * C[0]
        py = w1 * A[1] + w2 * B[1] + w3 * C[1]

        all_x.append(px)
        all_y.append(py)
        all_w3.append(w3)  # Zbieramy trzecią współrzędną do pokolorowania

    plt.figure(figsize=(10, 8))  # Trochę szerzej, żeby zmieścić legendę kolorów

    # Rysowanie boków trójkąta
    tri_x = [A[0], B[0], C[0], A[0]]
    tri_y = [A[1], B[1], C[1], A[1]]
    plt.plot(tri_x, tri_y, 'k-', linewidth=1, alpha=0.5)

    # 1. Rysowanie punktów z użyciem colormap
    # c=all_w3 -> wartości decydujące o kolorze
    # cmap='viridis' -> popularna, czytelna paleta (można zmienić na 'magma', 'inferno', 'jet')
    scatter = plt.scatter(all_x, all_y, c=all_w3, cmap='viridis',
                          s=100, alpha=0.8, edgecolors='none')

    plt.axis('equal')
    plt.grid(True, linestyle=':', alpha=0.3)
    plt.tight_layout()
    plt.show()


def fill_bary_random(n):
    return [[random.randint(0, 100) for _ in range(3)] for _ in range(n)]


def fill_bary_uniform(n):
    points = []
    for _ in range(n):
        u = [random.random() for _ in range(3)]
        e = [-math.log(x) for x in u]
        total = sum(e)
        points.append([val / total for val in e])
    return points


def plot_entropy(resolution=100):
    vertices = [[0, 0], [1, 0], [0.5, math.sqrt(3) / 2]]
    A, B, C = vertices

    points_x = []
    points_y = []
    entropies = []

    # Generujemy regularną siatkę punktów wewnątrz trójkąta
    for i in range(resolution + 1):
        for j in range(resolution + 1 - i):
            w1 = i / resolution
            w2 = j / resolution
            w3 = 1.0 - w1 - w2

            # Obliczanie współrzędnych kartezjańskich (x, y)
            px = w1 * A[0] + w2 * B[0] + w3 * C[0]
            py = w1 * A[1] + w2 * B[1] + w3 * C[1]

            # Obliczanie entropii (H = - suma p * log2(p))
            # Obsługa p=0 (lim p->0 p*log(p) = 0)
            h = 0
            for p in [w1, w2, w3]:
                if p > 0:
                    h -= p * math.log(p, 2)

            points_x.append(px)
            points_y.append(py)
            entropies.append(h)

    plt.figure(figsize=(10, 8))

    # Rysowanie krawędzi
    tri_x = [A[0], B[0], C[0], A[0]]
    tri_y = [A[1], B[1], C[1], A[1]]
    plt.plot(tri_x, tri_y, 'k-', lw=1, alpha=0.3)

    # Wyświetlanie mapy entropii
    sc = plt.scatter(points_x, points_y, c=entropies, cmap='hot', s=15)
    cbar = plt.colorbar(sc)
    cbar.set_label('Entropia (bity)', rotation=270, labelpad=15)

    plt.axis('equal')
    plt.axis('off')
    plt.show()


plot_entropy(100)
# plot_barycentric_colored(fill_bary_uniform(303))
