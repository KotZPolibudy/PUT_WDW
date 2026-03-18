import numpy as np
import math
import plotly.graph_objects as go


def get_tetrahedron_vertices():
    # wierzchołków czworościanu
    return np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0.5, math.sqrt(3) / 2, 0],
        [0.5, math.sqrt(3) / 6, math.sqrt(6) / 3]
    ])


def interactive_entropy_4d_labeled(resolution=30):
    vertices = get_tetrahedron_vertices()
    labels = ['A', 'B', 'C', 'D']

    all_points = []
    entropies = []

    # 1. Generowanie punktów (entropii)
    for i in range(resolution + 1):
        for j in range(resolution + 1 - i):
            for k in range(resolution + 1 - i - j):
                w1 = i / resolution
                w2 = j / resolution
                w3 = k / resolution
                w4 = 1.0 - w1 - w2 - w3

                p = (w1 * vertices[0] + w2 * vertices[1] +
                     w3 * vertices[2] + w4 * vertices[3])

                h = 0
                for prob in [w1, w2, w3, w4]:
                    if prob > 1e-10:
                        h -= prob * math.log2(prob)

                all_points.append(p)
                entropies.append(h)

    all_points = np.array(all_points)
    fig = go.Figure()

    # 2. Trace dla punktów entropii
    fig.add_trace(go.Scatter3d(
        x=all_points[:, 0], y=all_points[:, 1], z=all_points[:, 2],
        mode='markers',
        marker=dict(
            size=3,
            color=entropies,
            colorscale='Inferno',
            opacity=0.4,
            colorbar=dict(title="Entropia (bity)", x=0.85)
        ),
        name="Punkty entropii",
        hovertemplate="Entropia: %{marker.color:.4f}<extra></extra>"
    ))

    # 3. Trace dla krawędzi czworościanu
    edges = [(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)]
    for start, end in edges:
        v_start, v_end = vertices[start], vertices[end]
        fig.add_trace(go.Scatter3d(
            x=[v_start[0], v_end[0]], y=[v_start[1], v_end[1]], z=[v_start[2], v_end[2]],
            mode='lines',
            line=dict(color='rgba(50, 50, 50, 0.8)', width=3),
            showlegend=False, hoverinfo='skip'
        ))

    # 4. Trace dla WIERZCHOŁKÓW z etykietami
    fig.add_trace(go.Scatter3d(
        x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
        mode='markers+text',
        marker=dict(size=8, color='black'),
        text=labels,
        textposition="top center",
        textfont=dict(family="Arial Black", size=14, color="black"),
        name="Wierzchołki"
    ))

    # Konfiguracja widoku
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),  # bez osi wygląda ładniej
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))  # Początkowe ustawienie kamery
        ),
        margin=dict(l=0, r=0, b=0, t=50),
        showlegend=False
    )

    fig.show()


interactive_entropy_4d_labeled(30)
