import numpy as np
import math
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output


# --- 1. KONFIGURACJA MATEMATYCZNA ---

def get_tetrahedron_vertices():
    # Wierzchołki A, B, C, D w przestrzeni 3D
    return np.array([
        [0, 0, 0],  # A (w1)
        [1, 0, 0],  # B (w2)
        [0.5, math.sqrt(3) / 2, 0],  # C (w3)
        [0.5, math.sqrt(3) / 6, math.sqrt(6) / 3]  # D (w4)
    ])


def my_function(w):
    """Zmień tę funkcję na dowolną inną."""
    # Przykład: Entropia Shannona
    return -sum(p * math.log2(p) for p in w if p > 1e-9)


# Generowanie chmury punktów
RESOLUTION = 35
VERTICES = get_tetrahedron_vertices()
points, values, weights = [], [], []

for i in range(RESOLUTION + 1):
    for j in range(RESOLUTION + 1 - i):
        for k in range(RESOLUTION + 1 - i - j):
            w1 = i / RESOLUTION
            w2 = j / RESOLUTION
            w3 = k / RESOLUTION
            w4 = max(0, 1.0 - w1 - w2 - w3)

            p = (w1 * VERTICES[0] + w2 * VERTICES[1] +
                 w3 * VERTICES[2] + w4 * VERTICES[3])

            points.append(p)
            values.append(my_function([w1, w2, w3, w4]))
            weights.append([w1, w2, w3, w4])

points = np.array(points)
values = np.array(values)
weights = np.array(weights)

# --- 2. LAYOUT APLIKACJI DASH ---

app = Dash(__name__)

app.layout = html.Div([
    html.H2("Eksplorator Przekrojów Simpleksu 4D", style={'textAlign': 'center', 'fontFamily': 'sans-serif'}),

    html.Div([
        # Panel boczny z suwakami
        html.Div([
            html.B("Filtr wartości funkcji:"),
            dcc.RangeSlider(
                id='val-slider',
                min=float(min(values)), max=float(max(values)),
                value=[min(values), max(values)],
                tooltip={"placement": "bottom", "always_visible": True}
            ),
            html.Hr(),
            html.B("Minimalne udziały zmiennych (0.0 - 1.0):"),

            html.Div([html.Label("Zmienna A (Wierzchołek [0,0,0]):"),
                      dcc.Slider(id='w1-s', min=0, max=1, step=0.05, value=0)], style={'marginBottom': '10px'}),

            html.Div([html.Label("Zmienna B (Wierzchołek [1,0,0]):"),
                      dcc.Slider(id='w2-s', min=0, max=1, step=0.05, value=0)], style={'marginBottom': '10px'}),

            html.Div([html.Label("Zmienna C (Podstawa tył):"),
                      dcc.Slider(id='w3-s', min=0, max=1, step=0.05, value=0)], style={'marginBottom': '10px'}),

            html.Div([html.Label("Zmienna D (Szczyt):"),
                      dcc.Slider(id='w4-s', min=0, max=1, step=0.05, value=0)], style={'marginBottom': '10px'}),

            html.P(
                "Uwaga: Ponieważ A+B+C+D=1, ustawienie zbyt wysokich filtrów na wielu suwakach spowoduje brak punktów.",
                style={'fontSize': '12px', 'color': 'gray', 'marginTop': '20px'})

        ], style={'width': '25%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '20px',
                  'backgroundColor': '#f9f9f9'}),

        # Panel wykresu
        html.Div([
            dcc.Graph(id='main-graph', style={'height': '85vh'})
        ], style={'width': '70%', 'display': 'inline-block'})
    ], style={'display': 'flex'})
])


# --- 3. REAKTYWNOŚĆ ---

@app.callback(
    Output('main-graph', 'figure'),
    [Input('val-slider', 'value'),
     Input('w1-s', 'value'), Input('w2-s', 'value'),
     Input('w3-s', 'value'), Input('w4-s', 'value')]
)
def update_view(v_range, m1, m2, m3, m4):
    # Logika filtrowania (Maska)
    mask = (values >= v_range[0]) & (values <= v_range[1]) & \
           (weights[:, 0] >= m1) & (weights[:, 1] >= m2) & \
           (weights[:, 2] >= m3) & (weights[:, 3] >= m4)

    f_p = points[mask]
    f_v = values[mask]

    fig = go.Figure()

    # Chmura punktów
    fig.add_trace(go.Scatter3d(
        x=f_p[:, 0], y=f_p[:, 1], z=f_p[:, 2],
        mode='markers',
        marker=dict(
            size=4, color=f_v, colorscale='Viridis', opacity=0.7,
            colorbar=dict(title="Wartość", thickness=20)
        ),
        hovertemplate="A:%{customdata[0]:.2f}<br>B:%{customdata[1]:.2f}<br>C:%{customdata[2]:.2f}<br>D:%{customdata[3]:.2f}<br>Wynik: %{marker.color:.4f}<extra></extra>",
        customdata=weights[mask]
    ))

    # Szkielet czworościanu
    edges = [(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)]
    for s, e in edges:
        fig.add_trace(go.Scatter3d(
            x=[VERTICES[s][0], VERTICES[e][0]], y=[VERTICES[s][1], VERTICES[e][1]], z=[VERTICES[s][2], VERTICES[e][2]],
            mode='lines', line=dict(color='black', width=1), showlegend=False, hoverinfo='skip'
        ))

    fig.update_layout(
        scene=dict(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False),
        margin=dict(l=0, r=0, b=0, t=0),
        title=f"Widocznych punktów: {len(f_p)}"
    )

    return fig


if __name__ == '__main__':
    app.run(debug=True)