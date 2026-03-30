import numpy as np
import math
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output


# --- 1. KONFIGURACJA MATEMATYCZNA ---
def get_tetrahedron_vertices():
    return np.array([
        [0, 0, 0],  # A (w1)
        [1, 0, 0],  # B (w2)
        [0.5, math.sqrt(3) / 2, 0],  # C (w3)
        [0.5, math.sqrt(3) / 6, math.sqrt(6) / 3]  # D (w4)
    ])


def my_function(w):
    return -sum(p * math.log2(p) for p in w if p > 1e-9)  # Entropia


RESOLUTION = 40
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

# Stylizacja dla suwaków
slider_style = {'marginBottom': '25px', 'padding': '0 10px'}

app.layout = html.Div([
    html.H2("Precyzyjne Cięcie Simpleksu 4D", style={'textAlign': 'center', 'fontFamily': 'sans-serif'}),

    html.Div([
        # Panel boczny
        html.Div([
            html.B("Filtr wartości funkcji:"),
            dcc.RangeSlider(
                id='val-slider', min=float(min(values)), max=float(max(values)),
                value=[min(values), max(values)],
                tooltip={"placement": "bottom", "always_visible": True}
            ),
            html.Hr(),
            html.B("Zakresy udziałów zmiennych (min - max):"),

            html.Div([html.Label("Zmienna A (Baza Lewa):"),
                      dcc.RangeSlider(id='w1-r', min=0, max=1, step=0.05, value=[0, 1])], style=slider_style),

            html.Div([html.Label("Zmienna B (Baza Prawa):"),
                      dcc.RangeSlider(id='w2-r', min=0, max=1, step=0.05, value=[0, 1])], style=slider_style),

            html.Div([html.Label("Zmienna C (Baza Tył):"),
                      dcc.RangeSlider(id='w3-r', min=0, max=1, step=0.05, value=[0, 1])], style=slider_style),

            html.Div([html.Label("Zmienna D (Szczyt):"),
                      dcc.RangeSlider(id='w4-r', min=0, max=1, step=0.05, value=[0, 1])], style=slider_style),

        ], style={'width': '25%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '20px',
                  'backgroundColor': '#f4f4f4', 'borderRadius': '10px'}),

        # Panel wykresu
        html.Div([
            dcc.Graph(id='main-graph', style={'height': '85vh'})
        ], style={'width': '70%', 'display': 'inline-block'})
    ], style={'display': 'flex', 'justifyContent': 'space-around', 'padding': '10px'})
])


# --- 3. REAKTYWNOŚĆ (CALLBACK) ---

@app.callback(
    Output('main-graph', 'figure'),
    [Input('val-slider', 'value'),
     Input('w1-r', 'value'), Input('w2-r', 'value'),
     Input('w3-r', 'value'), Input('w4-r', 'value')]
)
def update_view(v_range, r1, r2, r3, r4):
    # Logika filtrowania z użyciem zakresów [min, max]
    mask = (values >= v_range[0]) & (values <= v_range[1]) & \
           (weights[:, 0] >= r1[0]) & (weights[:, 0] <= r1[1]) & \
           (weights[:, 1] >= r2[0]) & (weights[:, 1] <= r2[1]) & \
           (weights[:, 2] >= r3[0]) & (weights[:, 2] <= r3[1]) & \
           (weights[:, 3] >= r4[0]) & (weights[:, 3] <= r4[1])

    f_p = points[mask]
    f_v = values[mask]
    f_w = weights[mask]

    fig = go.Figure()

    # Chmura punktów
    fig.add_trace(go.Scatter3d(
        x=f_p[:, 0], y=f_p[:, 1], z=f_p[:, 2],
        mode='markers',
        marker=dict(
            size=4, color=f_v, colorscale='Plasma', opacity=0.7,
            colorbar=dict(title="Wartość", thickness=20)
        ),
        customdata=f_w,
        hovertemplate="A:%{customdata[0]:.2f} B:%{customdata[1]:.2f}<br>" +
                      "C:%{customdata[2]:.2f} D:%{customdata[3]:.2f}<br>" +
                      "<b>Wynik: %{marker.color:.4f}</b><extra></extra>"
    ))

    # Szkielet czworościanu (obramowanie)
    edges = [(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)]
    for s, e in edges:
        fig.add_trace(go.Scatter3d(
            x=[VERTICES[s][0], VERTICES[e][0]], y=[VERTICES[s][1], VERTICES[e][1]], z=[VERTICES[s][2], VERTICES[e][2]],
            mode='lines', line=dict(color='rgba(0,0,0,0.3)', width=2), showlegend=False, hoverinfo='skip'
        ))

    fig.update_layout(
        scene=dict(
            xaxis_visible=False, yaxis_visible=False, zaxis_visible=False,
            camera=dict(eye=dict(x=1.8, y=1.8, z=1.8))
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        title=f"Punktów w widoku: {len(f_p)}"
    )

    return fig


if __name__ == '__main__':
    app.run(debug=True)