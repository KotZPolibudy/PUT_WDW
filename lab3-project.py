import numpy as np
import math
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output


# --- 1. KONFIGURACJA MATEMATYCZNA ---

def get_tetrahedron_vertices():
    return np.array([
        [0, 0, 0], [1, 0, 0], [0.5, math.sqrt(3) / 2, 0], [0.5, math.sqrt(3) / 6, math.sqrt(6) / 3]
    ])


FUNCTIONS = {
    'Entropia Shannona': lambda w: -sum(p * math.log2(p) for p in w if p > 0),
    'Gini Impurity': lambda w: 1 - sum(p ** 2 for p in w),
    'Accuracy (Klasa D)': lambda w: w[3],
    'Margin (D vs Reszta)': lambda w: w[3] - max(w[0], w[1], w[2]),
    'BŁĄD: Ratio Test (A+B / C+D)': lambda w: (w[0] + w[1]) / (w[2] + w[3]),
    'BŁĄD: Logit A/B': lambda w: math.log2(w[0] / w[1])
}

RESOLUTION = 40
VERTICES = get_tetrahedron_vertices()
weights_grid = []

for i in range(RESOLUTION + 1):
    for j in range(RESOLUTION + 1 - i):
        for k in range(RESOLUTION + 1 - i - j):
            w1 = i / RESOLUTION
            w2 = j / RESOLUTION
            w3 = k / RESOLUTION
            w4 = max(0, 1.0 - w1 - w2 - w3)
            weights_grid.append([w1, w2, w3, w4])

weights_grid = np.array(weights_grid)
points_3d = np.array(
    [(w[0] * VERTICES[0] + w[1] * VERTICES[1] + w[2] * VERTICES[2] + w[3] * VERTICES[3]) for w in weights_grid])

# --- 2. LAYOUT APLIKACJI ---

app = Dash(__name__)

app.layout = html.Div([
    html.H2("Diagnostyka Funkcji 4D - Wizualizacja Błędów", style={'textAlign': 'center', 'fontFamily': 'sans-serif'}),

    html.Div([
        html.Div([
            html.B("Wybierz funkcję:"),
            dcc.Dropdown(id='func-dropdown', options=[{'label': k, 'value': k} for k in FUNCTIONS.keys()],
                         value='Entropia Shannona'),

            html.Hr(),
            html.B("Filtr wartości (Floating Point):"),
            html.Div(id='slider-container', children=[
                dcc.RangeSlider(
                    id='val-slider', min=0, max=2, step=0.01, value=[0, 2],
                    marks={0: '0', 1: '1', 2: '2'},
                    tooltip={"placement": "bottom", "always_visible": True}
                )
            ]),

            html.Hr(),
            html.B("Zakresy wag (A, B, C, D):"),
            *[html.Div([
                html.Label(f"Zmienna {c}:"),
                dcc.RangeSlider(id=f'w{i + 1}-r', min=0, max=1, step=0.05, value=[0, 1])
            ], style={'marginBottom': '10px'}) for i, c in enumerate(['A', 'B', 'C', 'D'])],

            html.Div([
                html.B("Legenda Diagnostyczna:"),
                html.Div([html.Span("■ ", style={'color': '#FF0000'}), "Inf (Dzielenie przez 0)"]),
                html.Div([html.Span("■ ", style={'color': '#FF00FF'}), "Błąd domeny (np. log<=0)"]),
                html.Div([html.Span("■ ", style={'color': '#FFFFFF', 'textShadow': '0 0 2px black'}),
                          "NaN (Nieoznaczone 0/0)"]),
            ], style={'marginTop': '20px', 'padding': '15px', 'backgroundColor': '#333', 'color': 'white',
                      'borderRadius': '5px'})

        ], style={'width': '25%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '20px',
                  'backgroundColor': '#f4f4f4'}),

        html.Div([
            dcc.Graph(id='main-graph', style={'height': '85vh'})
        ], style={'width': '70%', 'display': 'inline-block'})
    ], style={'display': 'flex'})
])


# --- 3. REAKTYWNOŚĆ ---

# Callback do aktualizacji zakresu suwaka przy zmianie funkcji
@app.callback(
    [Output('val-slider', 'min'), Output('val-slider', 'max'), Output('val-slider', 'marks'),
     Output('val-slider', 'value')],
    Input('func-dropdown', 'value')
)
def update_slider_range(func_name):
    f = FUNCTIONS[func_name]
    temp_vals = []
    for w in weights_grid:
        try:
            res = f(w)
            if not np.isinf(res) and not np.isnan(res):
                temp_vals.append(res)
        except:
            pass

    if not temp_vals: return 0, 1, {0: '0', 1: '1'}, [0, 1]

    vmin, vmax = float(min(temp_vals)), float(max(temp_vals))
    # Tworzenie czytelnych etykiet (5 punktów kontrolnych)
    marks = {round(v, 2): str(round(v, 2)) for v in np.linspace(vmin, vmax, 5)}
    return vmin, vmax, marks, [vmin, vmax]


# Główny callback wykresu
@app.callback(
    Output('main-graph', 'figure'),
    [Input('func-dropdown', 'value'), Input('val-slider', 'value'),
     Input('w1-r', 'value'), Input('w2-r', 'value'), Input('w3-r', 'value'), Input('w4-r', 'value')]
)
def update_view(func_name, v_range, r1, r2, r3, r4):
    f = FUNCTIONS[func_name]
    vals, err_types = [], []

    for w in weights_grid:
        try:
            res = f(w)
            if np.isnan(res):
                (vals.append(0), err_types.append(3))
            elif np.isinf(res):
                (vals.append(0), err_types.append(1))
            else:
                (vals.append(res), err_types.append(0))
        except ZeroDivisionError:
            (vals.append(0), err_types.append(1))
        except (ValueError, OverflowError):
            (vals.append(0), err_types.append(2))
        except:
            (vals.append(0), err_types.append(3))

    vals, err_types = np.array(vals), np.array(err_types)

    mask_w = (weights_grid[:, 0] >= r1[0]) & (weights_grid[:, 0] <= r1[1]) & \
             (weights_grid[:, 1] >= r2[0]) & (weights_grid[:, 1] <= r2[1]) & \
             (weights_grid[:, 2] >= r3[0]) & (weights_grid[:, 2] <= r3[1]) & \
             (weights_grid[:, 3] >= r4[0]) & (weights_grid[:, 3] <= r4[1])

    fig = go.Figure()

    # 1. DANE POPRAWNE (Paleta Viridis - od błękitu do żółtego)
    valid_mask = (err_types == 0) & mask_w
    v_vals = vals[valid_mask]
    # Filtr wartości ze slidera
    val_filter = (v_vals >= v_range[0]) & (v_vals <= v_range[1])

    fig.add_trace(go.Scatter3d(
        x=points_3d[valid_mask][val_filter, 0], y=points_3d[valid_mask][val_filter, 1],
        z=points_3d[valid_mask][val_filter, 2],
        mode='markers', name='Poprawne',
        marker=dict(size=4, color=v_vals[val_filter], colorscale='Viridis', opacity=0.7,
                    colorbar=dict(title="Wynik", x=0.85)),
        customdata=weights_grid[valid_mask][val_filter],
        hovertemplate="A:%{customdata[0]:.2f} B:%{customdata[1]:.2f}<br>Wynik: %{marker.color:.4f}<extra></extra>"
    ))

    # 2. DANE BŁĘDNE (Neonowe barwy dla kontrastu)
    errors = [
        (1, '#FF0000', 'Inf (Dzielenie przez 0)'),  # Jasny czerwony
        (2, '#FF00FF', 'Błąd domeny'),  # Magenta/Fuksja
        (3, '#FFFFFF', 'NaN / Nieoznaczone')  # Biały
    ]

    for code, color, name in errors:
        e_mask = (err_types == code) & mask_w
        if any(e_mask):
            fig.add_trace(go.Scatter3d(
                x=points_3d[e_mask, 0], y=points_3d[e_mask, 1], z=points_3d[e_mask, 2],
                mode='markers', name=name,
                marker=dict(size=6, color=color, line=dict(color='black', width=1), opacity=1.0),
                customdata=weights_grid[e_mask],
                hovertemplate="<b>%{name}</b><br>Wagi: %{customdata}<extra></extra>"
            ))

    # Obramowanie
    edges = [(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)]
    for s, e in edges:
        fig.add_trace(go.Scatter3d(
            x=[VERTICES[s][0], VERTICES[e][0]], y=[VERTICES[s][1], VERTICES[e][1]], z=[VERTICES[s][2], VERTICES[e][2]],
            mode='lines', line=dict(color='black', width=1), showlegend=False, hoverinfo='skip'
        ))

    fig.update_layout(
        scene=dict(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False, bgcolor='white'),
        margin=dict(l=0, r=0, b=0, t=30),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )

    return fig


if __name__ == '__main__':
    app.run(debug=True)