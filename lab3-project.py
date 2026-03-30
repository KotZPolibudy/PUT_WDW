import numpy as np
import math
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State


# --- 1. KONFIGURACJA ---
def get_tetrahedron_vertices():
    return np.array([
        [0, 0, 0], [1, 0, 0], [0.5, math.sqrt(3) / 2, 0], [0.5, math.sqrt(3) / 6, math.sqrt(6) / 3]
    ])


VERTICES = get_tetrahedron_vertices()
CENTER = np.mean(VERTICES, axis=0)

# Predefiniowane funkcje
PRESETS = {
    'Entropia': '-sum(p * log2(p) for p in [A,B,C,D] if p > 1e-12)',
    'Gini': '1 - (A**2 + B**2 + C**2 + D**2)',
    'Odległość od środka': 'np.linalg.norm(np.dot([A,B,C,D], VERTICES) - CENTER)',
    'D': 'D',
    'A/D': 'A / D'
}


def safe_eval(expr, w_list):
    w = [round(x, 12) for x in w_list]  # Unikanie błędów

    # Wszystko ląduje w jednym słowniku, który będzie służył jako 'globals'
    ctx = {
        'a': w[0], 'b': w[1], 'c': w[2], 'd': w[3],
        'A': w[0], 'B': w[1], 'C': w[2], 'D': w[3],
        'np': np,
        'math': math,
        'sqrt': math.sqrt,
        'log2': math.log2,
        'exp': math.exp,
        'abs': abs,
        'sum': sum,
        'max': max,
        'min': min,
        'VERTICES': VERTICES,
        'CENTER': CENTER
    }
    return eval(expr, ctx)


def generate_simplex_data(res, expr):
    weights, points, values, err_types = [], [], [], []
    # Próg, powyżej którego liczba jest traktowana jako błąd Inf (np. przy dzieleniu przez 1e-15)
    INF_THRESHOLD = 1e9

    for i in range(res + 1):
        for j in range(res + 1 - i):
            for k in range(res + 1 - i - j):
                w1 = i / res
                w2 = j / res
                w3 = k / res
                w4 = max(0, 1.0 - w1 - w2 - w3)
                w = [w1, w2, w3, w4]

                p = w[0] * VERTICES[0] + w[1] * VERTICES[1] + w[2] * VERTICES[2] + w[3] * VERTICES[3]

                try:
                    res_val = safe_eval(expr, w)

                    if np.isnan(res_val):
                        (val, err) = (0, 3)
                    elif np.isinf(res_val) or abs(res_val) > INF_THRESHOLD:
                        (val, err) = (0, 1)  # Czerwony (Inf)
                    else:
                        (val, err) = (float(res_val), 0)  # Poprawne
                except ZeroDivisionError:
                    (val, err) = (0, 1)  # Czerwony
                except (ValueError, NameError, SyntaxError, TypeError):
                    (val, err) = (0, 2)  # Magenta (Błąd formuły)
                except:
                    (val, err) = (0, 3)  # Biały (NaN)

                weights.append(w)
                points.append(p)
                values.append(val)
                err_types.append(err)

    return np.array(weights), np.array(points), np.array(values), np.array(err_types)


# --- 2. LAYOUT APLIKACJI ---

app = Dash(__name__)

app.layout = html.Div([
    html.H2("Custom Function Simplex Explorer 4D", style={'textAlign': 'center', 'fontFamily': 'sans-serif'}),

    html.Div([
        html.Div([
            html.B("Wybierz preset lub wpisz własną funkcję:"),
            dcc.Dropdown(id='preset-dropdown', options=[{'label': k, 'value': v} for k, v in PRESETS.items()],
                         value=PRESETS['Entropia']),

            html.Div([
                html.Label("Formuła (zmienne A, B, C, D):"),
                dcc.Input(id='custom-func-input', type='text', value=PRESETS['Entropia'],
                          style={'width': '100%', 'marginTop': '5px', 'fontFamily': 'monospace'})
            ], style={'marginTop': '10px'}),

            html.Hr(),
            html.B("Gęstość (Resolution):"),
            dcc.Slider(id='res-slider', min=10, max=60, step=5, value=30),

            html.Hr(),
            html.B("Filtr wartości:"),
            html.Div(id='slider-container', children=[
                dcc.RangeSlider(id='val-slider', min=0, max=2, step=0.01, value=[0, 2],
                                tooltip={"placement": "bottom", "always_visible": True})
            ]),

            html.Hr(),
            html.B("Zakresy wag (A, B, C, D):"),
            *[html.Div([
                html.Label(f"Zmienna {c}:"),
                dcc.RangeSlider(id=f'w{i + 1}-r', min=0, max=1, step=0.05, value=[0, 1])
            ], style={'marginBottom': '10px'}) for i, c in enumerate(['A', 'B', 'C', 'D'])],

            html.Div([
                html.B("Diagnostyka Błędów:"),
                html.Div([html.Span("■ ", style={'color': '#FF0000'}), "Inf (np. A/0)"]),
                html.Div([html.Span("■ ", style={'color': '#FF00FF'}), "Błąd formuły (Syntax/Domain)"]),
                html.Div([html.Span("■ ", style={'color': '#FFFFFF', 'textShadow': '0 0 2px black'}), "NaN"]),
            ], style={'marginTop': '20px', 'padding': '15px', 'backgroundColor': '#333', 'color': 'white',
                      'borderRadius': '5px'})

        ], style={'width': '25%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '20px',
                  'backgroundColor': '#f4f4f4'}),

        html.Div([
            dcc.Loading(id="loading-graph", children=[dcc.Graph(id='main-graph', style={'height': '85vh'})],
                        type="circle")
        ], style={'width': '70%', 'display': 'inline-block'})
    ], style={'display': 'flex'})
])


# --- 3. REAKTYWNOŚĆ ---

# Synchronizacja dropdownu z inputem tekstowym
@app.callback(
    Output('custom-func-input', 'value'),
    Input('preset-dropdown', 'value')
)
def update_input_from_preset(preset_val):
    return preset_val


# Aktualizacja zakresu suwaka wartości
@app.callback(
    [Output('val-slider', 'min'), Output('val-slider', 'max'), Output('val-slider', 'marks'),
     Output('val-slider', 'value')],
    [Input('custom-func-input', 'value'), Input('res-slider', 'value')]
)
def update_slider_range(expr, res):
    try:
        _, _, values, err_types = generate_simplex_data(15, expr)  # Mała rozdzielczość dla szybkości
        valid_vals = values[err_types == 0]
        if len(valid_vals) == 0: return 0, 1, {0: '0', 1: '1'}, [0, 1]
        vmin, vmax = float(np.min(valid_vals)), float(np.max(valid_vals))
        if vmin == vmax: vmax += 0.01
        marks = {round(v, 2): str(round(v, 2)) for v in np.linspace(vmin, vmax, 5)}
        return vmin, vmax, marks, [vmin, vmax]
    except:
        return 0, 1, {0: '0', 1: '1'}, [0, 1]


# Główny wykres
@app.callback(
    Output('main-graph', 'figure'),
    [Input('custom-func-input', 'value'), Input('res-slider', 'value'), Input('val-slider', 'value'),
     Input('w1-r', 'value'), Input('w2-r', 'value'), Input('w3-r', 'value'), Input('w4-r', 'value')]
)
def update_view(expr, res, v_range, r1, r2, r3, r4):
    weights, points, values, err_types = generate_simplex_data(res, expr)

    mask_w = (weights[:, 0] >= r1[0]) & (weights[:, 0] <= r1[1]) & \
             (weights[:, 1] >= r2[0]) & (weights[:, 1] <= r2[1]) & \
             (weights[:, 2] >= r3[0]) & (weights[:, 2] <= r3[1]) & \
             (weights[:, 3] >= r4[0]) & (weights[:, 3] <= r4[1])

    fig = go.Figure()

    # Dane poprawne
    v_mask = (err_types == 0) & mask_w
    v_vals = values[v_mask]
    val_filter = (v_vals >= v_range[0]) & (v_vals <= v_range[1])

    fig.add_trace(go.Scatter3d(
        x=points[v_mask][val_filter, 0], y=points[v_mask][val_filter, 1], z=points[v_mask][val_filter, 2],
        mode='markers', name='Poprawne',
        marker=dict(size=4 if res < 40 else 3, color=v_vals[val_filter], colorscale='Viridis', opacity=0.7,
                    colorbar=dict(title="Wynik")),
        customdata=weights[v_mask][val_filter],
        hovertemplate="A:%{customdata[0]:.2f} B:%{customdata[1]:.2f} C:%{customdata[2]:.2f} D:%{customdata[3]:.2f}<br>Wynik: %{marker.color:.4f}<extra></extra>"
    ))

    # Dane błędne
    errors = [(1, '#FF0000', 'Inf'), (2, '#FF00FF', 'Błąd formuły'), (3, '#FFFFFF', 'NaN')]
    for code, color, name in errors:
        e_mask = (err_types == code) & mask_w
        if any(e_mask):
            fig.add_trace(go.Scatter3d(
                x=points[e_mask, 0], y=points[e_mask, 1], z=points[e_mask, 2],
                mode='markers', name=name,
                marker=dict(size=5, color=color, line=dict(color='black', width=1), opacity=1.0),
                customdata=weights[e_mask]
            ))

    # Szkielet
    edges = [(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)]
    for s, e in edges:
        fig.add_trace(go.Scatter3d(
            x=[VERTICES[s][0], VERTICES[e][0]], y=[VERTICES[s][1], VERTICES[e][1]], z=[VERTICES[s][2], VERTICES[e][2]],
            mode='lines', line=dict(color='black', width=1), showlegend=False
        ))

    fig.update_layout(scene=dict(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False, bgcolor='white'),
                      margin=dict(l=0, r=0, b=0, t=30))
    return fig


if __name__ == '__main__':
    app.run(debug=True)