import numpy as np
import math
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output


# --- 1. KONFIGURACJA MATEMATYCZNA ---

def get_tetrahedron_vertices():
    return np.array([
        [0, 0, 0], [1, 0, 0], [0.5, math.sqrt(3) / 2, 0], [0.5, math.sqrt(3) / 6, math.sqrt(6) / 3]
    ])


VERTICES = get_tetrahedron_vertices()
CENTER = np.mean(VERTICES, axis=0)

FUNCTIONS = {
    'Entropia Shannona': lambda w: -sum(p * math.log2(p) for p in w if p > 0),
    'Gini Impurity': lambda w: 1 - sum(p ** 2 for p in w),
    'Accuracy (Klasa D)': lambda w: w[3],
    'Margin (D vs Reszta)': lambda w: w[3] - max(w[0], w[1], w[2]),
    'BŁĄD: Ratio Test (A+B / C+D)': lambda w: (w[0] + w[1]) / (w[2] + w[3]),
    'BŁĄD: Logit A/B': lambda w: math.log2(w[0] / w[1]),
    'Odległość od środka': lambda w: np.linalg.norm(np.dot(w, VERTICES) - CENTER),
    'Kwadrat odległości od środka': lambda w: np.sum((np.dot(w, VERTICES) - CENTER) ** 2)
}


def generate_simplex_data(res, func_name):
    """Generuje wagi, pozycje 3D i wartości dla danej rozdzielczości."""
    f = FUNCTIONS[func_name]
    weights, points, values, err_types = [], [], [], []

    for i in range(res + 1):
        for j in range(res + 1 - i):
            for k in range(res + 1 - i - j):
                w1 = i / res
                w2 = j / res
                w3 = k / res
                w4 = max(0, 1.0 - w1 - w2 - w3)
                w = [w1, w2, w3, w4]

                # Pozycja 3D
                p = w[0] * VERTICES[0] + w[1] * VERTICES[1] + w[2] * VERTICES[2] + w[3] * VERTICES[3]

                try:
                    res_val = f(w)
                    if np.isnan(res_val):
                        (val, err) = (0, 3)
                    elif np.isinf(res_val):
                        (val, err) = (0, 1)
                    else:
                        (val, err) = (res_val, 0)
                except ZeroDivisionError:
                    (val, err) = (0, 1)
                except (ValueError, OverflowError):
                    (val, err) = (0, 2)
                except:
                    (val, err) = (0, 3)

                weights.append(w)
                points.append(p)
                values.append(val)
                err_types.append(err)

    return np.array(weights), np.array(points), np.array(values), np.array(err_types)


# --- 2. LAYOUT APLIKACJI ---

app = Dash(__name__)

app.layout = html.Div([
    html.H2("Eksplorator Funkcji 4D - Pełna Kontrola Gęstości",
            style={'textAlign': 'center', 'fontFamily': 'sans-serif'}),

    html.Div([
        html.Div([
            html.B("Wybierz funkcję:"),
            dcc.Dropdown(id='func-dropdown', options=[{'label': k, 'value': k} for k in FUNCTIONS.keys()],
                         value='Entropia Shannona'),

            html.Hr(),
            html.B("Gęstość punktów (Resolution):"),
            dcc.Slider(id='res-slider', min=10, max=60, step=5, value=30,
                       marks={i: str(i) for i in range(10, 61, 10)},
                       tooltip={"placement": "bottom", "always_visible": True}),

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
                  'backgroundColor': '#f4f4f4', 'borderRadius': '10px'}),

        html.Div([
            dcc.Loading(id="loading-graph", children=[dcc.Graph(id='main-graph', style={'height': '85vh'})],
                        type="circle")
        ], style={'width': '70%', 'display': 'inline-block'})
    ], style={'display': 'flex'})
])


# --- 3. REAKTYWNOŚĆ ---

@app.callback(
    [Output('val-slider', 'min'), Output('val-slider', 'max'), Output('val-slider', 'marks'),
     Output('val-slider', 'value')],
    [Input('func-dropdown', 'value'), Input('res-slider', 'value')]
)
def update_slider_range(func_name, res):
    # Generujemy małą próbkę danych, by określić zakresy dla suwaka
    _, _, values, err_types = generate_simplex_data(res, func_name)
    valid_vals = values[err_types == 0]

    if len(valid_vals) == 0: return 0, 1, {0: '0', 1: '1'}, [0, 1]

    vmin, vmax = float(np.min(valid_vals)), float(np.max(valid_vals))
    marks = {round(v, 2): str(round(v, 2)) for v in np.linspace(vmin, vmax, 5)}
    return vmin, vmax, marks, [vmin, vmax]


@app.callback(
    Output('main-graph', 'figure'),
    [Input('func-dropdown', 'value'), Input('res-slider', 'value'), Input('val-slider', 'value'),
     Input('w1-r', 'value'), Input('w2-r', 'value'), Input('w3-r', 'value'), Input('w4-r', 'value')]
)
def update_view(func_name, res, v_range, r1, r2, r3, r4):
    # Generowanie danych dla wybranej rozdzielczości
    weights, points, values, err_types = generate_simplex_data(res, func_name)

    # Maska zakresów wag
    mask_w = (weights[:, 0] >= r1[0]) & (weights[:, 0] <= r1[1]) & \
             (weights[:, 1] >= r2[0]) & (weights[:, 1] <= r2[1]) & \
             (weights[:, 2] >= r3[0]) & (weights[:, 2] <= r3[1]) & \
             (weights[:, 3] >= r4[0]) & (weights[:, 3] <= r4[1])

    fig = go.Figure()

    # 1. DANE POPRAWNE
    valid_mask = (err_types == 0) & mask_w
    v_vals = values[valid_mask]
    val_filter = (v_vals >= v_range[0]) & (v_vals <= v_range[1])

    fig.add_trace(go.Scatter3d(
        x=points[valid_mask][val_filter, 0], y=points[valid_mask][val_filter, 1],
        z=points[valid_mask][val_filter, 2],
        mode='markers', name='Poprawne',
        marker=dict(size=4 if res < 40 else 3, color=v_vals[val_filter], colorscale='Viridis', opacity=0.7,
                    colorbar=dict(title="Wynik", x=0.85)),
        customdata=weights[valid_mask][val_filter],
        hovertemplate="A:%{customdata[0]:.2f} B:%{customdata[1]:.2f} C:%{customdata[2]:.2f} D:%{customdata[3]:.2f}<br>Wynik: %{marker.color:.4f}<extra></extra>"
    ))

    # 2. DANE BŁĘDNE
    errors = [(1, '#FF0000', 'Inf'), (2, '#FF00FF', 'Błąd domeny'), (3, '#FFFFFF', 'NaN')]

    for code, color, name in errors:
        e_mask = (err_types == code) & mask_w
        if any(e_mask):
            fig.add_trace(go.Scatter3d(
                x=points[e_mask, 0], y=points[e_mask, 1], z=points[e_mask, 2],
                mode='markers', name=name,
                marker=dict(size=5, color=color, line=dict(color='black', width=1), opacity=1.0),
                customdata=weights[e_mask],
                hovertemplate="<b>%{name}</b><br>Wagi: %{customdata}<extra></extra>"
            ))

    # Obramowanie czworościanu
    edges = [(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)]
    for s, e in edges:
        fig.add_trace(go.Scatter3d(
            x=[VERTICES[s][0], VERTICES[e][0]], y=[VERTICES[s][1], VERTICES[e][1]], z=[VERTICES[s][2], VERTICES[e][2]],
            mode='lines', line=dict(color='black', width=1), showlegend=False, hoverinfo='skip'
        ))

    fig.update_layout(
        scene=dict(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False, bgcolor='white'),
        margin=dict(l=0, r=0, b=0, t=30),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        title=f"Liczba punktów w chmurze: {len(points)}"
    )

    return fig


if __name__ == '__main__':
    app.run(debug=True)