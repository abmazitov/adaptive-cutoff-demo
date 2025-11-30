#!/usr/bin/env python3
"""
Interactive Dashboard for Adaptive Cutoff Visualization

This script creates an interactive dashboard to visualize adaptive cutoff behavior
with adjustable parameters using Plotly Dash.
"""

import os
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html
from metatomic.torch import NeighborListOptions
from .utils import compute_adaptive_cutoff, create_atom_configuration


# Global configuration
GLOBAL_CUTOFF = 7.0
CUTOFF_WIDTH = 1.0
options = NeighborListOptions(cutoff=GLOBAL_CUTOFF, full_list=True, strict=True)


# Initialize Dash app
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# For deployment
server = app.server

# App layout
app.layout = dbc.Container(
    [
        html.H1(
            "Adaptive Cutoff Visualization Dashboard", className="text-center my-4"
        ),
        # Hidden component to store the random seed
        dcc.Store(id="random-seed", data=42),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H4("Controls", className="mb-3"),
                        html.Label("Number of Random Atoms (1-10):"),
                        dcc.Slider(
                            id="num-atoms-slider",
                            min=1,
                            max=10,
                            step=1,
                            value=5,
                            marks={i: str(i) for i in range(1, 11)},
                            tooltip={"placement": "bottom", "always_visible": True},
                            updatemode="drag",
                        ),
                        html.Br(),
                        html.Label("Special Atom Y Position (-10 to 10):"),
                        dcc.Slider(
                            id="special-y-slider",
                            min=-10,
                            max=10,
                            step=0.1,
                            value=2.0,
                            marks={i: str(i) for i in range(-10, 11)},
                            tooltip={"placement": "bottom", "always_visible": True},
                            updatemode="drag",
                        ),
                        html.Br(),
                        html.Label("Weight Function:"),
                        dcc.RadioItems(
                            id="weight-function",
                            options=[
                                {"label": " Gaussian", "value": "gaussian"},
                                {"label": " Exponential", "value": "exponential"},
                            ],
                            value="gaussian",
                            inline=True,
                        ),
                        html.Br(),
                        html.Label("Max Number of Neighbors:"),
                        dcc.Slider(
                            id="max-neighbors-slider",
                            min=0.5,
                            max=10,
                            step=0.5,
                            value=2.0,
                            marks={i: str(i) for i in range(0, 11, 2)},
                            tooltip={"placement": "bottom", "always_visible": True},
                            updatemode="drag",
                        ),
                        html.Br(),
                        html.Label("Width:"),
                        dcc.Slider(
                            id="width-slider",
                            min=0.1,
                            max=2.0,
                            step=0.1,
                            value=0.5,
                            marks={i * 0.5: str(i * 0.5) for i in range(0, 5)},
                            tooltip={"placement": "bottom", "always_visible": True},
                            updatemode="drag",
                        ),
                        html.Br(),
                        html.Div(
                            id="beta-container",
                            children=[
                                html.Label("Beta (Exponential only):"),
                                dcc.Slider(
                                    id="beta-slider",
                                    min=0.1,
                                    max=5.0,
                                    step=0.1,
                                    value=1.0,
                                    marks={i: str(i) for i in range(0, 6)},
                                    tooltip={
                                        "placement": "bottom",
                                        "always_visible": True,
                                    },
                                    updatemode="drag",
                                ),
                            ],
                        ),
                        html.Br(),
                        dbc.Button(
                            "Regenerate Random Atoms",
                            id="regenerate-button",
                            color="primary",
                            className="w-100 mb-3",
                        ),
                        html.Div(
                            id="cutoff-info",
                            className="alert alert-info",
                            style={"margin-top": "20px"},
                        ),
                    ],
                    width=4,
                ),
                dbc.Col(
                    [
                        dcc.Graph(id="atom-visualization", style={"height": "500px"}),
                        dcc.Graph(id="nef-plot", style={"height": "400px"}),
                    ],
                    width=8,
                ),
            ]
        ),
    ],
    fluid=True,
)


@app.callback(
    Output("random-seed", "data"),
    Input("regenerate-button", "n_clicks"),
    prevent_initial_call=True,
)
def regenerate_seed(n_clicks):
    """Generate a new random seed when button is clicked."""
    return np.random.randint(0, 100000)


@app.callback(
    [
        Output("atom-visualization", "figure"),
        Output("nef-plot", "figure"),
        Output("cutoff-info", "children"),
    ],
    [
        Input("num-atoms-slider", "value"),
        Input("special-y-slider", "value"),
        Input("weight-function", "value"),
        Input("max-neighbors-slider", "value"),
        Input("width-slider", "value"),
        Input("beta-slider", "value"),
        Input("random-seed", "data"),
    ],
)
def update_visualization(
    num_atoms, special_y, weight_function, max_neighbors, width, beta, seed
):
    """Update the visualization based on slider values."""

    # Create atomic configuration with the current seed
    atoms = create_atom_configuration(num_atoms, special_y, seed=seed)
    positions = atoms.get_positions()

    # Compute adaptive cutoff
    cutoff_value, eff_num_neighbors, probe_cutoffs = compute_adaptive_cutoff(
        atoms,
        options,
        weight_function=weight_function,
        max_num_neighbors=max_neighbors,
        width=width,
        beta=beta,
    )

    # Create atom visualization
    fig_atoms = go.Figure()

    # Plot cutoff circle for central atom
    theta = np.linspace(0, 2 * np.pi, 100)
    circle_x = cutoff_value * np.cos(theta)
    circle_y = cutoff_value * np.sin(theta)

    fig_atoms.add_trace(
        go.Scatter(
            x=circle_x,
            y=circle_y,
            mode="lines",
            line=dict(color="rgba(100, 100, 255, 0.5)", width=2, dash="dash"),
            name=f"Adaptive Cutoff (r={cutoff_value:.2f} Å)",
            hoverinfo="name",
        )
    )

    # Plot atoms
    # Central atom
    fig_atoms.add_trace(
        go.Scatter(
            x=[positions[0, 0]],
            y=[positions[0, 1]],
            mode="markers",
            marker=dict(
                size=20,
                color="red",
                symbol="circle",
                line=dict(color="darkred", width=2),
            ),
            name="Central Atom",
            text=["Central Atom"],
            hovertemplate="<b>%{text}</b><br>x: %{x:.2f}<br>y: %{y:.2f}<extra></extra>",
        )
    )

    # Random atoms
    if num_atoms > 1:
        fig_atoms.add_trace(
            go.Scatter(
                x=positions[1:num_atoms, 0],
                y=positions[1:num_atoms, 1],
                mode="markers",
                marker=dict(
                    size=15,
                    color="lightblue",
                    symbol="circle",
                    line=dict(color="darkblue", width=1),
                ),
                name="Random Atoms",
                text=[f"Atom {i + 1}" for i in range(num_atoms - 1)],
                hovertemplate="<b>%{text}</b><br>x: %{x:.2f}<br>y: %{y:.2f}<extra></extra>",
            )
        )

    # Special atom
    fig_atoms.add_trace(
        go.Scatter(
            x=[positions[-1, 0]],
            y=[positions[-1, 1]],
            mode="markers",
            marker=dict(
                size=18,
                color="gold",
                symbol="star",
                line=dict(color="orange", width=2),
            ),
            name="Special Atom",
            text=["Special Atom"],
            hovertemplate="<b>%{text}</b><br>x: %{x:.2f}<br>y: %{y:.2f}<extra></extra>",
        )
    )

    fig_atoms.update_layout(
        title="Atomic Configuration (Top View, z=0)",
        xaxis=dict(title="x (Å)", range=[-6, 6], scaleanchor="y", scaleratio=1),
        yaxis=dict(title="y (Å)", range=[-10, 10]),
        showlegend=True,
        hovermode="closest",
        plot_bgcolor="rgba(240, 240, 240, 0.5)",
        height=500,
    )

    # Create effective number of neighbors plot
    fig_nef = go.Figure()

    fig_nef.add_trace(
        go.Scatter(
            x=probe_cutoffs,
            y=eff_num_neighbors,
            mode="lines",
            line=dict(color="darkgreen", width=3),
            name="Effective # Neighbors",
        )
    )

    # Add horizontal line for max_neighbors
    fig_nef.add_hline(
        y=max_neighbors,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Max N = {max_neighbors}",
        annotation_position="right",
    )

    # Add vertical line for adaptive cutoff
    fig_nef.add_vline(
        x=cutoff_value,
        line_dash="dot",
        line_color="blue",
        annotation_text=f"Adaptive Cutoff = {cutoff_value:.2f} Å",
        annotation_position="top",
    )

    fig_nef.update_layout(
        title="Effective Number of Neighbors vs Probe Cutoff",
        xaxis_title="Probe Cutoff (Å)",
        yaxis_title="Effective Number of Neighbors",
        showlegend=True,
        plot_bgcolor="rgba(240, 240, 240, 0.5)",
        height=400,
    )

    # Info text
    info_text = html.Div(
        [
            html.H5("Current Configuration:", className="alert-heading"),
            html.P(
                [
                    html.Strong("Total atoms: "),
                    f"{num_atoms + 1} (1 central + {num_atoms - 1} random + 1 special)",
                    html.Br(),
                    html.Strong("Special atom position: "),
                    f"[0.0, {special_y:.1f}, 0.0]",
                    html.Br(),
                    html.Strong("Weight function: "),
                    weight_function.capitalize(),
                    html.Br(),
                    html.Strong("Adaptive cutoff: "),
                    f"{cutoff_value:.3f} Å",
                    html.Br(),
                    html.Strong("Max neighbors: "),
                    f"{max_neighbors}",
                ]
            ),
        ]
    )

    return fig_atoms, fig_nef, info_text


def main():
    """Main entry point for the dashboard."""
    print("Starting Adaptive Cutoff Dashboard...")

    # Get port from environment variable (for deployment) or use default
    port = int(os.environ.get("PORT", 8050))
    host = os.environ.get("HOST", "0.0.0.0")

    print(f"Server running on: http://{host}:{port}")
    app.run(debug=False, host=host, port=port)


if __name__ == "__main__":
    main()
