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
from dash import Dash, Input, Output, State, dcc, html
from metatomic.torch import NeighborListOptions
from .utils import (
    compute_adaptive_cutoff,
    create_atom_configuration,
    compute_special_atom_cutoffs_vs_position,
)


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
        # Cache for cutoff vs position curve data
        dcc.Store(id="cutoff-curve-cache", data=None),
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
                            updatemode="mouseup",
                        ),
                        html.Br(),
                        html.Label("Special Atom Y Position (0 to 10):"),
                        dcc.Slider(
                            id="special-y-slider",
                            min=0,
                            max=10,
                            step=0.1,
                            value=2.0,
                            marks={i: str(i) for i in range(-10, 11)},
                            tooltip={"placement": "bottom", "always_visible": True},
                            updatemode="mouseup",
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
                            updatemode="mouseup",
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
                            updatemode="mouseup",
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
                                    updatemode="mouseup",
                                ),
                            ],
                        ),
                        html.Br(),
                        html.Label("Probe Cutoff Step Size:"),
                        dcc.Slider(
                            id="step-size-slider",
                            min=0.01,
                            max=1.0,
                            step=0.01,
                            value=0.1,
                            marks={i * 0.2: str(i * 0.2) for i in range(0, 6)},
                            tooltip={"placement": "bottom", "always_visible": True},
                            updatemode="mouseup",
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
                        dbc.Row(
                            [
                                dbc.Col(
                                    dcc.Graph(id="nef-plot", style={"height": "400px"}),
                                    width=6,
                                ),
                                dbc.Col(
                                    dcc.Graph(
                                        id="cutoff-vs-position",
                                        style={"height": "400px"},
                                    ),
                                    width=6,
                                ),
                            ]
                        ),
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
    Output("cutoff-curve-cache", "data"),
    [
        Input("num-atoms-slider", "value"),
        Input("weight-function", "value"),
        Input("max-neighbors-slider", "value"),
        Input("width-slider", "value"),
        Input("beta-slider", "value"),
        Input("step-size-slider", "value"),
        Input("random-seed", "data"),
    ],
)

def compute_cutoff_curve(
    num_atoms, weight_function, max_neighbors, width, beta, step_size, seed
):
    """Compute and cache the cutoff vs position curve."""
    y_positions = np.linspace(0, 10, 50)
    cutoffs_vs_y = compute_special_atom_cutoffs_vs_position(
        num_atoms,
        y_positions,
        seed,
        options,
        weight_function=weight_function,
        max_num_neighbors=max_neighbors,
        width=width,
        beta=beta,
        step_size=step_size,
    )
    return {"y_positions": y_positions.tolist(), "cutoffs": cutoffs_vs_y.tolist()}


@app.callback(
    [
        Output("atom-visualization", "figure"),
        Output("nef-plot", "figure"),
        Output("cutoff-vs-position", "figure"),
        Output("cutoff-info", "children"),
    ],
    [
        Input("special-y-slider", "value"),
        Input("cutoff-curve-cache", "data"),
    ],
    [
        State("num-atoms-slider", "value"),
        State("weight-function", "value"),
        State("max-neighbors-slider", "value"),
        State("width-slider", "value"),
        State("beta-slider", "value"),
        State("step-size-slider", "value"),
        State("random-seed", "data"),
    ],
)
def update_visualization(
    special_y,
    curve_cache,
    num_atoms,
    weight_function,
    max_neighbors,
    width,
    beta,
    step_size,
    seed,
):
    """Update the visualization based on slider values."""

    # Create atomic configuration with the current seed
    atoms = create_atom_configuration(num_atoms, special_y, seed=seed)
    positions = atoms.get_positions()

    # Compute adaptive cutoffs for all atoms (single computation)
    all_cutoffs, eff_num_neighbors, probe_cutoffs, probe_weights = compute_adaptive_cutoff(
        atoms,
        options,
        weight_function=weight_function,
        max_num_neighbors=max_neighbors,
        width=width,
        beta=beta,
        step_size=step_size,
        atom_index=0,  # For effective neighbors plot
        return_all_cutoffs=True,
    )

    # Extract central atom cutoff
    cutoff_value = all_cutoffs[0]

    # Create atom visualization
    fig_atoms = go.Figure()

    # Plot cutoff circles for all atoms
    theta = np.linspace(0, 2 * np.pi, 100)
    for i, (pos, cutoff) in enumerate(zip(positions, all_cutoffs)):
        circle_x = pos[0] + cutoff * np.cos(theta)
        circle_y = pos[1] + cutoff * np.sin(theta)

        # Different styling for central atom
        if i == 0:
            line_color = "rgba(255, 0, 0, 0.3)"
            line_width = 2
        else:
            line_color = "rgba(100, 100, 100, 0.2)"
            line_width = 1

        fig_atoms.add_trace(
            go.Scatter(
                x=circle_x,
                y=circle_y,
                mode="lines",
                line=dict(color=line_color, width=line_width, dash="dash"),
                name="Adaptive Cutoffs" if i == 0 else None,
                showlegend=(i == 0),
                hoverinfo="skip",
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
        legend=dict(
            x=0.02,
            y=0.98,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(255, 255, 255, 0.8)",
        ),
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
            mode="lines+markers",
            line=dict(color="rgba(0, 100, 0, 0.4)", width=2),
            marker=dict(color="darkgreen", size=probe_weights[0]*500, line=dict(color="white", width=1)),
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
        legend=dict(
            x=0.02,
            y=0.98,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(255, 255, 255, 0.8)",
        ),
        plot_bgcolor="rgba(240, 240, 240, 0.5)",
        height=400,
    )

    # Get cached cutoff vs position curve
    if curve_cache is None:
        # Return empty figure if cache is not ready
        fig_cutoff_pos = go.Figure()
        fig_cutoff_pos.update_layout(
            title="Special Atom Cutoff vs Y Position",
            xaxis_title="Y Position (Å)",
            yaxis_title="Adaptive Cutoff (Å)",
        )
    else:
        y_positions = np.array(curve_cache["y_positions"])
        cutoffs_vs_y = np.array(curve_cache["cutoffs"])

        # Create cutoff vs position plot
        fig_cutoff_pos = go.Figure()

        fig_cutoff_pos.add_trace(
            go.Scatter(
                x=y_positions,
                y=cutoffs_vs_y,
                mode="lines+markers",
                line=dict(color="rgba(100, 0, 200, 0.6)", width=2),
                marker=dict(color="purple", size=6, line=dict(color="white", width=1)),
                name="Cutoff vs Y Position",
            )
        )

        # Add marker for current position
        # Compute cutoff for current special_y position
        current_cutoff = np.interp(special_y, y_positions, cutoffs_vs_y)

        fig_cutoff_pos.add_trace(
            go.Scatter(
                x=[special_y],
                y=[current_cutoff],
                mode="markers",
                marker=dict(
                    size=12,
                    color="gold",
                    symbol="star",
                    line=dict(color="orange", width=2),
                ),
                name="Current Position",
                hovertemplate="<b>Current Position</b><br>Y: %{x:.2f} Å<br>Cutoff: %{y:.3f} Å<extra></extra>",
            )
        )

        fig_cutoff_pos.update_layout(
            title="Special Atom Cutoff vs Y Position",
            xaxis_title="Y Position (Å)",
            yaxis_title="Adaptive Cutoff (Å)",
            showlegend=True,
            legend=dict(
                x=0.02,
                y=0.98,
                xanchor="left",
                yanchor="bottom",
                bgcolor="rgba(255, 255, 255, 0.8)",
            ),
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

    return fig_atoms, fig_nef, fig_cutoff_pos, info_text


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
