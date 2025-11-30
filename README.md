# Adaptive Cutoff Demo

An interactive dashboard for visualizing adaptive cutoff behavior in atomic systems using Plotly Dash.

## Overview

This dashboard provides an interactive visualization of adaptive cutoff mechanisms for molecular dynamics simulations. It allows users to:

- Adjust the number of atoms in a configuration
- Control the position of special atoms
- Choose between Gaussian and Exponential weight functions
- Tune parameters like max neighbors, width, and beta
- Visualize how adaptive cutoffs change based on atomic configurations

## Installation

### From Source

1. Clone this repository:
```bash
git clone https://github.com/abmazitov/adaptive-cutoff-demo.git
cd adaptive-cutoff-demo
```

2. Install the package and dependencies:
```bash
pip install -e .
```

### Requirements

This package depends on the `adaptive-cutoff` branch of the `metatrain` package from GitHub. The installation is handled automatically. 

## Usage

### Running Locally

After installation, you can run the dashboard using:

```bash
adaptive-cutoff-dashboard
```

Or directly with Python:

```bash
python -m adaptive_cutoff_demo.dashboard
```

The dashboard will be available at `http://localhost:8050` by default.

### Running with Custom Port

Set the PORT environment variable:

```bash
PORT=8080 adaptive-cutoff-dashboard
```

## Deployment

### Deploying to Render

1. Push this repository to GitHub
2. Go to [Render](https://render.com) and sign in
3. Create a new Web Service
4. Connect your GitHub repository
5. Use the following settings:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python -m adaptive_cutoff_demo.dashboard`
6. Click "Create Web Service"

Render will automatically deploy your dashboard and provide a public URL.

### Deploying to Railway

1. Push this repository to GitHub
2. Go to [Railway](https://railway.app) and sign in
3. Create a new project from your GitHub repository
4. Railway will auto-detect the Python app and deploy it

### Environment Variables

For deployment, you can set:
- `PORT`: The port to run the server on (default: 8050)
- `HOST`: The host to bind to (default: 0.0.0.0)

## Features

- **Interactive Controls**: Adjust parameters in real-time
- **Multiple Weight Functions**: Choose between Gaussian and Exponential approaches
- **Atomic Visualization**: See the 2D configuration of atoms and adaptive cutoff radius
- **Effective Neighbors Plot**: Visualize how the number of effective neighbors changes with cutoff distance

## Development

### Project Structure

```
adaptive-cutoff-demo/
├── adaptive_cutoff_demo/
│   ├── __init__.py
│   └── dashboard.py
├── pyproject.toml
└── README.md
```

### Dependencies

- ase: Atomic Simulation Environment
- dash: Web application framework
- dash-bootstrap-components: Bootstrap components for Dash
- numpy: Numerical computing
- plotly: Interactive plotting
- torch: PyTorch for tensor operations
- metatrain: Main package (from GitHub adaptive-cutoff branch)

## License

This project is part of the metatrain package ecosystem.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.
