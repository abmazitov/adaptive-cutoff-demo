import ase
import numpy as np
import torch
from typing import List, Optional, Union
from metatomic.torch import systems_to_torch, NeighborListOptions
from metatensor.torch import Labels
from metatrain.pet.modules.structures import (
    concatenate_structures,
    get_effective_num_neighbors,
    get_exponential_cutoff_weights,
    get_gaussian_cutoff_weights,
)
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists


def compute_adaptive_cutoff(
    atoms: ase.Atoms,
    options: NeighborListOptions,
    weight_function: str = "gaussian",
    max_num_neighbors: float = 2.0,
    width: float = 0.5,
    beta: float = 1.0,
):
    """Compute the adaptive cutoff for the central atom."""
    positions, centers, edge_distances, system_indices, num_nodes = atoms_to_tensors(
        atoms, options
    )

    probe_cutoffs = torch.arange(
        0.5,
        options.cutoff - 0.9,
        0.1,
        device=edge_distances.device,
        dtype=edge_distances.dtype,
    )

    effective_num_neighbors = get_effective_num_neighbors(
        edge_distances,
        probe_cutoffs,
        centers,
        num_nodes,
    )

    if weight_function == "gaussian":
        cutoffs_weights = get_gaussian_cutoff_weights(
            effective_num_neighbors,
            probe_cutoffs,
            max_num_neighbors,
            num_nodes,
            width=width,
        )
    else:  # exponential
        cutoffs_weights = get_exponential_cutoff_weights(
            effective_num_neighbors,
            probe_cutoffs,
            max_num_neighbors,
            width=width,
            beta=beta,
        )

    adapted_cutoffs = probe_cutoffs @ cutoffs_weights.T

    # Get cutoff for central atom (index 0)
    central_atom_cutoff = adapted_cutoffs[0].item()

    return (
        central_atom_cutoff,
        effective_num_neighbors[0].cpu().numpy(),
        probe_cutoffs.cpu().numpy(),
    )


def create_atom_configuration(num_atoms: int, special_atom_y: float, seed: int = 42):
    """Create atomic configuration with random positions on a plane."""
    np.random.seed(seed)

    positions = []
    # Central atom at origin
    positions.append([0.0, 0.0, 0.0])

    # Random atoms on the plane (excluding the special atom)
    for _ in range(num_atoms - 1):
        x = np.random.uniform(-5, 5)
        y = np.random.uniform(-5, 5)
        positions.append([x, y, 0.0])

    # Special atom at [0, special_atom_y, 0]
    positions.append([0.0, special_atom_y, 0.0])

    atoms = ase.Atoms("C" * (num_atoms + 1), positions=positions)
    return atoms


def atoms_to_tensors(
    atoms: Union[ase.Atoms, List[ase.Atoms]],
    options: NeighborListOptions,
    selected_atoms: Optional[Labels] = None,
):
    """Convert ASE Atoms to torch tensors with neighbor lists."""
    if isinstance(atoms, ase.Atoms):
        atoms = [atoms]
    systems = systems_to_torch(atoms)
    systems = [get_system_with_neighbor_lists(system, [options]) for system in systems]
    (
        positions,
        centers,
        neighbors,
        species,
        cells,
        cell_shifts,
        system_indices,
        sample_labels,
    ) = concatenate_structures(systems, options, selected_atoms)
    if len(cells) == 1:
        cell_contributions = cell_shifts.to(cells.dtype) @ cells[0]
    else:
        cell_contributions = torch.einsum(
            "ab, abc -> ac",
            cell_shifts.to(cells.dtype),
            cells[system_indices[centers]],
        )
    edge_vectors = positions[neighbors] - positions[centers] + cell_contributions
    edge_distances = torch.norm(edge_vectors, dim=-1) + 1e-15

    if selected_atoms is not None:
        num_nodes = int(centers.max()) + 1
    else:
        num_nodes = len(positions)
    return positions, centers, edge_distances, system_indices, num_nodes
