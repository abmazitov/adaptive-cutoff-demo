import ase
import numpy as np
import torch
from typing import List, Optional, Union
from metatomic.torch import systems_to_torch, NeighborListOptions
from metatensor.torch import Labels
from metatrain.pet.modules.structures import concatenate_structures
from metatrain.pet.modules.adaptive_cutoff import (
    get_effective_num_neighbors,
    get_exponential_cutoff_weights,
)
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists


def get_gaussian_cutoff_weights(
    effective_num_neighbors: torch.Tensor,
    probe_cutoffs: torch.Tensor,
    max_num_neighbors: float,
    num_nodes: int,
    width: float = 0.5,
) -> torch.Tensor:
    """
    Computes the weights for each probe cutoff based on
    the effective number of neighbors using Gaussian weights
    centered at the expected number of neighbors.

    :param effective_num_neighbors: Effective number of neighbors for each center atom
        and probe cutoff.
    :param probe_cutoffs: Probe cutoff distances.
    :param max_num_neighbors: Target maximum number of neighbors per atom.
    :param num_nodes: Total number of center atoms.
    :param width: Width of the Gaussian function.
    :return: Weights for each probe cutoff.
    """
    max_num_neighbors_t = torch.as_tensor(
        max_num_neighbors, device=effective_num_neighbors.device
    )

    diff = effective_num_neighbors - max_num_neighbors_t
    diff[:, -1] = torch.clamp(
        diff[:, -1], min=0.0
    )  # ensure last column is non-negative
    weights = torch.exp(-0.5 * (diff / width) ** 2)

    # row-wise normalization, with small epsilon to avoid division by zero
    weights_sum = weights.sum(dim=1, keepdim=True)
    weights = weights / (weights_sum + 1e-12)

    return weights


def compute_adaptive_cutoff(
    atoms: ase.Atoms,
    options: NeighborListOptions,
    weight_function: str = "gaussian",
    max_num_neighbors: float = 2.0,
    width: float = 0.5,
    beta: float = 1.0,
    step_size: float = 0.1,
    atom_index: int = 0,
    return_all_cutoffs: bool = False,
    return_weights: bool = False,
):
    """Compute the adaptive cutoff for a specific atom or all atoms.

    Args:
        atoms: Atomic structure
        options: Neighbor list options
        weight_function: Type of weight function ('gaussian' or 'exponential')
        max_num_neighbors: Maximum number of neighbors
        width: Width parameter for weight function
        beta: Beta parameter for exponential weight function
        step_size: Step size for probe cutoff grid
        atom_index: Index of atom to compute cutoff for (default: 0 for central atom)
        return_all_cutoffs: If True, return cutoffs for all atoms instead of just one
        return_weights: If True, also return the cutoff weights
    """
    positions, centers, edge_distances, system_indices, num_nodes = atoms_to_tensors(
        atoms, options
    )

    probe_cutoffs = torch.arange(
        0.5,
        options.cutoff - 0.9,
        step_size,
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

    if return_all_cutoffs:
        # Return cutoffs for all atoms plus additional data for specified atom
        result = (
            adapted_cutoffs.cpu().numpy(),
            effective_num_neighbors[atom_index].cpu().numpy(),
            probe_cutoffs.cpu().numpy(),
        )
        if return_weights:
            result = result + (cutoffs_weights[atom_index].cpu().numpy(),)
        return result
    else:
        # Get cutoff for specified atom
        atom_cutoff = adapted_cutoffs[atom_index].item()
        result = (
            atom_cutoff,
            effective_num_neighbors[atom_index].cpu().numpy(),
            probe_cutoffs.cpu().numpy(),
        )
        if return_weights:
            result = result + (cutoffs_weights[atom_index].cpu().numpy(),)
        return result


def compute_special_atom_cutoffs_vs_position(
    num_atoms: int,
    y_positions: np.ndarray,
    seed: int,
    options: NeighborListOptions,
    special_atom_idx: int = 0,
    weight_function: str = "gaussian",
    max_num_neighbors: float = 2.0,
    width: float = 0.5,
    beta: float = 1.0,
    step_size: float = 0.1,
):
    """Compute adaptive cutoff for special atom at different y positions.

    Args:
        num_atoms: Number of random atoms
        y_positions: Array of y positions to compute cutoff for
        seed: Random seed for atom positions
        options: Neighbor list options
        weight_function: Type of weight function
        max_num_neighbors: Maximum number of neighbors
        width: Width parameter
        beta: Beta parameter for exponential
        step_size: Step size for probe cutoff grid

    Returns:
        Array of cutoff values corresponding to each y position
    """
    cutoffs = []
    for y_pos in y_positions:
        atoms = create_atom_configuration(num_atoms, y_pos, seed=seed)
        # Special atom is at index -1 (last atom)
        cutoff, _, _ = compute_adaptive_cutoff(
            atoms,
            options,
            weight_function=weight_function,
            max_num_neighbors=max_num_neighbors,
            width=width,
            beta=beta,
            step_size=step_size,
            atom_index=special_atom_idx,
        )
        cutoffs.append(cutoff)
    return np.array(cutoffs)


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
