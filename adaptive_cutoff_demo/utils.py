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
    get_gaussian_cutoff_weights,
)
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists


def get_gaussian_cutoff_weights_smooth(
    effective_num_neighbors: torch.Tensor,
    max_num_neighbors: float,
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
    max_num_neighbors_t = torch.as_tensor(max_num_neighbors, device=effective_num_neighbors.device)

    diff = effective_num_neighbors - max_num_neighbors_t
    
    # pretend that the last cutoff has the exact number of neighbors, if it has fewer
    diff[:,-1][diff[:,-1]<0] = torch.max(diff[:,-1][diff[:,-1]<0], torch.zeros_like(diff[:,-1][diff[:,-1]<0]))
    weights = torch.exp(-0.5 * (diff / width) ** 2)
    
    # weights = 1/(1 + (diff / width) ** 2) 

    # row-wise normalization, with small epsilon to avoid division by zero
    weights_sum = weights.sum(dim=1, keepdim=True)
    weights = weights / weights_sum

    return weights

def cosine_cutoff(grid: torch.Tensor, r_cut: torch.Tensor, delta: float) -> torch.Tensor:
    """
    Cosine cutoff function.

    :param grid: Distances at which to evaluate the cutoff function.
    :param r_cut: Cutoff radius for each node.
    :param delta: Width of the cutoff region.
    :return: Values of the cutoff function at the specified distances.
    """
    mask_bigger = grid >= r_cut
    mask_smaller = grid <= r_cut - delta
    grid = (grid - r_cut + delta) / delta
    f = 0.5 + 0.5 * torch.cos(torch.pi * grid)

    f[mask_bigger] = 0.0
    f[mask_smaller] = 1.0
    return f

def smooth_cutoff(
    values: torch.Tensor, cutoff: torch.Tensor, 
    width: float=0.0
) -> torch.Tensor:
    """Compute the smooth delta function values.
    :param values: Input values (torch.Tensor).
    :param center: Center value (torch.Tensor).
    :param width: Width parameter (float).

    :return: Smooth delta function values (torch.Tensor).
    """
    x = torch.min(values / cutoff, torch.ones_like(values))
    x2 = x*x
    return torch.exp(-x2/(1-x2))

def bump_cutoff(values: torch.Tensor, cutoff: torch.Tensor, width: float) -> torch.Tensor:
    """
    Cosine cutoff function.

    :param grid: Distances at which to evaluate the cutoff function.
    :param r_cut: Cutoff radius for each node.
    :param delta: Width of the cutoff region.
    :return: Values of the cutoff function at the specified distances.
    """
    mask_bigger = values >= cutoff
    mask_smaller = values <= cutoff-width
    scaled_values = (values - (cutoff-width )) / width
    
    f = 0.5 *(1+torch.tanh(1/torch.tan(torch.pi*scaled_values)))
    #print("bump", values.shape, cutoff.shape, scaled_values.shape)
    #print("cutoff", cutoff, values, f[0], f[:,0])

    f[mask_bigger] = 0.0
    f[mask_smaller] = 1.0
    return f


def get_effective_num_neighbors_smooth(
    edge_distances: torch.Tensor,
    probe_cutoffs: torch.Tensor,
    centers: torch.Tensor,
    num_nodes: int,
    width: Optional[float] = None,
) -> torch.Tensor:
    """
    Computes the effective number of neighbors for each probe cutoff.

    :param edge_distances: Distances between centers and their neighbors.
    :param probe_cutoffs: Probe cutoff distances.
    :param centers: Indices of the center atoms.
    :param num_nodes: Total number of center atoms.
    :param width: Width of the cutoff function. If None, it will be
        automatically determined from the probe cutoff spacing.
    :return: Effective number of neighbors for each center atom and probe cutoff.
    """
    if width is None:
        # Automatically determine width from probe cutoff spacing
        # Use 2.5x the spacing for a smooth step function
        if len(probe_cutoffs) > 1:
            probe_spacing = probe_cutoffs[1] - probe_cutoffs[0]
            width = 2 * probe_spacing
        else:
            width = 0.5  # fallback for single probe cutoff    

    weights = bump_cutoff(
        edge_distances.unsqueeze(0), probe_cutoffs.unsqueeze(1),
        width
    )
    
    probe_num_neighbors = torch.zeros(
        (len(probe_cutoffs), num_nodes),
        dtype=edge_distances.dtype,
        device=edge_distances.device,
    )
    # Vectorized version: use scatter_add_ to accumulate weights for all probe
    # cutoffs at once
    centers_expanded = (
        centers.unsqueeze(0).expand(len(probe_cutoffs), -1).to(torch.int64)
    )
    probe_num_neighbors.scatter_add_(1, centers_expanded, weights)
    probe_num_neighbors = probe_num_neighbors.T.contiguous()
    return probe_num_neighbors  # / 0.286241 # normalization factor to account for the form of the cutoff function

def compute_adaptive_cutoff(
    atoms: ase.Atoms,
    options: NeighborListOptions,
    weight_function: str = "gaussian",
    max_num_neighbors: float = 2.0,
    cutoff_width: float = 0.5,
    width: float = 0.5,
    beta: float = 1.0,
    step_size: float = 0.1,
    atom_index: int = 0,
    return_all_cutoffs: bool = False,
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

    effective_num_neighbors = get_effective_num_neighbors_smooth(
        edge_distances,
        probe_cutoffs,
        centers,
        num_nodes,
        width=cutoff_width,
    )

    if weight_function == "gaussian":
        cutoffs_weights = get_gaussian_cutoff_weights_smooth(
            effective_num_neighbors,
            max_num_neighbors,
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
        return (
            adapted_cutoffs.cpu().numpy(),
            effective_num_neighbors[atom_index].cpu().numpy(),
            probe_cutoffs.cpu().numpy(),
            cutoffs_weights.cpu().numpy()
        )
    else:
        # Get cutoff for specified atom
        atom_cutoff = adapted_cutoffs[atom_index].item()
        return (
            atom_cutoff,
            effective_num_neighbors[atom_index].cpu().numpy(),
            probe_cutoffs.cpu().numpy(),
            cutoffs_weights.cpu().numpy()
        )


def compute_special_atom_cutoffs_vs_position(
    num_atoms: int,
    y_positions: np.ndarray,
    seed: int,
    options: NeighborListOptions,
    special_atom_idx: int = 0,
    weight_function: str = "gaussian",
    max_num_neighbors: float = 2.0,
    cutoff_width: float = 0.5,
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
        cutoff, _, _, _ = compute_adaptive_cutoff(
            atoms,
            options,
            weight_function=weight_function,
            max_num_neighbors=max_num_neighbors,
            cutoff_width=cutoff_width,
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
