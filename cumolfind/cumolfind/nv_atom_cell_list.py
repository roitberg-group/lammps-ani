# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
# type: ignore
import os

import nvtx
import torch
import warp as wp

from .nv_atomic_data import AtomicData
from .nv_cell_utils import (
    _cell_bin_atoms_pbc,
    _cell_build_neighbor_list,
    _cell_compute_cell_offsets,
    _cell_construct_bins,
    _cell_count_atoms_per_bin,
    _cell_query_neighbor_list,
)

# Initialize Warp explicitly
wp.init()

def _cell_neighbor_list(
    data: AtomicData,
    cutoff: float,
    max_nbins: int = 1000000,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the neighbor list for a single AtomicData environment

    This neighborlist can be run on either the CPU or the GPU, depending on the device
    of the Batch object.

    Parameters
    ----------
    data : AtomicData
        AtomicData representing a given atomic environment.
    cutoff : float
        Cutoff radius for the system.
    i : torch.Tensor | None, optional
        Index of the base atom
    j : torch.Tensor | None, optional
        Index of the neighboring atom
    u : torch.Tensor | None, optional
        Unit cell shift of the neighboring atom
    S : torch.Tensor | None, optional
        Shift of the neighboring atom
    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Returns `i`, `j`, `S` - the index of the base
        atom, the index of the neighboring atom,
        and the cell shifts between them.
    """

    # Get info from batch
    pos = data.positions
    pbc = data.pbc
    cell = data.cell.reshape(1, 3, 3).contiguous()

    # Get device and dtype
    device = str(pos.device)
    dtype = pos.dtype
    num_atoms = pos.shape[0]

    # Determine appropriate warp types
    if dtype == torch.float32:
        wp_dtype = wp.float32
        wp_vec_dtype = wp.vec3f
        wp_mat_dtype = wp.mat33f
    elif dtype == torch.float64:
        wp_dtype = wp.float64
        wp_vec_dtype = wp.vec3d
        wp_mat_dtype = wp.mat33d
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    # Create arrays for cell list
    with nvtx.annotate(message="creating warp arrays", color="blue"):
        wp_pos = wp.from_torch(pos, dtype=wp_vec_dtype)
        wp_cell = wp.from_torch(cell, dtype=wp_mat_dtype)
        wp_pbc = wp.from_torch(pbc.flatten(), dtype=wp.bool)

    # Compute cell list dimensions
    with nvtx.annotate(message="computing cell list dimensions", color="green"):
        cell_counts = wp.zeros((3,), dtype=wp.int32, device=device)
        neigh_search = wp.zeros((3,), dtype=wp.int32, device=device)
        wp.launch(
            _cell_construct_bins,
            dim=1,
            inputs=[
                wp_pos,
                wp_cell,
                wp_pbc,
                cell_counts,
                neigh_search,
                wp_dtype(cutoff),
                max_nbins,
            ],
            device=device,
        )

    # Create arrays for atom binning
    with nvtx.annotate(message="creating atom binning arrays", color="blue"):
        total_cells = int(torch.prod(wp.to_torch(cell_counts)))
        cell_atom_counts = wp.zeros((total_cells,), dtype=wp.int32, device=device)
        cell_shifts = wp.zeros((num_atoms,), dtype=wp.vec3i, device=device)
        wp.launch(
            _cell_count_atoms_per_bin,
            dim=num_atoms,
            inputs=[
                wp_pos,
                wp_cell,
                wp_pbc,
                cell_counts,
                cell_atom_counts,
                cell_shifts,
            ],
            device=device,
        )

    # Compute cell offsets
    with nvtx.annotate(message="computing cell offsets", color="green"):
        cell_offsets = wp.zeros((total_cells,), dtype=wp.int32, device=device)
        wp.launch(
            _cell_compute_cell_offsets,
            dim=total_cells,
            inputs=[cell_atom_counts, cell_offsets, total_cells],
            device=device,
        )

    # Allocate cell atom indices
    with nvtx.annotate(message="allocating cell atom indices", color="blue"):
        total_slots = int(
            wp.to_torch(cell_offsets)[-1] + wp.to_torch(cell_atom_counts)[-1]
        )
        cell_atom_indices = wp.zeros((total_slots,), dtype=wp.int32, device=device)

    # Reset cell atom counts
    cell_atom_counts.zero_()

    # Bin atoms into cells
    with nvtx.annotate(message="binning atoms into cells", color="green"):
        atom_cell_indices = wp.zeros((num_atoms,), dtype=wp.vec3i, device=device)
        wp.launch(
            _cell_bin_atoms_pbc,
            dim=num_atoms,
            inputs=[
                wp_pos,
                wp_cell,
                wp_pbc,
                cell_counts,
                atom_cell_indices,
                cell_atom_counts,
                cell_offsets,
                cell_atom_indices,
            ],
            device=device,
        )

    # Count neighbors
    with nvtx.annotate(message="counting neighbors", color="green"):
        wp_nneigh = wp.zeros((num_atoms,), dtype=wp.int32, device=device)
        wp.launch(
            _cell_query_neighbor_list,
            dim=num_atoms,
            inputs=[
                wp_pos,
                wp_cell,
                wp_pbc,
                wp_dtype(cutoff),
                cell_counts,
                cell_shifts,
                neigh_search,
                atom_cell_indices,
                cell_atom_counts,
                cell_offsets,
                cell_atom_indices,
                wp_nneigh,
            ],
            device=device,
        )

    # Set up output tensors
    with nvtx.annotate(message="creating ijS buffers", color="blue"):
        ## Need to build the i, j, S buffers
        offset = torch.zeros((num_atoms + 1,), device=device, dtype=torch.int32)
        nneigh = wp.to_torch(wp_nneigh)

        torch.cumsum(nneigh, dim=0, out=offset[1:])
        total_count = offset[-1].item()
        wp_offset = wp.from_torch(offset)

        i = torch.empty((total_count,), device=device, dtype=torch.int32)
        wp_i = wp.from_torch(i, dtype=wp.int32)

        j = torch.empty((total_count,), device=device, dtype=torch.int32)
        wp_j = wp.from_torch(j, dtype=wp.int32)

        dist = torch.empty((total_count,), device=device, dtype=torch.float32)
        wp_dist = wp.from_torch(dist, dtype=wp.float32)

        # u = torch.empty((total_count, 3), device=device, dtype=torch.int32)
        # wp_u = wp.from_torch(u, dtype=wp.vec3i)

        # S = torch.empty((total_count, 3), device=device, dtype=dtype)
        # wp_S = wp.from_torch(S, dtype=wp_vec_dtype)

        if os.environ.get("ALCHEMI_PROFILE_NVTX", False):
            wp.synchronize()

    with nvtx.annotate(message="building neighborlist", color="green"):
        wp.launch(
            _cell_build_neighbor_list,
            dim=num_atoms,
            inputs=[
                wp_pos,
                wp_cell,
                wp_pbc,
                wp_dtype(cutoff),
                cell_counts,
                cell_shifts,
                neigh_search,
                atom_cell_indices,
                cell_atom_counts,
                cell_offsets,
                cell_atom_indices,
                wp_offset,
                wp_i,
                wp_j,
                wp_dist,
                # wp_u,
                # wp_S,
            ],
            device=device,
        )
        if os.environ.get("ALCHEMI_PROFILE_NVTX", False):
            wp.synchronize()

    return i, j, dist
