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
import time as timetime

from .nv_batch import Batch
from .nv_batch_nl_utils import (
    _build_neighbor_list,
    _count_shifts,
)
from .nv_utils import (
    _compute_num_shifts,
    _cumsum,
    _split_tensors,
)

wp.init()

def batched_neighbor_list(
    batch: Batch,
    cutoff: torch.Tensor,
    offset: torch.Tensor | None = None,
    i: torch.Tensor | None = None,
    j: torch.Tensor | None = None,
    u: torch.Tensor | None = None,
    S: torch.Tensor | None = None,
    dist: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the neighbor list for a `Batch` (a batch of atomic systems)

    This neighborlist can be run on either the CPU or the GPU, depending on the device
    of the Batch object.

    Parameters
    ----------
    batch : Batch
        batch of atomic environments to compute the neighborlist for.
    cutoff : torch.Tensor
        Cutoff radius for each system.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Returns `i`, `j`, `S` - the index of the base
        atom, the index of the neighboring atom,
        and the cell shifts between them.
    """
    # Get info from batch
    pos = batch["positions"]
    pbc = batch["pbc"]
    cell = batch["cell"]

    # Get the dtype
    dtype = pos.dtype
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

    atom_ptr = batch.ptr
    num_systems = pbc.shape[0]
    total_num_atoms = pos.shape[0]
    device = str(pos.device)
    num_atoms = atom_ptr[1:] - atom_ptr[:-1]

    max_atom_per_system = num_atoms.max().item()

    with nvtx.annotate(message="getting num shifts", color="green"):
        cell = cell.reshape(num_systems, 3, 3)
        num_shifts = _compute_num_shifts(cell, cutoff, pbc)
        if os.environ.get("ALCHEMI_PROFILE_NVTX", False):
            torch.cuda.synchronize()

    # Prepare Warp arrays
    with nvtx.annotate(message="preparing warp arrays", color="blue"):
        wp_pos = wp.from_torch(pos, dtype=wp_vec_dtype)
        wp_cell = wp.from_torch(cell.reshape(num_systems, 3, 3), dtype=wp_mat_dtype)
        wp_atom_ptr = wp.from_torch(atom_ptr.to(torch.int32))
        wp_cutoff = wp.from_torch(cutoff, dtype=wp_dtype)
        wp_num_shifts = wp.from_torch(num_shifts, dtype=wp.vec3i)
        wp_nneigh = wp.zeros((total_num_atoms,), device=device, dtype=wp.int32)
        if os.environ.get("ALCHEMI_PROFILE_NVTX", False):
            wp.synchronize()

    with nvtx.annotate(message="querying neighborhoods", color="green"):
        dbg_cutoff = 0.0
        wp.launch(
            _count_shifts,
            dim=(num_systems, max_atom_per_system),
            inputs=[
                wp_pos,
                wp_cell,
                wp_num_shifts,
                wp_atom_ptr,
                wp_cutoff,
                wp_nneigh,
                dbg_cutoff,
            ],
            device=device,
        )
        print(f"dbg_cutoff: {dbg_cutoff}")
        if os.environ.get("ALCHEMI_PROFILE_NVTX", False):
            wp.synchronize()

    # Set up output tensors
    with nvtx.annotate(message="creating offset", color="blue"):
        ## Need to build the i, j, S buffers
        nneigh = wp.to_torch(wp_nneigh)
        if offset is None:
            offset = torch.zeros(
                (total_num_atoms + 1,), device=device, dtype=torch.int32
            )
        if os.environ.get("ALCHEMI_PROFILE_NVTX", False):
            torch.cuda.synchronize()

    with nvtx.annotate(message="cumsumming", color="blue"):
        total_count = _cumsum(nneigh, offset)
        if os.environ.get("ALCHEMI_PROFILE_NVTX", False):
            torch.cuda.synchronize()
    with nvtx.annotate(message="creating ijS buffers", color="blue"):
        wp_offset = wp.from_torch(offset)
        if i is None:
            wp_i = wp.zeros((total_count,), dtype=wp.int32, device=device)
        else:
            wp_i = wp.from_torch(i[:total_count], dtype=wp.int32)
        if j is None:
            wp_j = wp.zeros((total_count,), dtype=wp.int32, device=device)
        else:
            wp_j = wp.from_torch(j[:total_count], dtype=wp.int32)
        if u is None:
            wp_u = wp.zeros((total_count,), dtype=wp.vec3i, device=device)
        else:
            wp_u = wp.from_torch(u[:total_count], dtype=wp.vec3i)
        if S is None:
            wp_S = wp.zeros((total_count,), dtype=wp_vec_dtype, device=device)
        else:
            wp_S = wp.from_torch(S[:total_count], dtype=wp_vec_dtype)
        if dist is None:
            print("allocating wp_dist")
            wp_dist = wp.zeros((total_count,), dtype=wp.float32, device=device)
        else:
            wp_dist = wp.from_torch(i[:total_count], dtype=wp.float32)
        if os.environ.get("ALCHEMI_PROFILE_NVTX", False):
            wp.synchronize()

    with nvtx.annotate(message="building neighborlist", color="green"):
        wp.launch(
            _build_neighbor_list,
            dim=(num_systems, max_atom_per_system),
            inputs=[
                wp_pos,
                wp_cell,
                wp_num_shifts,
                wp_atom_ptr,
                wp_cutoff,
                wp_offset,
                wp_i,
                wp_j,
                wp_u,
                wp_S,
                wp_dist,
            ],
            device=device,
        )
        if os.environ.get("ALCHEMI_PROFILE_NVTX", False):
            wp.synchronize()


    if i is None:
        with nvtx.annotate(message="converting ijS to pytorch", color="blue"):
            # Now safely convert to PyTorch
            i, j, u, S, dist = (
                wp.to_torch(wp_i),
                wp.to_torch(wp_j),
                wp.to_torch(wp_u),
                wp.to_torch(wp_S),
                wp.to_torch(wp_dist),
            )

            if os.environ.get("ALCHEMI_PROFILE_NVTX", False):
                torch.cuda.synchronize()

        with nvtx.annotate(message="splitting tensors", color="green"):
            i, j, u, S, dist = _split_tensors(offset, atom_ptr, i, j, u, S, dist)

            if os.environ.get("ALCHEMI_PROFILE_NVTX", False):
                torch.cuda.synchronize()

    return i, j, u, S, dist
