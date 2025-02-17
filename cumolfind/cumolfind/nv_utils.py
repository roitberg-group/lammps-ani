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
import torch


@torch.jit.script
def _estimate_max_neighbors(
    cell: torch.Tensor,
    cutoff: torch.Tensor,
    atoms_per_system: torch.Tensor,
    multiplicative_factor: float = 1.2,
) -> torch.Tensor:
    """Estimate maximum possible neighbors based on cutoff and typical atomic density"""
    volume_per_atom = cell.view(-1, 3, 3).det() / atoms_per_system
    cutoff_volume = (4 / 3) * torch.pi * cutoff.pow(3)
    return (multiplicative_factor * cutoff_volume / volume_per_atom).to(torch.int32)


@torch.jit.script
def _compute_num_shifts(
    cell: torch.Tensor, cutoff: torch.Tensor, pbc: torch.Tensor
) -> torch.Tensor:  # pragma: no cover
    rcell = cell.inverse().transpose(-2, -1)
    inv_d = rcell.norm(2, -1)
    num_shifts = torch.ceil(cutoff[:, None] * inv_d).to(torch.int32)
    num_shifts *= pbc.to(torch.int32)
    return num_shifts


@torch.jit.script
def _cumsum(
    nneigh: torch.Tensor,
    offset: torch.Tensor,
) -> int:  # pragma: no cover
    """Computes cumulative sum of an integer array."""
    torch.cumsum(nneigh, dim=0, out=offset[1:])
    return offset[-1].item()


@torch.jit.script
def _split_tensors(
    offset: torch.Tensor,
    atom_ptr: torch.Tensor,
    i: torch.Tensor,
    j: torch.Tensor,
    u: torch.Tensor,
    S: torch.Tensor,
):  # pragma: no cover

    split_sizes = [
        int(offset[a_end] - offset[a_0])
        for a_0, a_end in zip(atom_ptr[:-1], atom_ptr[1:])
    ]

    i, j, u, S = (
        torch.split(i, split_sizes),
        torch.split(j, split_sizes),
        torch.split(u, split_sizes),
        torch.split(S, split_sizes),
    )
    return i, j, u, S

