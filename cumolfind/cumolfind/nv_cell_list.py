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

from .nv_batch import Batch
from .nv_cell_list_utils import (
    # NLBin,
    # SortedAtoms,
    _build_neighbor_list,
    _construct_bins,
    _prepare_bins,
    _query_neighbor_list,
    _sort_atoms_into_bins,
)
from .nv_utils import _estimate_max_neighbors

wp.init()


class CellList:
    """A cell list implementation for efficient neighbor list computation.

    This class partitions 3D space into cells and sorts atoms into these cells
    to efficiently find neighboring atoms within a cutoff distance.

    Parameters
    ----------
    batch : Batch
        Batch of atomic systems
    cutoff : float
        Cutoff radius for neighbor search
    max_bins : int, optional
        Maximum number of bins allowed, by default 1_000_000
    """

    def __init__(self, batch: Batch, cutoff: torch.Tensor, max_nbins: int = 1_000_000):
        self.batch = batch

        # Get the dtype
        self.dtype = batch.positions.dtype
        if self.dtype == torch.float32:
            self.wp_dtype = wp.float32
            self.wp_vec_dtype = wp.vec3f
            self.wp_mat_dtype = wp.mat33f
        elif self.dtype == torch.float64:
            self.wp_dtype = wp.float64
            self.wp_vec_dtype = wp.vec3d
            self.wp_mat_dtype = wp.mat33d

        self.cutoff = cutoff.to(self.dtype)

        self.max_nbins = int(max_nbins)
        self.device = str(batch.positions.device)
        self.num_systems = self.batch.num_graphs
        self.num_atoms = self.batch.num_nodes
        self.atoms_per_system = self.batch.ptr[1:] - self.batch.ptr[:-1]
        self.max_atoms_per_system = self.atoms_per_system.max()

        # Pre-allocate buffers for the maximum possible neighbors
        self.max_possible_neighbor = (
            (self.batch.ptr[1:] - self.batch.ptr[:-1])
            * _estimate_max_neighbors(
                self.batch["cell"], self.cutoff, self.atoms_per_system
            )
        ).sum()
        self.result_count = torch.zeros(
            self.num_atoms, dtype=torch.int32, device=self.device
        )

        self._initialize()

    def _initialize(
        self,
    ):
        """Initialize the cell list."""
        self.nbins = torch.zeros(
            self.num_systems, dtype=torch.int32, device=self.device
        )
        self.nbins_xyz = torch.zeros(
            (self.num_systems, 3), dtype=torch.int32, device=self.device
        )
        self.bin_size = torch.zeros(
            self.num_systems, dtype=self.dtype, device=self.device
        )
        self.bin_index = torch.zeros(
            self.num_atoms, dtype=torch.int32, device=self.device
        )
        self.bin_index_xyz = torch.zeros(
            (self.num_atoms, 3), dtype=torch.int32, device=self.device
        )
        self.neigh_search = torch.zeros(
            (self.num_systems, 3), dtype=torch.int32, device=self.device
        )
        self.max_natoms_per_bin = 0
        self.atom_list = torch.zeros(
            self.num_atoms, dtype=torch.int32, device=self.device
        )
        self.cell_shift = torch.zeros(
            (self.num_atoms, 3), dtype=torch.int32, device=self.device
        )

        with nvtx.annotate(message="launching construct bins", color="green"):
            # Convert to wp tensors
            wp_nbins = wp.from_torch(self.nbins, dtype=wp.int32)
            wp_nbins_xyz = wp.from_torch(self.nbins_xyz, dtype=wp.vec3i)
            wp_bin_size = wp.from_torch(self.bin_size, dtype=self.wp_dtype)
            wp_cell = wp.from_torch(
                self.batch["cell"].reshape(-1, 3, 3), dtype=self.wp_mat_dtype
            )
            wp_cutoff = wp.from_torch(self.cutoff, dtype=self.wp_dtype)
            wp.launch(
                _construct_bins,
                dim=self.num_systems,
                inputs=[
                    wp_nbins,
                    wp_nbins_xyz,
                    wp_bin_size,
                    wp_cell,
                    wp_cutoff,
                    self.max_nbins,
                ],
                device=self.device,
            )

        # Estimate the maximum number of atoms per bin
        self.max_natoms_per_bin_per_system = torch.ceil(
            1.5 * self.atoms_per_system / self.nbins
        ).to(torch.int32)
        self.max_natoms_per_bin = self.max_natoms_per_bin_per_system.max().item()

        self.bin_ptr = torch.zeros(
            self.num_systems + 1, dtype=torch.int32, pin_memory=True
        ).to(self.device)
        torch.cumsum(self.nbins, dim=0, out=self.bin_ptr[1:])
        self.atoms_in_bins = -1 * torch.ones(
            (self.bin_ptr[-1], self.max_natoms_per_bin),
            dtype=torch.int32,
            pin_memory=True,
        ).to(self.device)

    def _prepare_bins(
        self,
    ):
        """Set up the cell list."""

        with nvtx.annotate(message="launching prepare bins", color="green"):
            # Convert to wp tensors
            wp_nbins = wp.from_torch(self.nbins, dtype=wp.int32)
            wp_nbins_xyz = wp.from_torch(self.nbins_xyz, dtype=wp.vec3i)
            wp_neigh_search = wp.from_torch(self.neigh_search, dtype=wp.vec3i)
            wp_bin_size = wp.from_torch(self.bin_size, dtype=self.wp_dtype)
            wp_bin_index = wp.from_torch(self.bin_index, dtype=wp.int32)
            wp_bin_index_xyz = wp.from_torch(self.bin_index_xyz, dtype=wp.vec3i)
            wp_cell_shift = wp.from_torch(self.cell_shift, dtype=wp.vec3i)
            wp_atom_list = wp.from_torch(self.atom_list, dtype=wp.int32)

            wp_pos = wp.from_torch(self.batch["positions"], dtype=self.wp_vec_dtype)
            wp_pbc = wp.from_torch(self.batch["pbc"])
            wp_cell = wp.from_torch(
                self.batch["cell"].reshape(-1, 3, 3), dtype=self.wp_mat_dtype
            )
            wp_atom_ptr = wp.from_torch(self.batch.ptr.to(torch.int32))
            wp_cutoff = wp.from_torch(self.cutoff, dtype=self.wp_dtype)
            wp_max_natoms_per_bin = wp.from_torch(
                self.max_natoms_per_bin_per_system, dtype=wp.int32
            )

            wp.launch(
                _prepare_bins,
                dim=self.num_systems,
                inputs=[
                    wp_bin_size,
                    wp_nbins,
                    wp_nbins_xyz,
                    wp_neigh_search,
                    wp_bin_index,
                    wp_bin_index_xyz,
                    wp_cell_shift,
                    wp_atom_list,
                    wp_pos,
                    wp_cell,
                    wp_pbc,
                    wp_atom_ptr,
                    wp_cutoff,
                    self.max_nbins,
                    wp_max_natoms_per_bin,
                ],
                device=self.device,
            )

            if os.environ.get("ALCHEMI_PROFILE_NVTX", False):
                torch.cuda.synchronize()

        self.max_natoms_per_bin = self.max_natoms_per_bin_per_system.max().item()
        if self.max_natoms_per_bin > self.atoms_in_bins.shape[1]:
            self.atoms_in_bins = torch.cat(
                [
                    self.atoms_in_bins,
                    -1
                    * torch.ones(
                        (
                            self.atoms_in_bins.shape[0],
                            self.max_natoms_per_bin - self.atoms_in_bins.shape[1],
                        ),
                        dtype=torch.int32,
                        device=self.device,
                    ),
                ],
                dim=1,
            )

    def _sort_atoms_into_bins(
        self,
    ):
        """Sort the atoms into particular bins"""
        with nvtx.annotate(message="Sorting Atoms", color="green"):
            wp_atom_ptr = wp.from_torch(self.batch.ptr.to(torch.int32))
            wp_bin_ptr = wp.from_torch(self.bin_ptr.to(torch.int32))
            wp.launch(
                _sort_atoms_into_bins,
                dim=self.num_systems,
                inputs=[
                    self.bin_index,
                    self.atom_list,
                    self.atoms_in_bins,
                    wp_atom_ptr,
                    wp_bin_ptr,
                ],
                device=self.device,
            )
            if os.environ.get("ALCHEMI_PROFILE_NVTX", False):
                torch.cuda.synchronize()

    def _get_neighbor_list(
        self,
    ):

        wp_pos = wp.from_torch(self.batch["positions"], dtype=self.wp_vec_dtype)
        wp_cell = wp.from_torch(
            self.batch["cell"].reshape(-1, 3, 3), dtype=self.wp_mat_dtype
        )
        wp_atom_ptr = wp.from_torch(self.batch.ptr.to(torch.int32))
        wp_bin_ptr = wp.from_torch(self.bin_ptr.to(torch.int32))
        wp_cutoff = wp.from_torch(self.cutoff, dtype=self.wp_dtype)

        result_count = torch.zeros(
            self.num_atoms, dtype=torch.int32, device=self.device
        )
        offset = torch.zeros(self.num_atoms + 1, dtype=torch.int32, device=self.device)
        with nvtx.annotate(message="querying neighbor list", color="green"):
            wp_result_count = wp.from_torch(result_count, dtype=wp.int32)
            wp_atoms_in_bin = wp.from_torch(self.atoms_in_bins, dtype=wp.int32)
            wp_nbins_xyz = wp.from_torch(self.nbins_xyz, dtype=wp.vec3i)
            wp_bin_index_xyz = wp.from_torch(self.bin_index_xyz, dtype=wp.vec3i)
            wp_neigh_search = wp.from_torch(self.neigh_search, dtype=wp.vec3i)
            wp_cell_shift = wp.from_torch(self.cell_shift, dtype=wp.vec3i)
            wp.launch(
                _query_neighbor_list,
                dim=(self.num_systems, self.max_atoms_per_system),
                inputs=[
                    wp_nbins_xyz,
                    wp_bin_index_xyz,
                    wp_neigh_search,
                    wp_cell_shift,
                    wp_atoms_in_bin,
                    wp_pos,
                    wp_cell,
                    wp_atom_ptr,
                    wp_bin_ptr,
                    wp_cutoff,
                    wp_result_count,
                    self.max_natoms_per_bin,
                ],
                device=self.device,
            )
            if os.environ.get("ALCHEMI_PROFILE_NVTX", False):
                torch.cuda.synchronize()

        with nvtx.annotate(message="cumsumming", color="blue"):
            torch.cumsum(result_count, dim=0, out=offset[1:])
            total_count = offset[-1]
            if os.environ.get("ALCHEMI_PROFILE_NVTX", False):
                torch.cuda.synchronize()

        with nvtx.annotate(message="creating ijS buffers", color="blue"):

            wp_offset = wp.from_torch(offset)

            i = torch.zeros((total_count,), dtype=torch.int32, device=self.device)
            j = torch.zeros((total_count,), dtype=torch.int32, device=self.device)
            u = torch.zeros((total_count, 3), dtype=torch.int32, device=self.device)
            S = torch.zeros((total_count, 3), dtype=self.dtype, device=self.device)

            wp_i = wp.from_torch(i, dtype=wp.int32)
            wp_j = wp.from_torch(j, dtype=wp.int32)
            wp_u = wp.from_torch(u, dtype=wp.vec3i)
            wp_S = wp.from_torch(S, dtype=self.wp_vec_dtype)
            if os.environ.get("ALCHEMI_PROFILE_NVTX", False):
                torch.cuda.synchronize()

        with nvtx.annotate(message="building neighborlist", color="green"):
            wp.launch(
                _build_neighbor_list,
                dim=(self.num_systems, self.max_atoms_per_system),
                inputs=[
                    wp_nbins_xyz,
                    wp_bin_index_xyz,
                    wp_neigh_search,
                    wp_cell_shift,
                    wp_atoms_in_bin,
                    wp_pos,
                    wp_cell,
                    wp_atom_ptr,
                    wp_bin_ptr,
                    wp_cutoff,
                    wp_offset,
                    wp_i,
                    wp_j,
                    wp_u,
                    wp_S,
                    self.max_natoms_per_bin,
                ],
                device=self.device,
            )
            if os.environ.get("ALCHEMI_PROFILE_NVTX", False):
                torch.cuda.synchronize()
        # # MA ADDED:
        # mask = i < j  # Ensure each pair appears once
        # i = i[mask]
        # j = j[mask]
        # u = u[mask]
        # S = S[mask]
        return i, j, u, S


def batched_cell_list(
    batch: Batch, cutoff: torch.Tensor, cell_list: CellList | None = None
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
    cell_list : CellList | None
        If provided, the cell list will be used to compute the neighbor list.
        Otherwise, a new cell list will be created.
    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Returns `i`, `j`, `S` - the index of the base
        atom, the index of the neighboring atom,
        and the cell shifts between them.
    """
    if cell_list is None:
        cell_list = CellList(batch, cutoff)

    cell_list._prepare_bins()
    cell_list._sort_atoms_into_bins()
    return cell_list._get_neighbor_list()
