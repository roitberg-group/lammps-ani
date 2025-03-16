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
from typing import Any

import warp as wp


@wp.func
def wpdivmod(a: int, b: int):  # type: ignore
    """Warp implementation of the divmod utility."""
    div = int(a / b)
    mod = a % b
    if mod < 0:
        div -= 1
        mod = b + mod
    return div, mod


@wp.kernel
def _cell_construct_bins(
    coord: wp.array(dtype=Any),
    cell: wp.array(dtype=Any),
    pbc: wp.array(dtype=Any),
    cell_counts: wp.array(dtype=Any),
    neigh_search: wp.array(dtype=Any),
    cell_size: Any,
    max_nbins: Any,
):
    """This utility method prepares the neighborlist bins
    based on the cell, pbc, and position info of the passed systems.

    This method constructs the size of the bins, the number of bins,
    the bin indicies, and the maximum number of atoms per bin.

    Parameters
    ----------
    coord : wp.array
        Array of coordinates
    cell : wp.array
        Array of 3x3 cell matrices.
    pbc : wp.array
        Array of bools represented whether the x- y- or z-
        directions are periodic.
    cutoff : wp.array
        An array of cutoff values for each system.
    cell_counts : wp.array
        Array counting the number of bins
    neigh_search : wp.array
        Array of the number of bins to search for each dimension
    cell_size : float
    max_nbins : int
        Maximum number of bins allowed
    """

    inv_cell = wp.transpose(wp.inverse(cell[0]))

    for i in range(3):
        face_dist = type(cell_size)(1.0) / wp.length(inv_cell[i])
        cell_counts[i] = max(wp.int32(face_dist / cell_size), 1)

    nbins = int(cell_counts[0] * cell_counts[1] * cell_counts[2])

    while nbins > max_nbins:
        for i in range(3):
            cell_counts[i] = max(cell_counts[i] // 2, 1)

        nbins = int(cell_counts[0] * cell_counts[1] * cell_counts[2])

    for i in range(3):
        if cell_counts[i] == 1 and not pbc[i]:
            neigh_search[i] = 0
        else:
            neigh_search[i] = wp.int32(
                wp.ceil(cell_size * type(cell_size)(cell_counts[i]) / face_dist)
            )


@wp.overload
def _cell_construct_bins(  # noqa: F811
    coord: wp.array(dtype=wp.vec3d),
    cell: wp.array(dtype=wp.mat33d),
    pbc: wp.array(dtype=bool),
    cell_counts: wp.array(dtype=wp.int32),
    neigh_search: wp.array(dtype=wp.int32),
    cell_size: wp.float64,
    max_nbins: int,
):  #  pragma: no cover
    """This utility method prepares the neighborlist bins (float64 version)"""
    ...


@wp.overload
def _cell_construct_bins(  # noqa: F811
    coord: wp.array(dtype=wp.vec3f),
    cell: wp.array(dtype=wp.mat33f),
    pbc: wp.array(dtype=bool),
    cell_counts: wp.array(dtype=wp.int32),
    neigh_search: wp.array(dtype=wp.int32),
    cell_size: wp.float32,
    max_nbins: int,
):  #  pragma: no cover
    """This utility method prepares the neighborlist bins (float32 version)"""
    ...


@wp.kernel
def _cell_count_atoms_per_bin(
    positions: wp.array(dtype=Any),
    cell: wp.array(dtype=Any),
    pbc: wp.array(dtype=Any),
    cell_counts: wp.array(dtype=Any),
    cell_atom_counts: wp.array(dtype=Any),
    cell_shifts: wp.array(dtype=Any),
):
    tid = wp.tid()

    inv_cell = wp.transpose(wp.inverse(cell[0]))

    sp = inv_cell * positions[tid]

    cell_index = wp.vec3i(0, 0, 0)
    for i in range(3):
        cell_index[i] = wp.int32(wp.floor(sp[i] * type(sp[i])(cell_counts[i])))

        if pbc[i]:
            a = cell_index[i]
            b = cell_counts[i]
            div, mod = wpdivmod(a, b)
            cell_shifts[tid][i] = div
            cell_index[i] = mod
        else:
            cell_shifts[tid][i] = 0
            cell_index[i] = wp.clamp(cell_index[i], 0, cell_counts[i] - 1)

    linear_idx = cell_index[0] + cell_counts[0] * (
        cell_index[1] + cell_counts[1] * (cell_index[2])
    )

    if linear_idx >= len(cell_atom_counts):
        return

    # Atomically increment count
    wp.atomic_add(cell_atom_counts, linear_idx, 1)


@wp.overload
def _cell_count_atoms_per_bin(
    positions: wp.array(dtype=wp.vec3f),
    cell: wp.array(dtype=wp.mat33f),
    pbc: wp.array(dtype=wp.bool),
    cell_counts: wp.array(dtype=wp.int32),
    cell_atom_counts: wp.array(dtype=wp.int32),
    cell_shifts: wp.array(dtype=wp.vec3i),
):  # pragma: no cover
    """First pass: count atoms per cell (float32 version)"""
    ...


@wp.overload
def _cell_count_atoms_per_bin(
    positions: wp.array(dtype=wp.vec3d),
    cell: wp.array(dtype=wp.mat33d),
    pbc: wp.array(dtype=wp.bool),
    cell_counts: wp.array(dtype=wp.int32),
    cell_atom_counts: wp.array(dtype=wp.int32),
    cell_shifts: wp.array(dtype=wp.vec3i),
):  # pragma: no cover
    """First pass: count atoms per cell (float64 version)"""
    ...


@wp.kernel
def _cell_compute_cell_offsets(
    cell_atom_counts: wp.array(dtype=wp.int32),
    cell_offsets: wp.array(dtype=wp.int32),
    num_cells: int,
):
    """Compute exclusive prefix sum of cell counts"""
    tid = wp.tid()
    if tid < num_cells:
        sum = int(0)
        for i in range(tid):
            sum += cell_atom_counts[i]
        cell_offsets[tid] = sum


@wp.kernel
def _cell_bin_atoms_pbc(
    positions: wp.array(dtype=Any),
    cell: wp.array(dtype=Any),
    pbc: wp.array(dtype=Any),
    cell_counts: wp.array(dtype=Any),
    atom_cell_indices: wp.array(dtype=Any),
    cell_atom_counts: wp.array(dtype=Any),
    cell_offsets: wp.array(dtype=Any),
    cell_atom_indices: wp.array(dtype=Any),
):
    tid = wp.tid()

    if tid >= positions.shape[0]:
        return

    inv_cell = wp.transpose(wp.inverse(cell[0]))

    sp = inv_cell * positions[tid]

    cell_index = wp.vec3i(0, 0, 0)
    for i in range(3):
        cell_index[i] = wp.int32(wp.floor(sp[i] * type(sp[i])(cell_counts[i])))

        if pbc[i]:
            a = cell_index[i]
            b = cell_counts[i]
            div, mod = wpdivmod(a, b)
            cell_index[i] = mod
        else:
            cell_index[i] = wp.clamp(cell_index[i], 0, cell_counts[i] - 1)

    atom_cell_indices[tid] = cell_index

    linear_idx = cell_index[0] + cell_counts[0] * (
        cell_index[1] + cell_counts[1] * (cell_index[2])
    )

    if linear_idx >= len(cell_atom_counts):
        return

    # Atomically add atom to cell and get its position in the cell
    pos_in_cell = wp.atomic_add(cell_atom_counts, linear_idx, 1)

    final_idx = cell_offsets[linear_idx] + pos_in_cell

    if final_idx >= len(cell_atom_indices):
        return

    # Store atom index in cell's list
    cell_atom_indices[final_idx] = tid


@wp.overload
def _cell_bin_atoms_pbc(
    positions: wp.array(dtype=wp.vec3f),
    cell: wp.array(dtype=wp.mat33f),
    pbc: wp.array(dtype=wp.bool),
    cell_counts: wp.array(dtype=wp.int32),
    atom_cell_indices: wp.array(dtype=wp.vec3i),
    cell_atom_counts: wp.array(dtype=wp.int32),
    cell_offsets: wp.array(dtype=wp.int32),
    cell_atom_indices: wp.array(dtype=wp.int32),
):  # pragma: no cover
    """Bin atoms into cells with PBC handling (float32 version)"""
    ...


@wp.overload
def _cell_bin_atoms_pbc(
    positions: wp.array(dtype=wp.vec3d),
    cell: wp.array(dtype=wp.mat33d),
    pbc: wp.array(dtype=wp.bool),
    cell_counts: wp.array(dtype=wp.int32),
    atom_cell_indices: wp.array(dtype=wp.vec3i),
    cell_atom_counts: wp.array(dtype=wp.int32),
    cell_offsets: wp.array(dtype=wp.int32),
    cell_atom_indices: wp.array(dtype=wp.int32),
):  # pragma: no cover
    """Bin atoms into cells with PBC handling (float64 version)"""
    ...


@wp.kernel
def _cell_query_neighbor_list(
    coord: wp.array(dtype=Any),
    cell: wp.array(dtype=Any),
    pbc: wp.array(dtype=Any),
    cutoff: Any,
    cell_counts: wp.array(dtype=Any),
    cell_shifts: wp.array(dtype=Any),
    neigh_search: wp.array(dtype=Any),
    atom_cell_indices: wp.array(dtype=Any),
    cell_atom_counts: wp.array(dtype=Any),
    cell_offsets: wp.array(dtype=Any),
    cell_atom_indices: wp.array(dtype=Any),
    nneigh: wp.array(dtype=Any),
):
    """Count neighbors using cell lists with PBC support"""
    tid = wp.tid()

    ctf = cutoff
    pos_i = coord[tid]
    cell_i = atom_cell_indices[tid]
    cell_t = wp.transpose(cell[0])
    cell_shift_i = cell_shifts[tid]

    count = int(0)
    for dz in range(-neigh_search[2], neigh_search[2] + 1):
        for dy in range(-neigh_search[1], neigh_search[1] + 1):
            for dx in range(-neigh_search[0], neigh_search[0] + 1):
                ax = cell_i[0] + dx
                ay = cell_i[1] + dy
                az = cell_i[2] + dz
                cx = cell_counts[0]
                cy = cell_counts[1]
                cz = cell_counts[2]

                divx, mx = wpdivmod(ax, cx)
                divy, my = wpdivmod(ay, cy)
                divz, mz = wpdivmod(az, cz)

                # Calculate linear index
                linear_idx = mx + cell_counts[0] * (my + cell_counts[1] * mz)

                # Get range of atoms in this cell using cell_offsets
                cell_start = cell_offsets[linear_idx]
                num_atoms_in_cell = cell_atom_counts[linear_idx]

                # Iterate over atoms in neighboring cell
                for atom_idx in range(num_atoms_in_cell):
                    aj = cell_atom_indices[cell_start + atom_idx]
                    if tid >= aj:
                        continue
                    global_shift = cell_shift_i - cell_shifts[aj]

                    shift = type(pos_i)(
                        type(ctf)(divx + global_shift[0]),
                        type(ctf)(divy + global_shift[1]),
                        type(ctf)(divz + global_shift[2]),
                    )
                    cshift = cell_t * shift

                    if not (
                        (aj == tid)
                        and (cshift[0] == 0 and cshift[1] == 0 and cshift[2] == 0)
                    ):
                        dist = wp.length(coord[aj] - pos_i + cshift)

                        if dist < ctf:
                            count += 1

    nneigh[tid] = count


@wp.overload
def _cell_query_neighbor_list(
    coord: wp.array(dtype=wp.vec3f),
    cell: wp.array(dtype=wp.mat33f),
    pbc: wp.array(dtype=wp.bool),
    cutoff: wp.float32,
    cell_counts: wp.array(dtype=wp.int32),
    cell_shifts: wp.array(dtype=wp.vec3i),
    neigh_search: wp.array(dtype=wp.int32),
    atom_cell_indices: wp.array(dtype=wp.vec3i),
    cell_atom_counts: wp.array(dtype=wp.int32),
    cell_offsets: wp.array(dtype=wp.int32),
    cell_atom_indices: wp.array(dtype=wp.int32),
    nneigh: wp.array(dtype=wp.int32),
):  # pragma: no cover
    """Count neighbors using cell lists with PBC support (float32 version)"""
    ...


@wp.overload
def _cell_query_neighbor_list(
    coord: wp.array(dtype=wp.vec3d),
    cell: wp.array(dtype=wp.mat33d),
    pbc: wp.array(dtype=wp.bool),
    cutoff: wp.float64,
    cell_counts: wp.array(dtype=wp.int32),
    cell_shifts: wp.array(dtype=wp.vec3i),
    neigh_search: wp.array(dtype=wp.int32),
    atom_cell_indices: wp.array(dtype=wp.vec3i),
    cell_atom_counts: wp.array(dtype=wp.int32),
    cell_offsets: wp.array(dtype=wp.int32),
    cell_atom_indices: wp.array(dtype=wp.int32),
    nneigh: wp.array(dtype=wp.int32),
):  #  pragma: no cover
    """Count neighbors using cell lists with PBC support (float64 version)"""
    ...


@wp.kernel
def _cell_build_neighbor_list(
    coord: wp.array(dtype=Any),
    cell: wp.array(dtype=Any),
    pbc: wp.array(dtype=Any),
    cutoff: Any,
    cell_counts: wp.array(dtype=Any),
    cell_shifts: wp.array(dtype=Any),
    neigh_search: wp.array(dtype=Any),
    atom_cell_indices: wp.array(dtype=Any),
    cell_atom_counts: wp.array(dtype=Any),
    cell_offsets: wp.array(dtype=Any),
    cell_atom_indices: wp.array(dtype=Any),
    offset: wp.array(dtype=Any),
    i: wp.array(dtype=Any),
    j: wp.array(dtype=Any),
    dist_ij: wp.array(dtype=Any),
    coord_i: wp.array(dtype=Any),
    coord_j: wp.array(dtype=Any),
):
    """Determine the number of neighbors for each atom"""
    tid = wp.tid()

    ctf = cutoff
    pos_i = coord[tid]
    cell_i = atom_cell_indices[tid]
    cell_t = wp.transpose(cell[0])
    cell_shift_i = cell_shifts[tid]
    offset_tid = offset[tid]

    count = int(0)
    for dz in range(-neigh_search[2], neigh_search[2] + 1):
        for dy in range(-neigh_search[1], neigh_search[1] + 1):
            for dx in range(-neigh_search[0], neigh_search[0] + 1):

                ax, ay, az = cell_i[0] + dx, cell_i[1] + dy, cell_i[2] + dz
                cx, cy, cz = (
                    cell_counts[0],
                    cell_counts[1],
                    cell_counts[2],
                )

                divx, mx = wpdivmod(ax, cx)
                divy, my = wpdivmod(ay, cy)
                divz, mz = wpdivmod(az, cz)

                # Calculate linear index
                linear_idx = mx + cell_counts[0] * (my + cell_counts[1] * mz)

                # Get range of atoms in this cell using cell_offsets
                cell_start = cell_offsets[linear_idx]
                num_atoms_in_cell = cell_atom_counts[linear_idx]

                # Iterate over atoms in neighboring cell
                for atom_idx in range(num_atoms_in_cell):
                    aj = cell_atom_indices[cell_start + atom_idx]
                    if tid >= aj:
                        continue
                    global_shift = cell_shift_i - cell_shifts[aj]

                    shift = type(pos_i)(
                        type(ctf)(divx + global_shift[0]),
                        type(ctf)(divy + global_shift[1]),
                        type(ctf)(divz + global_shift[2]),
                    )
                    cshift = cell_t * shift

                    if not (
                        (aj == tid)
                        and (cshift[0] == 0 and cshift[1] == 0 and cshift[2] == 0)
                    ):
                        dist = wp.length(coord[aj] - pos_i + cshift)
                        if dist < ctf:
                            i[offset_tid + count] = tid
                            j[offset_tid + count] = aj
                            dist_ij[offset_tid + count] = dist
                            coord_i[offset_tid + count] = pos_i
                            coord_j[offset_tid + count] = coord[aj]
                            count += 1


@wp.overload
def _cell_build_neighbor_list(  # noqa: F811
    coord: wp.array(dtype=wp.vec3f),
    cell: wp.array(dtype=wp.mat33f),
    pbc: wp.array(dtype=wp.bool),
    cutoff: wp.float32,
    cell_counts: wp.array(dtype=wp.int32),
    cell_shifts: wp.array(dtype=wp.vec3i),
    neigh_search: wp.array(dtype=wp.int32),
    atom_cell_indices: wp.array(dtype=wp.vec3i),
    cell_atom_counts: wp.array(dtype=wp.int32),
    cell_offsets: wp.array(dtype=wp.int32),
    cell_atom_indices: wp.array(dtype=wp.int32),
    offset: wp.array(dtype=int),
    i: wp.array(dtype=int),
    j: wp.array(dtype=int),
    dist_ij: wp.array(dtype=wp.float32),
    coord_i: wp.array(dtype=wp.vec3f),
    coord_j: wp.array(dtype=wp.vec3f),
):  #  pragma: no cover
    """Determine the number of neighbors for each atom (float32 version)"""
    ...


@wp.overload
def _cell_build_neighbor_list(  # noqa: F811
    coord: wp.array(dtype=wp.vec3d),
    cell: wp.array(dtype=wp.mat33d),
    pbc: wp.array(dtype=wp.bool),
    cutoff: wp.float64,
    cell_counts: wp.array(dtype=wp.int32),
    cell_shifts: wp.array(dtype=wp.vec3i),
    neigh_search: wp.array(dtype=wp.int32),
    atom_cell_indices: wp.array(dtype=wp.vec3i),
    cell_atom_counts: wp.array(dtype=wp.int32),
    cell_offsets: wp.array(dtype=wp.int32),
    cell_atom_indices: wp.array(dtype=wp.int32),
    offset: wp.array(dtype=int),
    i: wp.array(dtype=int),
    j: wp.array(dtype=int),
    dist_ij: wp.array(dtype=wp.float64),
    coord_i: wp.array(dtype=wp.vec3d),
    coord_j: wp.array(dtype=wp.vec3d),
):  #  pragma: no cover
    """Determine the number of neighbors for each atom (float64 version)"""
    ...
