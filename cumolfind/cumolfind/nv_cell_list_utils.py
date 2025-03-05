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
def _batch_construct_bins(
    positions: wp.array(dtype=Any),
    cell: wp.array(dtype=Any),
    pbc: wp.array2d(dtype=Any),
    cell_counts: wp.array(dtype=Any),
    neigh_search: wp.array(dtype=Any),
    cell_size: wp.array(dtype=Any),
    max_nbins: Any,
):
    """Construct bins for each structure in batch"""
    # tid is structure index
    tid = wp.tid()

    # Get cell matrix for this structure
    struct_cell = cell[tid]
    inv_cell = wp.transpose(wp.inverse(struct_cell))
    cs = cell_size[tid]

    # Compute bins for this structure
    for i in range(3):
        face_dist = type(cs)(1.0) / wp.length(inv_cell[i])
        cell_counts[tid][i] = max(wp.int32(face_dist / cs), 1)

    # Adjust if too many bins
    nbins = int(cell_counts[tid][0] * cell_counts[tid][1] * cell_counts[tid][2])
    while nbins > max_nbins:
        for i in range(3):
            cell_counts[tid][i] = max(cell_counts[tid][i] // 2, 1)
        nbins = int(cell_counts[tid][0] * cell_counts[tid][1] * cell_counts[tid][2])

    # Compute neighbor search range
    for i in range(3):
        if cell_counts[tid][i] == 1 and not pbc[tid, i]:
            neigh_search[tid][i] = 0
        else:
            neigh_search[tid][i] = wp.int32(
                wp.ceil(cs * type(cs)(cell_counts[tid][i]) / face_dist)
            )


@wp.overload
def _batch_construct_bins(
    positions: wp.array(dtype=wp.vec3f),
    cell: wp.array(dtype=wp.mat33f),
    pbc: wp.array2d(dtype=bool),
    cell_counts: wp.array(dtype=wp.vec3i),
    neigh_search: wp.array(dtype=wp.vec3i),
    cell_size: wp.array(dtype=wp.float32),
    max_nbins: wp.int32,
):
    """Construct bins for each structure in batch (float32 version)"""
    ...


@wp.overload
def _batch_construct_bins(
    positions: wp.array(dtype=wp.vec3d),
    cell: wp.array(dtype=wp.mat33d),
    pbc: wp.array2d(dtype=bool),
    cell_counts: wp.array(dtype=wp.vec3i),
    neigh_search: wp.array(dtype=wp.vec3i),
    cell_size: wp.array(dtype=wp.float64),
    max_nbins: wp.int32,
):
    """Construct bins for each structure in batch (float64 version)"""
    ...


@wp.kernel
def _batch_count_atoms_per_bin(
    positions: wp.array(dtype=Any),
    cell: wp.array(dtype=Any),
    pbc: wp.array2d(dtype=Any),
    ptr: wp.array(dtype=Any),
    cell_counts_per_struct: wp.array(dtype=Any),
    cell_offsets_ptr: wp.array(dtype=Any),
    cell_atom_counts: wp.array(dtype=Any),
    cell_shifts: wp.array(dtype=Any),
):
    """Count atoms per bin for batched structures"""
    tid = wp.tid()

    # Find which structure this atom belongs to
    struct_idx = int(0)
    while tid >= ptr[struct_idx + 1]:
        struct_idx += 1

    # Get cell matrix and counts for this structure
    struct_cell = cell[struct_idx]
    struct_counts = cell_counts_per_struct[struct_idx]
    cell_offset = cell_offsets_ptr[struct_idx]

    # Rest follows similar to non-batch version but with offsets
    inv_cell = wp.transpose(wp.inverse(struct_cell))
    sp = inv_cell * positions[tid]

    cell_index = wp.vec3i(0, 0, 0)
    for i in range(3):
        cell_index[i] = wp.int32(wp.floor(sp[i] * type(sp[i])(struct_counts[i])))

        if pbc[struct_idx, i]:
            a = cell_index[i]
            b = struct_counts[i]
            div, mod = wpdivmod(a, b)
            cell_shifts[tid][i] = div
            cell_index[i] = mod
        else:
            cell_shifts[tid][i] = 0
            cell_index[i] = wp.clamp(cell_index[i], 0, struct_counts[i] - 1)

    # Compute linear index with offset for this structure
    linear_idx = (
        cell_offset
        + cell_index[0]
        + struct_counts[0] * (cell_index[1] + struct_counts[1] * cell_index[2])
    )

    wp.atomic_add(cell_atom_counts, linear_idx, 1)


@wp.overload
def _batch_count_atoms_per_bin(
    positions: wp.array(dtype=wp.vec3f),
    cell: wp.array(dtype=wp.mat33f),
    pbc: wp.array2d(dtype=bool),
    ptr: wp.array(dtype=wp.int32),
    cell_counts_per_struct: wp.array(dtype=wp.vec3i),
    cell_offsets_ptr: wp.array(dtype=wp.int32),
    cell_atom_counts: wp.array(dtype=wp.int32),
    cell_shifts: wp.array(dtype=wp.vec3i),
):
    """Count atoms per bin for batched structures (float32 version)"""
    ...


@wp.overload
def _batch_count_atoms_per_bin(
    positions: wp.array(dtype=wp.vec3d),
    cell: wp.array(dtype=wp.mat33d),
    pbc: wp.array2d(dtype=bool),
    ptr: wp.array(dtype=wp.int32),
    cell_counts_per_struct: wp.array(dtype=wp.vec3i),
    cell_offsets_ptr: wp.array(dtype=wp.int32),
    cell_atom_counts: wp.array(dtype=wp.int32),
    cell_shifts: wp.array(dtype=wp.vec3i),
):
    """Count atoms per bin for batched structures (float64 version)"""
    ...


@wp.kernel
def _batch_bin_atoms_pbc(
    positions: wp.array(dtype=Any),
    cell: wp.array(dtype=Any),
    pbc: wp.array2d(dtype=Any),
    ptr: wp.array(dtype=wp.int32),
    cell_counts_per_struct: wp.array(dtype=Any),
    cell_offsets_ptr: wp.array(dtype=wp.int32),
    atom_cell_indices: wp.array(dtype=Any),
    cell_atom_counts: wp.array(dtype=Any),
    cell_offsets: wp.array(dtype=Any),
    cell_atom_indices: wp.array(dtype=Any),
):
    """Bin atoms into cells for batched structures"""
    tid = wp.tid()

    # Find which structure this atom belongs to
    struct_idx = int(0)
    while tid >= ptr[struct_idx + 1]:
        struct_idx += 1

    # Get structure-specific information
    struct_cell = cell[struct_idx]
    struct_counts = cell_counts_per_struct[struct_idx]
    cell_offset = cell_offsets_ptr[struct_idx]

    inv_cell = wp.transpose(wp.inverse(struct_cell))
    sp = inv_cell * positions[tid]

    cell_index = wp.vec3i(0, 0, 0)
    for i in range(3):
        cell_index[i] = wp.int32(wp.floor(sp[i] * type(sp[i])(struct_counts[i])))

        if pbc[struct_idx, i]:
            a = cell_index[i]
            b = struct_counts[i]
            div, mod = wpdivmod(a, b)
            cell_index[i] = mod
        else:
            cell_index[i] = wp.clamp(cell_index[i], 0, struct_counts[i] - 1)

    atom_cell_indices[tid] = cell_index

    # Compute linear index with structure offset
    linear_idx = (
        cell_offset
        + cell_index[0]
        + struct_counts[0] * (cell_index[1] + struct_counts[1] * cell_index[2])
    )

    if linear_idx >= len(cell_atom_counts):
        return

    # Atomically add atom to cell and get its position
    pos_in_cell = wp.atomic_add(cell_atom_counts, linear_idx, 1)
    final_idx = cell_offsets[linear_idx] + pos_in_cell

    if final_idx >= len(cell_atom_indices):
        return

    cell_atom_indices[final_idx] = tid


@wp.overload
def _batch_bin_atoms_pbc(
    positions: wp.array(dtype=wp.vec3f),
    cell: wp.array(dtype=wp.mat33f),
    pbc: wp.array2d(dtype=bool),
    ptr: wp.array(dtype=wp.int32),
    cell_counts_per_struct: wp.array(dtype=wp.vec3i),
    cell_offsets_ptr: wp.array(dtype=wp.int32),
    atom_cell_indices: wp.array(dtype=wp.vec3i),
    cell_atom_counts: wp.array(dtype=wp.int32),
    cell_offsets: wp.array(dtype=wp.int32),
    cell_atom_indices: wp.array(dtype=wp.int32),
):
    """Bin atoms into cells for batched structures (float32 version)"""
    ...


@wp.overload
def _batch_bin_atoms_pbc(
    positions: wp.array(dtype=wp.vec3d),
    cell: wp.array(dtype=wp.mat33d),
    pbc: wp.array2d(dtype=bool),
    ptr: wp.array(dtype=wp.int32),
    cell_counts_per_struct: wp.array(dtype=wp.vec3i),
    cell_offsets_ptr: wp.array(dtype=wp.int32),
    atom_cell_indices: wp.array(dtype=wp.vec3i),
    cell_atom_counts: wp.array(dtype=wp.int32),
    cell_offsets: wp.array(dtype=wp.int32),
    cell_atom_indices: wp.array(dtype=wp.int32),
):
    """Bin atoms into cells for batched structures (float64 version)"""
    ...


@wp.kernel
def _batch_query_neighbor_list(
    coord: wp.array(dtype=Any),
    cell: wp.array(dtype=Any),
    pbc: wp.array2d(dtype=Any),
    ptr: wp.array(dtype=Any),
    cutoff: wp.array(dtype=Any),
    cell_counts_per_struct: wp.array(dtype=Any),
    cell_offsets_ptr: wp.array(dtype=Any),
    cell_shifts: wp.array(dtype=Any),
    neigh_search_per_struct: wp.array(dtype=Any),
    atom_cell_indices: wp.array(dtype=Any),
    cell_atom_counts: wp.array(dtype=Any),
    cell_offsets: wp.array(dtype=Any),
    cell_atom_indices: wp.array(dtype=Any),
    nneigh: wp.array(dtype=Any),
):
    """Count neighbors using cell lists for batched structures"""
    tid = wp.tid()

    # Find which structure this atom belongs to
    struct_idx = int(0)
    while tid >= ptr[struct_idx + 1]:
        struct_idx += 1

    pos_i = coord[tid]
    ctf = cutoff[struct_idx]
    cell_i = atom_cell_indices[tid]
    cell_t = wp.transpose(cell[struct_idx])
    struct_counts = cell_counts_per_struct[struct_idx]
    cell_offset = cell_offsets_ptr[struct_idx]
    neigh_search = neigh_search_per_struct[struct_idx]
    cell_shift_i = cell_shifts[tid]

    count = int(0)
    for dz in range(-neigh_search[2], neigh_search[2] + 1):
        for dy in range(-neigh_search[1], neigh_search[1] + 1):
            for dx in range(-neigh_search[0], neigh_search[0] + 1):
                ax = cell_i[0] + dx
                ay = cell_i[1] + dy
                az = cell_i[2] + dz

                divx, mx = wpdivmod(ax, struct_counts[0])
                divy, my = wpdivmod(ay, struct_counts[1])
                divz, mz = wpdivmod(az, struct_counts[2])

                # Calculate linear index with structure offset
                linear_idx = (
                    cell_offset + mx + struct_counts[0] * (my + struct_counts[1] * mz)
                )

                # Get range of atoms in this cell
                cell_start = cell_offsets[linear_idx]
                num_atoms_in_cell = cell_atom_counts[linear_idx]

                # Iterate over atoms in neighboring cell
                for atom_idx in range(num_atoms_in_cell):
                    aj = cell_atom_indices[cell_start + atom_idx]
                    global_shift = cell_shift_i - cell_shifts[aj]

                    # Only consider atoms in the same structure
                    if aj < ptr[struct_idx] or aj >= ptr[struct_idx + 1]:
                        continue

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
                        if dist < ctf and tid < aj:
                            count += 1

    nneigh[tid] = count


@wp.overload
def _batch_query_neighbor_list(
    coord: wp.array(dtype=wp.vec3f),
    cell: wp.array(dtype=wp.mat33f),
    pbc: wp.array2d(dtype=bool),
    ptr: wp.array(dtype=wp.int32),
    cutoff: wp.array(dtype=wp.float32),
    cell_counts_per_struct: wp.array(dtype=wp.vec3i),
    cell_offsets_ptr: wp.array(dtype=wp.int32),
    cell_shifts: wp.array(dtype=wp.vec3i),
    neigh_search_per_struct: wp.array(dtype=wp.vec3i),
    atom_cell_indices: wp.array(dtype=wp.vec3i),
    cell_atom_counts: wp.array(dtype=wp.int32),
    cell_offsets: wp.array(dtype=wp.int32),
    cell_atom_indices: wp.array(dtype=wp.int32),
    nneigh: wp.array(dtype=wp.int32),
):
    """Count neighbors using cell lists for batched structures (float32 version)"""
    ...


@wp.overload
def _batch_query_neighbor_list(
    coord: wp.array(dtype=wp.vec3d),
    cell: wp.array(dtype=wp.mat33d),
    pbc: wp.array2d(dtype=bool),
    ptr: wp.array(dtype=wp.int32),
    cutoff: wp.array(dtype=wp.float64),
    cell_counts_per_struct: wp.array(dtype=wp.vec3i),
    cell_offsets_ptr: wp.array(dtype=wp.int32),
    cell_shifts: wp.array(dtype=wp.vec3i),
    neigh_search_per_struct: wp.array(dtype=wp.vec3i),
    atom_cell_indices: wp.array(dtype=wp.vec3i),
    cell_atom_counts: wp.array(dtype=wp.int32),
    cell_offsets: wp.array(dtype=wp.int32),
    cell_atom_indices: wp.array(dtype=wp.int32),
    nneigh: wp.array(dtype=wp.int32),
):
    """Count neighbors using cell lists for batched structures (float64 version)"""
    ...


@wp.kernel
def _batch_build_neighbor_list(
    coord: wp.array(dtype=Any),
    cell: wp.array(dtype=Any),
    pbc: wp.array2d(dtype=Any),
    ptr: wp.array(dtype=Any),
    cutoff: wp.array(dtype=Any),
    cell_counts_per_struct: wp.array(dtype=Any),
    cell_offsets_ptr: wp.array(dtype=Any),
    cell_shifts: wp.array(dtype=Any),
    neigh_search_per_struct: wp.array(dtype=Any),
    atom_cell_indices: wp.array(dtype=Any),
    cell_atom_counts: wp.array(dtype=Any),
    cell_offsets: wp.array(dtype=Any),
    cell_atom_indices: wp.array(dtype=Any),
    offset: wp.array(dtype=Any),
    i: wp.array(dtype=Any),
    j: wp.array(dtype=Any),
    dist_ij: wp.array(dtype=Any),
    # u: wp.array(dtype=Any),
    # S: wp.array(dtype=Any),
):
    """Build neighbor list for batched structures"""
    tid = wp.tid()

    # Find which structure this atom belongs to
    struct_idx = int(0)
    while tid >= ptr[struct_idx + 1]:
        struct_idx += 1

    a_0 = ptr[struct_idx]
    ctf = cutoff[struct_idx]
    pos_i = coord[tid]
    cell_i = atom_cell_indices[tid]
    cell_t = wp.transpose(cell[struct_idx])
    struct_counts = cell_counts_per_struct[struct_idx]
    cell_offset = cell_offsets_ptr[struct_idx]
    neigh_search = neigh_search_per_struct[struct_idx]
    cell_shift_i = cell_shifts[tid]
    offset_tid = offset[tid]

    count = int(0)
    # for dz in range(-neigh_search[2], neigh_search[2] + 1):
    #     for dy in range(-neigh_search[1], neigh_search[1] + 1):
    #         for dx in range(-neigh_search[0], neigh_search[0] + 1):

    # ONLY NEIGHBORING BINS
    for dz in range(-1, 2):
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                ax = cell_i[0] + dx
                ay = cell_i[1] + dy
                az = cell_i[2] + dz

                # skip bins that are invalid
                if (ax < 0 or ay < 0 or az < 0 or ax >= struct_counts[0] or ay >= struct_counts[1] or az >= struct_counts[2]):
                    continue  
                divx, mx = wpdivmod(ax, struct_counts[0])
                divy, my = wpdivmod(ay, struct_counts[1])
                divz, mz = wpdivmod(az, struct_counts[2])

                # Calculate linear index with structure offset
                linear_idx = (
                    cell_offset + mx + struct_counts[0] * (my + struct_counts[1] * mz)
                )

                # Get range of atoms in this cell
                cell_start = cell_offsets[linear_idx]
                num_atoms_in_cell = cell_atom_counts[linear_idx]

                # Iterate over atoms in neighboring cell
                for atom_idx in range(num_atoms_in_cell):
                    aj = cell_atom_indices[cell_start + atom_idx]
                    global_shift = cell_shift_i - cell_shifts[aj]
                    # Only consider atoms in the same structure
                    if aj < ptr[struct_idx] or aj >= ptr[struct_idx + 1]:
                        continue

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
                            i_val, j_val = tid - a_0, aj - a_0
                            if i_val < j_val:
                                i[offset_tid + count] = i_val
                                j[offset_tid + count] = j_val
                                dist_ij[offset_tid + count] = dist
                                # u[offset_tid + count] = wp.vec3i(
                                #     int(shift[0]), int(shift[1]), int(shift[2])
                                # )
                                # S[offset_tid + count] = cshift
                                count += 1


@wp.overload
def _batch_build_neighbor_list(
    coord: wp.array(dtype=wp.vec3f),
    cell: wp.array(dtype=wp.mat33f),
    pbc: wp.array2d(dtype=bool),
    ptr: wp.array(dtype=wp.int32),
    cutoff: wp.array(dtype=wp.float32),
    cell_counts_per_struct: wp.array(dtype=wp.vec3i),
    cell_offsets_ptr: wp.array(dtype=wp.int32),
    cell_shifts: wp.array(dtype=wp.vec3i),
    neigh_search_per_struct: wp.array(dtype=wp.vec3i),
    atom_cell_indices: wp.array(dtype=wp.vec3i),
    cell_atom_counts: wp.array(dtype=wp.int32),
    cell_offsets: wp.array(dtype=wp.int32),
    cell_atom_indices: wp.array(dtype=wp.int32),
    offset: wp.array(dtype=wp.int32),
    i: wp.array(dtype=wp.int32),
    j: wp.array(dtype=wp.int32),
    dist_ij: wp.array(dtype=wp.float32),
    # u: wp.array(dtype=wp.vec3i),
    # S: wp.array(dtype=wp.vec3f),
):
    """Build neighbor list for batched structures (float32 version)"""
    ...


@wp.overload
def _batch_build_neighbor_list(
    coord: wp.array(dtype=wp.vec3d),
    cell: wp.array(dtype=wp.mat33d),
    pbc: wp.array2d(dtype=bool),
    ptr: wp.array(dtype=wp.int32),
    cutoff: wp.array(dtype=wp.float64),
    cell_counts_per_struct: wp.array(dtype=wp.vec3i),
    cell_offsets_ptr: wp.array(dtype=wp.int32),
    cell_shifts: wp.array(dtype=wp.vec3i),
    neigh_search_per_struct: wp.array(dtype=wp.vec3i),
    atom_cell_indices: wp.array(dtype=wp.vec3i),
    cell_atom_counts: wp.array(dtype=wp.int32),
    cell_offsets: wp.array(dtype=wp.int32),
    cell_atom_indices: wp.array(dtype=wp.int32),
    offset: wp.array(dtype=wp.int32),
    i: wp.array(dtype=wp.int32),
    j: wp.array(dtype=wp.int32),
    dist_ij: wp.array(dtype=wp.float64),
    # u: wp.array(dtype=wp.vec3i),
    # S: wp.array(dtype=wp.vec3d),
):
    """Build neighbor list for batched structures (float64 version)"""
    ...
