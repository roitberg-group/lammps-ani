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
def _construct_bins(
    nbins: wp.array(dtype=Any),
    nbins_xyz: wp.array(dtype=Any),
    bin_size: wp.array(dtype=Any),
    cell: wp.array(dtype=Any),
    cutoff: wp.array(dtype=Any),
    max_nbins: Any,
):
    """This utility method prepares the neighborlist bins
    based on the cell, pbc, and position info of the passed systems.

    This method constructs the size of the bins, the number of bins,
    the bin indicies, and the maximum number of atoms per bin.

    Parameters
    ----------
    nbins : wp.array
        Array counting the number of bins
    nbins_xyz : wp.array
        Array counting the number of bins in each dimension
    bin_size : wp.array
        Array of bin sizes
    cell : wp.array, optional
        Array of 3x3 cell matrices.
    cutoff : wp.array, optional
        An array of cutoff values for each system.
    max_nbins : int
        Maximum number of bins allowed
    """
    tid = wp.tid()

    bin_size[tid] = max(cutoff[tid], type(cutoff[tid])(3))
    inv_cell = wp.transpose(wp.inverse(cell[tid]))

    for i in range(3):
        face_dist = type(cutoff[tid])(1.0) / wp.length(inv_cell[i])
        nbins_xyz[tid][i] = max(wp.int32(face_dist / bin_size[tid]), 1)

    nbins[tid] = nbins_xyz[tid][0] * nbins_xyz[tid][1] * nbins_xyz[tid][2]

    while nbins[tid] > max_nbins:
        for i in range(3):
            nbins_xyz[tid][i] = max(nbins_xyz[tid][i] // 2, 1)

        nbins[tid] = nbins_xyz[tid][0] * nbins_xyz[tid][1] * nbins_xyz[tid][2]


@wp.overload
def _construct_bins(  # noqa: F811
    nbins: wp.array(dtype=int),
    nbins_xyz: wp.array(dtype=wp.vec3i),
    bin_size: wp.array(dtype=wp.float64),
    cell: wp.array(dtype=wp.mat33d),
    cutoff: wp.array(dtype=wp.float64),
    max_nbins: int,
):  #  pragma: no cover
    """This utility method prepares the neighborlist bins (float64 version)"""
    ...


@wp.overload
def _construct_bins(  # noqa: F811
    nbins: wp.array(dtype=int),
    nbins_xyz: wp.array(dtype=wp.vec3i),
    bin_size: wp.array(dtype=wp.float32),
    cell: wp.array(dtype=wp.mat33f),
    cutoff: wp.array(dtype=wp.float32),
    max_nbins: int,
):  #  pragma: no cover
    """This utility method prepares the neighborlist bins (float32 version)"""
    ...


@wp.kernel
def _prepare_bins(
    bin_size: wp.array(dtype=Any),
    nbins: wp.array(dtype=Any),
    nbins_xyz: wp.array(dtype=Any),
    neigh_search: wp.array(dtype=Any),
    bin_index: wp.array(dtype=Any),
    bin_index_xyz: wp.array(dtype=Any),
    cell_shift: wp.array(dtype=Any),
    atom_list: wp.array(dtype=Any),
    positions: wp.array(dtype=Any),
    cell: wp.array(dtype=Any),
    pbc: wp.array2d(dtype=Any),
    atom_ptr: wp.array(dtype=Any),
    cutoff: wp.array(dtype=Any),
    max_nbins: Any,
    max_natoms_per_bin: wp.array(dtype=Any),
):
    """This utility method prepares the neighborlist bins
    based on the cell, pbc, and position info of the passed systems.

    This method constructs the size of the bins, the number of bins,
    the bin indicies, and the maximum number of atoms per bin.

    Parameters
    ----------
    max_nbins : int
        Maximum number of bins allowed
    bins : wp.array
        Array of initialized NLBin objects
    positions : wp.array
        Array of positions, concatenated by system
    cell : wp.array, optional
        Array of 3x3 cell matrices.
    pbc : wp.array2d, optional
        Array of bools represented whether the x- y- or z-
        directions are periodic.
    atom_ptr : wp.array, optional
        An array of indices pointing to the atoms for a particular
        system.
    cutoff : wp.array, optional
        An array of cutoff values for each system.
    """
    tid = wp.tid()
    ctf = cutoff[tid]
    bin_size[tid] = max(ctf, type(ctf)(3))
    a_0 = atom_ptr[tid]
    a_n = atom_ptr[tid + 1]
    inv_cell = wp.transpose(wp.inverse(cell[tid]))

    for i in range(3):
        face_dist = type(ctf)(1.0) / wp.length(inv_cell[i])
        if nbins_xyz[tid][i] == 1 and not pbc[tid, i]:
            neigh_search[tid][i] = 0
        else:
            neigh_search[tid][i] = wp.int32(
                wp.ceil(bin_size[tid] * type(ctf)(nbins_xyz[tid][i]) / face_dist)
            )

    for ind in range(a_0, a_n):
        scaled_positions = inv_cell * positions[ind]

        atom_list[ind] = ind - a_0
        for j in range(3):
            bin_index_xyz[ind][j] = 0

            bin_index_xyz[ind][j] = wp.int32(
                wp.floor(scaled_positions[j] * type(ctf)(nbins_xyz[tid][j]))
            )

            if pbc[tid, j]:
                a = bin_index_xyz[ind][j]
                b = nbins_xyz[tid][j]
                div, mod = wpdivmod(a, b)
                cell_shift[ind][j] = div
                bin_index_xyz[ind][j] = mod
            else:
                cell_shift[ind][j] = 0
                bin_index_xyz[ind][j] = wp.clamp(
                    bin_index_xyz[ind][j], 0, nbins_xyz[tid][j] - 1
                )

        bin_index[ind] = bin_index_xyz[ind][0] + nbins_xyz[tid][0] * (
            bin_index_xyz[ind][1] + nbins_xyz[tid][1] * (bin_index_xyz[ind][2])
        )

    for ind in range(a_n - a_0):
        for jnd in range(a_0, a_n - ind - 1):
            if bin_index[jnd] > bin_index[jnd + 1]:
                temp = bin_index[jnd]
                bin_index[jnd] = bin_index[jnd + 1]
                bin_index[jnd + 1] = temp

                temp = atom_list[jnd]
                atom_list[jnd] = atom_list[jnd + 1]
                atom_list[jnd + 1] = temp

    current_bin = int(0)
    count = int(0)
    for ind in range(a_0, a_n):
        if bin_index[ind] != current_bin:
            if max_natoms_per_bin[tid] <= count:
                max_natoms_per_bin[tid] = count
            count = 0
            current_bin = bin_index[ind]

        count += 1

    if max_natoms_per_bin[tid] < count:
        max_natoms_per_bin[tid] = count


@wp.overload
def _prepare_bins(  # noqa: F811
    bin_size: wp.array(dtype=wp.float32),
    nbins: wp.array(dtype=int),
    nbins_xyz: wp.array(dtype=wp.vec3i),
    neigh_search: wp.array(dtype=wp.vec3i),
    bin_index: wp.array(dtype=int),
    bin_index_xyz: wp.array(dtype=wp.vec3i),
    cell_shift: wp.array(dtype=wp.vec3i),
    atom_list: wp.array(dtype=int),
    positions: wp.array(dtype=wp.vec3f),
    cell: wp.array(dtype=wp.mat33f),
    pbc: wp.array2d(dtype=bool),
    atom_ptr: wp.array(dtype=int),
    cutoff: wp.array(dtype=wp.float32),
    max_nbins: int,
    max_natoms_per_bin: wp.array(dtype=int),
):  #  pragma: no cover
    """This utility method prepares the neighborlist bins (float32 version)"""
    ...


@wp.overload
def _prepare_bins(  # noqa: F811
    bin_size: wp.array(dtype=wp.float64),
    nbins: wp.array(dtype=int),
    nbins_xyz: wp.array(dtype=wp.vec3i),
    neigh_search: wp.array(dtype=wp.vec3i),
    bin_index: wp.array(dtype=int),
    bin_index_xyz: wp.array(dtype=wp.vec3i),
    cell_shift: wp.array(dtype=wp.vec3i),
    atom_list: wp.array(dtype=int),
    positions: wp.array(dtype=wp.vec3d),
    cell: wp.array(dtype=wp.mat33d),
    pbc: wp.array2d(dtype=bool),
    atom_ptr: wp.array(dtype=int),
    cutoff: wp.array(dtype=wp.float64),
    max_nbins: int,
    max_natoms_per_bin: wp.array(dtype=int),
):  #  pragma: no cover
    """This utility method prepares the neighborlist bins (float64 version)"""
    ...


@wp.kernel
def _sort_atoms_into_bins(
    bin_index: wp.array(dtype=int),
    atom_list: wp.array(dtype=int),
    atoms_in_bin: wp.array2d(dtype=int),
    atom_ptr: wp.array(dtype=int),
    bin_ptr: wp.array(dtype=int),
):
    """Sort the atoms into particular bins

    Parameters
    ----------
    bins : wp.array
        Array of NLBin objects, should be filled by the
        `prepare_bins` method.
    sa : wp.array
        Array of SortedAtoms objects, should be initialized
        but unfilled.
    """
    tid = wp.tid()
    a_0, a_n = atom_ptr[tid], atom_ptr[tid + 1]
    b_0 = bin_ptr[tid]
    count = int(0)
    current_bin = int(0)

    for ind in range(a_0, a_n):
        bi_n = bin_index[ind]
        ai_n = atom_list[ind]

        if bi_n != current_bin:
            count = int(0)
            current_bin = bi_n

        atoms_in_bin[b_0 + bi_n, count] = ai_n
        count += 1


@wp.kernel
def _query_neighbor_list(
    nbins_xyz: wp.array(dtype=Any),
    bin_index_xyz: wp.array(dtype=Any),
    neigh_search: wp.array(dtype=Any),
    cell_shift: wp.array(dtype=Any),
    atoms_in_bin: wp.array2d(dtype=Any),
    positions: wp.array(dtype=Any),
    cell: wp.array(dtype=Any),
    atom_ptr: wp.array(dtype=Any),
    bin_ptr: wp.array(dtype=Any),
    cutoff: wp.array(dtype=Any),
    result_count: wp.array(dtype=Any),
    max_natoms_per_bin: Any,
):
    """Determine the number of neighbors for each atom

    This method loops over all neighboring bins.

    Parameters
    ----------
    bins : wp.array
        Array of NLBin objects, should be filled by the
        `prepare_bins` method.
    sas : wp.array
        Array of SortedAtoms objects, should be initialized
        and filled by the `sort_atoms_into_bins` method.
    positions : wp.array
        Array of positions, concatenated by system
    cell : wp.array
        Array of 3x3 cell matrices.
    atom_ptr : wp.array
        An array of indices pointing to the atoms for a particular
        system.
    cutoff : wp.array
        An array of cutoff values for each system.
    result_count : wp.array
        Array to fill the neighbor count for each atom
    """
    tid, tjd = wp.tid()
    a_0 = atom_ptr[tid]
    a_n = atom_ptr[tid + 1]
    b_0 = bin_ptr[tid]
    num_atoms = a_n - a_0

    ctf = cutoff[tid]
    pos_i = positions[a_0 + tjd]
    cellt = wp.transpose(cell[tid])
    cellshift_i = cell_shift[a_0 + tjd]
    if tjd < num_atoms:
        bi_xyz = bin_index_xyz[a_0 + tjd]

        count = int(0)
        cc = int(0)
        for dz in range(-neigh_search[tid][2], neigh_search[tid][2] + 1):
            for dy in range(-neigh_search[tid][1], neigh_search[tid][1] + 1):
                for dx in range(-neigh_search[tid][0], neigh_search[tid][0] + 1):

                    cc += 1
                    ax = bi_xyz[0] + dx
                    ay = bi_xyz[1] + dy
                    az = bi_xyz[2] + dz
                    cx = nbins_xyz[tid][0]
                    cy = nbins_xyz[tid][1]
                    cz = nbins_xyz[tid][2]

                    divx, mx = wpdivmod(ax, cx)
                    divy, my = wpdivmod(ay, cy)
                    divz, mz = wpdivmod(az, cz)

                    neighbin_b = mx + nbins_xyz[tid][0] * (my + nbins_xyz[tid][1] * mz)
                    for ni in range(max_natoms_per_bin):
                        aj = atoms_in_bin[b_0 + neighbin_b, ni]

                        # MA modified to remove bidirectionality
                        # if aj != -1:
                        if aj != -1 and (a_0 + tjd) < aj:
                            global_shift = cellshift_i - cell_shift[a_0 + aj]

                            shift = type(pos_i)(
                                type(ctf)(divx + global_shift[0]),
                                type(ctf)(divy + global_shift[1]),
                                type(ctf)(divz + global_shift[2]),
                            )

                            cshift = cellt * shift

                            if not (
                                (aj == tjd)
                                and (
                                    cshift[0] == 0 and cshift[1] == 0 and cshift[2] == 0
                                )
                            ):
                                dist = wp.length(positions[a_0 + aj] - pos_i + cshift)

                                if dist < ctf:
                                    count += 1

        result_count[a_0 + tjd] = count


@wp.overload
def _query_neighbor_list(
    nbins_xyz: wp.array(dtype=wp.vec3i),
    bin_index_xyz: wp.array(dtype=wp.vec3i),
    neigh_search: wp.array(dtype=wp.vec3i),
    cell_shift: wp.array(dtype=wp.vec3i),
    atoms_in_bin: wp.array2d(dtype=int),
    positions: wp.array(dtype=wp.vec3d),
    cell: wp.array(dtype=wp.mat33d),
    atom_ptr: wp.array(dtype=int),
    bin_ptr: wp.array(dtype=int),
    cutoff: wp.array(dtype=wp.float64),
    result_count: wp.array(dtype=wp.int32),
    max_natoms_per_bin: int,
):  #  pragma: no cover
    """Determine the number of neighbors for each atom (float64 version)"""
    ...


@wp.kernel
def _build_neighbor_list(
    nbins_xyz: wp.array(dtype=Any),
    bin_index_xyz: wp.array(dtype=Any),
    neigh_search: wp.array(dtype=Any),
    cell_shift: wp.array(dtype=Any),
    atoms_in_bin: wp.array2d(dtype=Any),
    positions: wp.array(dtype=Any),
    cell: wp.array(dtype=Any),
    atom_ptr: wp.array(dtype=Any),
    bin_ptr: wp.array(dtype=Any),
    cutoff: wp.array(dtype=Any),
    offset: wp.array(dtype=Any),
    i: wp.array(dtype=Any),
    j: wp.array(dtype=Any),
    u: wp.array(dtype=Any),
    S: wp.array(dtype=Any),
    max_natoms_per_bin: Any,
):
    """Determine the number of neighbors for each atom

    This method loops over all neighboring bins.

    Parameters
    ----------
    bins : wp.array
        Array of NLBin objects, should be filled by the
        `prepare_bins` method.
    sas : wp.array
        Array of SortedAtoms objects, should be initialized
        and filled by the `sort_atoms_into_bins` method.
    positions : wp.array
        Array of positions, concatenated by system
    cell : wp.array
        Array of 3x3 cell matrices.
    atom_ptr : wp.array
        An array of indices pointing to the atoms for a particular
        system.
    cutoff : wp.array
        An array of cutoff values for each system.
    offset : wp.array
        Offset of elements in i, j, S for each atom
    i : wp.array
        Array of indices representing the base atom
    j : wp.array
        Array of indices representing the neighboring atom
    u : wp.array
        Array of unit cell shifts
    S : wp.array
        Array of cell shifts
    """
    tid, tjd = wp.tid()
    a_0 = atom_ptr[tid]
    a_n = atom_ptr[tid + 1]
    b_0 = bin_ptr[tid]
    num_atoms = a_n - a_0

    ctf = cutoff[tid]
    pos_i = positions[a_0 + tjd]
    cellt = wp.transpose(cell[tid])
    cellshift_i = cell_shift[a_0 + tjd]
    offset_tid = offset[a_0 + tjd]

    if tjd < num_atoms:
        bi_xyz = bin_index_xyz[a_0 + tjd]
        count = int(0)
        cc = int(0)
        for dz in range(-neigh_search[tid][2], neigh_search[tid][2] + 1):
            for dy in range(-neigh_search[tid][1], neigh_search[tid][1] + 1):
                for dx in range(-neigh_search[tid][0], neigh_search[tid][0] + 1):

                    ax, ay, az = bi_xyz[0] + dx, bi_xyz[1] + dy, bi_xyz[2] + dz
                    cx, cy, cz = (
                        nbins_xyz[tid][0],
                        nbins_xyz[tid][1],
                        nbins_xyz[tid][2],
                    )
                    cc += 1

                    divx, mx = wpdivmod(ax, cx)
                    divy, my = wpdivmod(ay, cy)
                    divz, mz = wpdivmod(az, cz)

                    neighbin_b = mx + nbins_xyz[tid][0] * (my + nbins_xyz[tid][1] * mz)
                    for ni in range(max_natoms_per_bin):
                        aj = atoms_in_bin[b_0 + neighbin_b, ni]

                        if aj != -1:
                            global_shift = cellshift_i - cell_shift[a_0 + aj]

                            shift = type(pos_i)(
                                type(ctf)(divx + global_shift[0]),
                                type(ctf)(divy + global_shift[1]),
                                type(ctf)(divz + global_shift[2]),
                            )
                            cshift = cellt * shift

                            if not (
                                (aj == tjd)
                                and (
                                    cshift[0] == 0 and cshift[1] == 0 and cshift[2] == 0
                                )
                            ):
                                dist = wp.length(positions[a_0 + aj] - pos_i + cshift)
                                if dist < ctf:
                                    i[offset_tid + count] = tjd
                                    j[offset_tid + count] = aj
                                    u[offset_tid + count] = wp.vec3i(
                                        int(shift[0]), int(shift[1]), int(shift[2])
                                    )
                                    S[offset_tid + count] = cshift
                                    count += 1


@wp.overload
def _build_neighbor_list(  # noqa: F811
    nbins_xyz: wp.array(dtype=wp.vec3i),
    bin_index_xyz: wp.array(dtype=wp.vec3i),
    neigh_search: wp.array(dtype=wp.vec3i),
    cell_shift: wp.array(dtype=wp.vec3i),
    atoms_in_bin: wp.array2d(dtype=int),
    positions: wp.array(dtype=wp.vec3f),
    cell: wp.array(dtype=wp.mat33f),
    atom_ptr: wp.array(dtype=int),
    bin_ptr: wp.array(dtype=int),
    cutoff: wp.array(dtype=wp.float32),
    offset: wp.array(dtype=int),
    i: wp.array(dtype=int),
    j: wp.array(dtype=int),
    u: wp.array(dtype=wp.vec3i),
    S: wp.array(dtype=wp.vec3f),
    max_natoms_per_bin: int,
):  #  pragma: no cover
    """Determine the number of neighbors for each atom (float32 version)"""
    ...


@wp.overload
def _build_neighbor_list(  # noqa: F811
    nbins_xyz: wp.array(dtype=wp.vec3i),
    bin_index_xyz: wp.array(dtype=wp.vec3i),
    neigh_search: wp.array(dtype=wp.vec3i),
    cell_shift: wp.array(dtype=wp.vec3i),
    atoms_in_bin: wp.array2d(dtype=int),
    positions: wp.array(dtype=wp.vec3d),
    cell: wp.array(dtype=wp.mat33d),
    atom_ptr: wp.array(dtype=int),
    bin_ptr: wp.array(dtype=int),
    cutoff: wp.array(dtype=wp.float64),
    offset: wp.array(dtype=int),
    i: wp.array(dtype=int),
    j: wp.array(dtype=int),
    u: wp.array(dtype=wp.vec3i),
    S: wp.array(dtype=wp.vec3d),
    max_natoms_per_bin: int,
):  #  pragma: no cover
    """Determine the number of neighbors for each atom (float64 version)"""
    ...

