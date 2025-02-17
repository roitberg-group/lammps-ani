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


@wp.kernel
def _count_shifts(
    coord: wp.array(dtype=Any),
    cell: wp.array(dtype=Any),
    nshift: wp.array(dtype=Any),
    atom_ptr: wp.array(dtype=Any),
    cutoff: wp.array(dtype=Any),
    nneigh: wp.array(dtype=Any),
    dbg_cutoff: float,
):
    """Computes a batch of shifts

    Parameters
    ----------
    coord : wp.array
        Array of coordinate positions, each entry of which
        represents the position in x-,y-,z- direction.
        Each batch member is concatenated in this array.
    nshift : wp.array
        An array of integer vectors, each entry of which
        represents the number of shifts in the x-,y-,z- direction.
    ptr: wp.array
        An array of integers corresponding to the 1st and n+1th
        element of an batch of length n.
    nneigh: wp.array
        An array to hold the number of neighbors for each atom in
        coord.
    """
    tid, tjd = wp.tid()

    # Get local information
    a_0, a_n = atom_ptr[tid], atom_ptr[tid + 1]
    a_i = a_0 + tjd
    num_atoms = a_n - a_0

    if tjd < num_atoms:
        l_nshift = nshift[tid]
        l_cutoff = cutoff[tid]
        debug_buf = l_cutoff
        l_cell_T = wp.transpose(cell[tid])
        pos_i = coord[a_i]

        count = int(0)
        for k1 in range(-l_nshift[0], l_nshift[0] + 1):
            for k2 in range(-l_nshift[1], l_nshift[1] + 1):
                for k3 in range(-l_nshift[2], l_nshift[2] + 1):

                    # Get the current shift
                    shift = type(pos_i)(
                        type(l_cutoff)(k1), type(l_cutoff)(k2), type(l_cutoff)(k3)
                    )

                    cshift = l_cell_T * shift

                    # Loop over all atoms and test
                    for a_j in range(a_0, a_n):
                        # if not ((a_j == a_i) and wp.length(cshift) < 1e-10):
                        # MA modified to remove bidirectionality!
                        if a_j > a_i: 
                            dist = wp.length(coord[a_j] - pos_i + cshift)

                            if dist < l_cutoff:
                                count += 1

        nneigh[a_i] = count


@wp.overload
def _count_shifts(  # noqa: F811
    coord: wp.array(dtype=wp.vec3f),
    cell: wp.array(dtype=wp.mat33f),
    nshift: wp.array(dtype=wp.vec3i),
    atom_ptr: wp.array(dtype=int),
    cutoff: wp.array(dtype=wp.float32),
    nneigh: wp.array(dtype=int),
    dbg_cutoff: float,
):  #  pragma: no cover
    """Computes a batch of shifts (float32 version)"""
    ...


# _count_shifts float64 version
@wp.overload
def _count_shifts(  # noqa: F811
    coord: wp.array(dtype=wp.vec3d),
    cell: wp.array(dtype=wp.mat33d),
    nshift: wp.array(dtype=wp.vec3i),
    atom_ptr: wp.array(dtype=int),
    cutoff: wp.array(dtype=wp.float64),
    nneigh: wp.array(dtype=int),
    dbg_cutoff: float,
):  #  pragma: no cover
    """Computes a batch of shifts (float64 version)"""
    ...


@wp.kernel
def _build_neighbor_list(
    coord: wp.array(dtype=Any),
    cell: wp.array(dtype=Any),
    nshift: wp.array(dtype=Any),
    atom_ptr: wp.array(dtype=Any),
    cutoff: wp.array(dtype=Any),
    offset: wp.array(dtype=Any),
    i: wp.array(dtype=Any),
    j: wp.array(dtype=Any),
    u: wp.array(dtype=Any),
    S: wp.array(dtype=Any),
):
    """Computes a batch of shifts

    Parameters
    ----------
    coord : wp.array
        Array of coordinate positions, each entry of which
        represents the position in x-,y-,z- direction.
        Each batch member is concatenated in this array.
    nshift : wp.array
        An array of integer vectors, each entry of which
        represents the number of shifts in the x-,y-,z- direction.
    ptr: wp.array
        An array of integers corresponding to the 1st and n+1th
        element of an batch of length n.
    nneigh: wp.array
        An array to hold the number of neighbors for each atom in
        coord.
    """
    tid, tjd, tkd = wp.tid()

    # Get local information
    a_0, a_n = atom_ptr[tid], atom_ptr[tid + 1]
    a_i = a_0 + tjd
    num_atoms = a_n - a_0
    offset_tid = offset[a_i]

    if tjd < num_atoms:
        l_nshift = nshift[tid]
        l_cutoff = cutoff[tid]
        l_cell_T = wp.transpose(cell[tid])
        pos_i = coord[a_i]

        count = int(0)
        for k1 in range(-l_nshift[0], l_nshift[0] + 1):
            for k2 in range(-l_nshift[1], l_nshift[1] + 1):
                for k3 in range(-l_nshift[2], l_nshift[2] + 1):

                    # Get the current shift
                    shift = type(pos_i)(
                        type(l_cutoff)(k1), type(l_cutoff)(k2), type(l_cutoff)(k3)
                    )

                    cshift = l_cell_T * shift

                    # Loop over all atoms and test
                    for a_j in range(a_0, a_n):

                        if not (
                            (a_j == a_i)
                            and (cshift[0] == 0 and cshift[1] == 0 and cshift[2] == 0)
                        ):
                            dist = wp.length(coord[a_j] - pos_i + cshift)

                            if dist < l_cutoff:
                                idx = offset_tid + count
                                i[idx] = tjd
                                j[idx] = a_j - a_0
                                u[idx] = type(l_nshift)(k1, k2, k3)
                                S[idx] = cshift
                                count += 1


@wp.overload
def _build_neighbor_list(  # noqa: F811
    coord: wp.array(dtype=wp.vec3f),
    cell: wp.array(dtype=wp.mat33f),
    nshift: wp.array(dtype=wp.vec3i),
    atom_ptr: wp.array(dtype=int),
    cutoff: wp.array(dtype=wp.float32),
    offset: wp.array(dtype=int),
    i: wp.array(dtype=int),
    j: wp.array(dtype=int),
    u: wp.array(dtype=wp.vec3i),
    S: wp.array(dtype=wp.vec3f),
):  #  pragma: no cover
    """Computes neighborlist for float32 version"""
    ...


@wp.overload
def _build_neighbor_list(  # noqa: F811
    coord: wp.array(dtype=wp.vec3d),
    cell: wp.array(dtype=wp.mat33d),
    nshift: wp.array(dtype=wp.vec3i),
    atom_ptr: wp.array(dtype=int),
    cutoff: wp.array(dtype=wp.float64),
    offset: wp.array(dtype=int),
    i: wp.array(dtype=int),
    j: wp.array(dtype=int),
    u: wp.array(dtype=wp.vec3i),
    S: wp.array(dtype=wp.vec3d),
):  #  pragma: no cover
    """Computes neighborlist for float64 version"""
    ...
