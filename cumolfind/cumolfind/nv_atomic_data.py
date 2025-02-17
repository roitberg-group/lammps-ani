# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

###########################################################################################
# Atomic Data Class for handling molecules as graphs
# Authors: Ilyes Batatia, Gregor Simm
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################
import warnings

import ase
import numpy as np
import torch.utils.data
import torch_geometric
from typing import Sequence

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    warnings.simplefilter("ignore", category=FutureWarning)
    # from mace.data.neighborhood import get_neighborhood
    # from mace.tools import (
    #     AtomicNumberTable,
    #     # atomic_numbers_to_indices,
    #     # to_one_hot,
    #     torch_geometric,
    #     voigt_to_matrix,
    # )

class AtomicNumberTable:
    def __init__(self, zs: Sequence[int]):
        self.zs = zs

    def __len__(self) -> int:
        return len(self.zs)

    def __str__(self):
        return f"AtomicNumberTable: {tuple(s for s in self.zs)}"

    def index_to_z(self, index: int) -> int:
        return self.zs[index]

    def z_to_index(self, atomic_number: str) -> int:
        return self.zs.index(atomic_number)


class AtomicData(torch_geometric.data.Data):
    """Atomic data, primiarily for use with MACE"""

    num_graphs: torch.Tensor
    batch: torch.Tensor
    atomic_numbers: torch.Tensor
    edge_index: torch.Tensor
    node_attrs: torch.Tensor
    edge_vectors: torch.Tensor
    edge_lengths: torch.Tensor
    positions: torch.Tensor
    shifts: torch.Tensor
    unit_shifts: torch.Tensor
    cell: torch.Tensor
    pbc: torch.Tensor
    forces: torch.Tensor
    energy: torch.Tensor
    # stress: torch.Tensor
    # virials: torch.Tensor
    # dipole: torch.Tensor
    charges: torch.Tensor
    info: dict

    def __init__(
        self,
        atomic_numbers: torch.Tensor,  # [n_nodes]
        positions: torch.Tensor,  # [n_nodes, 3]
        cell: torch.Tensor | None,  # [3,3]
        pbc: torch.Tensor | None,  # [3,]
        forces: torch.Tensor | None,  # [n_nodes, 3]
        energy: torch.Tensor | None,  # [1, ]
        # stress: torch.Tensor | None,  # [1,3,3]
        # virials: torch.Tensor | None,  # [1,3,3]
        # dipole: torch.Tensor | None,  # [1, 3]
        charges: torch.Tensor | None,  # [n_nodes, ]
        edge_index: torch.Tensor | None = None,  # [2, n_edges]
        node_attrs: torch.Tensor | None = None,  # [n_nodes, n_node_feats]
        shifts: torch.Tensor | None = None,  # [n_edges, 3],
        unit_shifts: torch.Tensor | None = None,  # [n_edges, 3]
        spin_multiplicity: torch.Tensor | None = None,  # [1,]
        info: dict = {},
    ):
        # Check shapes
        num_nodes = positions.shape[0]
        assert atomic_numbers.shape[0] == num_nodes  # noqa: S101
        assert positions.shape == (num_nodes, 3)  # noqa : S101
        assert (  # noqa : S101
            edge_index is None
            or edge_index.shape[0] == 2
            and len(edge_index.shape) == 2
        )
        assert shifts is None or shifts.shape[1] == 3  # noqa : S101
        assert unit_shifts is None or unit_shifts.shape[1] == 3  # noqa : S101
        assert node_attrs is None or len(node_attrs.shape) == 2  # noqa : S101
        assert cell is None or cell.shape == (3, 3)  # noqa : S101
        assert pbc is None or pbc.shape == (3,) or pbc.shape == (1, 3)  # noqa : S101
        assert forces is None or forces.shape == (num_nodes, 3)  # noqa : S101
        assert energy is None or len(energy.shape) == 1  # noqa : S101
        # assert stress is None or stress.shape == (1, 3, 3)  # noqa : S101
        # assert virials is None or virials.shape == (1, 3, 3)  # noqa : S101
        # assert dipole is None or dipole.shape[-1] == 3  # noqa : S101

        if (charges is not None) and charges.shape == (1,):
            charge = charges[0]
            charges = torch.zeros((num_nodes,), device=positions.device)
            charges[0] = charge

        assert charges is None or charges.shape == (num_nodes,)  # noqa : S101

        assert spin_multiplicity is None or spin_multiplicity.shape == (  # noqa: S101
            1,
        )

        # Aggregate data
        data = {
            "num_nodes": num_nodes,
            "atomic_numbers": atomic_numbers,
            "edge_index": edge_index,
            "positions": positions,
            "shifts": shifts,
            "unit_shifts": unit_shifts,
            "cell": cell,
            "pbc": pbc,
            "node_attrs": node_attrs,
            "forces": forces,
            "energy": energy,
            # "stress": stress,
            # "virials": virials,
            # "dipole": dipole,
            "charges": charges,
            "spin_multiplicity": spin_multiplicity,
            "info": info,
        }
        super().__init__(**data)

    @classmethod
    def from_atoms(
        cls,
        atoms: ase.Atoms,
        energy_key: str = "energy",
        forces_key: str = "forces",
        # stress_key: str = "stress",
        # virials_key: str = "virials",
        # dipole_key: str = "dipole",
        charges_key: str = "charges",
        spin_multiplicity_key: str = "spin_multiplicity",
        device: torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
        compute_graph_embedding: bool = True,
        cutoff: float = 5.0,
        z_table: AtomicNumberTable | None = None,
    ) -> "AtomicData":
        """Creates AtomicData from a ase.Atoms data structure.

        Parameters
        ----------
        atoms : ase.Atoms

        Returns
        -------
        AtomicData
        """

        # Get base components from ase.Atoms object
        atomic_numbers = torch.as_tensor(atoms.arrays["numbers"], device=device, dtype = torch.int32)
        positions = torch.as_tensor(atoms.arrays["positions"], device=device, dtype=dtype)
        pbc = torch.as_tensor(atoms.get_pbc().reshape(1, 3), device=device)
        cell = torch.as_tensor(np.array(atoms.get_cell(complete=True)), device=device, dtype=dtype)

        # Get info from ase.Atoms
        energy = atoms.info.get(energy_key, None)  # eV
        forces = atoms.arrays.get(forces_key, None)  # eV / Ang
        # stress = atoms.info.get(stress_key, None)  # eV / Ang ^ 3
        # virials = atoms.info.get(virials_key, None)
        # dipole = atoms.info.get(dipole_key, None)  # Debye
        charges = atoms.arrays.get(charges_key, None)
        spin_multiplicity = atoms.info.get(spin_multiplicity_key, None)

        # fill in missing quantities but set their weight to 0.0
        if energy is None:
            energy = torch.as_tensor([0.0], device=device, dtype=dtype)
        if forces is None:
            forces = torch.zeros_like(positions)
        # if stress is None:
        #     stress = torch.zeros(6, device=device, dtype=dtype)
        if charges is None:
            charges = torch.as_tensor(
                atoms.info.get(charges_key, np.zeros((len(atomic_numbers),))),
                device=device,
                dtype = torch.int32
            )
        if spin_multiplicity is None:
            spin_multiplicity = torch.ones((1,), device=device, dtype=torch.int32)

        # stress = voigt_to_matrix(stress).unsqueeze(0).to(device)

        # if virials is None:
        #     virials = torch.zeros((3, 3), device=device, dtype=dtype)

        # virials = voigt_to_matrix(virials).unsqueeze(0)

        # if dipole is None:
        #     dipole = torch.zeros((1, 3), device=device, dtype = torch.int32)

        edge_index, shifts, unit_shifts, node_attrs = None, None, None, None
        # if compute_graph_embedding:
        #     if z_table is None:
        #         raise ValueError(
        #             "Compute Graph Embedding Option Requires an Atomic Z Table."
        #         )
        #     edge_index, shifts, unit_shifts, _ = get_neighborhood(
        #         positions=atoms.arrays["positions"],
        #         cutoff=cutoff,
        #         pbc=atoms.get_pbc(),
        #         cell=np.array(atoms.get_cell(complete=True)),
        #     )
        #     edge_index = torch.tensor(edge_index, dtype=torch.long, device=device)
        #     shifts = torch.tensor(
        #         shifts, dtype=torch.get_default_dtype(), device=device
        #     )
        #     unit_shifts = torch.tensor(
        #         unit_shifts, dtype=torch.get_default_dtype(), device=device
        #     )

        if z_table is not None:
            indices = torch.as_tensor(
                atomic_numbers_to_indices(atoms.arrays["numbers"], z_table=z_table),
                device=device,
            )
            node_attrs = to_one_hot(
                indices.unsqueeze(-1),
                num_classes=len(z_table),
            )

        return cls(
            atomic_numbers=atomic_numbers,
            positions=positions,
            edge_index=edge_index,
            shifts=shifts,
            unit_shifts=unit_shifts,
            cell=cell,
            pbc=pbc,
            node_attrs=node_attrs,
            forces=forces,
            energy=energy,
            # stress=stress,
            # virials=virials,
            # dipole=dipole,
            charges=charges,
            spin_multiplicity=spin_multiplicity,
            info=atoms.info,
        )

    def to_one_hot(indices: torch.Tensor, num_classes: int) -> torch.Tensor:
        """
        Generates one-hot encoding with <num_classes> classes from <indices>
        :param indices: (N x 1) tensor
        :param num_classes: number of classes
        :param device: torch device
        :return: (N x num_classes) tensor
        """
        shape = indices.shape[:-1] + (num_classes,)
        oh = torch.zeros(shape, device=indices.device).view(shape)

        # scatter_ is the in-place version of scatter
        oh.scatter_(dim=-1, index=indices, value=1)

        return oh.view(*shape)

    def atomic_numbers_to_indices(
        atomic_numbers: np.ndarray, z_table: AtomicNumberTable
    ) -> np.ndarray:
        to_index_fn = np.vectorize(z_table.z_to_index)
        return to_index_fn(atomic_numbers)

    def to_atoms(
        self,
    ) -> ase.Atoms:
        """Convert AtomicData object to ASE Atoms object.

        Returns
        -------
        ase.Atoms
        """
        info = dict(
            energy=self.energy.cpu().numpy(),
            forces=self.forces.cpu().numpy(),
            stress=self.stress.cpu().numpy(),
            virials=self.virials.cpu().numpy(),
            dipole=self.dipole.cpu().numpy(),
        )
        return ase.Atoms(
            numbers=self.atomic_numbers.cpu().numpy(),
            positions=self.positions.detach().cpu().numpy(),
            cell=self.cell.cpu().numpy(),
            pbc=self.pbc.cpu().numpy(),
            charges=self.charges.cpu().numpy(),
            info=info | self.info,
        )

