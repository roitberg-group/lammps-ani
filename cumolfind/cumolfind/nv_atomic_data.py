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
from collections.abc import Sequence

import ase
import numpy as np
import torch.utils.data

from .nv_data import Data  # type: ignore


class AtomicNumberTable:
    """
    Atomic number table
    """

    def __init__(self, zs: Sequence[int]):
        self.zs = zs

    def __len__(self) -> int:
        return len(self.zs)

    def __str__(self) -> str:
        return f"AtomicNumberTable: {tuple(s for s in self.zs)}"

    def index_to_z(self, index: int) -> int:
        """
        Convert index to atomic number
        """
        return self.zs[index]

    def z_to_index(self, atomic_number: str) -> int:
        """
        Convert atomic number to index
        """
        return self.zs.index(atomic_number)


class AtomicData(Data):
    """Atomic data structure for molecular systems.

    Represents molecular systems as graphs with atomic properties and interactions.
    Inherits from Data class and adds specific functionality for atomic systems.

    Attributes
    ----------
    num_graphs : torch.Tensor
        Number of graphs in the batch
    atomic_numbers : torch.Tensor
        Atomic numbers of each atom [n_nodes]
    positions : torch.Tensor
        Cartesian coordinates [n_nodes, 3]
    forces : torch.Tensor
        Atomic forces [n_nodes, 3]
    energy : torch.Tensor
        Total energy [1]
    cell : torch.Tensor
        Unit cell vectors [3, 3]
    pbc : torch.Tensor
        Periodic boundary conditions [3]
    stress : torch.Tensor
        Stress tensor [1, 3, 3]
    virials : torch.Tensor
        Virial tensor [1, 3, 3]
    dipole : torch.Tensor
        Dipole moment [1, 3]
    charges : torch.Tensor
        Charges on each atom [n_nodes]
    edge_index : torch.Tensor
        Edge index [2, n_edges]
    node_attrs : torch.Tensor
        Node attributes [n_nodes, n_node_feats]
    shifts : torch.Tensor
        Shifts for each edge [n_edges, 3]
    unit_shifts : torch.Tensor
        Additional shifts for each edge [n_edges, 3]
    spin_multiplicity : torch.Tensor
        Spin multiplicity [1]
    dtype : torch.dtype
        Data type of the tensors
    device : torch.device
        Device on which the data is stored
    info : dict
        Additional information about the system
    """

    num_graphs: torch.Tensor
    batch: torch.Tensor
    atomic_numbers: torch.Tensor
    atomic_masses: torch.Tensor
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
    stress: torch.Tensor
    virials: torch.Tensor
    dipole: torch.Tensor
    charges: torch.Tensor
    info: dict

    def __init__(
        self,
        atomic_numbers: torch.Tensor,  # [n_nodes]
        positions: torch.Tensor,  # [n_nodes, 3]
        **kwargs: dict[str, torch.Tensor | None],
    ):

        # Validate data
        self._validate_data(atomic_numbers, positions, **kwargs)

        # Initialize defaults
        self._initialize_defaults(atomic_numbers, positions, **kwargs)

        # Store data
        super().__init__(**self._prepare_data_dict())

    def _validate_data(
        self,
        atomic_numbers: torch.Tensor,
        positions: torch.Tensor,
        **kwargs: dict[str, torch.Tensor | None],
    ) -> None:
        """Validate input tensors shapes and types.

        Parameters
        ----------
        atomic_numbers : torch.Tensor
            Atomic numbers [n_nodes]
        positions : torch.Tensor
            Cartesian coordinates [n_nodes, 3]
        atomic_masses : torch.Tensor | None
            Atomic masses [n_nodes]
        **kwargs : dict
            Additional attributes to validate

        Raises
        ------
        AssertionError
            If tensor shapes or types are invalid
        """
        # Basic shape validation
        num_nodes = positions.shape[0]
        assert atomic_numbers.shape[0] == num_nodes, (  # noqa: S101
            f"Atomic numbers shape {atomic_numbers.shape} does not match "
            f"number of nodes {num_nodes}"
        )
        assert positions.shape == (num_nodes, 3), (  # noqa: S101
            f"Positions shape {positions.shape} does not match "
            f"expected shape ({num_nodes}, 3)"
        )

        # Optional inputs validation
        atomic_masses: torch.Tensor | None = kwargs.get("atomic_masses")
        if atomic_masses is not None:
            assert atomic_masses.shape[0] == num_nodes, (  # noqa: S101
                f"Atomic masses shape {atomic_masses.shape} does not match "
                f"number of nodes {num_nodes}"
            )

        # Validate optional tensors from kwargs
        edge_index: torch.Tensor | None = kwargs.get("edge_index")
        if edge_index is not None:
            assert (  # noqa: S101
                edge_index.shape[0] == 2 and len(edge_index.shape) == 2
            ), (
                f"Edge index shape {edge_index.shape} does not match "
                "expected shape (2, n_edges)"
            )

        energy: torch.Tensor | None = kwargs.get("energy")
        if energy is not None:
            assert energy.shape == (1,), (  # noqa: S101
                f"Energy shape {energy.shape} does not match " "expected shape (1,)"
            )

        forces: torch.Tensor | None = kwargs.get("forces")
        if forces is not None:
            assert forces.shape == (num_nodes, 3), (  # noqa: S101
                f"Forces shape {forces.shape} does not match "
                f"expected shape ({num_nodes}, 3)"
            )

        cell: torch.Tensor | None = kwargs.get("cell")
        if cell is not None:
            assert cell.shape == (3, 3), (  # noqa: S101
                f"Cell shape {cell.shape} does not match " "expected shape (3, 3)"
            )

        pbc: torch.Tensor | None = kwargs.get("pbc")
        if pbc is not None:
            assert pbc.shape == (3,) or pbc.shape == (1, 3), (  # noqa: S101
                f"PBC shape {pbc.shape} does not match " "expected shape (3,) or (1, 3)"
            )

        shifts: torch.Tensor | None = kwargs.get("shifts")
        if shifts is not None:
            num_edges = shifts.shape[0]
            assert shifts.shape == (num_edges, 3), (  # noqa: S101
                f"Shifts shape {shifts.shape} does not match "
                f"expected shape ({num_edges}, 3)"
            )

        unit_shifts: torch.Tensor | None = kwargs.get("unit_shifts")
        if unit_shifts is not None:
            num_edges = unit_shifts.shape[0]
            assert unit_shifts.shape == (num_edges, 3), (  # noqa: S101
                f"Unit shifts shape {unit_shifts.shape} does not match "
                f"expected shape ({num_edges}, 3)"
            )

        node_attrs: torch.Tensor | None = kwargs.get("node_attrs")
        if node_attrs is not None:
            assert node_attrs.shape[0] == num_nodes, (  # noqa: S101
                f"Node attributes shape {node_attrs.shape} does not match "
                f"expected shape ({num_nodes}, n_node_feats)"
            )

        stress: torch.Tensor | None = kwargs.get("stress")
        if stress is not None:
            assert stress.shape == (1, 3, 3), (  # noqa: S101
                f"Stress shape {stress.shape} does not match "
                "expected shape (1, 3, 3)"
            )

        virials: torch.Tensor | None = kwargs.get("virials")
        if virials is not None:
            assert virials.shape == (1, 3, 3), (  # noqa: S101
                f"Virials shape {virials.shape} does not match "
                "expected shape (1, 3, 3)"
            )

        dipole: torch.Tensor | None = kwargs.get("dipole")
        if dipole is not None:
            assert dipole.shape == (1, 3), (  # noqa: S101
                f"Dipole shape {dipole.shape} does not match " "expected shape (1, 3)"
            )

    def _initialize_defaults(
        self,
        atomic_numbers: torch.Tensor,
        positions: torch.Tensor,
        **kwargs: dict[str, torch.Tensor | None],
    ) -> None:
        """Initialize default values for optional attributes.

        Parameters
        ----------
        atomic_numbers : torch.Tensor
            Atomic numbers [n_nodes]
        positions : torch.Tensor
            Cartesian coordinates [n_nodes, 3]
        **kwargs : dict
            Optional attributes with user-provided values
        """
        num_nodes = positions.shape[0]
        self.device = kwargs.get("device", positions.device)
        self.dtype = kwargs.get("dtype", positions.dtype)
        int_dtype = torch.int32 if self.dtype == torch.float32 else torch.int64

        # Initialize required attributes
        self.atomic_numbers = atomic_numbers
        self.positions = positions

        # Initialize optional attributes with defaults if not provided
        self.atomic_masses = kwargs.get(
            "atomic_masses",
            torch.ones((num_nodes,), device=self.device, dtype=self.dtype),
        )

        self.cell = kwargs.get(
            "cell", torch.eye(3, device=self.device, dtype=self.dtype)
        )

        self.pbc = kwargs.get(
            "pbc", torch.zeros((3,), dtype=torch.bool, device=self.device)
        )

        self.forces = kwargs.get(
            "forces", torch.zeros((num_nodes, 3), device=self.device, dtype=self.dtype)
        )

        self.energy = kwargs.get(
            "energy", torch.zeros((1,), device=self.device, dtype=self.dtype)
        )

        self.stress = kwargs.get(
            "stress", torch.zeros((1, 3, 3), device=self.device, dtype=self.dtype)
        )

        self.virials = kwargs.get(
            "virials", torch.zeros((1, 3, 3), device=self.device, dtype=self.dtype)
        )

        self.dipole = kwargs.get(
            "dipole", torch.zeros((1, 3), device=self.device, dtype=self.dtype)
        )

        self.charges = kwargs.get(
            "charges", torch.zeros((num_nodes,), device=self.device, dtype=int_dtype)
        )

        # if charges.shape == (1,):
        #     self.charges = torch.zeros(
        #         (num_nodes,), device=self.device, dtype=int_dtype
        #     )
        #     self.charges[0] = charges[0]
        # else:
        #     self.charges = charges

        self.spin_multiplicity = kwargs.get(
            "spin_multiplicity", torch.ones((1,), device=self.device, dtype=int_dtype)
        )

        self.edge_index = kwargs.get(
            "edge_index", torch.zeros((2, 0), device=self.device, dtype=torch.int64)
        )

        self.node_attrs = kwargs.get(
            "node_attrs",
            torch.zeros((num_nodes, 0), device=self.device, dtype=self.dtype),
        )

        self.shifts = kwargs.get(
            "shifts", torch.zeros((0, 3), device=self.device, dtype=self.dtype)
        )

        self.unit_shifts = kwargs.get(
            "unit_shifts", torch.zeros((0, 3), device=self.device, dtype=self.dtype)
        )

        # Store additional info
        self.info = kwargs.get("info", {})

    def _prepare_data_dict(self) -> dict:
        """Prepare dictionary of attributes for parent class.

        Returns
        -------
        dict
            Dictionary containing all attributes to be stored in Data class

        Notes
        -----
        This method aggregates all attributes into a single dictionary
        that will be passed to the parent Data class constructor.
        """
        return {
            "num_nodes": len(self.atomic_numbers),
            "atomic_numbers": self.atomic_numbers,
            "atomic_masses": self.atomic_masses,
            "positions": self.positions,
            "cell": self.cell,
            "pbc": self.pbc,
            "edge_index": self.edge_index,
            "node_attrs": self.node_attrs,
            "shifts": self.shifts,
            "unit_shifts": self.unit_shifts,
            "forces": self.forces,
            "energy": self.energy,
            "stress": self.stress,
            "virials": self.virials,
            "dipole": self.dipole,
            "charges": self.charges,
            "spin_multiplicity": self.spin_multiplicity,
            "info": self.info,
        }

    @classmethod
    def from_atoms(
        cls,
        atoms: ase.Atoms,
        energy_key: str = "energy",
        forces_key: str = "forces",
        stress_key: str = "stress",
        virials_key: str = "virials",
        dipole_key: str = "dipole",
        charges_key: str = "charges",
        spin_multiplicity_key: str = "spin_multiplicity",
        device: torch.device = "cuda",
        dtype: torch.dtype = torch.float32,
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
        atomic_numbers = torch.as_tensor(
            atoms.arrays["numbers"], device=device, dtype=torch.int32
        )
        positions = torch.as_tensor(
            atoms.arrays["positions"], device=device, dtype=dtype
        )
        pbc = torch.as_tensor(atoms.get_pbc().reshape(1, 3), device=device)
        cell = torch.as_tensor(
            np.array(atoms.get_cell(complete=True)), device=device, dtype=dtype
        )

        # Get info from ase.Atoms
        energy = torch.as_tensor(
            atoms.info.get(energy_key, [0.0]), device=device, dtype=dtype
        )  # eV
        forces = torch.as_tensor(
            atoms.arrays.get(
                forces_key,
                torch.zeros((len(atomic_numbers), 3), device=device, dtype=dtype),
            ),
            device=device,
            dtype=dtype,
        )  # eV / Ang
        stress = torch.as_tensor(
            atoms.info.get(stress_key, torch.zeros((3, 3), device=device, dtype=dtype)),
            device=device,
            dtype=dtype,
        )  # eV / Ang ^ 3
        virials = torch.as_tensor(
            atoms.info.get(
                virials_key, torch.zeros((3, 3), device=device, dtype=dtype)
            ),
            device=device,
            dtype=dtype,
        )
        dipole = torch.as_tensor(
            atoms.info.get(dipole_key, torch.zeros((1, 3), device=device, dtype=dtype)),
            device=device,
            dtype=dtype,
        )  # Debye
        charges = torch.as_tensor(
            atoms.arrays.get(
                charges_key,
                torch.zeros((len(atomic_numbers),), device=device, dtype=torch.int32),
            ),
            device=device,
            dtype=torch.int32,
        )
        spin_multiplicity = torch.as_tensor(
            atoms.info.get(
                spin_multiplicity_key,
                torch.ones((1,), device=device, dtype=torch.int32),
            ),
            device=device,
            dtype=torch.int32,
        )

        stress = voigt_to_matrix(stress).unsqueeze(0)
        virials = voigt_to_matrix(virials).unsqueeze(0)

        node_attrs = None
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
            cell=cell,
            pbc=pbc,
            node_attrs=node_attrs,  # type: ignore
            forces=forces,
            energy=energy,
            stress=stress,
            virials=virials,
            dipole=dipole,
            charges=charges,
            spin_multiplicity=spin_multiplicity,
            info=atoms.info,
        )

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

    def to(
        self, device: torch.device | None = None, dtype: torch.dtype | None = None
    ) -> "AtomicData":
        """Convert AtomicData object to a different device and dtype.

        Parameters
        ----------
        device : torch.device
        dtype : torch.dtype

        Returns
        -------
        AtomicData
        """
        if device is None:
            device = self.device
        if dtype is None:
            dtype = self.dtype

        if dtype.itemsize == 4:
            int_dtype = torch.int32
            float_dtype = torch.float32
        else:
            int_dtype = torch.int64
            float_dtype = torch.float64

        self.device = device
        self.dtype = dtype

        for key in self.keys:
            if key in ["device", "dtype"]:
                continue
            elif key in ["edge_index"]:
                self[key] = self[key].to(device)  # Don't change edge_index dtype.
            elif isinstance(self[key], torch.Tensor):
                dtype = float_dtype if "float" in str(self[key].dtype) else int_dtype
                self[key] = self[key].to(device, dtype)
            elif isinstance(key, dict):
                for k, v in key.items():
                    if isinstance(v, torch.Tensor):
                        dtype = float_dtype if "float" in str(v.dtype) else int_dtype
                        key[k] = v.to(device, dtype)

        return self


def to_one_hot(indices: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Generates one-hot encoding
    """
    shape = indices.shape[:-1] + (num_classes,)
    oh = torch.zeros(shape, device=indices.device).view(shape)

    # scatter_ is the in-place version of scatter
    oh.scatter_(dim=-1, index=indices, value=1)

    return oh.view(*shape)


def voigt_to_matrix(t: torch.Tensor) -> torch.Tensor:
    """
    Convert voigt notation to matrix notation
    """
    if t.shape == (3, 3):
        return t
    if t.shape == (6,):
        return torch.tensor(
            [
                [t[0], t[5], t[4]],
                [t[5], t[1], t[3]],
                [t[4], t[3], t[2]],
            ],
            dtype=t.dtype,
        )
    if t.shape == (9,):
        return t.view(3, 3)

    raise ValueError(
        f"Stress tensor must be of shape (6,) or (3, 3), or (9,) but has shape {t.shape}"
    )


def atomic_numbers_to_indices(
    atomic_numbers: np.ndarray, z_table: AtomicNumberTable
) -> np.ndarray:
    """
    Convert atomic numbers to indices
    """
    to_index_fn = np.vectorize(z_table.z_to_index)
    return to_index_fn(atomic_numbers)
