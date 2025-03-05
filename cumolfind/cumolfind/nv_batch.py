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
# type: ignore
import re
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from torch import Tensor

from .nv_atomic_data import AtomicData
from .nv_data import Data

try:
    from typing import Self  # type: ignore
except ImportError:
    from typing_extensions import Self



class Batch(Data):
    """A batch of molecular graphs combined into one large graph.

    Combines multiple AtomicData instances into a single graph with batch
    indices tracking which atoms belong to which molecule.

    Attributes
    ----------
    batch : torch.Tensor
        Batch assignment for each node
    ptr : torch.Tensor
        Cumulative nodes per graph
    """

    def __init__(
        self,
        batch: torch.Tensor | None = None,
        ptr: torch.Tensor | None = None,
        **kwargs: dict,
    ):
        super().__init__(**kwargs)

        for key, item in kwargs.items():
            if key == "num_nodes":
                self.__num_nodes__ = item
            else:
                self[key] = item

        self.batch = batch
        self.ptr = ptr
        self.__data_class__ = AtomicData
        self.__slices__: list | None = None
        self.__cumsum__: list | None = None
        self.__cat_dims__: list | None = None
        self.__num_nodes_list__: list | None = None
        self.__num_graphs__: int | None = None

        self._excluded_keys = ["batch", "ptr", "device", "dtype"]

    @classmethod
    def from_data_list(
        cls,
        data_list: list[AtomicData],
        follow_batch: list[str] = [],
        exclude_keys: list[str] = [],
    ) -> "Batch":
        """Constructs a batch from a list of AtomicData objects.

        Parameters
        ----------
        data_list : list[AtomicData]
            List of AtomicData objects to batch
        follow_batch : list[str]
            Keys to track batch assignments for
        exclude_keys : list[str]
            Keys to exclude from batching

        Returns
        -------
        Batch
            Batched data structure
        """
        _excluded_keys = ["batch", "ptr", "device", "dtype"]
        excluded_keys = set(_excluded_keys + exclude_keys)
        batch = cls._initialize_batch(data_list, excluded_keys)
        device = cls._validate_devices(data_list)

        # Initialize tracking structures
        slices, cumsum, cat_dims = cls._initialize_tracking_structures(
            data_list[0], excluded_keys
        )
        num_nodes_list = []
        print("device in from data list", device)

        # Process each data object
        for i, data in enumerate(data_list):
            cls._process_data_object(
                i,
                data,
                batch,
                slices,
                cumsum,
                cat_dims,
                follow_batch,
                num_nodes_list,
                device,
            )

        # Finalize batch
        cls._finalize_batch(batch, data_list, slices, cumsum, cat_dims, num_nodes_list)

        return batch.contiguous()

    @classmethod
    def _initialize_batch(
        cls, data_list: list[AtomicData], excluded_keys: list[str]
    ) -> "Batch":
        """Initialize empty batch structure.

        Parameters
        ----------
        data_list : list[AtomicData]
            List of data objects to batch
        exclude_keys : list[str]
            Keys to exclude from batching

        Returns
        -------
        Batch
            Empty initialized batch
        """
        keys = list(set(data_list[0].keys) - excluded_keys)
        assert (  # noqa: S101
            "batch" not in keys and "ptr" not in keys
        ), "batch and ptr must not be present"

        batch = cls()
        # Copy metadata from first data object
        for key in data_list[0].__dict__.keys():
            if key[:2] != "__" and key[-2:] != "__":
                batch[key] = None

        batch.__num_graphs__ = len(data_list)
        batch.__data_class__ = data_list[0].__class__

        # Initialize lists for each key
        for key in keys + ["batch"]:
            batch[key] = []
        batch["ptr"] = [0]

        return batch

    @staticmethod
    def _validate_devices(data_list: list[AtomicData]) -> torch.device:
        """Ensure all data objects are on same device.

        Parameters
        ----------
        data_list : list[AtomicData]
            List of data objects to validate

        Returns
        -------
        torch.device
            Common device for all objects

        Raises
        ------
        AssertionError
            If devices don't match
        """
        device = data_list[0].device
        assert all(  # noqa: S101
            data.device == device for data in data_list
        ), "Device mismatch"
        return device

    @staticmethod
    def _initialize_tracking_structures(
        data: AtomicData, excluded_keys: list[str]
    ) -> tuple[dict, dict, dict]:
        """Initialize structures for tracking batch assembly.

        Parameters
        ----------
        data : AtomicData
            First data object (used as reference)
        excluded_keys : list[str]
            Keys to exclude

        Returns
        -------
        tuple[dict, dict, dict]
            Slices, cumsum, and cat_dims dictionaries
        """
        keys = list(set(data.keys) - excluded_keys)
        slices = {key: [0] for key in keys}
        cumsum = {key: [0] for key in keys}
        cat_dims = {}
        return slices, cumsum, cat_dims

    @classmethod
    def _process_data_object(
        cls,
        idx: int,
        data: AtomicData,
        batch: "Batch",
        slices: dict,
        cumsum: dict,
        cat_dims: dict,
        follow_batch: list[str],
        num_nodes_list: list,
        device: torch.device,
    ) -> None:
        """Process single data object for batching.

        Parameters
        ----------
        idx : int
            Index of current data object
        data : AtomicData
            Data object to process
        batch : Batch
            Batch being constructed
        slices : dict
            Slicing information
        cumsum : dict
            Cumulative sums
        cat_dims : dict
            Concatenation dimensions
        follow_batch : list[str]
            Keys to track batch assignments for
        num_nodes_list : list
            List of number of nodes
        device : torch.device
            Device for tensors
        """
        # Process each key in the data object
        for key in slices.keys():
            item = data[key]
            cls._update_slices_and_cumsum(key, item, slices, cumsum, cat_dims, data)
            batch[key].append(item)

            if key in follow_batch:
                cls._handle_follow_batch(key, item, idx, batch, device)

        # Track number of nodes
        if hasattr(data, "__num_nodes__"):
            num_nodes_list.append(data.__num_nodes__)
        else:
            num_nodes_list.append(None)

        # Update batch assignments
        num_nodes = data.num_nodes
        if num_nodes is not None:
            item = torch.full((num_nodes,), idx, dtype=torch.long, device=device)
            batch.batch.append(item)
            batch.ptr.append(batch.ptr[-1] + num_nodes)

    @staticmethod
    def _handle_follow_batch(
        key: str,
        item: Any,
        batch_idx: int,
        batch: "Batch",
        device: torch.device,
    ) -> None:
        """Track batch assignments for specific attributes.

        Creates batch assignment tensors for specified keys to track which
        elements belong to which batch entry.

        Parameters
        ----------
        key : str
            The attribute key being tracked
        item : Any
            The attribute value
        batch_idx : int
            Current batch index
        batch : Batch
            Batch being constructed
        device : torch.device
            Device for tensors

        Examples
        --------
        If tracking edge indices across batches:
        >>> data1 = AtomicData(edge_index=[[0,1], [1,0]])
        >>> data2 = AtomicData(edge_index=[[0,1], [1,0]])
        >>> batch = Batch.from_data_list([data1, data2], follow_batch=['edge_index'])
        >>> batch.edge_index_batch
        tensor([0, 0, 1, 1])  # Shows which edges belong to which graph
        """
        if isinstance(item, Tensor) and item.dtype != torch.bool:
            # Get number of elements to track
            if item.dim() == 0:
                batch[f"{key}_batch"] = []
            else:
                batch[f"{key}_batch"].append(
                    torch.full(
                        (item.size(0),),  # Size of first dimension
                        batch_idx,
                        dtype=torch.long,
                        device=device,
                    )
                )

    @staticmethod
    def _update_slices_and_cumsum(
        key: str,
        item: Any,
        slices: dict,
        cumsum: dict,
        cat_dims: dict,
        data: AtomicData,
    ) -> None:
        """Update slicing and cumulative sum information.

        Parameters
        ----------
        key : str
            Current key being processed
        item : Any
            Item from data object
        slices : dict
            Slicing information
        cumsum : dict
            Cumulative sums
        cat_dims : dict
            Concatenation dimensions
        data : AtomicData
            Current data object
        """
        # Get size and cat_dim
        size = 1
        cat_dim = data.__cat_dim__(key, item)

        if isinstance(item, Tensor) and item.dim() == 0:
            cat_dim = None
        cat_dims[key] = cat_dim

        # Handle different item types
        if isinstance(item, Tensor):
            size = item.size(cat_dim) if cat_dim is not None else 1

        slices[key].append(size + slices[key][-1])

        # Update cumsum
        inc = data.__inc__(key, item)
        if isinstance(inc, (tuple, list)):
            inc = torch.tensor(inc)
        cumsum[key].append(inc + cumsum[key][-1])

    @classmethod
    def _finalize_batch(
        cls,
        batch: "Batch",
        data_list: list[AtomicData],
        slices: dict,
        cumsum: dict,
        cat_dims: dict,
        num_nodes_list: list,
    ) -> None:
        """Finalize batch by concatenating tensors and setting attributes.

        Parameters
        ----------
        batch : Batch
            Batch to finalize
        data_list : list[AtomicData]
            Original data list
        slices : dict
            Slicing information
        cumsum : dict
            Cumulative sums
        cat_dims : dict
            Concatenation dimensions
        num_nodes_list : list
            List of number of nodes
        """

        # Concatenate all tensors
        ref_data = data_list[0]
        device = ref_data.device
        dtype = ref_data.dtype

        # Set batch attributes
        batch.batch = None if len(batch.batch) == 0 else torch.cat(batch.batch, dim=0)
        batch.ptr = (
            None
            if len(batch.ptr) == 1
            else torch.tensor(batch.ptr, device=device, dtype=torch.int32)
        )
        batch.__slices__ = slices
        batch.__cumsum__ = cumsum
        batch.__cat_dims__ = cat_dims
        batch.__num_nodes_list__ = num_nodes_list
        batch.device = device
        batch.dtype = dtype

        for key in batch.keys:
            if key in ["ptr", "batch", "device", "dtype"]:
                continue
            items = batch[key]
            item = items[0]
            cat_dim = ref_data.__cat_dim__(key, item)
            cat_dim = 0 if cat_dim is None else cat_dim

            if isinstance(item, Tensor):
                batch[key] = torch.cat(items, cat_dim)
            elif isinstance(item, (int, float)):
                batch[key] = torch.tensor(items, device=device)

    def add_key(
        self,
        key: str,
        values: list[torch.Tensor],
        follow_batch: bool = False,
        cat_dim: int | None = None,
        overwrite: bool = False,
    ) -> None:
        """Add a new key-value pair to the batch.

        Parameters
        ----------
        key : str
            Name of the new attribute to add
        values : list[torch.Tensor]
            List of values for each graph in the batch
        follow_batch : bool, default=False
            Whether to track batch assignments for this key
        cat_dim : int | None, default=None
            Dimension along which to concatenate tensors. If None,
            will be inferred from the first tensor.
        overwrite : bool, default=False
            Whether to overwrite if key already exists

        Raises
        ------
        ValueError
            If key exists and overwrite=False
            If number of values doesn't match batch size
            If tensors have inconsistent shapes/devices
        """
        # Check if key already exists
        if key in self and not overwrite:
            raise ValueError(
                f"Key '{key}' already exists in batch. "
                "Set overwrite=True to replace existing values."
            )

        # Clean up existing tracking if overwriting
        if key in self and overwrite:
            self._cleanup_existing_key(key)

        if len(values) != self.num_graphs:
            raise ValueError(
                f"Number of values ({len(values)}) must match "
                f"number of graphs in batch ({self.num_graphs})"
            )

        # Initialize tracking for the new key
        self.__slices__[key] = [0]
        self.__cumsum__[key] = [0]

        # Determine concatenation dimension if not provided
        if cat_dim is None and isinstance(values[0], torch.Tensor):
            if values[0].dim() == 0:
                cat_dim = None
            else:
                cat_dim = 0
        self.__cat_dims__[key] = cat_dim

        # Process values
        items = []
        device = self.device

        for i, value in enumerate(values):
            # Ensure consistent device
            if isinstance(value, torch.Tensor):
                value = value.to(device)

            # Increase values by cumsum
            cum = self.__cumsum__[key][-1]
            if isinstance(value, torch.Tensor):
                if not isinstance(cum, int) or cum != 0:
                    value = value + cum
            elif isinstance(value, (int, float)):
                value = value + cum

            # Update slices and cumsum
            if isinstance(value, torch.Tensor):
                size = value.size(cat_dim) if cat_dim is not None else 1
            else:
                size = 1
            self.__slices__[key].append(size + self.__slices__[key][-1])

            # Handle increments (usually for indices)
            if hasattr(self.__data_class__, "__inc__"):
                inc = (
                    self.__num_nodes_list__[i]
                    if bool(re.search("(index|face)", key))
                    else 0
                )
                if isinstance(inc, (tuple, list)):
                    inc = torch.tensor(inc)
            else:
                inc = 0
            self.__cumsum__[key].append(inc + self.__cumsum__[key][-1])

            # Store value
            items.append(value)

            # Handle batch tracking if requested
            if (
                follow_batch
                and isinstance(value, torch.Tensor)
                and value.dtype != torch.bool
            ):
                if value.dim() == 0:
                    self[f"{key}_batch"] = []
                else:
                    if f"{key}_batch" not in self:
                        self[f"{key}_batch"] = []
                    self[f"{key}_batch"].append(
                        torch.full((value.size(0),), i, dtype=torch.long, device=device)
                    )

        # Concatenate values
        if isinstance(items[0], torch.Tensor):
            if cat_dim is None:
                self[key] = torch.stack(items)
            else:
                self[key] = torch.cat(items, dim=cat_dim)
        else:
            self[key] = items

        # Finalize batch tracking
        if follow_batch and f"{key}_batch" in self:
            self[f"{key}_batch"] = torch.cat(self[f"{key}_batch"], dim=0)

    def _cleanup_existing_key(self, key: str) -> None:
        """Remove all tracking information for an existing key.

        Parameters
        ----------
        key : str
            Key to cleanup
        """
        # Remove main value
        del self[key]

        # Remove from tracking structures
        if hasattr(self, "__slices__") and key in self.__slices__:
            del self.__slices__[key]
        if hasattr(self, "__cumsum__") and key in self.__cumsum__:
            del self.__cumsum__[key]
        if hasattr(self, "__cat_dims__") and key in self.__cat_dims__:
            del self.__cat_dims__[key]

        # Remove batch tracking if it exists
        batch_key = f"{key}_batch"
        if batch_key in self:
            del self[batch_key]

    def index_select(
        self, idx: int | slice | Tensor | list[int] | np.ndarray | Sequence[int]
    ) -> Self:
        """Select a subset of the current batch structure.

        Parameters
        ----------
        idx :
            Can be one of:
            - int: Single index
            - slice: Python slice object
            - Tensor: Boolean or integer tensor
            - list[int]: List of indices
            - np.ndarray: Boolean or integer array
            - Sequence[int]: Any sequence of integers

        Returns
        -------
        Batch
            New Batch data structure containing selected graphs

        Raises
        ------
        IndexError
            If index type is not supported
        RuntimeError
            If batch wasn't created using from_data_list
        """

        # Convert different index types to list of integers
        idx = self._normalize_index(idx)

        # Validate batch structure
        if self.__slices__ is None:
            raise RuntimeError(
                "Cannot reconstruct data list from batch because the batch "
                "object was not created using `Batch.from_data_list()`."
            )

        # Initialize new batch
        data = self._initialize_selected_batch(idx)

        # Process each key in the batch
        self._process_keys_for_selection(idx, data)

        # Finalize and return
        data.device = self.device
        data.dtype = self.dtype
        return data

    def _normalize_index(
        self, idx: int | slice | Tensor | list[int] | np.ndarray | Sequence[int]
    ) -> list[int]:
        """Convert various index types to list of integers.

        Parameters
        ----------
        idx : int | slice | Tensor | list[int] | np.ndarray | Sequence[int]
            Input index of various types

        Returns
        -------
        list[int]
            Normalized list of integer indices

        Raises
        ------
        IndexError
            If index type is not supported
        """
        if isinstance(idx, slice):
            idx = list(range(self.num_graphs)[idx])

        elif isinstance(idx, Tensor):
            if idx.dtype == torch.long:
                idx = idx.flatten().tolist()
            elif idx.dtype == torch.bool:
                idx = idx.flatten().nonzero(as_tuple=False).flatten().tolist()
            else:
                raise IndexError(f"Tensor index must be long or bool, got {idx.dtype}")

        elif isinstance(idx, np.ndarray):
            if idx.dtype == np.int64:
                idx = idx.flatten().tolist()
            elif idx.dtype == np.bool_:
                idx = idx.flatten().nonzero()[0].flatten().tolist()
            else:
                raise IndexError(f"NumPy index must be int64 or bool, got {idx.dtype}")

        elif isinstance(idx, int):
            idx = [idx]

        elif isinstance(idx, Sequence) and not isinstance(idx, str):
            idx = list(idx)

        else:
            raise IndexError(
                f"Only integers, slices (':'), list, tuples, torch.tensor and "
                f"np.ndarray of dtype long or bool are valid indices (got "
                f"'{type(idx).__name__}')"
            )

        if len(idx) == 0:
            raise IndexError("Index is empty")

        # Handle negative indices
        idx = [self.num_graphs + i if i < 0 else i for i in idx]
        return idx

    def _initialize_selected_batch(self, idx: list[int]) -> "Batch":
        """Initialize new batch for selected indices.

        Parameters
        ----------
        idx : list[int]
            List of indices to select

        Returns
        -------
        Batch
            Initialized batch structure
        """
        data = Batch()
        data.__data_class__ = AtomicData
        data.ptr = [0]
        data.batch = []
        data.__num_graphs__ = len(idx)

        # Initialize tracking structures
        keys = self.__slices__.keys()
        data.__slices__ = {key: [0] for key in keys}
        data.__cumsum__ = {key: [0] for key in keys}
        data.__cat_dims__ = {}
        data.__num_nodes_list__ = []

        return data

    def _process_keys_for_selection(self, idx: list[int], data: "Batch") -> None:
        """Process each key for the selected indices.

        Parameters
        ----------
        idx : list[int]
            List of indices to select
        data : Batch
            Batch being constructed
        """
        device = self.batch.device

        # Process each key
        for key in self.__slices__.keys():
            data[key] = self._select_key_items(
                key, idx, data.__slices__[key], data.__cumsum__[key]
            )
            data.__cat_dims__[key] = self.__cat_dims__[key]

        # Process node information
        for i, index in enumerate(idx):
            num_nodes = self.__num_nodes_list__[index]
            data.__num_nodes_list__.append(num_nodes)

            if num_nodes is not None:
                data.batch.append(
                    torch.full((num_nodes,), i, dtype=torch.long, device=device)
                )
                data.ptr.append(data.ptr[-1] + num_nodes)

        # Finalize batch and ptr tensors
        data.batch = torch.cat(data.batch, 0) if data.batch else None
        data.ptr = torch.tensor(data.ptr, device=device) if len(data.ptr) > 1 else None
        data["num_nodes"] = sum(data.__num_nodes_list__)

    def _select_key_items(
        self, key: str, idx: list[int], slices: list[int], cumsum: list[int]
    ) -> Any:
        """Select items for a specific key.

        Parameters
        ----------
        key : str
            Key to process
        idx : list[int]
            Indices to select
        slices : list[int]
            Slice information for new batch
        cumsum : list[int]
            Cumsum information for new batch

        Returns
        -------
        Any
            Selected and processed items
        """
        item = self[key]
        items = []

        if self.__cat_dims__[key] is None:
            # Item was concatenated along batch dimension
            return item[idx]

        # Process each selected index
        for index in idx:
            # Extract item slice
            start = self.__slices__[key][index]
            end = self.__slices__[key][index + 1]

            if isinstance(item, Tensor):
                dim = self.__cat_dims__[key]
                items.append(item.narrow(dim, start, end - start))
            else:
                temp_item = item[start:end]
                items.append(temp_item[0] if len(temp_item) == 1 else temp_item)

            slices.append(slices[-1] + end - start)

            # Update cumsum
            cum_start = self.__cumsum__[key][index]
            cum_end = self.__cumsum__[key][index + 1]

            if isinstance(items[-1], Tensor) and items[-1].dtype != torch.bool:
                if cum_start != 0:
                    items[-1] = items[-1] - cum_start
                    items[-1] = items[-1] + cumsum[-1]
            elif isinstance(item, (int, float)):
                items[-1] = items[-1] - cum_start
                items[-1] = items[-1] + cumsum[-1]

            cumsum.append(cumsum[-1] + cum_end - cum_start)

        # Combine items
        if isinstance(item, Tensor):
            return torch.cat(items, self.__cat_dims__[key]).contiguous()
        elif isinstance(item, np.ndarray):
            return torch.tensor(items, device=self.device).contiguous()
        return items

    def __getitem__(
        self, idx: int | slice | Tensor | list[int] | np.ndarray | Sequence[int]
    ):
        if isinstance(idx, str):
            return super().__getitem__(idx)
        elif isinstance(idx, (int, np.integer)):
            return self.index_select([idx])
        else:
            return self.index_select(idx)

    def get_data(self, idx: int) -> AtomicData:
        r"""Reconstructs the :class:`torch_geometric.data.Data` object at index
        :obj:`idx` from the batch object.
        The batch object must have been created via :meth:`from_data_list` in
        order to be able to reconstruct the initial objects."""

        if self.__slices__ is None:
            raise RuntimeError(
                "Cannot reconstruct data list from batch because the batch "
                "object was not created using `Batch.from_data_list()`."
            )

        data = {}
        idx = self.num_graphs + idx if idx < 0 else idx
        for key in self.__slices__.keys():
            item = self[key]
            if self.__cat_dims__[key] is None:
                # The item was concatenated along a new batch dimension,
                # so just index in that dimension:
                item = item[idx]
            else:
                # Narrow the item based on the values in `__slices__`.
                if isinstance(item, Tensor):
                    dim = self.__cat_dims__[key]
                    start = self.__slices__[key][idx]
                    end = self.__slices__[key][idx + 1]
                    item = item.narrow(dim, start, end - start)
                else:
                    start = self.__slices__[key][idx]
                    end = self.__slices__[key][idx + 1]
                    item = item[start:end]
                    item = item[0] if len(item) == 1 else item

            # Decrease its value by `cumsum` value:
            cum = self.__cumsum__[key][idx]
            if isinstance(item, Tensor):
                if not isinstance(cum, int) or cum != 0:
                    item = item - cum
            elif isinstance(item, (int, float)):
                item = item - cum

            data[key] = item
        data = self.__data_class__(**data)
        if self.__num_nodes_list__[idx] is not None:
            data["num_nodes"] = self.__num_nodes_list__[idx]

        return data

    def to_data_list(self) -> list[AtomicData]:
        r"""Reconstructs the list of :class:`torch_geometric.data.Data` objects
        from the batch object.
        The batch object must have been created via :meth:`from_data_list` in
        order to be able to reconstruct the initial objects."""
        return [self.get_data(i) for i in range(self.num_graphs)]

    @property
    def num_graphs(self) -> int:
        """Returns the number of graphs in the batch."""
        if self.__num_graphs__ is not None:
            return self.__num_graphs__
        elif self.ptr is not None:
            return self.ptr.numel() - 1
        elif self.batch is not None:
            return int(self.batch.max()) + 1
        else:
            raise ValueError

    def append_data(
        self,
        data_list: list[AtomicData],
        exclude_keys: list[str] = [],
    ):
        """Adds data to the current batch

        Parameters
        ----------
        data_list : list[Data]
            List of data structures to append to batch.
        exclude_keys : list[str], optional
            Keys in Data to not include in batch
        """
        _excluded_keys = ["batch", "ptr", "device", "dtype"]
        keys = list(set(data_list[0].keys) - set(_excluded_keys + exclude_keys))
        if not "batch" not in keys and "ptr" not in keys:
            raise AssertionError()

        num_graphs_before = self.__num_graphs__
        if num_graphs_before is None:
            raise ValueError("Batch must not be empty")

        self.__num_graphs__ += len(data_list)
        device = self.batch.device
        for i, data in enumerate(data_list):
            for key in keys:
                item = data[key]
                if isinstance(item, Tensor):
                    item = item.to(device)

                # Increase values by `cumsum` value
                cum = self.__cumsum__[key][-1]
                if isinstance(item, Tensor) and item.dtype != torch.bool:
                    if not isinstance(cum, int) or cum != 0:
                        item = item + cum
                elif isinstance(item, (int, float)):
                    item = item + cum

                # Gather the size of the `cat` dimension.
                size = 1
                cat_dim = self.__cat_dims__[key]

                # Add a batch dimension to items whose `cat_dim` is `None`:
                if isinstance(item, Tensor) and cat_dim is None:
                    cat_dim = 0  # Concatenate along this new batch dimension.
                    item = item.unsqueeze(0)

                elif isinstance(item, Tensor):
                    size = item.size(cat_dim)

                if isinstance(item, Tensor):
                    self[key] = torch.cat([self[key], item], cat_dim)
                elif isinstance(item, (int, float)):
                    self[key] = torch.cat([self[key], torch.tensor([item])], cat_dim)
                elif isinstance(item, dict):
                    self[key].append(item)
                self.__slices__[key].append(size + self.__slices__[key][-1])

                inc = data.__inc__(key, item)
                if isinstance(inc, (tuple, list)):
                    inc = torch.tensor(inc)
                self.__cumsum__[key].append(inc + self.__cumsum__[key][-1])

            if hasattr(data, "__num_nodes__"):
                self.__num_nodes_list__.append(data.__num_nodes__)
            else:
                self.__num_nodes_list__.append(None)

            num_nodes = data.num_nodes
            if num_nodes is not None:
                item = torch.full(
                    (num_nodes,), num_graphs_before + i, dtype=torch.long, device=device
                )
                self.batch = torch.cat((self.batch, item), 0)
                self.ptr = torch.cat(
                    (self.ptr, torch.tensor([self.ptr[-1] + num_nodes], device=device)),
                    0,
                )

        self.num_nodes = sum(self.__num_nodes_list__)
