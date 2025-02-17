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
import warnings
from collections.abc import Sequence
from typing import Union

import numpy as np
import torch

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    warnings.simplefilter("ignore", category=FutureWarning)
    # from mace.tools.torch_geometric.data import Data
    # from mace.tools.torch_geometric.dataset import IndexType

from torch import Tensor

from .nv_atomic_data import AtomicData
from torch_geometric.data import Data

try:
    from typing import Self  # type: ignore
except ImportError:
    from typing_extensions import Self

IndexType = Union[slice, Tensor, np.ndarray, Sequence]

class Batch(Data):
    r"""A plain old python object modeling a batch of graphs as one big
    (disconnected) graph. With :class:`torch_geometric.data.Data` being the
    base class, all its methods can also be used here.
    In addition, single graphs can be reconstructed via the assignment vector
    :obj:`batch`, which maps each node to its respective graph identifier.
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

    def update_key(self, key: str, item: list[int] | list[float] | Tensor):
        """Update an existing key."""

    def add_key(
        self,
        key: str,
        data: list[int] | list[float] | list[Tensor],
        cat_dim: int = 0,
    ):
        """Add a new key to the batch data."""
        cumsum = [0]
        slices = [0]
        batch = []
        for item in data:
            # Increase values by `cumsum` value.
            cum = cumsum[-1]
            if isinstance(item, Tensor) and item.dtype != torch.bool:
                if not isinstance(cum, int) or cum != 0:
                    item = item + cum
            elif isinstance(item, (int, float)):
                item = item + cum

            # Gather the size of the `cat` dimension.
            size = 1
            # 0-dimensional tensors have no dimension along which to
            # concatenate, so we set `cat_dim` to `None`.
            if isinstance(item, Tensor) and item.dim() == 0:
                cat_dim = None

            # Add a batch dimension to items whose `cat_dim` is `None`:
            if isinstance(item, Tensor) and cat_dim is None:
                cat_dim = 0  # Concatenate along this new batch dimension.
                item = item.unsqueeze(0)
            elif isinstance(item, Tensor):
                size = item.size(cat_dim)

            batch.append(item)  # Append item to the attribute list.

            slices.append(size + slices[-1])
            inc = data.__inc__(key, item)
            if isinstance(inc, (tuple, list)):
                inc = torch.tensor(inc)
            cumsum[key].append(inc + cumsum[key][-1])

    @classmethod
    def from_data_list(
        cls,
        data_list: list[AtomicData],
        follow_batch: list[str] = [],
        exclude_keys: list[str] = [],
    ):
        r"""Constructs a batch object from a python list holding
        :class:`torch_geometric.data.Data` objects.
        The assignment vector :obj:`batch` is created on the fly.
        Additionally, creates assignment batch vectors for each key in
        :obj:`follow_batch`.
        Will exclude any keys given in :obj:`exclude_keys`."""

        keys = list(set(data_list[0].keys()) - set(exclude_keys))
        if not "batch" not in keys and "ptr" not in keys:
            raise AssertionError()

        batch = cls()
        for key in data_list[0].__dict__.keys():
            if key[:2] != "__" and key[-2:] != "__":
                batch[key] = None

        batch.__num_graphs__ = len(data_list)
        batch.__data_class__ = data_list[0].__class__
        for key in keys + ["batch"]:
            batch[key] = []
        batch["ptr"] = [0]

        device = None
        slices = {key: [0] for key in keys}
        cumsum = {key: [0] for key in keys}
        cat_dims = {}
        num_nodes_list = []
        for i, data in enumerate(data_list):
            for key in keys:
                item = data[key]

                # Increase values by `cumsum` value.
                cum = cumsum[key][-1]
                if isinstance(item, Tensor) and item.dtype != torch.bool:
                    if not isinstance(cum, int) or cum != 0:
                        item = item + cum
                elif isinstance(item, (int, float)):
                    item = item + cum

                # Gather the size of the `cat` dimension.
                size = 1
                cat_dim = data.__cat_dim__(key, data[key])
                # 0-dimensional tensors have no dimension along which to
                # concatenate, so we set `cat_dim` to `None`.
                if isinstance(item, Tensor) and item.dim() == 0:
                    cat_dim = None
                cat_dims[key] = cat_dim

                # Add a batch dimension to items whose `cat_dim` is `None`:
                if isinstance(item, Tensor) and cat_dim is None:
                    cat_dim = 0  # Concatenate along this new batch dimension.
                    item = item.unsqueeze(0)
                    device = item.device
                elif isinstance(item, Tensor):
                    size = item.size(cat_dim)
                    device = item.device

                batch[key].append(item)  # Append item to the attribute list.

                slices[key].append(size + slices[key][-1])
                inc = data.__inc__(key, item)
                if isinstance(inc, (tuple, list)):
                    inc = torch.tensor(inc)
                cumsum[key].append(inc + cumsum[key][-1])

                if key in follow_batch:
                    if isinstance(size, Tensor):
                        for j, size in enumerate(size.tolist()):
                            tmp = f"{key}_{j}_batch"
                            batch[tmp] = [] if i == 0 else batch[tmp]
                            batch[tmp].append(
                                torch.full((size,), i, dtype=torch.long, device=device)
                            )
                    else:
                        tmp = f"{key}_batch"
                        batch[tmp] = [] if i == 0 else batch[tmp]
                        batch[tmp].append(
                            torch.full((size,), i, dtype=torch.long, device=device)
                        )

            if hasattr(data, "__num_nodes__"):
                num_nodes_list.append(data.__num_nodes__)
            else:
                num_nodes_list.append(None)

            num_nodes = data.num_nodes
            if num_nodes is not None:
                item = torch.full((num_nodes,), i, dtype=torch.long, device=device)
                batch.batch.append(item)
                batch.ptr.append(batch.ptr[-1] + num_nodes)

        batch.batch = None if len(batch.batch) == 0 else batch.batch
        batch.ptr = None if len(batch.ptr) == 1 else batch.ptr
        batch.__slices__ = slices
        batch.__cumsum__ = cumsum
        batch.__cat_dims__ = cat_dims
        batch.__num_nodes_list__ = num_nodes_list

        ref_data = data_list[0]
        # for key in batch.keys:
        for key in batch.keys():
            items = batch[key]
            item = items[0]
            cat_dim = ref_data.__cat_dim__(key, item)
            cat_dim = 0 if cat_dim is None else cat_dim
            if isinstance(item, Tensor):
                batch[key] = torch.cat(items, cat_dim)
            elif isinstance(item, (int, float)):
                batch[key] = torch.tensor(items, device=device)

        # if torch_geometric.is_debug_enabled():
        #     batch.debug()
        return batch.contiguous()

    def index_select(self, idx: IndexType) -> Self:
        """Select a subset of the current batch structure.

        Parameters
        ----------
        idx : IndexType
           Indexes to subselect

        Returns
        -------
        Batch
            New Batch data structure.
        Raises
        ------
        IndexError
            _description_
        RuntimeError
            _description_
        """
        if isinstance(idx, slice):
            idx = list(range(self.num_graphs)[idx])

        elif isinstance(idx, Tensor) and idx.dtype == torch.long:
            idx = idx.flatten().tolist()

        elif isinstance(idx, Tensor) and idx.dtype == torch.bool:
            idx = idx.flatten().nonzero(as_tuple=False).flatten().tolist()

        elif isinstance(idx, np.ndarray) and idx.dtype == np.int64:
            idx = idx.flatten().tolist()

        elif isinstance(idx, np.ndarray) and idx.dtype == np.bool_:
            idx = idx.flatten().nonzero()[0].flatten().tolist()

        elif isinstance(idx, int):
            idx = [idx]

        elif isinstance(idx, Sequence) and not isinstance(idx, str):
            pass

        else:
            raise IndexError(
                f"Only integers, slices (':'), list, tuples, torch.tensor and "
                f"np.ndarray of dtype long or bool are valid indices (got "
                f"'{type(idx).__name__}')"
            )

        if self.__slices__ is None:
            raise RuntimeError(
                "Cannot reconstruct data list from batch because the batch "
                "object was not created using `Batch.from_data_list()`."
            )

        device = self.batch.device
        data = Batch()  # type: ignore
        data.__data_class__ = AtomicData
        data.ptr = [0]
        data.batch = []
        idx = [self.num_graphs + index if index < 0 else index for index in idx]
        data.__num_graphs__ = len(idx)
        keys = self.__slices__.keys()
        slices = {key: [0] for key in keys}
        cumsum = {key: [0] for key in keys}
        cat_dims = {}
        num_nodes_list = []
        for key in keys:
            item = self[key]
            items = []
            cat_dims[key] = self.__cat_dims__[key]
            if self.__cat_dims__[key] is None:
                # The item was concatenated along a new batch dimension,
                # so just index in that dimension:
                item = item[idx]
            else:
                for index in idx:
                    if isinstance(item, Tensor):
                        dim = self.__cat_dims__[key]
                        start = self.__slices__[key][index]
                        end = self.__slices__[key][index + 1]
                        items.append(item.narrow(dim, start, end - start))
                    else:
                        start = self.__slices__[key][index]
                        end = self.__slices__[key][index + 1]
                        temp_item = item[start:end]
                        temp_item = temp_item[0] if len(temp_item) == 1 else temp_item
                        items.append(temp_item)
                    slices[key].append(slices[key][-1] + end - start)

            # Decrease its value by `cumsum` value:
            for i, index in enumerate(idx):

                cum_start = self.__cumsum__[key][index]
                cum_end = self.__cumsum__[key][index + 1]

                if isinstance(items[i], Tensor) and items[i].dtype != torch.bool:
                    if not isinstance(start, int) or cum_start != 0:
                        items[i] = items[i] - cum_start
                        # Add new cumsum
                        items[i] = items[i] + cumsum[key][-1]

                elif isinstance(item, (int, float)):
                    items[i] = items[i] - cum_start

                    # Add new cumsum
                    items[i] = items[i] + cumsum[key][-1]

                cumsum[key].append(cumsum[key][-1] + cum_end - cum_start)

            if isinstance(item, Tensor):
                item = torch.cat(items, cat_dims[key]).contiguous()
            elif isinstance(item, np.ndarray):
                item = torch.tensor(items, device=device).contiguous()

            data[key] = item

        for i, index in enumerate(idx):
            num_nodes = self.__num_nodes_list__[index]
            num_nodes_list += [num_nodes]
            data["batch"].append(
                torch.full((num_nodes,), i, dtype=torch.long, device=device)
            )
            data.ptr.append(data.ptr[-1] + num_nodes)

        data.batch = torch.cat(data.batch, 0)
        data.batch = None if len(data.batch) == 0 else data.batch
        data.ptr = None if len(data.ptr) == 1 else data.ptr
        data.ptr = torch.tensor(data.ptr, device=device)
        data.__slices__ = slices
        data.__cumsum__ = cumsum
        data.__cat_dims__ = cat_dims
        data.__num_nodes_list__ = num_nodes_list
        data["num_nodes"] = sum(num_nodes_list)

        return data.to(device)

    def __getitem__(self, idx: IndexType):
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

        keys = list(set(data_list[0].keys()) - set(exclude_keys))
        if not "batch" not in keys and "ptr" not in keys:
            raise AssertionError()

        num_graphs_before = self.__num_graphs__
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

