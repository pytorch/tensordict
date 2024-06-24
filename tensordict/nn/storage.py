# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import abc
from typing import Callable, Dict, Generic, List, Optional, TypeVar

import torch

import torch.nn as nn

from tensordict import NestedKey, TensorDict, TensorDictBase
from tensordict.nn.common import TensorDictModuleBase

K = TypeVar("K")
V = TypeVar("V")


class TensorStorage(abc.ABC, Generic[K, V]):
    """An Abstraction for implementing different storage.

    This class is for internal use, please use derived classes instead.
    """

    def clear(self) -> None:
        raise NotImplementedError

    def __getitem__(self, item: K) -> V:
        raise NotImplementedError

    def __setitem__(self, key: K, value: V) -> None:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def contain(self, item: K) -> torch.Tensor:
        raise NotImplementedError


class DynamicStorage(TensorStorage[torch.Tensor, torch.Tensor]):
    """A Dynamic Tensor Storage.

    This is a storage that save its tensors in cpu memories. It
    expands as necessary.
    """

    def __init__(self, default_tensor: torch.Tensor):
        self.tensor_dict: Dict[int, torch.Tensor] = {}
        self.default_tensor = default_tensor

    def clear(self) -> None:
        self.tensor_dict.clear()

    def __getitem__(self, indices: torch.Tensor) -> torch.Tensor:
        values: List[torch.Tensor] = []
        for index in indices.tolist():
            value = self.tensor_dict.get(index)
            if value is None:
                value = self.default_tensor.clone()
            values.append(value)

        return torch.stack(values)

    def __setitem__(self, indices: torch.Tensor, values: torch.Tensor) -> None:
        for index, value in zip(indices.tolist(), values.unbind(0)):
            self.tensor_dict[index] = value

    def __len__(self) -> None:
        return len(self.tensor_dict)

    def contain(self, indices: torch.Tensor) -> torch.Tensor:
        res: List[bool] = []
        for index in indices.tolist():
            res.append(index in self.tensor_dict)

        return torch.tensor(res, dtype=torch.int64)


class FixedStorage(nn.Module, TensorStorage[torch.Tensor, torch.Tensor]):
    """A Fixed Tensor Storage.

    This is storage that backed by nn.Embedding and hence can be in any device that
    nn.Embedding supports. The size of storage is fixed and cannot be extended.
    """

    def __init__(
        self, embedding: nn.Embedding, init_fm: Callable[[torch.Tensor], torch.Tensor]
    ):
        super().__init__()
        self.embedding = embedding
        self.num_embedding = embedding.num_embeddings
        self.flag = None
        self.init_fm = init_fm
        self.clear()

    def clear(self):
        self.init_fm(self.embedding.weight)
        self.flag = torch.zeros((self.embedding.num_embeddings, 1), dtype=torch.int64)

    def to_index(self, item: torch.Tensor) -> torch.Tensor:
        return torch.remainder(item.to(torch.int64), self.num_embedding).to(torch.int64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(self.to_index(x))

    def __getitem__(self, item: torch.Tensor) -> torch.Tensor:
        return self.forward(item)

    def __setitem__(self, item: torch.Tensor, value: torch.Tensor) -> None:
        if value.shape[-1] != self.embedding.embedding_dim:
            raise ValueError(
                "The shape value does not match with storage cell shape, "
                f"expected {self.embedding.embedding_dim} but got {value.shape[-1]}!"
            )
        index = self.to_index(item)
        with torch.no_grad():
            self.embedding.weight[index, :] = value
            self.flag[index] = 1

    def __len__(self) -> int:
        return torch.sum(self.flag).item()

    def contain(self, item: torch.Tensor) -> torch.Tensor:
        index = self.to_index(item)
        return self.flag[index]


class BinaryToDecimal(torch.nn.Module):
    """A Module to convert binaries encoded tensors to decimals.

    This is a utility class that allow to convert a binary encoding tensor (e.g. `1001`) to
    its decimal value (e.g. `9`)
    """

    def __init__(
        self,
        num_bits: int,
        device: torch.device,
        dtype: torch.dtype,
        convert_to_binary: bool,
    ):
        super().__init__()
        self.convert_to_binary = convert_to_binary
        self.bases = 2 ** torch.arange(num_bits - 1, -1, -1, device=device, dtype=dtype)
        self.num_bits = num_bits
        self.zero_tensor = torch.zeros((1,))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        num_features = features.shape[-1]
        if self.num_bits > num_features:
            raise ValueError(f"{num_features=} is less than {self.num_bits=}")
        elif num_features % self.num_bits != 0:
            raise ValueError(f"{num_features=} is not divisible by {self.num_bits=}")

        binary_features = (
            torch.heaviside(features, self.zero_tensor)
            if self.convert_to_binary
            else features
        )
        feature_parts = binary_features.reshape(shape=(-1, self.num_bits))
        digits = torch.sum(self.bases * feature_parts, -1)
        digits = digits.reshape(shape=(-1, features.shape[-1] // self.num_bits))
        aggregated_digits = torch.sum(digits, dim=-1)
        return aggregated_digits


class SipHash(torch.nn.Module):
    """A Module to Compute SipHash values for given tensors.

    A hash function module based on SipHash implementation in python.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hash_values = []
        for x_i in torch.unbind(x):
            hash_value = hash(x_i.detach().numpy().tobytes())
            hash_values.append(hash_value)

        return torch.tensor(hash_values, dtype=torch.int64)


class QueryModule(TensorDictModuleBase):
    """A Module to generate compatible indices for storage.

    A module that queries a storage and return required index of that storage.
    Currently, it only outputs integer indices (torch.int64).
    """

    def __init__(
        self,
        in_keys: List[NestedKey],
        index_key: NestedKey,
        hash_module: torch.nn.Module,
        aggregation_module: torch.nn.Module | None = None,
    ):
        self.in_keys = in_keys if isinstance(in_keys, List) else [in_keys]
        self.out_keys = [index_key]

        super().__init__()

        self.aggregation_module = (
            aggregation_module if aggregation_module else hash_module
        )

        self.hash_module = hash_module
        self.index_key = index_key

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        hash_values = []

        for k in self.in_keys:
            hash_values.append(self.hash_module(tensordict[k]))

        td_hash_value = self.aggregation_module(
            torch.stack(
                hash_values,
                dim=-1,
            ),
        )

        output = tensordict.clone(False)
        output[self.index_key] = td_hash_value
        return output


class TensorDictStorage(
    TensorDictModuleBase, TensorStorage[TensorDictModuleBase, TensorDictModuleBase]
):
    """A Storage for TensorDict.

    This module resembles a storage. It takes a tensordict as its input and
    returns another tensordict as output similar to TensorDictModuleBase. However,
    it provides additional functionality like python map:

    Examples:
        >>> import torch
        >>> from tensordict import TensorDict
        >>> mlp = torch.nn.LazyLinear(out_features=64, bias=True)
        >>> binary_to_decimal = BinaryToDecimal(
        ...     num_bits=8, device="cpu", dtype=torch.int32, convert_to_binary=True
        ... )
        >>> query_module = QueryModule(
        ...     in_keys=["key1", "key2"],
        ...     index_key="index",
        ...     hash_module=torch.nn.Sequential(mlp, binary_to_decimal),
        ... )
        >>> embedding_storage = FixedStorage(
        ...     torch.nn.Embedding(num_embeddings=23, embedding_dim=1),
        ...     lambda x: torch.nn.init.constant_(x, 0),
        ... )
        >>> tensor_dict_storage = TensorDictStorage(
        ...     query_module=query_module,
        ...     key_to_storage={"index": embedding_storage},
        ... )
        >>> index = TensorDict(
        ...     {
        ...         "key1": torch.Tensor([[-1], [1], [3], [-3]]),
        ...         "key2": torch.Tensor([[0], [2], [4], [-4]]),
        ...     },
        ...     batch_size=(4,),
        ... )
        >>> value = TensorDict(
        ...     {"index": torch.Tensor([[10], [20], [30], [40]])}, batch_size=(4,)
        ... )
        >>> tensor_dict_storage[index] = value
        >>> tensor_dict_storage[index]
        >>> assert torch.sum(tensor_dict_storage.contain(index)).item() == 4
        >>> new_index = index.clone(True)
        >>> new_index["key3"] = torch.Tensor([[4], [5], [6], [7]])
        >>> retrieve_value = tensor_dict_storage[new_index]
        >>> assert (retrieve_value["index"] == value["index"]).all()
    """

    def __init__(
        self,
        query_module: QueryModule,
        key_to_storage: Dict[NestedKey, TensorStorage[torch.Tensor, torch.Tensor]],
    ):
        self.in_keys = query_module.in_keys
        self.out_keys = list(key_to_storage.keys())

        super().__init__()

        for k in self.out_keys:
            assert k in key_to_storage, f"{k} has not been assigned to a memory"
        self.query_module = query_module
        self.index_key = query_module.index_key
        self.key_to_storage = key_to_storage
        self.batch_added = False

    def clear(self) -> None:
        for mem in self.key_to_storage.values():
            mem.clear()

    def to_index(self, item: TensorDictBase) -> torch.Tensor:
        return self.query_module(item)[self.index_key]

    def maybe_add_batch(
        self, item: TensorDictBase, value: TensorDictBase | None
    ) -> TensorDictBase:
        self.batch_added = False
        if len(item.batch_size) == 0:
            self.batch_added = True

            item = item.unsqueeze(dim=0)
            if value is not None:
                value = value.unsqueeze(dim=0)

        return item, value

    def maybe_remove_batch(self, item: TensorDictBase) -> TensorDictBase:
        if self.batch_added:
            item = item.squeeze(dim=0)
        return item

    def __getitem__(self, item: TensorDictBase) -> TensorDictBase:
        item, _ = self.maybe_add_batch(item, None)

        index = self.to_index(item)

        res = TensorDict({}, batch_size=item.batch_size)
        for k in self.out_keys:
            res[k] = self.key_to_storage[k][index]

        res = self.maybe_remove_batch(res)
        return res

    def __setitem__(self, item: TensorDictBase, value: TensorDictBase):
        item, value = self.maybe_add_batch(item, value)

        index = self.to_index(item)
        for k in self.out_keys:
            self.key_to_storage[k][index] = value[k]

    def __len__(self):
        return len(next(iter(self.key_to_storage.values())))

    def contain(self, item: TensorDictBase) -> torch.Tensor:
        item, _ = self.maybe_add_batch(item, None)
        index = self.to_index(item)

        res = next(iter(self.key_to_storage.values())).contain(index)
        res = self.maybe_remove_batch(res)
        return res
