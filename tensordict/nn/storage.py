# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import abc
from abc import abstractmethod
from typing import Callable, Dict, Generic, List, TypeVar

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

    @abstractmethod
    def clear(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, item: K) -> V:
        raise NotImplementedError

    @abstractmethod
    def __setitem__(self, key: K, value: V) -> None:
        raise NotImplementedError

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def contains(self, item: K) -> torch.Tensor:
        raise NotImplementedError

    def __contains__(self, item):
        return self.contains(item)


class DynamicStorage(TensorStorage[torch.Tensor, torch.Tensor]):
    """A Dynamic Tensor Storage.

    Indices can be of any pytorch dtype.

    This is a storage that save its tensors in cpu memories. It
    expands as necessary.

    It is assumed that all values in the storage can be stacked together
    using :func:`~torch.stack`.

    Args:
        default_tensor (torch.Tensor): the default value to return when
            an index cannot be found. This value will not be set in the
            storage.

    Examples:
        >>> storage = DynamicStorage(default_tensor=torch.zeros((1,)))
        >>> index = torch.randn((3,))
        >>> # set a value with a mismatching shape: it will be expanded to (3, 2, 1) shape
        >>> value = torch.rand((2, 1))
        >>> storage[index] = value
        >>> assert len(storage) == 3
        >>> assert (storage[index.clone()] == value).all()

    """

    def __init__(self, default_tensor: torch.Tensor):
        self.tensor_dict: Dict[int, torch.Tensor] = {}
        self.default_tensor = default_tensor

    def clear(self) -> None:
        self.tensor_dict.clear()

    def _check_indices(self, indices: torch.Tensor) -> None:
        if len(indices.shape) != 1:
            raise ValueError(
                f"Indices have to be a one-d vector but got {indices.shape}"
            )

    def __getitem__(self, indices: torch.Tensor) -> torch.Tensor:
        self._check_indices(indices)
        values: List[torch.Tensor] = []
        for index in indices.tolist():
            value = self.tensor_dict.get(index, self.default_tensor)
            values.append(value)

        return torch.stack(values)

    def __setitem__(self, indices: torch.Tensor, values: torch.Tensor) -> None:
        self._check_indices(indices)
        if not indices.ndim:
            self.tensor_dict[indices.item()] = values
            return
        if not values.ndim:
            values = values.expand(indices.shape[0])
        if values.shape[0] != indices.shape[0]:
            values = values.expand(indices.shape[0], *values.shape)
        for index, value in zip(indices.tolist(), values.unbind(0)):
            self.tensor_dict[index] = value

    def __len__(self) -> None:
        return len(self.tensor_dict)

    def contains(self, indices: torch.Tensor) -> torch.Tensor:
        self._check_indices(indices)
        res: List[bool] = []
        for index in indices.tolist():
            res.append(index in self.tensor_dict)

        return torch.tensor(res, dtype=torch.bool)


class FixedStorage(nn.Module, TensorStorage[torch.Tensor, torch.Tensor]):
    """A Fixed Tensor Storage.

    Indices must be of ``torch.long`` dtype.

    This is storage that backed by nn.Embedding and hence can be in any device that
    nn.Embedding supports. The size of storage is fixed and cannot be extended.

    Args:
        embedding (torch.nn.Embedding): the embedding module, or equivalent.
        init_fn (Callable[[torch.Tensor], torch.Tensor], optional): an init function
            for the embedding weights. Defaults to
            :func:`~torch.nn.init.normal_`, like `nn.Embedding`.

    Examples:
        >>> embedding_storage = FixedStorage(
        ...     torch.nn.Embedding(num_embeddings=10, embedding_dim=2),
        ...     lambda x: torch.nn.init.constant_(x, 0),
        ... )
        >>> index = torch.Tensor([1, 2], dtype=torch.long)
        >>> assert len(embedding_storage) == 0
        >>> assert not (embedding_storage[index] == torch.ones(size=(2, 2))).all()
        >>> embedding_storage[index] = torch.ones(size=(2, 2))
        >>> assert torch.sum(embedding_storage.contains(index)).item() == 2
        >>> assert (embedding_storage[index] == torch.ones(size=(2, 2))).all()
        >>> assert len(embedding_storage) == 2
        >>> embedding_storage.clear()
        >>> assert len(embedding_storage) == 0
        >>> assert not (embedding_storage[index] == torch.ones(size=(2, 2))).all()
    """

    def __init__(
        self,
        embedding: nn.Embedding,
        init_fm: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ):
        super().__init__()
        self.embedding = embedding
        self.num_embedding = embedding.num_embeddings
        self.flag = None
        if init_fm is None:
            init_fm = torch.nn.init.normal_
        self.init_fm = init_fm
        self.clear()

    def clear(self):
        self.init_fm(self.embedding.weight)
        self.flag = torch.zeros((self.embedding.num_embeddings, 1), dtype=torch.bool)

    def _to_index(self, item: torch.Tensor) -> torch.Tensor:
        return torch.remainder(item.to(torch.int64), self.num_embedding).to(torch.int64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(self._to_index(x))

    def __getitem__(self, item: torch.Tensor) -> torch.Tensor:
        return self.forward(item)

    def __setitem__(self, item: torch.Tensor, value: torch.Tensor) -> None:
        if value.shape[-1] != self.embedding.embedding_dim:
            raise ValueError(
                "The shape value does not match with storage cell shape, "
                f"expected {self.embedding.embedding_dim} but got {value.shape[-1]}!"
            )
        index = self._to_index(item)
        with torch.no_grad():
            self.embedding.weight[index, :] = value
            self.flag[index] = True

    def __len__(self) -> int:
        return torch.sum(self.flag).item()

    def contains(self, item: torch.Tensor) -> torch.Tensor:
        index = self._to_index(item)
        return self.flag[index]


class BinaryToDecimal(torch.nn.Module):
    """A Module to convert binaries encoded tensors to decimals.

    This is a utility class that allow to convert a binary encoding tensor (e.g. `1001`) to
    its decimal value (e.g. `9`)

    Args:
        num_bits (int): the number of bits to use for the bases table.
            The number of bits must be lower or equal to the input length and the input length
            must be divisible by ``num_bits``. If ``num_bits`` is lower than the number of
            bits in the input, the end result will be aggregated on the last dimension using
            :func:`~torch.sum`.
        device (torch.device): the device where inputs and outputs are to be expected.
        dtype (torch.dtype): the output dtype.
        convert_to_binary (bool, optional): if ``True``, the input to the ``forward``
            method will be cast to a binary input using :func:`~torch.heavyside`.
            Defaults to ``False``.

    Examples:
        >>> binary_to_decimal = BinaryToDecimal(
        ...    num_bits=4, device="cpu", dtype=torch.int32, convert_to_binary=True
        ... )
        >>> binary = torch.Tensor([[0, 0, 1, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 10, 0]])
        >>> decimal = binary_to_decimal(binary)
        >>> assert decimal.shape == (2,)
        >>> assert (decimal == torch.Tensor([3, 2])).all()
    """

    def __init__(
        self,
        num_bits: int,
        device: torch.device,
        dtype: torch.dtype,
        convert_to_binary: bool = False,
    ):
        super().__init__()
        self.convert_to_binary = convert_to_binary
        self.bases = 2 ** torch.arange(num_bits - 1, -1, -1, device=device, dtype=dtype)
        self.num_bits = num_bits
        self.zero_tensor = torch.zeros((1,), device=device)

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
        digits = torch.vmap(torch.dot, (None, 0))(
            self.bases, feature_parts.to(self.bases.dtype)
        )
        digits = digits.reshape(shape=(-1, features.shape[-1] // self.num_bits))
        aggregated_digits = torch.sum(digits, dim=-1)
        return aggregated_digits


class SipHash(torch.nn.Module):
    """A Module to Compute SipHash values for given tensors.

    A hash function module based on SipHash implementation in python.

    .. warning:: This module relies on the builtin ``hash`` function.
        To get reproducible results across runs, the ``PYTHONHASHSEED`` environment
        variable must be set before the code is run (changing this value during code
        execution is without effect).

    Examples:
        >>> # Assuming we set PYTHONHASHSEED=0 prior to running this code
        >>> a = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        >>> b = a.clone()
        >>> hash_module = SipHash()
        >>> hash_a = hash_module(a)
        >>> hash_a
        tensor([-4669941682990263259, -3778166555168484291, -9122128731510687521])
        >>> hash_b = hash_module(b)
        >>> assert (hash_a == hash_b).all()
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hash_values = []
        for x_i in x.detach().cpu().numpy():
            hash_value = hash(x_i.tobytes())
            hash_values.append(hash_value)

        return torch.tensor(hash_values, dtype=torch.int64)


class RandomProjectionHash(SipHash):
    """A module that combines random projections with SipHash to get a low-dimensional tensor, easier to embed through SipHash.

    This module requires sklearn to be installed.

    """

    def __init__(
        self,
        n_components=16,
        projection_type: str = "gaussian",
        dtype_cast=torch.float16,
        **kwargs,
    ):
        super().__init__()
        from sklearn.random_projection import (
            GaussianRandomProjection,
            SparseRandomProjection,
        )

        self.dtype_cast = dtype_cast
        if projection_type == "gaussian":
            self.transform = GaussianRandomProjection(
                n_components=n_components, **kwargs
            )
        elif projection_type == "sparse_random":
            self.transform = SparseRandomProjection(n_components=n_components, **kwargs)
        else:
            raise ValueError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.transform.transform(x)
        x = torch.as_tensor(x, dtype=self.dtype_cast)
        return super().forward(x)


class QueryModule(TensorDictModuleBase):
    """A Module to generate compatible indices for storage.

    A module that queries a storage and return required index of that storage.
    Currently, it only outputs integer indices (torch.int64).

    Args:
        in_keys (list of NestedKeys): keys of the input tensordict that
            will be used to generate the hash value.
        index_key (NestedKey): the output key where the hash value will be written.

    Keyword Args:
        hash_module (nn.Module or Callable[[torch.Tensor], torch.Tensor]): a hash
            module similar to :class:`~tensordict.nn.SipHash` (default).
        aggregation_module (torch.nn.Module or Callable[[torch.Tensor], torch.Tensor]): a
            method to aggregate the hash values. Defaults to the value of ``hash_module``.
            If only one ``in_Keys`` is provided, this module will be ignored.
        clone (bool, optional): if ``True``, a shallow clone of the input TensorDict will be
            returned. Defaults to ``False``.

    Examples:
        >>> query_module = QueryModule(
        ...     in_keys=["key1", "key2"],
        ...     index_key="index",
        ...     hash_module=SipHash(),
        ... )
        >>> query = TensorDict(
        ...     {
        ...         "key1": torch.Tensor([[1], [1], [1], [2]]),
        ...         "key2": torch.Tensor([[3], [3], [2], [3]]),
        ...         "other": torch.randn(4),
        ...     },
        ...     batch_size=(4,),
        ... )
        >>> res = query_module(query)
        >>> # The first two pairs of key1 and key2 match
        >>> assert res["index"][0] == res["index"][1]
        >>> # The last three pairs of key1 and key2 have at least one mismatching value
        >>> assert res["index"][1] != res["index"][2]
        >>> assert res["index"][2] != res["index"][3]
    """

    def __init__(
        self,
        in_keys: List[NestedKey],
        index_key: NestedKey,
        *,
        hash_module: torch.nn.Module | None = None,
        aggregation_module: torch.nn.Module | None = None,
        clone: bool = False,
    ):
        self.in_keys = in_keys if isinstance(in_keys, List) else [in_keys]
        if len(in_keys) == 0:
            raise ValueError("`in_keys` cannot be empty.")
        self.out_keys = [index_key]

        super().__init__()

        if hash_module is None:
            hash_module = SipHash()

        self.aggregation_module = (
            aggregation_module if aggregation_module else hash_module
        )

        self.hash_module = hash_module
        self.index_key = index_key
        self.clone = clone

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        hash_values = []

        i = -1  # to make linter happy
        for k in self.in_keys:
            hash_values.append(self.hash_module(tensordict.get(k)))

        if i > 0:
            td_hash_value = self.aggregation_module(
                torch.stack(
                    hash_values,
                    dim=-1,
                ),
            )
        else:
            td_hash_value = hash_values[0]

        if self.clone:
            output = tensordict.copy()
        else:
            output = tensordict

        output.set(self.index_key, td_hash_value)
        return output


class TensorDictStorage(
    TensorDictModuleBase, TensorStorage[TensorDictModuleBase, TensorDictModuleBase]
):
    """A Storage for TensorDict.

    This module resembles a storage. It takes a tensordict as its input and
    returns another tensordict as output similar to TensorDictModuleBase. However,
    it provides additional functionality like python map:

    Args:
        query_module (TensorDictModuleBase): a query module, typically an instance of
            :class:`~tensordict.nn.QueryModule`, used to map a set of tensordict
            entries to a hash key.
        key_to_storage (Dict[NestedKey, TensorStorage[torch.Tensor, torch.Tensor]]):
            a dictionary representing the map from an index key to a tensor storage.

    Examples:
        >>> import torch
        >>> from tensordict import TensorDict
        >>> from typing import cast
        >>> query_module = QueryModule(
        ...     in_keys=["key1", "key2"],
        ...     index_key="index",
        ...     hash_module=SipHash(),
        ... )
        >>> embedding_storage = DynamicStorage(
        ...     default_tensor=torch.zeros((1,)),
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
        TensorDict(
            fields={
                index: Tensor(shape=torch.Size([4, 1]), device=cpu, dtype=torch.float32, is_shared=False)},
            batch_size=torch.Size([4]),
            device=None,
            is_shared=False)
        >>> assert torch.sum(tensor_dict_storage.contains(index)).item() == 4
        >>> new_index = index.clone(True)
        >>> new_index["key3"] = torch.Tensor([[4], [5], [6], [7]])
        >>> retrieve_value = tensor_dict_storage[new_index]
        >>> assert cast(torch.Tensor, retrieve_value["index"] == value["index"]).all()
    """

    def __init__(
        self,
        query_module: QueryModule,
        key_to_storage: Dict[NestedKey, TensorStorage[torch.Tensor, torch.Tensor]],
    ):
        self.in_keys = query_module.in_keys
        self.out_keys = list(key_to_storage.keys())

        super().__init__()

        self.query_module = query_module
        self.index_key = query_module.index_key
        self.key_to_storage = key_to_storage
        self.batch_added = False

    def clear(self) -> None:
        for mem in self.key_to_storage.values():
            mem.clear()

    def _to_index(self, item: TensorDictBase) -> torch.Tensor:
        return self.query_module(item)[self.index_key]

    def _maybe_add_batch(
        self, item: TensorDictBase, value: TensorDictBase | None
    ) -> TensorDictBase:
        self.batch_added = False
        if len(item.batch_size) == 0:
            self.batch_added = True

            item = item.unsqueeze(dim=0)
            if value is not None:
                value = value.unsqueeze(dim=0)

        return item, value

    def _maybe_remove_batch(self, item: TensorDictBase) -> TensorDictBase:
        if self.batch_added:
            item = item.squeeze(dim=0)
        return item

    def __getitem__(self, item: TensorDictBase) -> TensorDictBase:
        item, _ = self._maybe_add_batch(item, None)

        index = self._to_index(item)

        res = TensorDict({}, batch_size=item.batch_size)
        for k in self.out_keys:
            storage: FixedStorage = self.key_to_storage[k]
            res[k] = storage[index]

        res = self._maybe_remove_batch(res)
        return res

    def __setitem__(self, item: TensorDictBase, value: TensorDictBase):
        item, value = self._maybe_add_batch(item, value)

        index = self._to_index(item)
        for k in self.out_keys:
            self.key_to_storage[k][index] = value[k]

    def __len__(self):
        return len(next(iter(self.key_to_storage.values())))

    def contains(self, item: TensorDictBase) -> torch.Tensor:
        item, _ = self._maybe_add_batch(item, None)
        index = self._to_index(item)

        res = next(iter(self.key_to_storage.values())).contains(index)
        res = self._maybe_remove_batch(res)
        return res
