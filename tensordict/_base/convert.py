# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Conversions to and from other containers, file formats and nn.Module parameters of :class:`~tensordict.TensorDictBase`.

The methods live on a mixin that ``tensordict.base`` imports before it
defines ``TensorDictBase``, so this module imports the helpers it needs from
``tensordict.base`` without needing the class itself. A method that uses
``TensorDictBase`` at run time imports it locally. The comment above the
mixin imports in ``tensordict/base.py`` gives the other rules.
"""

from __future__ import annotations

import collections
import importlib.util
import numbers
import uuid
import weakref
from collections import UserDict
from copy import copy
from typing import Any, List, Literal, Sequence, Type, TYPE_CHECKING

import numpy as np
import torch
from tensordict._datasets import to_mds
from tensordict._nestedkey import NestedKey
from tensordict._tabular import (
    _columns_to_tensordict,
    _dataframe_to_tensordict,
    _read_csv,
    _read_json,
    _read_parquet,
    _tensordict_to_dataframe,
    _write_csv,
    _write_json,
    _write_parquet,
)
from tensordict.base import (
    __base__setattr__,
    _has_h5,
    _is_leaf_nontensor,
    _is_tensor_collection,
    _maybe_preserve_module_state,
    _set_tensor_dict,
    Buffer,
    is_compiling,
    is_tensor_collection,
    NO_DEFAULT,
    Self,
)
from tensordict.utils import (
    _as_context_manager,
    _is_dataclass as is_dataclass,
    _is_list_tensor_compatible,
    _is_namedtuple,
    _is_namedtuple_class,
    _is_non_tensor,
    _is_unbatched,
    _maybe_correct_neg_dim,
    _set_max_batch_size,
    is_non_tensor,
    LinkedList,
    set_lazy_legacy,
)
from torch import nn, Tensor
from torch.nn.parameter import UninitializedTensorMixin
from torch.nn.utils._named_member_accessor import swap_tensor
from torch.utils._pytree import is_structseq_instance

if TYPE_CHECKING:
    from tensordict.base import TensorDictBase

try:
    from functorch import dim as ftdim
except ImportError:
    from tensordict.utils import _ftdim_mock as ftdim


def _int_batch_dims(batch_dims):
    # for backward compatibility, from_any ignores a batch_dims that is not an
    # int for dataclasses and h5py files
    if isinstance(batch_dims, numbers.Integral) and not isinstance(batch_dims, bool):
        return batch_dims
    return None


class _Conversion:
    """Conversions to and from other containers, file formats and nn.Module parameters."""

    @classmethod
    def from_list(
        cls,
        input,
        *,
        auto_batch_size: bool | None = None,
        batch_size: torch.Size | None = None,
        device: torch.device | None = None,
        batch_dims: int | None = None,
        names: list[str] | None = None,
        lazy: bool | None = None,
    ) -> Self:
        from tensordict.base import TensorDictBase

        if lazy is None:
            stack = cls.maybe_dense_stack
        elif lazy:
            stack = cls.lazy_stack
        else:
            stack = torch.stack
        if batch_size is not None:
            if isinstance(batch_size, int):
                batch_size = torch.Size([batch_size])
            if batch_size[0] != len(input):
                raise ValueError(
                    f"The provided batch size ({batch_size}) does not match the length of the list ({len(input)})."
                )
            bsz = batch_size[1:]
        else:
            bsz = None
        if batch_dims is not None:
            batch_dims -= 1
        if names is not None:
            names = names[1:]
        if cls is TensorDictBase:
            from tensordict import TensorDict

            cls = TensorDict
        input = [
            (
                cls.from_dict(
                    d,
                    auto_batch_size=auto_batch_size,
                    batch_size=bsz,
                    batch_dims=batch_dims,
                    device=device,
                    names=names,
                )
                if not is_tensor_collection(d)
                else d
            )
            for d in input
        ]
        return stack(input)

    def from_dict_instance(
        self,
        input_dict,
        *,
        auto_batch_size: bool | None = None,
        batch_size=None,
        device=None,
        batch_dims=None,
        names=None,
    ):
        """Instance method version of :meth:`~tensordict.TensorDict.from_dict`.

        Unlike :meth:`~tensordict.TensorDict.from_dict`, this method will
        attempt to keep the tensordict types within the existing tree (for
        any existing leaf).

        Examples:
            >>> from tensordict import TensorDict, tensorclass
            >>> import torch
            >>>
            >>> @tensorclass
            ... class MyClass:
            ...     x: torch.Tensor
            ...     y: int
            >>>
            >>> td = TensorDict({"a": torch.randn(()), "b": MyClass(x=torch.zeros(()), y=1)})
            >>> print(td.from_dict_instance(td.to_dict()))
            TensorDict(
                fields={
                    a: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
                    b: MyClass(
                        x=Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
                        y=Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int64, is_shared=False),
                        batch_size=torch.Size([]),
                        device=None,
                        is_shared=False)},
                batch_size=torch.Size([]),
                device=None,
                is_shared=False)
            >>> print(td.from_dict(td.to_dict()))
            TensorDict(
                fields={
                    a: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
                    b: TensorDict(
                        fields={
                            x: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
                            y: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int64, is_shared=False)},
                        batch_size=torch.Size([]),
                        device=None,
                        is_shared=False)},
                batch_size=torch.Size([]),
                device=None,
                is_shared=False)

        """
        if batch_dims is not None and batch_size is not None:
            raise ValueError(
                "Cannot pass both batch_size and batch_dims to `from_dict`."
            )
        from tensordict import TensorDict

        batch_size_set = torch.Size(()) if batch_size is None else batch_size
        if is_compiling():
            input_dict = type(input_dict)(input_dict)
        else:
            input_dict = copy(input_dict)
        for key, value in list(input_dict.items()):
            if isinstance(value, (dict,)):
                cur_value = self.get(key)
                if cur_value is not None:
                    input_dict[key] = cur_value.from_dict_instance(
                        value,
                        device=device,
                        auto_batch_size=False,
                    )
                    continue
                else:
                    # we don't know if another tensor of smaller size is coming
                    # so we can't be sure that the batch-size will still be valid later
                    input_dict[key] = TensorDict.from_dict(
                        value,
                        device=device,
                        auto_batch_size=False,
                    )
            else:
                input_dict[key] = TensorDict.from_any(
                    value,
                    auto_batch_size=False,
                )

        out = TensorDict.from_dict(
            input_dict,
            batch_size=batch_size_set,
            device=device,
            names=names,
        )
        if batch_size is None:
            if auto_batch_size is None and batch_dims is None:
                auto_batch_size = False
            elif auto_batch_size is None:
                auto_batch_size = True
            if auto_batch_size:
                _set_max_batch_size(out, batch_dims)
        else:
            out.batch_size = batch_size
        return out

    @classmethod
    def from_pytree(
        cls,
        pytree,
        *,
        batch_size: torch.Size | None = None,
        auto_batch_size: bool = False,
        batch_dims: int | None = None,
    ):
        """Converts a pytree to a TensorDict instance.

        This method is designed to keep the pytree nested structure as much as possible.

        Additional non-tensor keys are added to keep track of each level's identity, providing
        a built-in pytree-to-tensordict bijective transform API.

        Accepted classes currently include lists, tuples, named tuples and dict.

        .. note::
            For dictionaries, non-NestedKey keys are registered separately as :class:`~tensordict.NonTensorData`
            instances.

        .. note::
            Tensor-castable types (such as int, float or np.ndarray) will be converted to torch.Tensor instances.
            Note that this transformation is surjective: transforming back the tensordict to a pytree will not
            recover the original types.

        Examples:
            >>> # Create a pytree with tensor leaves, and one "weird"-looking dict key
            >>> class WeirdLookingClass:
            ...     pass
            ...
            >>> weird_key = WeirdLookingClass()
            >>> # Make a pytree with tuple, lists, dict and namedtuple
            >>> pytree = (
            ...     [torch.randint(10, (3,)), torch.zeros(2)],
            ...     {
            ...         "tensor": torch.randn(
            ...             2,
            ...         ),
            ...         "td": TensorDict({"one": 1}),
            ...         weird_key: torch.randint(10, (2,)),
            ...         "list": [1, 2, 3],
            ...     },
            ...     {"named_tuple": TensorDict({"two": torch.ones(1) * 2}).to_namedtuple()},
            ... )
            >>> # Build a TensorDict from that pytree
            >>> td = TensorDict.from_pytree(pytree)
            >>> # Recover the pytree
            >>> pytree_recon = td.to_pytree()
            >>> # Check that the leaves match
            >>> def check(v1, v2):
            ...     assert (v1 == v2).all()
            ...
            >>> torch.utils._pytree.tree_map(check, pytree, pytree_recon)
            >>> assert weird_key in pytree_recon[1]

        """
        if is_tensor_collection(pytree):
            return pytree
        if isinstance(pytree, (torch.Tensor,)):
            return pytree

        from tensordict._td import TensorDict

        result = None
        if _is_namedtuple(pytree):
            result = TensorDict.from_namedtuple(named_tuple=pytree)
            # Without batch_dims, batch_size is ignored here, unlike for lists
            # and dicts: applying it would raise when it does not fit the leaves.
            if batch_dims is not None and batch_size is not None:
                result.batch_size = batch_size
            result["_pytree_type"] = type(pytree)
        elif isinstance(pytree, (list, tuple)):
            source = {str(i): cls.from_pytree(elt) for i, elt in enumerate(pytree)}
            source["_pytree_type"] = type(pytree)
            result = TensorDict(source, batch_size=batch_size)
        elif isinstance(pytree, dict):
            source = {}
            for key, item in pytree.items():
                if isinstance(key, NestedKey):
                    source[key] = cls.from_pytree(item)
                else:
                    subs_key = "<NON_NESTED>" + str(uuid.uuid1())
                    source[subs_key] = TensorDict(
                        {"value": cls.from_pytree(item), "key": key}
                    )
            source["_pytree_type"] = type(pytree)
            result = TensorDict(source, batch_size=batch_size)
        if result is not None:
            if auto_batch_size:
                result.auto_batch_size_(batch_dims)
            return result
        if isinstance(pytree, (int, float, np.ndarray)):
            return torch.as_tensor(pytree)
        raise NotImplementedError(f"Unknown type {type(pytree)}.")

    def to_pytree(self):
        """Converts a tensordict to a PyTree.

        If the tensordict was not created from a pytree, this method just returns ``self`` without modification.

        See :meth:`~.from_pytree` for more information and examples.

        """
        _pytree_type = self._get_str("_pytree_type", default=None)
        if _pytree_type is None:
            return self
        _pytree_type = _pytree_type.data
        items = {key: val for (key, val) in self.items() if key != "_pytree_type"}
        items = {
            key: val if not is_tensor_collection(val) else val.to_pytree()
            for key, val in items.items()
        }
        if _pytree_type in (list, tuple):
            return _pytree_type((items[str(i)] for i in range(len(items))))
        if _pytree_type is dict:
            items = dict(
                (
                    (
                        (val["key"], val["value"])
                        if key.startswith("<NON_NESTED>")
                        else (key, val)
                    )
                    for (key, val) in items.items()
                )
            )
            return items
        if _is_namedtuple_class(_pytree_type):
            from tensordict._td import TensorDict

            return TensorDict(items).to_namedtuple(dest_cls=_pytree_type)
        raise NotImplementedError(f"unknown type {_pytree_type}")

    @classmethod
    def from_h5(
        cls,
        filename,
        *,
        mode: str = "r",
        auto_batch_size: bool = False,
        batch_dims: int | None = None,
        batch_size: torch.Size | None = None,
    ):
        """Creates a PersistentTensorDict from a h5 file.

        Args:
            filename (str): The path to the h5 file.

        Keyword Arguments:
            mode (str, optional): Reading mode. Defaults to ``"r"``.
            auto_batch_size (bool, optional): If ``True``, the batch size will be computed automatically.
                Defaults to ``False``.
            batch_dims (int, optional): If auto_batch_size is ``True``, defines how many dimensions the output
                tensordict should have. Defaults to ``None`` (full batch-size at each level).
            batch_size (torch.Size, optional): The batch size of the TensorDict. Defaults to ``None``.

        Returns:
            A PersistentTensorDict representation of the input h5 file.

        Examples:
            >>> td = TensorDict.from_h5("path/to/file.h5")
            >>> print(td)
            PersistentTensorDict(
                fields={
                    key1: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False),
                    key2: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False)},
                batch_size=torch.Size([]),
                device=None,
                is_shared=False)
        """
        from tensordict.persistent import PersistentTensorDict

        result = PersistentTensorDict.from_h5(
            filename, mode=mode, batch_size=batch_size
        )
        if auto_batch_size:
            if batch_size is not None:
                raise TypeError(cls._CONFLICTING_BATCH_SIZES.format("from_h5"))
            result.auto_batch_size_(batch_dims=batch_dims)
        return result

    @classmethod
    def from_zarr(
        cls,
        filename,
        *,
        mode: str = "r",
        auto_batch_size: bool = False,
        batch_dims: int | None = None,
        batch_size: torch.Size | None = None,
    ):
        """Creates a PersistentTensorDict from a zarr store.

        Requires ``zarr>=3.0`` to be installed.

        Args:
            filename (str, path or zarr store): The path to the zarr store (a
                directory), or a ``zarr.abc.store.Store`` instance (e.g. a
                ``zarr.storage.ZipStore``).

        Keyword Arguments:
            mode (str, optional): Reading mode. Defaults to ``"r"``.
            auto_batch_size (bool, optional): If ``True``, the batch size will be computed automatically.
                Defaults to ``False``.
            batch_dims (int, optional): If auto_batch_size is ``True``, defines how many dimensions the output
                tensordict should have. Defaults to ``None`` (full batch-size at each level).
            batch_size (torch.Size, optional): The batch size of the TensorDict. Defaults to ``None``
                (read from the metadata written by :meth:`~.to_zarr`, or automatically determined
                for stores written by other tools).

        Returns:
            A PersistentTensorDict representation of the input zarr store.

        Examples:
            >>> td = TensorDict.from_zarr("path/to/store.zarr")
            >>> print(td)
            PersistentTensorDict(
                fields={
                    key1: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False),
                    key2: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False)},
                batch_size=torch.Size([]),
                device=None,
                is_shared=False)
        """
        from tensordict.persistent import PersistentTensorDict

        result = PersistentTensorDict.from_zarr(
            filename, mode=mode, batch_size=batch_size
        )
        if auto_batch_size:
            if batch_size is not None:
                raise TypeError(cls._CONFLICTING_BATCH_SIZES.format("from_zarr"))
            result.auto_batch_size_(batch_dims=batch_dims)
        return result

    @classmethod
    def from_schema(
        cls,
        schema: dict[str, tuple[list[int] | torch.Size, torch.dtype]],
        *,
        batch_size: Sequence[int] | torch.Size | None = None,
        storage: str | None = None,
        device=None,
        **kwargs,
    ) -> TensorDictBase:
        """Pre-allocate a zero-filled TensorDict from a schema.

        Creates a :class:`TensorDictBase` whose storage backend is selected
        by ``storage``.  Each entry in ``schema`` maps a field name to an
        ``(element_shape, dtype)`` pair; the full stored shape is
        ``[*batch_size, *element_shape]``.

        Args:
            schema: Mapping from field name to ``(element_shape, dtype)``.
                ``element_shape`` is the per-element shape (excluding
                ``batch_size``).

        Keyword Args:
            batch_size: Overall batch dimensions prepended to every element
                shape.  Defaults to ``()``.
            storage (str or None): Backend selector:

                - ``None`` -- plain :class:`TensorDict` with regular tensors.
                - ``"memmap"`` -- memory-mapped tensors on disk.
                  Pass ``prefix=<dir>`` in *kwargs*.
                - ``"h5"`` -- HDF5 via :class:`PersistentTensorDict`.
                  Pass ``filename=<path>`` in *kwargs*.
                - ``"zarr"`` -- zarr (requires ``zarr>=3.0``) via
                  :class:`PersistentTensorDict`. Pass ``filename=<path or store>``
                  in *kwargs*.
                - ``"shared"`` -- CPU shared-memory tensors.
                - ``"redis"`` / ``"dragonfly"`` -- delegates to
                  :meth:`TensorDictStore.from_schema`.

            device: Device for the resulting tensors (ignored by some
                backends).
            **kwargs: Backend-specific arguments forwarded to the
                underlying constructor (e.g. ``prefix`` for memmap,
                ``filename`` for h5, ``host``/``port`` for redis).

        Returns:
            A new :class:`TensorDictBase` subclass instance with
            pre-allocated (zero-filled) keys.

        Examples:
            >>> td = TensorDict.from_schema(
            ...     {"obs": ([84, 84, 3], torch.uint8),
            ...      "reward": ([], torch.float32)},
            ...     batch_size=[1000],
            ... )
            >>> td["obs"].shape
            torch.Size([1000, 84, 84, 3])

            >>> import tempfile
            >>> with tempfile.TemporaryDirectory() as d:
            ...     td_mm = TensorDict.from_schema(
            ...         {"obs": ([4], torch.float32)},
            ...         batch_size=[8],
            ...         storage="memmap",
            ...         prefix=d,
            ...     )
            ...     assert td_mm.is_memmap()

        """
        from tensordict._td import TensorDict

        if batch_size is None:
            batch_size = torch.Size(())
        else:
            batch_size = torch.Size(batch_size)

        def _full_shape(elem_shape):
            return [*batch_size, *elem_shape]

        if storage is None:
            source = {
                key: torch.zeros(_full_shape(es), dtype=dt, device=device)
                for key, (es, dt) in schema.items()
            }
            return TensorDict(source, batch_size=batch_size, device=device)

        if storage == "memmap":
            source = {
                key: torch.zeros((), dtype=dt).expand(_full_shape(es))
                for key, (es, dt) in schema.items()
            }
            td = TensorDict(source, batch_size=batch_size)
            prefix = kwargs.pop("prefix", None)
            return td.memmap_like(prefix, **kwargs)

        if storage == "h5":
            from tensordict.persistent import PersistentTensorDict

            source = {
                key: torch.zeros((), dtype=dt).expand(_full_shape(es))
                for key, (es, dt) in schema.items()
            }
            filename = kwargs.pop("filename")
            return PersistentTensorDict.from_dict(
                source, filename, batch_size=batch_size, device=device, **kwargs
            )

        if storage == "zarr":
            from tensordict.persistent import PersistentTensorDict

            source = {
                key: torch.zeros((), dtype=dt).expand(_full_shape(es))
                for key, (es, dt) in schema.items()
            }
            filename = kwargs.pop("filename")
            return PersistentTensorDict.from_dict(
                source,
                filename,
                batch_size=batch_size,
                device=device,
                backend="zarr",
                **kwargs,
            )

        if storage == "shared":
            source = {
                key: torch.zeros(_full_shape(es), dtype=dt, device=device)
                for key, (es, dt) in schema.items()
            }
            return TensorDict(
                source, batch_size=batch_size, device=device
            ).share_memory_()

        if storage in ("redis", "dragonfly"):
            from tensordict.store._store import TensorDictStore

            return TensorDictStore.from_schema(
                schema,
                batch_size=batch_size,
                backend=storage,
                device=device,
                **kwargs,
            )

        raise ValueError(
            f"Unknown storage backend {storage!r}. Expected one of "
            f"None, 'memmap', 'h5', 'zarr', 'shared', 'redis', 'dragonfly'."
        )

    # Module interaction
    @classmethod
    def from_module(
        cls,
        module,
        as_module: bool = False,
        lock: bool = True,
        use_state_dict: bool = False,
    ):
        """Copies the params and buffers of a module in a tensordict.

        Args:
            module (nn.Module): the module to get the parameters from.
            as_module (bool, optional): if ``True``, a :class:`~tensordict.nn.TensorDictParams`
                instance will be returned which can be used to store parameters
                within a :class:`torch.nn.Module`. Defaults to ``False``.
            lock (bool, optional): if ``True``, the resulting tensordict will be locked.
                Defaults to ``True``.
            use_state_dict (bool, optional): if ``True``, the state-dict from the
                module will be used and unflattened into a TensorDict with
                the tree structure of the model. Defaults to ``False``.

                .. note::
                    This is particularly useful when state-dict hooks have to be used.

        Examples:
            >>> from torch import nn
            >>> module = nn.TransformerDecoder(
            ...     decoder_layer=nn.TransformerDecoderLayer(nhead=4, d_model=4),
            ...     num_layers=1
            ... )
            >>> params = TensorDict.from_module(module)
            >>> print(params["layers", "0", "linear1"])
            TensorDict(
                fields={
                    bias: Parameter(shape=torch.Size([2048]), device=cpu, dtype=torch.float32, is_shared=False),
                    weight: Parameter(shape=torch.Size([2048, 4]), device=cpu, dtype=torch.float32, is_shared=False)},
                batch_size=torch.Size([]),
                device=None,
                is_shared=False)

        """
        ...

    @classmethod
    def from_modules(
        cls,
        *modules,
        as_module: bool = False,
        lock: bool = True,
        use_state_dict: bool = False,
        lazy_stack: bool = False,
        expand_identical: bool = False,
    ):
        """Retrieves the parameters of several modules for ensebmle learning/feature of expects applications through vmap.

        Args:
            modules (sequence of nn.Module): the modules to get the parameters from.
                If the modules differ in their structure, a lazy stack is needed
                (see the ``lazy_stack`` argument below).

        Keyword Args:
            as_module (bool, optional): if ``True``, a :class:`~tensordict.nn.TensorDictParams`
                instance will be returned which can be used to store parameters
                within a :class:`torch.nn.Module`. Defaults to ``False``.
            lock (bool, optional): if ``True``, the resulting tensordict will be locked.
                Defaults to ``True``.
            use_state_dict (bool, optional): if ``True``, the state-dict from the
                module will be used and unflattened into a TensorDict with
                the tree structure of the model. Defaults to ``False``.

                .. note::
                    This is particularly useful when state-dict hooks have to be used.

            lazy_stack (bool, optional): whether parameters should be densly or
                lazily stacked. Defaults to ``False`` (dense stack).

                .. note::
                    ``lazy_stack`` and ``as_module`` are exclusive features.

                .. warning::
                    There is a crucial difference between lazy and non-lazy outputs
                    in that non-lazy output will reinstantiate parameters with the
                    desired batch-size, while ``lazy_stack`` will just represent
                    the parameters as lazily stacked. This means that whilst the
                    original parameters can safely be passed to an optimizer
                    when ``lazy_stack=True``, the new parameters need to be passed
                    when it is set to ``True``.

                .. warning::
                    Whilst it can be tempting to use a lazy stack to keep the
                    orignal parameter references, remember that lazy stack
                    perform a stack each time :meth:`~.get` is called. This will
                    require memory (N times the size of the parameters, more if a
                    graph is built) and time to be computed.
                    It also means that the optimizer(s) will contain more
                    parameters, and operations like :meth:`~torch.optim.Optimizer.step`
                    or :meth:`~torch.optim.Optimizer.zero_grad` will take longer
                    to be executed. In general, ``lazy_stack`` should be reserved
                    to very few use cases.

            expand_identical (bool, optional): if ``True`` and the same parameter (same
                identity) is being stacked to itself, an expanded version of this parameter
                will be returned instead. This argument is ignored when ``lazy_stack=True``.

        Examples:
            >>> from torch import nn
            >>> from tensordict import TensorDict
            >>> torch.manual_seed(0)
            >>> empty_module = nn.Linear(3, 4, device="meta")
            >>> n_models = 2
            >>> modules = [nn.Linear(3, 4) for _ in range(n_models)]
            >>> params = TensorDict.from_modules(*modules)
            >>> print(params)
            TensorDict(
                fields={
                    bias: Parameter(shape=torch.Size([2, 4]), device=cpu, dtype=torch.float32, is_shared=False),
                    weight: Parameter(shape=torch.Size([2, 4, 3]), device=cpu, dtype=torch.float32, is_shared=False)},
                batch_size=torch.Size([2]),
                device=None,
                is_shared=False)
            >>> # example of batch execution
            >>> def exec_module(params, x):
            ...     with params.to_module(empty_module):
            ...         return empty_module(x)
            >>> x = torch.randn(3)
            >>> y = torch.vmap(exec_module, (0, None))(params, x)
            >>> assert y.shape == (n_models, 4)
            >>> # since lazy_stack = False, backprop leaves the original params untouched
            >>> y.sum().backward()
            >>> assert params["weight"].grad.norm() > 0
            >>> assert modules[0].weight.grad is None

        With ``lazy_stack=True``, things are slightly different:

            >>> params = TensorDict.from_modules(*modules, lazy_stack=True)
            >>> print(params)
            LazyStackedTensorDict(
                fields={
                    bias: Tensor(shape=torch.Size([2, 4]), device=cpu, dtype=torch.float32, is_shared=False),
                    weight: Tensor(shape=torch.Size([2, 4, 3]), device=cpu, dtype=torch.float32, is_shared=False)},
                exclusive_fields={
                },
                batch_size=torch.Size([2]),
                device=None,
                is_shared=False,
                stack_dim=0)
            >>> # example of batch execution
            >>> y = torch.vmap(exec_module, (0, None))(params, x)
            >>> assert y.shape == (n_models, 4)
            >>> y.sum().backward()
            >>> assert modules[0].weight.grad is not None


        """
        param_list = [
            cls.from_module(module, use_state_dict=use_state_dict) for module in modules
        ]
        if lazy_stack:
            from tensordict._lazy import LazyStackedTensorDict

            for param in param_list:
                if any(
                    isinstance(tensor, UninitializedTensorMixin)
                    for tensor in param.values(True, True)
                ):
                    raise RuntimeError(
                        "lasy_stack=True is not compatible with lazy modules."
                    )
            params = LazyStackedTensorDict.lazy_stack(param_list)
        elif expand_identical:
            from tensordict._torch_func import _stack_uninit_params

            # Check the keys
            #  If not expand_identical, `stack` takes care of that check but
            #  here we use apply which will ignore keys that are in one TD but not another
            sets = [set(param.keys(True, True)) for param in param_list]
            for set_ in sets[1:]:
                if set_ != sets[0]:
                    raise ValueError(
                        f"All key sets must match. "
                        f"Got {set_.symmetric_difference(sets[0])} in one but not another."
                    )

            def maybe_stack(*params):
                param = params[0]
                if isinstance(param, UninitializedTensorMixin):
                    return _stack_uninit_params(params, 0)
                if len(set(params)) == 1:
                    return param.expand((len(params), *param.shape))
                result = torch.stack(params)
                if isinstance(param, nn.Parameter):
                    return nn.Parameter(result.detach(), param.requires_grad)
                return Buffer(result)

            params = param_list[0]._fast_apply(
                maybe_stack,
                *param_list[1:],
                batch_size=torch.Size([len(param_list), *param_list[0].batch_size]),
            )
        else:
            with set_lazy_legacy(False), torch.no_grad():
                params = torch.stack(param_list)

            # Make sure params are params, buffers are buffers
            def make_param(param, orig_param):
                if isinstance(param, UninitializedTensorMixin):
                    return param
                if isinstance(orig_param, nn.Parameter):
                    return nn.Parameter(param.detach(), orig_param.requires_grad)
                return Buffer(param)

            params = params._fast_apply(make_param, param_list[0], propagate_lock=True)
        if as_module:
            from tensordict.nn import TensorDictParams

            params = TensorDictParams(params, no_convert=True)
        if lock:
            params.lock_()
        return params

    @_as_context_manager()
    def to_module(
        self,
        module: nn.Module,
        *,
        inplace: bool | None = None,
        return_swap: bool = True,
        swap_dest=None,
        use_state_dict: bool = False,
        non_blocking: bool = False,
        preserve_module_state: bool | None = True,
        memo=None,  # deprecated
    ):
        """Writes the content of a TensorDictBase instance onto a given nn.Module attributes, recursively.

        ``to_module`` can also be used a context manager to temporarily populate a module with a collection of
        parameters/buffers (see example below).

        Args:
            module (nn.Module): a module to write the parameters into.

        Keyword Args:
            inplace (bool, optional): if ``True``, the parameters or tensors
                in the module are updated in-place. Defaults to ``None``, which
                behaves like ``False``. Cannot be passed together with
                ``use_state_dict=True``.
            return_swap (bool, optional): if ``True``, the old parameter configuration
                will be returned. Defaults to ``True``.
            swap_dest (TensorDictBase, optional): if ``return_swap`` is ``True``,
                the tensordict where the swap should be written.
            use_state_dict (bool, optional): if ``True``, state-dict API will be
                used to load the parameters (including the state-dict hooks).
                Defaults to ``False``.
            non_blocking (bool, optional): if ``True`` and this copy is between
                different devices, the copy may occur asynchronously with respect
                to the host.
            preserve_module_state (bool, optional): if ``True``, existing
                :class:`~torch.nn.Parameter` and buffer registrations are
                preserved when writing tensor leaves to ``module``: parameters
                remain parameters with their original ``requires_grad`` value,
                and buffers remain registered buffers. Leaves that are not
                parameters but require gradients, or that are batched by
                :func:`~torch.vmap`, are registered as they are, so that
                gradients still reach them. If ``False``, tensor
                leaves are written with the historical replacement semantics,
                which may deregister an existing parameter when the source leaf
                is not an :class:`~torch.nn.Parameter`. Defaults to ``True``.
                Pass ``False`` to retain the historical replacement behavior.

        Examples:
            >>> from torch import nn
            >>> module = nn.TransformerDecoder(
            ...     decoder_layer=nn.TransformerDecoderLayer(nhead=4, d_model=4),
            ...     num_layers=1)
            >>> params = TensorDict.from_module(module)
            >>> params.data.zero_()
            >>> params.to_module(module, preserve_module_state=True)
            >>> assert (module.layers[0].linear1.weight == 0).all()

        Using a tensordict as a context manager can be useful to make functional calls:
        Examples:
            >>> from tensordict import from_module
            >>> module = nn.TransformerDecoder(
            ...     decoder_layer=nn.TransformerDecoderLayer(nhead=4, d_model=4),
            ...     num_layers=1)
            >>> params = TensorDict.from_module(module)
            >>> params = params.data * 0 # Use TensorDictParams to remake these tensors regular nn.Parameter instances
            >>> with params.to_module(module, preserve_module_state=True):
            ...     # Call the module with zeroed params
            ...     y = module(*inputs)
            >>> # The module is repopulated with its original params
            >>> assert (TensorDict.from_module(module) != 0).any()

        Returns:
            A tensordict containing the values from the module if ``return_swap`` is ``True``, ``None`` otherwise.

        """
        if memo is not None:
            raise RuntimeError("memo cannot be passed to the public to_module anymore.")
        hooks = torch.nn.modules.module._global_parameter_registration_hooks
        memo = {"hooks": tuple(hooks.values())}
        return self._to_module(
            module=module,
            inplace=inplace,
            return_swap=return_swap,
            swap_dest=swap_dest,
            memo=memo,
            use_state_dict=use_state_dict,
            non_blocking=non_blocking,
            preserve_module_state=preserve_module_state,
        )

    def _to_module(
        self,
        module: nn.Module,
        *,
        inplace: bool | None = None,
        return_swap: bool = True,
        swap_dest=None,
        memo=None,
        use_state_dict: bool = False,
        non_blocking: bool = False,
        preserve_module_state: bool | None = True,
        is_dynamo: bool | None = None,
    ):
        from tensordict.base import TensorDictBase

        if is_dynamo is None:
            is_dynamo = is_compiling()

        if not use_state_dict and isinstance(module, TensorDictBase):
            if return_swap:
                swap = module.copy()
                module._param_td = getattr(self, "_param_td", self)
                return swap
            else:
                module.update(self)
                return

        hooks = memo["hooks"]
        if return_swap:
            _swap = {}
            if not is_dynamo:
                memo[weakref.ref(module)] = _swap

        if use_state_dict:
            if inplace is not None:
                raise RuntimeError(
                    "inplace argument cannot be passed when use_state_dict=True."
                )
            # execute module's pre-hooks
            state_dict = self.flatten_keys(".")
            prefix = ""
            strict = True
            local_metadata = {}
            missing_keys = []
            unexpected_keys = []
            error_msgs = []
            for hook in module._load_state_dict_pre_hooks.values():
                hook(
                    state_dict,
                    prefix,
                    local_metadata,
                    strict,
                    missing_keys,
                    unexpected_keys,
                    error_msgs,
                )

            def convert_type(x, y):
                if isinstance(y, nn.Parameter):
                    return nn.Parameter(x)
                if isinstance(y, Buffer):
                    return Buffer(x)
                return x

            input = state_dict.unflatten_keys(".")._fast_apply(
                convert_type, self, propagate_lock=True
            )
        else:
            input = self
            inplace = bool(inplace)

        # we use __dict__ directly to avoid the getattr/setattr overhead whenever we can
        if not is_dynamo and type(module).__setattr__ is __base__setattr__:
            # if type(module).__setattr__ is __base__setattr__:
            __dict__ = module.__dict__
            _parameters = __dict__["_parameters"]
            _buffers = __dict__["_buffers"]
        else:
            __dict__ = None

        for key, value in input.items():
            if isinstance(value, (Tensor, ftdim.Tensor)):
                # For Dynamo, we use regular set/delattr as we're not
                #  much afraid by overhead (and dynamo doesn't like those
                #  hacks we're doing).
                if __dict__ is not None:
                    # if setattr is the native nn.Module.setattr, we can rely on _set_tensor_dict
                    local_out = _set_tensor_dict(
                        __dict__,
                        _parameters,
                        _buffers,
                        hooks,
                        module,
                        key,
                        value,
                        inplace,
                        return_swap=return_swap,
                        preserve_module_state=preserve_module_state,
                        memo=memo,
                    )
                else:
                    if not inplace:
                        value = _maybe_preserve_module_state(
                            module,
                            key,
                            value,
                            preserve_module_state=preserve_module_state,
                            memo=memo,
                        )
                        local_out = swap_tensor(module, key, value)
                    else:
                        new_val = local_out
                        if return_swap:
                            local_out = local_out.clone()
                        new_val.data.copy_(value.data, non_blocking=non_blocking)
            else:
                if __dict__ is not None:
                    child = __dict__["_modules"][key]
                else:
                    child = module._modules.get(key)

                if not is_dynamo:
                    local_out = memo.get(weakref.ref(child), NO_DEFAULT)

                if is_dynamo or local_out is NO_DEFAULT:
                    local_out = value._to_module(
                        child,
                        inplace=inplace,
                        return_swap=return_swap,
                        swap_dest={},  # we'll be calling update later
                        memo=memo,
                        use_state_dict=use_state_dict,
                        non_blocking=non_blocking,
                        preserve_module_state=preserve_module_state,
                        is_dynamo=is_dynamo,
                    )

            if return_swap:
                _swap[key] = local_out

        if return_swap:
            if isinstance(swap_dest, dict):
                return _swap
            elif swap_dest is not None:

                def _quick_set(swap_dict, swap_td):
                    for key, val in swap_dict.items():
                        if isinstance(val, dict):
                            _quick_set(val, swap_td._get_str(key, default=NO_DEFAULT))
                        elif swap_td._get_str(key, None) is not val:
                            swap_td._set_str(
                                key,
                                val,
                                inplace=False,
                                validated=True,
                                non_blocking=non_blocking,
                            )

                _quick_set(_swap, swap_dest)
                return swap_dest
            else:
                return self._new_unsafe(_swap, batch_size=torch.Size(()))

    @classmethod
    def fromkeys(cls, keys: List[NestedKey], value: Any = 0):
        """Creates a tensordict from a list of keys and a single value.

        Args:
            keys (list of NestedKey): An iterable specifying the keys of the new dictionary.
            value (compatible type, optional): The value for all keys. Defaults to ``0``.
        """
        from tensordict._td import TensorDict

        return TensorDict(dict.fromkeys(keys, value), batch_size=[])

    to_mds = to_mds

    def _convert_to_tensordict(
        self, dict_value: dict[str, Any], non_blocking: bool | None = None
    ) -> Self:
        from tensordict._td import TensorDict

        return TensorDict(
            dict_value,
            batch_size=self.batch_size,
            device=self.device,
            names=self._maybe_names(),
            lock=self.is_locked,
            non_blocking=non_blocking,
        )

    def to_tensordict(self, *, retain_none: bool | None = None) -> Self:
        """Returns a regular TensorDict instance from the TensorDictBase.

        Args:
            retain_none (bool): if ``True``, the ``None`` values from tensorclass instances
                will be written in the tensordict.
                Otherwise they will be discarded. Default: ``True``.

        Returns:
            a new TensorDict object containing the same values.

        """
        from tensordict import TensorDict

        return TensorDict(
            {
                key: (
                    value.clone()
                    if not _is_tensor_collection(type(value))
                    else (
                        value
                        if is_non_tensor(value)
                        else (
                            value.clone()
                            if _is_unbatched(value)
                            else value.to_tensordict(retain_none=retain_none)
                        )
                    )
                )
                for key, value in self.items(is_leaf=_is_leaf_nontensor)
            },
            device=self.device,
            batch_size=self.batch_size,
            names=self._maybe_names(),
        )

    def to_lazystack(self, dim: int = 0):
        """Converts a TensorDict to a LazyStackedTensorDict or equivalent.

        .. note::
            This method can be used to swap the stack dimension of a LazyStackedTensorDict.
            For example, if you have a LazyStackedTensorDict with stack_dim=1, you can use this method to swap it to stack_dim=0:

            >>> td = TensorDict({"a": torch.zeros(2, 3), "b": torch.ones(2, 3)}, batch_size=(2, 3))
            >>> td2 = td.to_lazystack()
            >>> td2.batch_size
            torch.Size([2, 3])
            >>> assert isinstance(td2, LazyStackedTensorDict)
            >>> assert td2.stack_dim == 0
            >>> td3 = td2.to_lazystack(1)
            >>> assert td3.stack_dim == 1
            >>> td3.batch_size
            torch.Size([2, 3])

        Args:
            dim (int, optional): the dimension along which to stack the tensordict.
                Defaults to ``0``.

        Returns:
            A LazyStackedTensorDict instance.

        Examples:
            >>> from tensordict import TensorDict
            >>> td = TensorDict({"a": torch.zeros(2, 3), "b": torch.ones(2, 3)}, batch_size=(2, 3))
            >>> td2 = td.to_lazystack()
            >>> td2.batch_size
            torch.Size([2, 3])
            >>> assert isinstance(td2, LazyStackedTensorDict)

        """
        from tensordict import lazy_stack, LazyStackedTensorDict
        from tensordict.tensorclass import _is_tensorclass

        dim = _maybe_correct_neg_dim(dim, ndim=self.ndim, shape=None)
        if (isinstance(self, LazyStackedTensorDict) and self.stack_dim == dim) or (
            _is_tensorclass(type(self))
            and isinstance(self._tensordict, LazyStackedTensorDict)
            and self._tensordict.stack_dim == dim
        ):
            return self
        return lazy_stack(self.unbind(dim), dim=dim)

    def to_dict(
        self,
        *,
        retain_none: bool = True,
        convert_tensors: bool | Literal["numpy"] = False,
        tolist_first: bool = False,
    ) -> dict[str, Any]:
        """Returns a dictionary with key-value pairs matching those of the tensordict.

        Args:
            retain_none (bool): if ``True``, the ``None`` values from tensorclass instances
                will be written in the dictionary.
                Otherwise, they will be discarded. Default: ``True``.
            convert_tensors (bool, "numpy"): if ``True``, tensors will be converted to lists when creating the dictionary.
                If "numpy", tensors will be converted to numpy arrays.
                Otherwise, they will remain as tensors. Default: ``False``.
            tolist_first (bool): if ``True``, the tensordict will be converted to a list first when
                it has batch dimensions. Default: ``False``.

        Returns:
            A dictionary representation of the tensordict.

        .. seealso:: :meth:`~tensordict.TensorDictBase.tolist`

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>>
            >>> td = TensorDict(
            ...     a=torch.arange(6).view(2, 3),
            ...     b=TensorDict(c=torch.arange(4).reshape(2, 2), batch_size=(2, 2)),
            ...     batch_size=(2,)
            ... )
            >>> print(td.to_dict())
            {'a': tensor([[0, 1, 2],
                    [3, 4, 5]]), 'b': {'c': tensor([[0, 1],
                    [2, 3]])}}
            >>> print(td.to_dict(convert_tensors=True))
            {'a': [[0, 1, 2], [3, 4, 5]], 'b': {'c': [[0, 1], [2, 3]]}}

        """
        result = {}
        for key, value in self.items():
            if _is_tensor_collection(type(value)):
                # NonTensorStack.data raises AttributeError when the stacked
                # values differ: such a stack is not None, so it is kept.
                if (
                    not retain_none
                    and _is_non_tensor(type(value))
                    and getattr(value, "data", NO_DEFAULT) is None
                ):
                    continue
                if tolist_first:
                    value = value.tolist(convert_tensors=convert_tensors)
                else:
                    value = value.to_dict(
                        retain_none=retain_none, convert_tensors=convert_tensors
                    )
            elif convert_tensors:
                if isinstance(value, torch.Tensor) and convert_tensors == "numpy":
                    value = value.numpy()
                elif hasattr(value, "tolist"):
                    value = value.tolist()
            result[key] = value
        return result

    def tolist(
        self,
        *,
        convert_nodes: bool = True,
        convert_tensors: bool | Literal["numpy"] = False,
        tolist_first: bool = False,
        as_linked_list: bool = False,
    ) -> List[Any]:
        """Returns a nested list representation of the tensordict.

        If the tensordict has no batch dimensions, this method returns a single list or dictionary.
        Otherwise, it returns a nested list where each inner list represents a batch dimension.

        Args:
            convert_nodes (bool): if ``True``, leaf nodes will be converted to dictionaries.
                Otherwise, they will be returned as tensordicts. Default: ``True``.
            convert_tensors (bool, "numpy"): if ``True``, tensors will be converted to lists when creating the dictionary.
                If "numpy", tensors will be converted to numpy arrays.
                Otherwise, they will remain as tensors. Default: ``False``.
            tolist_first (bool): if ``True``, the tensordict will be converted to a list first when
                it has batch dimensions. Default: ``False``.
            as_linked_list (bool): if ``True``, the list will be converted to a :class:`tensordict.utils.LinkedList`
                which will automatically update the tensordict when the list is modified. Default: ``False``.

        Returns:
            A nested list representation of the tensordict.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>>
            >>> td = TensorDict(
            ...     a=torch.arange(24).view(2, 3, 4),
            ...     b=TensorDict(c=torch.arange(12).reshape(2, 3, 2), batch_size=(2, 3, 2)),
            ...     batch_size=(2, 3)
            ... )
            >>> print(td.tolist(tolist_first=True))
            [[{'a': tensor([0, 1, 2, 3]), 'b': [{'c': tensor(0)}, {'c': tensor(1)}]}, {'a': tensor([4, 5, 6, 7]), 'b': [{'c': tensor(2)}, {'c': tensor(3)}]}, {'a': tensor([ 8,  9, 10, 11]), 'b': [{'c': tensor(4)}, {'c': tensor(5)}]}], [{'a': tensor([12, 13, 14, 15]), 'b': [{'c': tensor(6)}, {'c': tensor(7)}]}, {'a': tensor([16, 17, 18, 19]), 'b': [{'c': tensor(8)}, {'c': tensor(9)}]}, {'a': tensor([20, 21, 22, 23]), 'b': [{'c': tensor(10)}, {'c': tensor(11)}]}]]
            >>> print(td.tolist(tolist_first=False))
            [[{'a': tensor([0, 1, 2, 3]), 'b': {'c': tensor([0, 1])}}, {'a': tensor([4, 5, 6, 7]), 'b': {'c': tensor([2, 3])}}, {'a': tensor([ 8,  9, 10, 11]), 'b': {'c': tensor([4, 5])}}], [{'a': tensor([12, 13, 14, 15]), 'b': {'c': tensor([6, 7])}}, {'a': tensor([16, 17, 18, 19]), 'b': {'c': tensor([8, 9])}}, {'a': tensor([20, 21, 22, 23]), 'b': {'c': tensor([10, 11])}}]]
            >>> print(td.tolist(convert_tensors=True))
            [[{'a': [0, 1, 2, 3], 'b': {'c': [0, 1]}}, {'a': [4, 5, 6, 7], 'b': {'c': [2, 3]}}, {'a': [8, 9, 10, 11], 'b': {'c': [4, 5]}}], [{'a': [12, 13, 14, 15], 'b': {'c': [6, 7]}}, {'a': [16, 17, 18, 19], 'b': {'c': [8, 9]}}, {'a': [20, 21, 22, 23], 'b': {'c': [10, 11]}}]]
            >>> print(td.tolist(convert_nodes=False)[0][0])
            TensorDict(
                fields={
                    a: Tensor(shape=torch.Size([4]), device=cpu, dtype=torch.int64, is_shared=False),
                    b: TensorDict(
                        fields={
                            c: Tensor(shape=torch.Size([2]), device=cpu, dtype=torch.int64, is_shared=False)},
                        batch_size=torch.Size([2]),
                        device=None,
                        is_shared=False)},
                batch_size=torch.Size([]),
                device=None,
                is_shared=False)

        """
        if convert_tensors and not convert_nodes:
            raise TypeError("convert_tensors requires convert_nodes to be set to True")
        if not self.batch_dims:
            if convert_nodes:
                return self.to_dict(
                    convert_tensors=convert_tensors, tolist_first=tolist_first
                )
            return self

        q = collections.deque()
        result = []
        q.append((self, result))
        while len(q):
            val, _result = q.popleft()
            vals = val.unbind(0)
            if val.ndim == 1:
                if convert_nodes:
                    vals = [
                        v.to_dict(
                            convert_tensors=convert_tensors, tolist_first=tolist_first
                        )
                        for v in vals
                    ]
                else:
                    vals = list(vals)
                _result.extend(vals)
            else:
                for local_val in vals:
                    local_res = []
                    _result.append(local_res)
                    q.append((local_val, local_res))
        if as_linked_list:
            return LinkedList(result, td=self)
        return result

    def numpy(self) -> np.ndarray | dict[str, Any]:
        """Converts a tensordict to a (possibly nested) dictionary of numpy arrays.

        Non-tensor data is exposed as such.

        Examples:
            >>> from tensordict import TensorDict
            >>> import torch
            >>> data = TensorDict({"a": {"b": torch.zeros(()), "c": "a string!"}})
            >>> print(data)
            TensorDict(
                fields={
                    a: TensorDict(
                        fields={
                            b: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
                            c: NonTensorData(data=a string!, batch_size=torch.Size([]), device=None)},
                        batch_size=torch.Size([]),
                        device=None,
                        is_shared=False)},
                batch_size=torch.Size([]),
                device=None,
                is_shared=False)
            >>> print(data.numpy())
            {'a': {'b': array(0., dtype=float32), 'c': 'a string!'}}

        seealso: :meth:`~tensordict.TensorDictBase.to_struct_array` to convert to a struct array.

        """
        as_dict = self.to_dict(retain_none=False)

        def to_numpy(x):
            if isinstance(x, torch.Tensor):
                if x.is_nested:
                    return tuple(_x.numpy() for _x in x)
                return x.numpy()
            if hasattr(x, "numpy"):
                return x.numpy()
            return x

        return torch.utils._pytree.tree_map(to_numpy, as_dict)

    def to_namedtuple(self, dest_cls: type | None = None) -> Any:
        """Converts a tensordict to a namedtuple.

        Args:
            dest_cls (Type, optional): an optional namedtuple class to use.

        Examples:
            >>> from tensordict import TensorDict
            >>> import torch
            >>> data = TensorDict({
            ...     "a_tensor": torch.zeros((3)),
            ...     "nested": {"a_tensor": torch.zeros((3)), "a_string": "zero!"}}, [3])
            >>> data.to_namedtuple()
            GenericDict(a_tensor=tensor([0., 0., 0.]), nested=GenericDict(a_tensor=tensor([0., 0., 0.]), a_string='zero!'))

        """

        def dict_to_namedtuple(dictionary):
            for key, value in dictionary.items():
                if isinstance(value, dict):
                    dictionary[key] = dict_to_namedtuple(value)
            cls = (
                collections.namedtuple("GenericDict", dictionary.keys())
                if dest_cls is None
                else dest_cls
            )
            return cls(**dictionary)

        return dict_to_namedtuple(self.to_dict(retain_none=False))

    @classmethod
    def from_any(
        cls,
        obj,
        *,
        auto_batch_size: bool = False,
        batch_dims: int | None = None,
        device: torch.device | None = None,
        batch_size: torch.Size | None = None,
    ):
        """Recursively converts any object to a TensorDict.

        .. note::  ``from_any`` is less restrictive than the regular TensorDict constructor. It can cast data structures like
            dataclasses or tuples to a tensordict using custom heuristics. This approach may incur some extra overhead and
            involves more opinionated choices in terms of mapping strategies.

        .. note:: This method recursively converts the input object to a TensorDict. If the object is already a
            TensorDict (or any similar tensor collection object), it will be returned as is.

        Args:
            obj: The object to be converted.

        Keyword Args:
            auto_batch_size (bool, optional): if ``True``, the batch size will be computed automatically.
                Defaults to ``False``.
            batch_dims (int, optional): If auto_batch_size is ``True``, defines how many dimensions the output tensordict
                should have. Defaults to ``None`` (full batch-size at each level).
            device (torch.device, optional): The device on which the TensorDict will be created.
            batch_size (torch.Size, optional): The batch size of the TensorDict.
                Exclusive with ``auto_batch_size``.

        Returns:
            A TensorDict representation of the input object.

        Supported objects:

        - Dataclasses through :meth:`~.from_dataclass` (dataclasses will be converted to TensorDict instances, not tensorclasses).
        - Namedtuples through :meth:`~.from_namedtuple`.
        - Dictionaries through :meth:`~.from_dict`.
        - Tuples through :meth:`~.from_tuple`.
        - NumPy's structured arrays through :meth:`~.from_struct_array`.
        - HDF5 objects through :meth:`~.from_h5`.

        """
        if type(obj) is Tensor or is_tensor_collection(obj):
            # Conversions from non-tensor data must be done manually
            # if is_non_tensor(obj):
            #     from tensordict.tensorclass import LazyStackedTensorDict
            #     if isinstance(obj, LazyStackedTensorDict):
            #         return obj
            #     return cls.from_any(obj.data, auto_batch_size=auto_batch_size)
            return obj
        if isinstance(obj, dict):
            return cls.from_dict(
                obj,
                auto_batch_size=auto_batch_size,
                batch_dims=batch_dims,
                device=device,
                batch_size=batch_size,
            )
        if isinstance(obj, UserDict):
            return cls.from_dict(
                dict(obj),
                auto_batch_size=auto_batch_size,
                batch_dims=batch_dims,
                device=device,
                batch_size=batch_size,
            )
        if (
            isinstance(obj, np.ndarray)
            and hasattr(obj.dtype, "names")
            and obj.dtype.names is not None
        ):
            return cls.from_struct_array(
                obj,
                auto_batch_size=auto_batch_size,
                batch_dims=batch_dims,
                device=device,
                batch_size=batch_size,
            )
        if isinstance(obj, tuple):
            if _is_namedtuple(obj):
                return cls.from_namedtuple(
                    obj,
                    auto_batch_size=auto_batch_size,
                    batch_dims=batch_dims,
                    device=device,
                    batch_size=batch_size,
                )
            return cls.from_tuple(
                obj,
                auto_batch_size=auto_batch_size,
                batch_dims=batch_dims,
                device=device,
                batch_size=batch_size,
            )
        if isinstance(obj, list):
            if _is_list_tensor_compatible(obj)[0]:
                return torch.tensor(obj)
            else:
                from tensordict.tensorclass import NonTensorStack

                return NonTensorStack.from_list(obj)
        if is_dataclass(obj):
            dataclass_batch_dims = _int_batch_dims(batch_dims)
            if auto_batch_size and dataclass_batch_dims is not None:
                try:
                    return cls.from_dataclass(
                        obj,
                        auto_batch_size=auto_batch_size,
                        batch_dims=dataclass_batch_dims,
                        device=device,
                        batch_size=batch_size,
                    )
                except Exception:
                    # for backward compatibility, a batch_dims that cannot be
                    # applied, for example to a nested NonTensorStack, is ignored
                    pass
            return cls.from_dataclass(
                obj,
                auto_batch_size=auto_batch_size,
                device=device,
                batch_size=batch_size,
            )
        if not is_compiling() and importlib.util.find_spec("pandas") is not None:
            import pandas as pd

            if isinstance(obj, pd.DataFrame):
                return cls.from_pandas(
                    obj,
                    auto_batch_size=auto_batch_size,
                    batch_dims=batch_dims,
                    device=device,
                    batch_size=batch_size,
                )
        if _has_h5:
            import h5py

            if isinstance(obj, h5py.File):
                from tensordict import TensorDict
                from tensordict.persistent import PersistentTensorDict

                if not auto_batch_size and batch_size is not None:
                    try:
                        return PersistentTensorDict(
                            group=obj,
                            batch_size=TensorDict._parse_batch_size(None, batch_size),
                        )
                    except Exception:
                        # for backward compatibility, a batch size that is not a
                        # size or that the file does not take is ignored
                        pass
                obj = PersistentTensorDict(group=obj)
                if auto_batch_size:
                    obj.auto_batch_size_(batch_dims=_int_batch_dims(batch_dims))
                return obj
        return obj

    @classmethod
    def from_tuple(
        cls,
        obj,
        *,
        auto_batch_size: bool = False,
        batch_dims: int | None = None,
        device: torch.device | None = None,
        batch_size: torch.Size | None = None,
    ):
        """Converts a tuple to a TensorDict.

        Args:
            obj: The tuple instance to be converted.

        Keyword Args:
            auto_batch_size (bool, optional): If ``True``, the batch size will be computed automatically. Defaults to ``False``.
            batch_dims (int, optional): If auto_batch_size is ``True``, defines how many dimensions the output tensordict
                should have. Defaults to ``None`` (full batch-size at each level).
            device (torch.device, optional): The device on which the TensorDict will be created. Defaults to ``None``.
            batch_size (torch.Size, optional): The batch size of the TensorDict. Defaults to ``None``.

        Returns:
            A TensorDict representation of the input tuple.

        Examples:
            >>> my_tuple = (1, 2, 3)
            >>> td = TensorDict.from_tuple(my_tuple)
            >>> print(td)
            TensorDict(
                fields={
                    0: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int64, is_shared=False),
                    1: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int64, is_shared=False),
                    2: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int64, is_shared=False)},
                batch_size=torch.Size([]),
                device=None,
                is_shared=False)

        """
        from tensordict import TensorDict

        result = TensorDict(
            {
                str(i): cls.from_any(item, batch_size=batch_size, device=device)
                for i, item in enumerate(obj)
            },
            batch_size=batch_size,
            device=device,
        )
        if auto_batch_size:
            if batch_size is not None:
                raise TypeError(cls._CONFLICTING_BATCH_SIZES.format("from_tuple"))
            result.auto_batch_size_(batch_dims=batch_dims)
        return result

    @classmethod
    def from_dataclass(
        cls,
        dataclass,
        *,
        dest_cls: Type | None = None,
        auto_batch_size: bool = False,
        batch_dims: int | None = None,
        as_tensorclass: bool = False,
        device: torch.device | None = None,
        batch_size: torch.Size | None = None,
    ):
        """Converts a dataclass into a TensorDict instance.

        Args:
            dataclass: The dataclass instance to be converted.

        Keyword Args:
            dest_cls (tensorclass, optional): A tensorclass type to be used to map the data. If not provided, a new
                class is created. Without effect if :attr:`obj` is a type or as_tensorclass is `False`.
            auto_batch_size (bool, optional): If ``True``, automatically determines and applies batch size to the
                resulting TensorDict. Defaults to ``False``.
            batch_dims (int, optional): If ``auto_batch_size`` is ``True``, defines how many dimensions the output
                tensordict should have. Defaults to ``None`` (full batch-size at each level).
            as_tensorclass (bool, optional): If ``True``, delegates the conversion to the free function
                :func:`~tensordict.from_dataclass` and returns a tensor-compatible class (:func:`~tensordict.tensorclass`)
                or instance instead of a TensorDict. Defaults to ``False``.
            device (torch.device, optional): The device on which the TensorDict will be created.
                Defaults to ``None``.
            batch_size (torch.Size, optional): The batch size of the TensorDict.
                Defaults to ``None``.

        Returns:
            A TensorDict instance derived from the provided dataclass, unless `as_tensorclass` is True, in which case a tensor-compatible class or instance is returned.

        Raises:
            TypeError: If the provided input is not a dataclass instance.

        .. warning:: This method is distinct from the free function `from_dataclass` and serves a different purpose.
            While the free function returns a tensor-compatible class or instance, this method returns a TensorDict instance.

        .. note::
            - This method creates a new TensorDict instance with keys corresponding to the fields of the input dataclass.
            - Each key in the resulting TensorDict is initialized using the `cls.from_any` method.
            - The `auto_batch_size` option allows for automatic batch size determination and application to the
              resulting TensorDict.

        """
        if as_tensorclass:
            from tensordict.tensorclass import from_dataclass

            return from_dataclass(
                dataclass,
                auto_batch_size=auto_batch_size,
                dest_cls=dest_cls,
                batch_dims=batch_dims,
                batch_size=batch_size,
                device=device,
            )
        from dataclasses import fields

        from tensordict import TensorDict

        if not is_dataclass(dataclass):
            raise TypeError(
                f"Expected a dataclass input, got a {type(dataclass)} input instead."
            )
        source = {}
        for field in fields(dataclass):
            source[field.name] = cls.from_any(
                getattr(dataclass, field.name), device=device, batch_size=batch_size
            )
        result = TensorDict(source, device=device, batch_size=batch_size)
        if auto_batch_size:
            if batch_size is not None:
                raise TypeError(cls._CONFLICTING_BATCH_SIZES.format("from_dataclass"))
            result.auto_batch_size_(batch_dims=batch_dims)
        return result

    @classmethod
    def from_namedtuple(
        cls,
        named_tuple,
        *,
        auto_batch_size: bool = False,
        batch_dims: int | None = None,
        device: torch.device | None = None,
        batch_size: torch.Size | None = None,
    ):
        """Converts a namedtuple to a TensorDict recursively.

        Args:
            named_tuple: The namedtuple instance to be converted.

        Keyword Args:
            auto_batch_size (bool, optional): if ``True``, the batch size will be computed automatically.
                Defaults to ``False``.
            batch_dims (int, optional): If ``auto_batch_size`` is ``True``, defines how many dimensions the output
                tensordict should have. Defaults to ``None`` (full batch-size at each level).
            device (torch.device, optional): The device on which the TensorDict will be created.
                Defaults to ``None``.
            batch_size (torch.Size, optional): The batch size of the TensorDict.
                Defaults to ``None``.

        Returns:
            A TensorDict representation of the input namedtuple.

        Examples:
            >>> from tensordict import TensorDict
            >>> import torch
            >>> data = TensorDict({
            ...     "a_tensor": torch.zeros((3)),
            ...     "nested": {"a_tensor": torch.zeros((3)), "a_string": "zero!"}}, [3])
            >>> nt = data.to_namedtuple()
            >>> print(nt)
            GenericDict(a_tensor=tensor([0., 0., 0.]), nested=GenericDict(a_tensor=tensor([0., 0., 0.]), a_string='zero!'))
            >>> TensorDict.from_namedtuple(nt, auto_batch_size=True)
            TensorDict(
                fields={
                    a_tensor: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False),
                    nested: TensorDict(
                        fields={
                            a_string: NonTensorData(data=zero!, batch_size=torch.Size([3]), device=None),
                            a_tensor: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False)},
                        batch_size=torch.Size([3]),
                        device=None,
                        is_shared=False)},
                batch_size=torch.Size([3]),
                device=None,
                is_shared=False)

        """
        from tensordict import TensorDict

        def namedtuple_to_dict(namedtuple_obj):
            if _is_namedtuple(namedtuple_obj):
                namedtuple_obj = namedtuple_obj._asdict()
            elif is_structseq_instance(namedtuple_obj):
                # torch.return_types (the results of max, sort, topk, ...)
                # are structseqs: named fields, but no _fields or _asdict.
                namedtuple_obj = {
                    name: getattr(namedtuple_obj, name)
                    for name in type(namedtuple_obj).__match_args__
                }
            for key, value in namedtuple_obj.items():
                namedtuple_obj[key] = cls.from_any(
                    value, device=device, batch_size=batch_size
                )
            return dict(namedtuple_obj)

        result = TensorDict(
            namedtuple_to_dict(named_tuple), device=device, batch_size=batch_size
        )
        if auto_batch_size:
            if batch_size is not None:
                raise TypeError(cls._CONFLICTING_BATCH_SIZES.format("from_namedtuple"))
            result.auto_batch_size_(batch_dims=batch_dims)
        return result

    @classmethod
    def from_struct_array(
        cls,
        struct_array: np.ndarray,
        *,
        auto_batch_size: bool = False,
        batch_dims: int | None = None,
        device: torch.device | None = None,
        batch_size: torch.Size | None = None,
    ) -> Self:
        """Converts a structured numpy array to a TensorDict.

        The resulting TensorDict will share the same memory content as the numpy array (it is a zero-copy operation).
        Changing values of the structured numpy array in-place will affect the content of the TensorDict.

        .. note:: This method performs a zero-copy operation, meaning that the resulting TensorDict will share the same memory
            content as the input numpy array. Therefore, changing values of the numpy array in-place will affect the content
            of the TensorDict.

        Args:
            struct_array (np.ndarray): The structured numpy array to be converted.

        Keyword Args:
            auto_batch_size (bool, optional): If ``True``, the batch size will be computed automatically. Defaults to ``False``.
            batch_dims (int, optional): If ``auto_batch_size`` is ``True``, defines how many dimensions the output
                tensordict should have. Defaults to ``None`` (full batch-size at each level).
            device (torch.device, optional): The device on which the TensorDict will be created.
                Defaults to ``None``.

                .. note::  Changing the device (i.e., specifying any device other than ``None`` or ``"cpu"``) will transfer the data,
                    resulting in a change to the memory location of the returned data.

            batch_size (torch.Size, optional): The batch size of the TensorDict. Defaults to None.

        Returns:
            A TensorDict representation of the input structured numpy array.

        Examples:
            >>> x = np.array(
            ...     [("Rex", 9, 81.0), ("Fido", 3, 27.0)],
            ...     dtype=[("name", "U10"), ("age", "i4"), ("weight", "f4")],
            ... )
            >>> td = TensorDict.from_struct_array(x)
            >>> x_recon = td.to_struct_array()
            >>> assert (x_recon == x).all()
            >>> assert x_recon.shape == x.shape
            >>> # Try modifying x age field and check effect on td
            >>> x["age"] += 1
            >>> assert (td["age"] == np.array([10, 4])).all()

        """
        from tensordict.base import TensorDictBase

        if cls is TensorDictBase:
            from tensordict._td import TensorDict

            cls = TensorDict
        td: Self = cls(
            {name: struct_array[name] for name in struct_array.dtype.names},
            batch_size=struct_array.shape if batch_size is None else batch_size,
            device=device,
        )
        if auto_batch_size:
            if batch_size is not None:
                raise TypeError(
                    cls._CONFLICTING_BATCH_SIZES.format("from_struct_array")
                )
            td.auto_batch_size_(batch_dims=batch_dims)
        return td

    def to_struct_array(self) -> np.ndarray:
        """Converts a tensordict to a numpy structured array.

        In a :meth:`.from_struct_array` - :meth:`.to_struct_array` loop, the content of the input and output arrays should match.
        However, `to_struct_array` will not keep the memory content of the original arrays.

        .. seealso:: :meth:`.from_struct_array` for more information.

        .. seealso:: :meth:`.numpy` to convert to a dictionary of numpy arrays.

        Returns:
            A numpy structured array representation of the input TensorDict.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict({'a': torch.tensor([1, 2, 3]), 'b': torch.tensor([4.0, 5.0, 6.0])}, batch_size=[3])
            >>> arr = td.to_struct_array()
            >>> print(arr)
            [(1, 4.) (2, 5.) (3, 6.)]

        """
        from tensordict.utils import _TORCH_TO_NUMPY_DTYPE_DICT

        keys, vals = zip(*self.items())
        _vals = []
        for v in vals:
            if is_tensor_collection(v):
                if is_non_tensor(v):
                    from tensordict import NonTensorDataBase

                    _vals.append(
                        v.data if isinstance(v, NonTensorDataBase) else v.tolist()
                    )
                    continue
                _vals.append(v.to_struct_array())
                continue
            _vals.append(v)
        vals = _vals
        del _vals
        vals = tuple(v if not is_non_tensor(v) else v.data for v in vals)

        # Convert values to numpy arrays and handle string inputs
        processed_vals = []
        for v in vals:
            if isinstance(v, torch.Tensor):
                processed_vals.append(v.detach().cpu().numpy())
            elif isinstance(v, (list, tuple, str)):
                # Handle lists/tuples which may contain strings, or strings
                processed_vals.append(np.array(v))
            else:
                # Keep other types as-is (already numpy arrays, etc.)
                processed_vals.append(v)
        vals = processed_vals

        def _get_dtype(val):
            if isinstance(val, np.ndarray):
                if val.dtype.kind in ["U", "S"]:  # Unicode or byte strings
                    # Calculate appropriate string length
                    if val.size > 0:
                        max_len = max(len(str(item)) for item in val.flat)
                        return (
                            f"U{max(10, max_len)}"  # At least U10, but longer if needed
                        )
                    return "U10"
                elif val.ndim > self.ndim:
                    # For arrays with more dimensions than batch dims, we need to specify shape
                    extra_shape = val.shape[self.ndim :]
                    return (val.dtype, extra_shape)
                return val.dtype
            elif isinstance(val, torch.Tensor):
                return _TORCH_TO_NUMPY_DTYPE_DICT.get(val.dtype, val.dtype)
            else:
                return "U10"

        dtype = [(key, _get_dtype(val)) for key, val in zip(keys, vals)]

        if self.ndim:
            # For multi-dimensional tensordicts, we need to create structured arrays properly
            # Reshape each value to have batch dimensions first, then flatten the batch dimensions
            batch_shape = self.shape
            batch_size = int(np.prod(batch_shape))

            # Reshape and prepare data for structured array
            reshaped_vals = []
            for val in vals:
                if isinstance(val, np.ndarray):
                    # Ensure the array has the right batch shape
                    if val.shape[: self.ndim] == batch_shape:
                        # Flatten batch dimensions
                        new_shape = (batch_size,) + val.shape[self.ndim :]
                        reshaped_vals.append(val.reshape(new_shape))
                    else:
                        # If shapes don't match, try to broadcast
                        try:
                            reshaped_vals.append(
                                np.broadcast_to(
                                    val, batch_shape + val.shape[self.ndim :]
                                ).reshape((batch_size,) + val.shape[self.ndim :])
                            )
                        except ValueError:
                            reshaped_vals.append(val)
                else:
                    reshaped_vals.append(val)

            # Create structured array
            result = np.empty(batch_size, dtype=dtype)
            for key, val in zip(keys, reshaped_vals):
                if isinstance(val, np.ndarray) and val.shape[0] == batch_size:
                    result[key] = val
                else:
                    result[key] = val

            # Reshape back to original batch shape
            return result.reshape(batch_shape)

        # For scalar case, create structured array properly
        result = np.empty((), dtype=dtype)
        for key, val in zip(keys, vals):
            if isinstance(val, np.ndarray) and val.ndim == 0:
                result[key] = val.item()
            elif isinstance(val, (np.ndarray, torch.Tensor)) and val.size == 1:
                result[key] = val.item()
            else:
                result[key] = val
        return result

    @classmethod
    def from_pandas(
        cls,
        dataframe,
        *,
        auto_batch_size: bool = False,
        batch_dims: int | None = None,
        device: torch.device | None = None,
        batch_size: torch.Size | None = None,
        separator: str | None = None,
        dtype: torch.dtype | None = None,
    ) -> Self:
        """Converts a pandas DataFrame to a TensorDict.

        Numeric columns become tensors, string/object columns become
        :class:`~tensordict.NonTensorData`.

        Args:
            dataframe (pd.DataFrame): The pandas DataFrame to convert.

        Keyword Args:
            auto_batch_size (bool, optional): If ``True``, the batch size will
                be computed automatically. Defaults to ``False``.
            batch_dims (int, optional): If ``auto_batch_size`` is ``True``,
                defines how many dimensions the output tensordict should have.
                Defaults to ``None``.
            device (torch.device, optional): The device for tensor data.
                Defaults to ``None``.
            batch_size (torch.Size, optional): The batch size. Defaults to
                ``[num_rows]``.
            separator (str, optional): If provided, column names are split on
                this separator to create nested TensorDicts. For example, with
                ``separator="."``, a column ``"obs.x"`` becomes
                ``td["obs", "x"]``. Defaults to ``None``.
            dtype (torch.dtype, optional): If provided, all numeric columns
                are cast to this dtype. Defaults to ``None``.

        Returns:
            A TensorDict representation of the DataFrame.

        Examples:
            >>> import pandas as pd
            >>> df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
            >>> td = TensorDict.from_pandas(df)
            >>> print(td)
            TensorDict(
                fields={
                    a: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.int64, is_shared=False),
                    b: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float64, is_shared=False)},
                batch_size=torch.Size([3]),
                device=None,
                is_shared=False)
        """
        from tensordict.base import TensorDictBase

        if cls is TensorDictBase:
            from tensordict._td import TensorDict

            cls = TensorDict

        result = _dataframe_to_tensordict(
            dataframe,
            cls=cls,
            device=device,
            batch_size=batch_size,
            separator=separator,
            dtype=dtype,
        )
        if auto_batch_size:
            if batch_size is not None:
                raise TypeError(cls._CONFLICTING_BATCH_SIZES.format("from_pandas"))
            result.auto_batch_size_(batch_dims=batch_dims)
        return result

    def to_pandas(self, *, separator: str | None = None):
        """Converts this TensorDict to a pandas DataFrame.

        Each leaf key becomes a column. Tensor values are converted to numpy
        arrays, :class:`~tensordict.NonTensorData` values are converted to
        Python lists.

        Keyword Args:
            separator (str, optional): If provided, nested keys are joined
                with this separator to produce flat column names. For example,
                ``td["obs", "x"]`` becomes column ``"obs.x"`` with
                ``separator="."``. Required when the TensorDict contains
                nested sub-TensorDicts. Defaults to ``None``.

        Returns:
            A pandas DataFrame.

        Examples:
            >>> td = TensorDict({"a": torch.arange(3), "b": torch.zeros(3)}, [3])
            >>> df = td.to_pandas()
            >>> print(df)
               a    b
            0  0  0.0
            1  1  0.0
            2  2  0.0
        """
        return _tensordict_to_dataframe(self, separator=separator)

    @classmethod
    def from_csv(
        cls,
        path,
        *,
        auto_batch_size: bool = False,
        batch_dims: int | None = None,
        device: torch.device | None = None,
        batch_size: torch.Size | None = None,
        separator: str | None = None,
        dtype: torch.dtype | None = None,
        **kwargs,
    ) -> Self:
        """Creates a TensorDict from a CSV file.

        Requires either pandas or pyarrow to be installed.

        Args:
            path (str or Path): Path to the CSV file.

        Keyword Args:
            auto_batch_size (bool, optional): If ``True``, the batch size will
                be computed automatically. Defaults to ``False``.
            batch_dims (int, optional): If ``auto_batch_size`` is ``True``,
                defines how many dimensions the output tensordict should have.
                Defaults to ``None``.
            device (torch.device, optional): The device for tensor data.
                Defaults to ``None``.
            batch_size (torch.Size, optional): The batch size. Defaults to
                ``[num_rows]``.
            separator (str, optional): If provided, column names are split on
                this separator to create nested TensorDicts. Defaults to ``None``.
            dtype (torch.dtype, optional): If provided, all numeric columns
                are cast to this dtype. Defaults to ``None``.
            **kwargs: Additional keyword arguments forwarded to the CSV reader
                (``pandas.read_csv`` or ``pyarrow.csv.read_csv``).

        Returns:
            A TensorDict representation of the CSV data.

        Examples:
            >>> td = TensorDict.from_csv("data.csv")
            >>> td = TensorDict.from_csv("data.csv", separator=".", dtype=torch.float32)
        """
        from tensordict.base import TensorDictBase

        if cls is TensorDictBase:
            from tensordict._td import TensorDict

            cls = TensorDict

        columns, num_rows = _read_csv(path, **kwargs)
        result = _columns_to_tensordict(
            columns,
            cls=cls,
            device=device,
            batch_size=batch_size,
            separator=separator,
            dtype=dtype,
            num_rows=num_rows,
        )
        if auto_batch_size:
            if batch_size is not None:
                raise TypeError(cls._CONFLICTING_BATCH_SIZES.format("from_csv"))
            result.auto_batch_size_(batch_dims=batch_dims)
        return result

    def to_csv(self, path, *, separator: str | None = None, **kwargs):
        """Writes this TensorDict to a CSV file.

        Requires pandas to be installed.

        Args:
            path (str or Path): Path to the output CSV file.

        Keyword Args:
            separator (str, optional): If provided, nested keys are joined
                with this separator. Defaults to ``None``.
            **kwargs: Additional keyword arguments forwarded to
                ``pandas.DataFrame.to_csv``.
        """
        _write_csv(self, path, separator=separator, **kwargs)

    @classmethod
    def from_parquet(
        cls,
        path,
        *,
        auto_batch_size: bool = False,
        batch_dims: int | None = None,
        device: torch.device | None = None,
        batch_size: torch.Size | None = None,
        separator: str | None = None,
        dtype: torch.dtype | None = None,
        columns: list[str] | None = None,
        **kwargs,
    ) -> Self:
        """Creates a TensorDict from a Parquet file.

        Requires either pyarrow or pandas to be installed. Prefers pyarrow
        when available for better performance.

        Args:
            path (str or Path): Path to the Parquet file.

        Keyword Args:
            auto_batch_size (bool, optional): If ``True``, the batch size will
                be computed automatically. Defaults to ``False``.
            batch_dims (int, optional): If ``auto_batch_size`` is ``True``,
                defines how many dimensions the output tensordict should have.
                Defaults to ``None``.
            device (torch.device, optional): The device for tensor data.
                Defaults to ``None``.
            batch_size (torch.Size, optional): The batch size. Defaults to
                ``[num_rows]``.
            separator (str, optional): If provided, column names are split on
                this separator to create nested TensorDicts. Defaults to ``None``.
            dtype (torch.dtype, optional): If provided, all numeric columns
                are cast to this dtype. Defaults to ``None``.
            columns (list of str, optional): If provided, only read these
                columns from the file. Defaults to ``None`` (all columns).
            **kwargs: Additional keyword arguments forwarded to the Parquet
                reader.

        Returns:
            A TensorDict representation of the Parquet data.

        Examples:
            >>> td = TensorDict.from_parquet("data.parquet")
            >>> td = TensorDict.from_parquet("data.parquet", columns=["obs", "reward"])
        """
        from tensordict.base import TensorDictBase

        if cls is TensorDictBase:
            from tensordict._td import TensorDict

            cls = TensorDict

        col_dict, num_rows = _read_parquet(path, columns=columns, **kwargs)
        result = _columns_to_tensordict(
            col_dict,
            cls=cls,
            device=device,
            batch_size=batch_size,
            separator=separator,
            dtype=dtype,
            num_rows=num_rows,
        )
        if auto_batch_size:
            if batch_size is not None:
                raise TypeError(cls._CONFLICTING_BATCH_SIZES.format("from_parquet"))
            result.auto_batch_size_(batch_dims=batch_dims)
        return result

    def to_parquet(self, path, *, separator: str | None = None, **kwargs):
        """Writes this TensorDict to a Parquet file.

        Requires either pyarrow or pandas to be installed.

        Args:
            path (str or Path): Path to the output Parquet file.

        Keyword Args:
            separator (str, optional): If provided, nested keys are joined
                with this separator. Defaults to ``None``.
            **kwargs: Additional keyword arguments forwarded to the Parquet
                writer.
        """
        _write_parquet(self, path, separator=separator, **kwargs)

    @classmethod
    def from_json(
        cls,
        path,
        *,
        auto_batch_size: bool = False,
        batch_dims: int | None = None,
        device: torch.device | None = None,
        batch_size: torch.Size | None = None,
        separator: str | None = None,
        dtype: torch.dtype | None = None,
        lines: bool = False,
        **kwargs,
    ) -> Self:
        """Creates a TensorDict from a JSON file.

        Supports both standard JSON (array of records) and JSON Lines format.
        For nested JSON objects, use :func:`from_dict` instead.

        Requires pandas for best results. Falls back to stdlib ``json``
        for simple cases.

        Args:
            path (str or Path): Path to the JSON file.

        Keyword Args:
            auto_batch_size (bool, optional): If ``True``, the batch size will
                be computed automatically. Defaults to ``False``.
            batch_dims (int, optional): If ``auto_batch_size`` is ``True``,
                defines how many dimensions the output tensordict should have.
                Defaults to ``None``.
            device (torch.device, optional): The device for tensor data.
                Defaults to ``None``.
            batch_size (torch.Size, optional): The batch size. Defaults to
                ``[num_rows]``.
            separator (str, optional): If provided, column names are split on
                this separator to create nested TensorDicts. Defaults to ``None``.
            dtype (torch.dtype, optional): If provided, all numeric columns
                are cast to this dtype. Defaults to ``None``.
            lines (bool, optional): If ``True``, reads the file as JSON Lines
                (one JSON object per line). Defaults to ``False``.
            **kwargs: Additional keyword arguments forwarded to the JSON
                reader.

        Returns:
            A TensorDict representation of the JSON data.

        Examples:
            >>> td = TensorDict.from_json("data.json")
            >>> td = TensorDict.from_json("data.jsonl", lines=True)
        """
        from tensordict.base import TensorDictBase

        if cls is TensorDictBase:
            from tensordict._td import TensorDict

            cls = TensorDict

        columns, num_rows = _read_json(path, lines=lines, **kwargs)
        result = _columns_to_tensordict(
            columns,
            cls=cls,
            device=device,
            batch_size=batch_size,
            separator=separator,
            dtype=dtype,
            num_rows=num_rows,
        )
        if auto_batch_size:
            if batch_size is not None:
                raise TypeError(cls._CONFLICTING_BATCH_SIZES.format("from_json"))
            result.auto_batch_size_(batch_dims=batch_dims)
        return result

    def to_json(
        self,
        path,
        *,
        separator: str | None = None,
        lines: bool = False,
        **kwargs,
    ):
        """Writes this TensorDict to a JSON file.

        Args:
            path (str or Path): Path to the output JSON file.

        Keyword Args:
            separator (str, optional): If provided, nested keys are joined
                with this separator. Defaults to ``None``.
            lines (bool, optional): If ``True``, writes in JSON Lines format.
                Defaults to ``False``.
            **kwargs: Additional keyword arguments forwarded to the JSON
                writer.
        """
        _write_json(self, path, separator=separator, lines=lines, **kwargs)

    def to_h5(
        self,
        filename,
        **kwargs,
    ) -> Any:
        """Converts a tensordict to a PersistentTensorDict with the h5 backend.

        Args:
            filename (str or path): path to the h5 file.
            **kwargs: kwargs to be passed to :meth:`h5py.File.create_dataset`.

        Returns:
            A :class:`~.tensordict.PersitentTensorDict` instance linked to the newly created file.

        Examples:
            >>> import tempfile
            >>> import timeit
            >>>
            >>> from tensordict import TensorDict, MemoryMappedTensor
            >>> td = TensorDict({
            ...     "a": MemoryMappedTensor.from_tensor(torch.zeros(()).expand(1_000_000)),
            ...     "b": {"c": MemoryMappedTensor.from_tensor(torch.zeros(()).expand(1_000_000, 3))},
            ... }, [1_000_000])
            >>>
            >>> file = tempfile.NamedTemporaryFile()
            >>> td_h5 = td.to_h5(file.name, compression="gzip", compression_opts=9)
            >>> print(td_h5)
            PersistentTensorDict(
                fields={
                    a: Tensor(shape=torch.Size([1000000]), device=cpu, dtype=torch.float32, is_shared=False),
                    b: PersistentTensorDict(
                        fields={
                            c: Tensor(shape=torch.Size([1000000, 3]), device=cpu, dtype=torch.float32, is_shared=False)},
                        batch_size=torch.Size([1000000]),
                        device=None,
                        is_shared=False)},
                batch_size=torch.Size([1000000]),
                device=None,
                is_shared=False)


        """
        from tensordict.persistent import PersistentTensorDict

        out = PersistentTensorDict.from_dict(
            self,
            filename=filename,
            **kwargs,
        )
        if self._has_names():
            out.names = self.names
        return out

    def to_zarr(
        self,
        filename,
        **kwargs,
    ) -> Any:
        """Converts a tensordict to a PersistentTensorDict with the zarr backend.

        Requires ``zarr>=3.0`` to be installed. The batch size (and dimension
        names) are persisted in the store attributes such that
        :meth:`~.from_zarr` restores them without inference.

        Args:
            filename (str, path or zarr store): path to the zarr store (a
                directory), or a ``zarr.abc.store.Store`` instance (e.g. a
                ``zarr.storage.ZipStore``).
            **kwargs: kwargs to be passed to :meth:`zarr.Group.create_array`.
                By default each tensor is stored as a single uncompressed chunk;
                pass ``chunks=...`` and/or ``compressors=...`` to override (e.g.
                for out-of-core row access or on-disk compression). Since leaves
                have different ranks, ``chunks`` constrains the leading
                dimensions of each leaf and trailing dimensions are left whole
                (e.g. ``chunks=(16,)`` chunks every leaf along its first
                dimension in blocks of 16).

        Returns:
            A :class:`~tensordict.PersistentTensorDict` instance linked to the newly created store.

        Examples:
            >>> import tempfile
            >>> import torch
            >>> from tensordict import TensorDict
            >>> td = TensorDict({
            ...     "a": torch.zeros(1000),
            ...     "b": {"c": torch.zeros(1000, 3)},
            ... }, [1000])
            >>> td_zarr = td.to_zarr(tempfile.mkdtemp() + "/store.zarr")
            >>> print(td_zarr)
            PersistentTensorDict(
                fields={
                    a: Tensor(shape=torch.Size([1000]), device=cpu, dtype=torch.float32, is_shared=False),
                    b: PersistentTensorDict(
                        fields={
                            c: Tensor(shape=torch.Size([1000, 3]), device=cpu, dtype=torch.float32, is_shared=False)},
                        batch_size=torch.Size([1000]),
                        device=None,
                        is_shared=False)},
                batch_size=torch.Size([1000]),
                device=None,
                is_shared=False)

        """
        from tensordict.persistent import PersistentTensorDict

        out = PersistentTensorDict.from_dict(
            self,
            filename=filename,
            backend="zarr",
            **kwargs,
        )
        if self._has_names():
            out.names = self.names
            out._write_attrs_metadata()
        return out

    def to_store(
        self,
        *,
        backend: str = "redis",
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        unix_socket_path: str | None = None,
        prefix: str = "tensordict",
        device=None,
        **kwargs,
    ) -> Any:
        """Upload this TensorDict to a key-value store (Redis, Dragonfly, etc.).

        Returns a :class:`~tensordict.store.TensorDictStore` (or
        :class:`~tensordict.store.LazyStackedTensorDictStore` for
        lazy stacks) backed by the uploaded data.

        For :class:`LazyStackedTensorDict` inputs, data is streamed in chunks
        to avoid materialising the full stack in memory.

        Keyword Args:
            backend (STORE_BACKENDS): Store backend — ``"redis"`` (default)
                or ``"dragonfly"``.
            host (str): Server hostname.  Defaults to ``"localhost"``.
            port (int): Server port.  Defaults to ``6379``.
            db (int): Database number.  Defaults to ``0``.
            unix_socket_path (str, optional): Unix domain socket path.
            prefix (str): Key namespace.  Defaults to ``"tensordict"``.
            device (torch.device, optional): Device override for retrieved
                tensors.  If ``None``, uses this TensorDict's device.
            **kwargs: Extra connection keyword arguments.

        Returns:
            A store-backed TensorDict instance.

        Examples:
            >>> from tensordict import TensorDict
            >>> td = TensorDict({"obs": torch.randn(10, 84)}, [10])
            >>> store_td = td.to_store(host="localhost")
            >>> store_td["obs"].shape
            torch.Size([10, 84])
            >>>
            >>> # Using Dragonfly instead of Redis
            >>> store_td = td.to_store(backend="dragonfly", host="dragonfly-host")
        """
        from tensordict.store._store import TensorDictStore

        return TensorDictStore.from_tensordict(
            self,
            backend=backend,
            host=host,
            port=port,
            db=db,
            unix_socket_path=unix_socket_path,
            prefix=prefix,
            device=device,
            **kwargs,
        )
