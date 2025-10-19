# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import functools
import inspect
import re
import weakref
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
from copy import copy
from functools import wraps
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    OrderedDict,
    Sequence,
    Type,
    TYPE_CHECKING,
)

import torch

from tensordict._lazy import _CustomOpTensorDict, LazyStackedTensorDict
from tensordict._nestedkey import NestedKey
from tensordict._td import _SubTensorDict, TensorDict
from tensordict._tensorcollection import TensorCollection
from tensordict._torch_func import TD_HANDLED_FUNCTIONS

from tensordict.base import (
    _default_is_leaf,
    _is_tensor_collection,
    _register_tensor_class,
    CompatibleType,
    NO_DEFAULT,
    T,
    TensorDictBase,
)

from tensordict.memmap import MemoryMappedTensor
from tensordict.utils import (
    _LOCK_ERROR,
    _zip_strict,
    BufferLegacy,
    erase_cache,
    implement_for,
    IndexType,
    is_batchedtensor,
    lock_blocked,
)
from torch import multiprocessing as mp, nn, Tensor
from torch.utils._pytree import tree_map

try:
    from functorch import dim as ftdim

    _has_funcdim = True
except ImportError:
    from tensordict.utils import _ftdim_mock as ftdim

    _has_funcdim = False

try:
    from torch.nn.parameter import Buffer
except ImportError:
    from tensordict.utils import Buffer


try:
    from torch.compiler import is_compiling
except ImportError:
    from torch._dynamo import is_compiling

if TYPE_CHECKING:
    from typing import Self
else:
    Self = Any


def _apply_leaves(data, fn):
    if isinstance(data, TensorDict):
        with data.unlock_():
            for key, val in list(data.items()):
                data._set_str(
                    key,
                    _apply_leaves(val, fn),
                    validated=True,
                    inplace=False,
                    non_blocking=False,
                )
        return data
    elif isinstance(data, LazyStackedTensorDict):
        # this is currently not implemented as the registration of params will only work
        # with plain TensorDict. The solution will be using pytree to get each independent
        # leaf
        raise RuntimeError(
            "Using a LazyStackedTensorDict within a TensorDictParams isn't permitted."
        )
        # for _data in data.tensordicts:
        #     _apply_leaves(_data, fn)
        # return data
    elif isinstance(data, _CustomOpTensorDict):
        _apply_leaves(data._source, fn)
        return data
    elif isinstance(data, _SubTensorDict):
        raise RuntimeError(
            "Using a _SubTensorDict within a TensorDictParams isn't permitted."
        )
    else:
        return fn(data)


def _get_args_dict(func, args, kwargs):
    signature = inspect.signature(func)
    bound_arguments = signature.bind(*args, **kwargs)
    bound_arguments.apply_defaults()

    args_dict = dict(bound_arguments.arguments)
    return args_dict


def _maybe_make_param(tensor):
    if isinstance(tensor, (Tensor, ftdim.Tensor)) and not isinstance(
        tensor, (nn.Parameter, Buffer, BufferLegacy)
    ):
        if tensor.dtype in (torch.float, torch.double, torch.half):
            tensor = nn.Parameter(tensor)
        elif not is_batchedtensor(tensor):
            # convert all non-parameters to buffers
            # dataptr = tensor.data.data_ptr()
            tensor = Buffer(tensor)
        else:
            # We want to keep the grad_fn of tensors, e.g. param.expand(10) should point to the original param
            tensor = BufferLegacy(tensor)
    return tensor


def _maybe_make_param_or_buffer(tensor):
    if isinstance(tensor, (Tensor, ftdim.Tensor)) and not isinstance(
        tensor, (nn.Parameter, Buffer)
    ):
        if not tensor.requires_grad and not is_batchedtensor(tensor):
            # convert all non-parameters to buffers
            # dataptr = tensor.data.data_ptr()
            tensor = Buffer(tensor)
        else:
            # We want to keep the grad_fn of tensors, e.g. param.expand(10) should point to the original param
            tensor = BufferLegacy(tensor)

        # assert tensor.data.data_ptr() == dataptr
    return tensor


class _unlock_and_set:
    # temporarily unlocks the nested tensordict to execute a function
    def __new__(cls, *args, **kwargs):
        if len(args) and callable(args[0]):
            return cls(**kwargs)(args[0])
        return super().__new__(cls)

    def __init__(self, **only_for_kwargs):
        self.only_for_kwargs = only_for_kwargs

    def __call__(self, func):
        name = func.__name__

        @wraps(func)
        def new_func(_self, *args, **kwargs):  # type: ignore[misc]
            if self.only_for_kwargs:
                arg_dict = _get_args_dict(func, (_self, *args), kwargs)
                for kwarg, exp_value in self.only_for_kwargs.items():
                    cur_val = arg_dict.get(kwarg, NO_DEFAULT)
                    if cur_val != exp_value:
                        # escape
                        meth = getattr(_self._param_td, name)
                        out = meth(*args, **kwargs)
                        return out
            if not _self.no_convert:
                args = tree_map(_maybe_make_param, args)
                kwargs = tree_map(_maybe_make_param, kwargs)
            else:
                args = tree_map(_maybe_make_param_or_buffer, args)
                kwargs = tree_map(_maybe_make_param_or_buffer, kwargs)
            if _self.is_locked:
                # if the root (TensorDictParams) is locked, we still want to raise an exception
                raise RuntimeError(_LOCK_ERROR)
            with (
                _self._param_td.unlock_()
                if _self._param_td.is_locked
                else nullcontext()
            ):
                meth = getattr(_self._param_td, name)
                out = meth(*args, **kwargs)
            _self._reset_params()
            if out is _self._param_td:
                return _self
            return out

        return new_func


def _get_post_hook(func):
    @wraps(func)
    def new_func(self, *args, **kwargs):  # type: ignore[misc]
        out = func(self, *args, **kwargs)
        return self._apply_get_post_hook(out)

    return new_func


def _fallback(func):
    """Calls the method on the nested tensordict."""
    name = func.__name__

    @wraps(func)
    def new_func(self, *args, **kwargs):  # type: ignore[misc]
        out = getattr(self._param_td, name)(*args, **kwargs)
        if out is self._param_td:
            # if the output does not change, return the wrapper
            return self
        return out

    return new_func


def _fallback_property(func):
    name = func.__name__

    @wraps(func)
    def new_func(self):  # type: ignore[misc]
        out = getattr(self._param_td, name)
        if out is self._param_td:
            return self
        return out

    def setter(self, value):  # type: ignore[misc]
        return getattr(type(self._param_td), name).fset(self._param_td, value)

    return property(new_func, setter)


def _replace(func):
    name = func.__name__

    @wraps(func)
    def new_func(self, *args, **kwargs):  # type: ignore[misc]
        out = getattr(self._param_td, name)(*args, **kwargs)
        if out is self._param_td:
            return self
        self._param_td = out
        return self

    return new_func


def _carry_over(func):
    name = func.__name__

    @wraps(func)
    def new_func(self, *args, **kwargs):  # type: ignore[misc]
        out = getattr(self._param_td, name)(*args, **kwargs)
        if out is self._param_td:
            return self
        if not isinstance(out, TensorDictParams):
            out = TensorDictParams(out, no_convert="skip")
            out.no_convert = self.no_convert
        return out

    return new_func


def _apply_on_data(func):
    @wraps(func)
    def new_func(self, *args, **kwargs):  # type: ignore[misc]
        getattr(self.data, func.__name__)(*args, **kwargs)
        return self

    return new_func


class TensorDictParams(TensorDictBase, nn.Module):  # type: ignore[override,misc,attr-defined]
    r"""A Wrapper for TensorDictBase with Parameter Exposure.

    This class is designed to hold a `TensorDictBase` instance that contains parameters, making them accessible to a
    parent :class:`~torch.nn.Module`. This allows for seamless integration of tensordict parameters into PyTorch modules,
    enabling operations like parameter iteration and optimization.

    Key Features:

    - Parameter Exposure: Parameters within the tensordict are exposed to the parent module, allowing them to be included
      in operations like `named_parameters()`.
    - Indexing: Indexing works similarly to the wrapped tensordict. However, parameter names (in :meth:`~.named_parameters`) are registered using
      `TensorDict.flatten_keys("_")`, which may result in different key names compared to the tensordict content.
    - Automatic Conversion: Any tensor set in the tensordict is automatically converted to a :class:`torch.nn.Parameter`,
      unless specified otherwise through the :attr:`no_convert` keyword argument.

    Args
        parameters (TensorDictBase or dict): The tensordict to represent as parameters. Values are converted to
            parameters unless `no_convert=True`. If a `dict` is provided, it is wrapped in a `TensorDict` instance.
            Keyword arguments can also be used.

    Keyword Args:
        no_convert (bool): If `True`, no conversion to `nn.Parameter` occurs and all non-parameter, non-buffer tensors
            will be converted to a :class:`~torch.nn.Buffer` instance.
            If ``False``, all tensors with non-integer dtypes will be converted to :class:`~torch.nn.Parameter`
            whereas integer dtypes will be converted to :class:`~torch.nn.Buffer` instances.
            Defaults to `False`.
        lock (bool): If `True`, the tensordict hosted by `TensorDictParams` is locked, preventing modifications and
            potentially impacting performance when `unlock_()` is required.
            Defaults to `False`.

            .. warning:: Because the inner tensordict isn't copied or locked by default, registering the tensordict
                in a ``TensorDictParams`` and modifying its content afterwards will __not__ update the values within
                the  ``TensorDictParams`` :meth:`.parameters` and :meth:`~.buffers` sequences.

        **kwargs: Key-value pairs to populate the `TensorDictParams`. Exclusive with the `parameters` input.

    Examples
        >>> from torch import nn
        >>> from tensordict import TensorDict
        >>> module = nn.Sequential(nn.Linear(3, 4), nn.Linear(4, 4))
        >>> params = TensorDict.from_module(module)
        >>> params.lock_()
        >>> p = TensorDictParams(params)
        >>> print(p)
        TensorDictParams(params=TensorDict(
            fields={
                0: TensorDict(
                    fields={
                        bias: Parameter(shape=torch.Size([4]), device=cpu, dtype=torch.float32, is_shared=False),
                        weight: Parameter(shape=torch.Size([4, 3]), device=cpu, dtype=torch.float32, is_shared=False)},
                    batch_size=torch.Size([]),
                    device=None,
                    is_shared=False),
                1: TensorDict(
                    fields={
                        bias: Parameter(shape=torch.Size([4]), device=cpu, dtype=torch.float32, is_shared=False),
                        weight: Parameter(shape=torch.Size([4, 4]), device=cpu, dtype=torch.float32, is_shared=False)},
                    batch_size=torch.Size([]),
                    device=None,
                    is_shared=False)},
            batch_size=torch.Size([]),
            device=None,
            is_shared=False))
        >>> class CustomModule(nn.Module):
        ...     def __init__(self, params):
        ...         super().__init__()
        ...         self.params = params
        >>> m = CustomModule(p)
        >>> # The wrapper supports assignment, and values are converted to Parameters
        >>> m.params['other'] = torch.randn(3)
        >>> assert isinstance(m.params['other'], nn.Parameter)

    """

    def __init__(
        self,
        parameters: TensorDictBase | dict | None = None,
        *,
        no_convert=False,
        lock: bool = False,
        **kwargs,
    ):
        nn.Module.__init__(self)
        if parameters is None:
            parameters = kwargs
        elif kwargs:
            raise TypeError(
                f"parameters cannot be passed along with extra keyword arguments, but got {kwargs.keys()} extra args."
            )

        params = None
        buffers = None
        if isinstance(parameters, dict):
            parameters = TensorDict(parameters)
        elif isinstance(parameters, TensorDictParams):
            params = dict(parameters._parameters)
            buffers = dict(parameters._buffers)
            parameters = parameters._param_td.copy().lock_()
            no_convert = "skip"

        self.no_convert = no_convert
        if no_convert != "skip":
            if not no_convert:
                func = _maybe_make_param
            else:
                func = _maybe_make_param_or_buffer
            self._param_td = _apply_leaves(parameters, lambda x: func(x))
        else:
            self._param_td = parameters

        self._lock_content = lock
        if lock:
            self._param_td.lock_()
        self._reset_params(params=params, buffers=buffers)
        self._is_locked = False
        self._locked_tensordicts = []
        self._get_post_hook = []

    @classmethod
    def _new_unsafe(
        cls,
        parameters: TensorDictBase,
        *,
        no_convert=None,
        lock: bool = False,
        params: dict | None = None,
        buffers: dict | None = None,
        **kwargs,
    ):
        if is_compiling():
            return TensorDictParams(parameters, no_convert="skip", lock=lock)

        if parameters is None:
            parameters = kwargs

        if isinstance(parameters, dict):
            parameters = TensorDict._new_unsafe(parameters, **kwargs)
            if no_convert is None:
                # Then _new_unsafe is called from somewhere that doesn't know
                #  that it's a TDParams and we return a TensorDict (eg, torch.gather)
                return parameters
        elif isinstance(parameters, TensorDictParams):
            if kwargs:
                raise TypeError(
                    f"parameters cannot be passed along with extra keyword arguments, but got {kwargs.keys()} extra args."
                )
            params = dict(parameters._parameters)
            buffers = dict(parameters._buffers)
            parameters = parameters._param_td
            no_convert = "skip"

        self = TensorDictParams.__new__(cls)
        nn.Module.__init__(self)

        self._param_td = parameters
        self.no_convert = no_convert
        if no_convert != "skip":
            raise RuntimeError("_new_unsafe requires no_convert to be set to 'skip'")
        self._lock_content = lock
        if lock:
            self._param_td.lock_()
        self._reset_params(params=params, buffers=buffers)
        self._is_locked = False
        self._locked_tensordicts = []
        self._get_post_hook = []
        return self

    def __iter__(self):
        yield from self._param_td.__iter__()

    def register_get_post_hook(self, hook):
        """Register a hook to be called after any get operation on leaf tensors."""
        if not callable(hook):
            raise ValueError("Hooks must be callables.")
        self._get_post_hook.append(hook)

    def _apply_get_post_hook(self, val):
        if not _is_tensor_collection(type(val)):
            for hook in self._get_post_hook:
                new_val = hook(self, val)
                if new_val is not None:
                    val = new_val
        return val

    def _reset_params(self, params: dict | None = None, buffers: dict | None = None):
        parameters = self._param_td

        self._parameters.clear()
        self._buffers.clear()

        if (params is not None) ^ (buffers is not None):
            raise RuntimeError("both params and buffers must either be None or not.")
        elif params is None:
            param_keys = []
            params = []
            buffer_keys = []
            buffers = []
            for key, value in parameters.items(True, True):
                # flatten key
                if isinstance(key, tuple):
                    key = ".".join(key)
                if isinstance(value, nn.Parameter):
                    param_keys.append(key)
                    params.append(value)
                else:
                    buffer_keys.append(key)
                    buffers.append(value)

            self._parameters.update(dict(_zip_strict(param_keys, params)))
            self._buffers.update(dict(_zip_strict(buffer_keys, buffers)))
        else:
            self._parameters.update(params)
            self._buffers.update(buffers)

    @classmethod
    def __torch_function__(
        cls,
        func: Callable,
        types: tuple[type, ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Callable:
        if kwargs is None:
            kwargs = {}
        if func not in TDPARAM_HANDLED_FUNCTIONS or not all(
            issubclass(t, (Tensor, ftdim.Tensor, TensorDictBase)) for t in types
        ):
            return NotImplemented
        return TDPARAM_HANDLED_FUNCTIONS[func](*args, **kwargs)

    @classmethod
    def _flatten_key(cls, key):
        def make_valid_identifier(s):
            # Replace invalid characters with underscores
            s = re.sub(r"\W|^(?=\d)", "_", s)

            # Ensure the string starts with a letter or underscore
            if not s[0].isalpha() and s[0] != "_":
                s = "_" + s

            return s

        key_flat = "_".join(key)
        if not key_flat.isidentifier():
            key_flat = make_valid_identifier(key_flat)
        return key_flat

    @lock_blocked
    @_unlock_and_set
    def __setitem__(  # type: ignore[misc]
        self,
        index: IndexType,
        value: Any,
    ) -> None: ...

    @lock_blocked
    @_unlock_and_set
    def set(
        self, key: NestedKey, item: CompatibleType, inplace: bool = False, **kwargs: Any
    ) -> TensorDictBase: ...

    @lock_blocked
    def update(
        self,
        input_dict_or_td: dict[str, CompatibleType] | TensorDictBase,
        clone: bool = False,
        inplace: bool = False,
        *,
        non_blocking: bool = False,
        keys_to_update: Sequence[NestedKey] | None = None,
        is_leaf: Callable[[Type], bool] | None = None,
        update_batch_size: bool = False,
        ignore_lock: bool = False,
    ) -> TensorDictBase:
        # Deprecating this since _set_tuple will do it thx to the decorator
        # if not self.no_convert:
        #     func = _maybe_make_param
        # else:
        #     func = _maybe_make_param_or_buffer
        # if _is_tensor_collection(type(input_dict_or_td)):
        #     input_dict_or_td = input_dict_or_td.apply(func)
        # else:
        #     input_dict_or_td = tree_map(func, input_dict_or_td)
        with self._param_td.unlock_():
            TensorDictBase.update(
                self,
                input_dict_or_td,
                clone=clone,
                inplace=inplace,
                keys_to_update=keys_to_update,
                non_blocking=non_blocking,
                is_leaf=is_leaf,
            )
            self._reset_params()
        return self

    @lock_blocked
    @_unlock_and_set
    def pop(self, key: NestedKey, default: Any = NO_DEFAULT) -> CompatibleType: ...

    @lock_blocked
    @_unlock_and_set
    def popitem(self): ...

    @lock_blocked
    @_unlock_and_set
    def rename_key_(
        self, old_key: NestedKey, new_key: NestedKey, safe: bool = False
    ) -> TensorDictBase: ...

    def map(
        self,
        fn: Callable,
        dim: int = 0,
        num_workers: int | None = None,
        chunksize: int | None = None,
        num_chunks: int | None = None,
        pool: mp.Pool = None,
        generator: torch.Generator | None = None,
        max_tasks_per_child: int | None = None,
        worker_threads: int = 1,
        mp_start_method: str | None = None,
    ):
        raise RuntimeError(
            "Cannot call map on a TensorDictParams object. Convert it "
            "to a detached tensordict first (through ``tensordict.data`` or ``tensordict.to_tensordict()``) and call "
            "map in a second time."
        )

    @_unlock_and_set(inplace=True)
    def apply(
        self,
        fn: Callable,
        *others: TensorDictBase,
        batch_size: Sequence[int] | None = None,
        device: torch.device | None = NO_DEFAULT,
        names: Sequence[str] | None = NO_DEFAULT,
        inplace: bool = False,
        default: Any = NO_DEFAULT,
        filter_empty: bool | None = None,
        call_on_nested: bool = False,
        **constructor_kwargs,
    ) -> TensorDictBase | None: ...

    @_unlock_and_set(inplace=True)
    def named_apply(
        self,
        fn: Callable,
        *others: TensorDictBase,
        batch_size: Sequence[int] | None = None,
        device: torch.device | None = NO_DEFAULT,
        names: Sequence[str] | None = NO_DEFAULT,
        inplace: bool = False,
        default: Any = NO_DEFAULT,
        filter_empty: bool | None = None,
        call_on_nested: bool = False,
        **constructor_kwargs,
    ) -> TensorDictBase | None: ...

    @_unlock_and_set(inplace=True)
    def _apply_nest(*args, **kwargs): ...

    @_fallback
    def _multithread_apply_flat(
        self,
        fn: Callable,
        *others: T,
        call_on_nested: bool = False,
        default: Any = NO_DEFAULT,
        named: bool = False,
        nested_keys: bool = False,
        prefix: tuple = (),
        is_leaf: Callable[[Type], bool] | None = None,
        executor: ThreadPoolExecutor,
        futures: List[Future],
        local_futures: List,
    ) -> None: ...

    @_fallback
    def _multithread_rebuild(
        self,
        *,
        batch_size: Sequence[int] | None = None,
        device: torch.device | None = NO_DEFAULT,
        names: Sequence[str] | None = NO_DEFAULT,
        inplace: bool = False,
        checked: bool = False,
        out: TensorDictBase | None = None,
        filter_empty: bool = False,
        executor: ThreadPoolExecutor,
        futures: List[Future],
        local_futures: List,
        subs_results: Dict[Future, Any] | None = None,
        multithread_set: bool = False,  # Experimental
        **constructor_kwargs,
    ) -> None: ...

    @_get_post_hook
    @_fallback
    def get(self, key: NestedKey, default: Any = None) -> CompatibleType: ...

    @_get_post_hook
    @_fallback
    def __getitem__(
        self, index: IndexType
    ) -> Self | Tensor | TensorCollection | Any: ...

    @_fallback
    def _set_device(self, device: torch.device) -> Self: ...  # type: ignore[misc]

    @_fallback
    def auto_device_(self) -> Self: ...

    __getitems__ = __getitem__

    def to(self, *args, **kwargs) -> TensorDictBase:
        params = self._param_td.to(*args, **kwargs)
        if params is self._param_td:
            return self
        return TensorDictParams(params)

    def cpu(self):
        params = self._param_td.cpu()
        if params is self._param_td:
            return self
        return TensorDictParams(params)

    def cuda(self, device=None):
        params = self._param_td.cuda(device=device)
        if params is self._param_td:
            return self
        return TensorDictParams(params)

    def _clone(self, recurse: bool = True) -> TensorDictBase:
        """Clones the TensorDictParams.

        .. warning::
            The effect of this call is different from a regular torch.Tensor.clone call
            in that it will create a TensorDictParams instance with a new copy of the
            parameters and buffers __detached__ from the current graph. For a
            regular clone (ie, cloning leaf parameters onto a new tensor that
            is part of the graph), simply call

                >>> params.apply(torch.clone)

        .. note::
            If a parameter is duplicated in the tree, ``clone`` will preserve this
            identity (ie, parameter tying is preserved).

        See :meth:`tensordict.TensorDictBase.clone` for more info on the clone
        method.

        """
        if not recurse:
            return TensorDictParams._new_unsafe(
                self._param_td._clone(False),
                no_convert="skip",
                params=dict(self._parameters),
                buffers=dict(self._buffers),
            )

        memo = {}

        def _clone(tensor, memo=memo):
            result = memo.get(tensor)
            if result is not None:
                return result

            if isinstance(tensor, nn.Parameter):
                result = nn.Parameter(
                    tensor.data.clone(), requires_grad=tensor.requires_grad
                )
            else:
                result = Buffer(tensor.data.clone())
            memo[tensor] = result
            return result

        return TensorDictParams(self._param_td.apply(_clone), no_convert="skip")

    @_fallback
    def chunk(self, chunks: int, dim: int = 0) -> tuple[TensorDictBase, ...]: ...

    @_fallback
    def _unbind(self, dim: int) -> tuple[TensorDictBase, ...]: ...

    @classmethod
    def from_dict(cls, *args, **kwargs):
        td = TensorDict.from_dict(*args, **kwargs)
        return TensorDictParams(td)

    @_fallback
    def to_tensordict(self, *, retain_none: bool | None = None): ...

    @_fallback
    def to_h5(
        self,
        filename,
        **kwargs,
    ): ...

    def __hash__(self):
        return hash((id(self), id(self.__dict__.get("_param_td"))))

    @_fallback
    def __eq__(self, other: object) -> TensorDictBase: ...

    @_fallback
    def __ne__(self, other: object) -> TensorDictBase: ...

    @_fallback
    def __xor__(self, other: object) -> TensorDictBase: ...

    @_fallback
    def __or__(self, other: object) -> TensorDictBase: ...

    @_fallback
    def __ge__(self, other: object) -> TensorDictBase: ...

    @_fallback
    def __gt__(self, other: object) -> TensorDictBase: ...

    @_fallback
    def __le__(self, other: object) -> TensorDictBase: ...

    @_fallback
    def __lt__(self, other: object) -> TensorDictBase: ...

    def __getattr__(self, item: str) -> Any:
        if not item.startswith("_"):
            try:
                return getattr(self.__dict__["_param_td"], item)
            except AttributeError:
                try:
                    return super().__getattr__(item)
                except AttributeError as e:
                    # During some state-dict loads, we may encounter cases where pytorch does a getattr
                    #  with the module name
                    if item in self.keys():
                        return TensorDictParams(self[item])
                    raise e
        else:
            return super().__getattr__(item)

    @_fallback
    def _change_batch_size(self, *args, **kwargs): ...

    @_fallback
    def _erase_names(self, *args, **kwargs): ...

    @_get_post_hook
    @_fallback
    def _get_str(self, *args, **kwargs): ...

    @_get_post_hook
    @_fallback
    def _get_tuple(self, *args, **kwargs): ...

    @_get_post_hook
    @_fallback
    def _get_at_str(self, key, idx, default, **kwargs): ...

    @_get_post_hook
    @_fallback
    def _get_at_tuple(self, key, idx, default, **kwargs): ...

    @_fallback
    def _add_batch_dim(self, *args, **kwargs): ...

    @_fallback
    def _convert_to_tensordict(self, *args, **kwargs): ...

    @_fallback
    def _get_names_idx(self, *args, **kwargs): ...

    @_fallback
    def _index_tensordict(self, *args, **kwargs): ...

    @_fallback
    def _remove_batch_dim(self, *args, **kwargs): ...

    @_fallback
    def _maybe_remove_batch_dim(self, *args, **kwargs): ...

    @_fallback
    def _has_names(self, *args, **kwargs): ...

    @_unlock_and_set
    def _rename_subtds(self, *args, **kwargs): ...

    @_unlock_and_set
    def _set_at_str(self, *args, **kwargs): ...

    @_fallback
    def _set_at_tuple(self, *args, **kwargs): ...

    @_unlock_and_set
    def _set_str(self, *args, **kwargs): ...

    @_unlock_and_set
    def _set_tuple(self, *args, **kwargs): ...

    @_unlock_and_set
    def _create_nested_str(self, *args, **kwargs): ...

    @_fallback_property
    def batch_size(self) -> torch.Size: ...

    @_fallback
    def contiguous(self, *args, **kwargs): ...

    @lock_blocked
    @_unlock_and_set
    def del_(self, *args, **kwargs): ...

    @_fallback
    def detach_(self, *args, **kwargs): ...

    @_fallback_property
    def device(self): ...

    @_fallback
    def entry_class(self, *args, **kwargs): ...

    @_fallback
    def is_contiguous(self, *args, **kwargs): ...

    @_fallback
    def keys(self, *args, **kwargs): ...

    @_fallback
    def masked_fill(self, *args, **kwargs): ...

    @_fallback
    def masked_fill_(self, *args, **kwargs): ...

    def memmap_(
        self,
        prefix: str | None = None,
        copy_existing: bool = False,
        num_threads: int = 0,
    ) -> TensorDictBase:
        raise RuntimeError(
            "Cannot build a memmap TensorDict in-place. Use memmap or memmap_like instead."
        )

    _memmap_ = TensorDict._memmap_

    _load_memmap = TensorDict._load_memmap

    def make_memmap(
        self,
        key: NestedKey,
        shape: torch.Size | torch.Tensor,
        *,
        dtype: torch.dtype | None = None,
    ) -> MemoryMappedTensor:
        raise RuntimeError(
            "Making a memory-mapped tensor after instantiation isn't currently allowed for TensorDictParams."
            "If this feature is required, open an issue on GitHub to trigger a discussion on the topic!"
        )

    def make_memmap_from_storage(
        self,
        key: NestedKey,
        storage: torch.UntypedStorage,
        shape: torch.Size | torch.Tensor,
        *,
        dtype: torch.dtype | None = None,
    ) -> MemoryMappedTensor:
        raise RuntimeError(
            "Making a memory-mapped tensor after instantiation isn't currently allowed for TensorDictParams."
            "If this feature is required, open an issue on GitHub to trigger a discussion on the topic!"
        )

    def make_memmap_from_tensor(
        self, key: NestedKey, tensor: torch.Tensor, *, copy_data: bool = True
    ) -> MemoryMappedTensor:
        raise RuntimeError(
            "Making a memory-mapped tensor after instantiation isn't currently allowed for TensorDictParams."
            "If this feature is required, open an issue on GitHub to trigger a discussion on the topic!"
        )

    @_fallback_property
    def names(self): ...

    def pin_memory(self, *args, **kwargs):
        if kwargs.get("inplace", False):
            raise RuntimeError(
                f"Cannot pin_memory in-place with {type(self).__name__}."
            )
        return _fallback(self.pin_memory)(self, *args, **kwargs)

    @_unlock_and_set
    def _select(self, *args, **kwargs): ...

    @_fallback
    def share_memory_(self, *args, **kwargs): ...

    @property
    def is_locked(self) -> bool:
        # Cannot be locked
        return self._is_locked

    @is_locked.setter
    def is_locked(self, value):
        self._is_locked = bool(value)

    @_fallback_property
    def is_shared(self) -> bool: ...

    @_fallback_property
    def is_memmap(self) -> bool: ...

    @property
    def _is_shared(self) -> bool:
        return self._param_td._is_shared

    @property
    def _is_memmap(self) -> bool:
        return self._param_td._is_memmap

    @_fallback_property
    def shape(self) -> torch.Size: ...

    def _propagate_lock(self, _lock_parents_weakrefs=None, *, is_compiling):
        """Registers the parent tensordict that handles the lock."""
        self._is_locked = True
        if not is_compiling:
            if _lock_parents_weakrefs is None:
                _lock_parents_weakrefs = []
            self._lock_parents_weakrefs += _lock_parents_weakrefs
            _lock_parents_weakrefs.append(weakref.ref(self))
        # we don't want to double-lock the _param_td attrbute which is locked by default
        if not self._param_td.is_locked:
            self._param_td._propagate_lock(
                _lock_parents_weakrefs, is_compiling=is_compiling
            )

    @erase_cache
    def _propagate_unlock(self):
        # if we end up here, we can clear the graph associated with this td
        self._is_locked = False

        if not self._lock_content:
            return self._param_td._propagate_unlock()
        return []

    unlock_ = TensorDict.unlock_
    lock_ = TensorDict.lock_

    @property
    def data(self) -> Self:
        return self._param_td._data()

    @property
    def grad(self):
        return self._param_td._grad()

    @_unlock_and_set(inplace=True)
    def flatten_keys(
        self, separator: str = ".", inplace: bool = False
    ) -> TensorDictBase: ...

    @_unlock_and_set(inplace=True)
    def unflatten_keys(
        self, separator: str = ".", inplace: bool = False
    ) -> TensorDictBase: ...

    @_unlock_and_set(inplace=True)
    def _exclude(
        self, *keys: NestedKey, inplace: bool = False, set_shared: bool = True
    ) -> TensorDictBase: ...

    @_carry_over
    def from_dict_instance(
        self,
        input_dict,
        *,
        auto_batch_size: bool = False,
        batch_size=None,
        device=None,
        batch_dims=None,
    ): ...

    @_carry_over
    def _legacy_transpose(self, dim0, dim1): ...

    @_fallback
    def _transpose(self, dim0, dim1): ...

    @_fallback
    def where(
        self,
        condition: Tensor,
        other: Tensor | TensorDictBase,
        *,
        out: TensorDictBase | None = None,
        pad: int | bool = None,
        update_batch_size: bool = False,
    ): ...

    @_fallback
    def _permute(
        self,
        *dims_list: int,
        dims: list[int] | None = None,
    ) -> TensorDictBase: ...

    @_carry_over
    def _legacy_permute(
        self,
        *dims_list: int,
        dims: list[int] | None = None,
    ) -> TensorDictBase: ...

    @_fallback
    def _squeeze(self, dim: int | None = None) -> TensorDictBase: ...

    @_carry_over
    def _legacy_squeeze(self, dim: int | None = None) -> TensorDictBase: ...

    @_fallback
    def _unsqueeze(self, dim: int) -> TensorDictBase: ...

    @_carry_over
    def _legacy_unsqueeze(self, dim: int) -> TensorDictBase: ...

    _check_device = TensorDict._check_device
    _check_is_shared = TensorDict._check_is_shared

    @_fallback
    def _cast_reduction(self, **kwargs): ...

    @_fallback
    def all(self, dim: int | None = None) -> bool | TensorDictBase: ...

    @_fallback
    def any(self, dim: int | None = None) -> bool | TensorDictBase: ...

    @_fallback
    def expand(self, *args, **kwargs) -> Self: ...

    @_fallback
    def masked_select(self, mask: Tensor) -> Self: ...

    @_fallback
    def memmap_like(
        self,
        prefix: str | None = None,
        copy_existing: bool = False,
        num_threads: int = 0,
    ) -> Self: ...

    @_fallback
    def reshape(self, *shape: int): ...

    @_fallback
    def repeat_interleave(self, *shape: int): ...

    @_fallback
    def _repeat(self, *repeats: int): ...

    @_fallback
    def split(
        self, split_size: int | list[int], dim: int = 0
    ) -> list[TensorDictBase]: ...

    @_fallback
    def _to_module(
        self,
        module,
        *,
        inplace: bool = False,
        return_swap: bool = True,
        swap_dest=None,
        memo=None,
        use_state_dict: bool = False,
        non_blocking: bool = False,
    ): ...

    @_fallback
    def _view(self, *args, **kwargs): ...

    @_carry_over
    def _legacy_view(self, *args, **kwargs): ...

    @_unlock_and_set
    def create_nested(self, key): ...

    def __repr__(self):
        return f"TensorDictParams(params={self._param_td})"

    def values(
        self,
        include_nested: bool = False,
        leaves_only: bool = False,
        is_leaf: Callable[[Type], bool] | None = None,
        *,
        sort: bool = False,
    ) -> Iterator[CompatibleType]:
        if is_leaf is None:
            is_leaf = _default_is_leaf
        for v in self._param_td.values(include_nested, leaves_only, sort=sort):
            if not is_leaf(type(v)):
                yield v
                continue
            yield self._apply_get_post_hook(v)

    def state_dict(
        self, *args, destination=None, prefix="", keep_vars=False, flatten=True
    ):
        # flatten must be True by default to comply with module's state-dict API
        # since we want all params to be visible at root
        return self._param_td.state_dict(
            destination=destination,
            prefix=prefix,
            keep_vars=keep_vars,
            flatten=flatten,
        )

    def load_state_dict(
        self, state_dict: OrderedDict[str, Any], strict=True, assign=False
    ):
        # The state-dict is presumably the result of a call to TensorDictParams.state_dict
        # but can't be sure.

        state_dict_tensors = {}
        state_dict = dict(state_dict)
        for k, v in list(state_dict.items()):
            if isinstance(v, torch.Tensor):
                del state_dict[k]
                state_dict_tensors[k] = v
        state_dict_tensors = dict(
            TensorDict(state_dict_tensors, []).unflatten_keys(".")
        )
        state_dict.update(state_dict_tensors)
        self.data.load_state_dict(state_dict, strict=True, assign=False)
        return self

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        data = TensorDict(
            {
                key: val
                for key, val in state_dict.items()
                if key.startswith(prefix) and val is not None
            },
            [],
        ).unflatten_keys(".")
        prefix = tuple(key for key in prefix.split(".") if key)
        if prefix:
            data = data.get(prefix)
        self.data.load_state_dict(data)

    def items(
        self,
        include_nested: bool = False,
        leaves_only: bool = False,
        is_leaf: Callable[[Type], bool] | None = None,
        *,
        sort: bool = False,
    ) -> Iterator[CompatibleType]:
        if is_leaf is None:
            is_leaf = _default_is_leaf
        for k, v in self._param_td.items(include_nested, leaves_only, sort=sort):
            if not is_leaf(type(v)):
                yield k, v
                continue
            yield k, self._apply_get_post_hook(v)

    @_apply_on_data
    def zero_(self) -> Self: ...

    @_apply_on_data
    def fill_(self, key: NestedKey, value: float | bool) -> Self: ...

    @_apply_on_data
    def copy_(self, tensordict: T, non_blocking: bool | None = None) -> T: ...

    @_apply_on_data
    def set_at_(
        self, key: NestedKey, value: CompatibleType, index: IndexType
    ) -> Self: ...

    @_apply_on_data
    def set_(
        self,
        key: NestedKey,
        item: CompatibleType,
    ) -> Self: ...

    @_apply_on_data
    def _stack_onto_(
        self,
        list_item: list[CompatibleType],
        dim: int,
    ) -> Self: ...

    @_apply_on_data
    def _stack_onto_at_(
        self,
        key: NestedKey,
        list_item: list[CompatibleType],
        dim: int,
        idx: IndexType,
    ) -> Self: ...

    @_apply_on_data
    def update_(
        self,
        input_dict_or_td: dict[str, CompatibleType] | T,
        clone: bool = False,
        *,
        non_blocking: bool = False,
        keys_to_update: Sequence[NestedKey] | None = None,
    ) -> T: ...

    @_apply_on_data
    def update_at_(
        self,
        input_dict_or_td: dict[str, CompatibleType] | T,
        idx: IndexType,
        clone: bool = False,
        *,
        non_blocking: bool = False,
        keys_to_update: Sequence[NestedKey] | None = None,
    ) -> T: ...

    @_apply_on_data
    def apply_(self, fn: Callable, *others, **kwargs) -> Self: ...

    @implement_for("torch", "2.1")
    def _apply(self, fn, recurse=True):
        self._param_td._erase_cache()
        param_td = self._param_td
        self._param_td = param_td.copy()
        # Keep a list of buffers to update .data only
        bufs = dict(self._buffers)
        out: TensorDictBase = super()._apply(fn, recurse=recurse)
        for key, val in bufs.items():
            val.data = self._buffers[key].data
            self._buffers[key] = val
        # Check device and shape
        cbs = out._check_batch_size(raise_exception=False)
        if not cbs:
            out.auto_batch_size_()
        cd = out._check_device(raise_exception=False)
        if not cd:
            out.auto_device_()
        return out

    @implement_for("torch", None, "2.1")
    def _apply(self, fn):  # noqa: F811
        self._param_td._erase_cache()
        param_td = self._param_td
        self._param_td = param_td.copy()
        # Keep a list of buffers to update .data only
        bufs = dict(self._buffers)
        out: TensorDictBase = super()._apply(fn)
        for key, val in bufs.items():
            val.data = self._buffers[key].data
            self._buffers[key] = val
        # Check device and shape
        cbs = out._check_batch_size(raise_exception=False)
        if not cbs:
            out.auto_batch_size_()
        cd = out._check_device(raise_exception=False)
        if not cd:
            out.auto_device_()
        return out


TDPARAM_HANDLED_FUNCTIONS = copy(TD_HANDLED_FUNCTIONS)


def implements_for_tdparam(torch_function: Callable) -> Callable[[Callable], Callable]:
    """Register a torch function override for TensorDictParams."""

    @functools.wraps(torch_function)
    def decorator(func: Callable) -> Callable:
        TDPARAM_HANDLED_FUNCTIONS[torch_function] = func
        return func

    return decorator


@implements_for_tdparam(torch.empty_like)
def _empty_like(td: TensorDictBase, *args, **kwargs) -> TensorDictBase:
    return td.apply(
        lambda x: torch.empty_like(x, *args, **kwargs),
        device=kwargs.pop("device", NO_DEFAULT),
    )


_register_tensor_class(TensorDictParams)
