# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import collections
import functools
import inspect
import weakref
from collections.abc import Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
from copy import copy
from functools import wraps
from pathlib import Path
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
from tensordict._deprecation import deprecated_attributes
from tensordict._lazy import _CustomOpTensorDict, LazyStackedTensorDict
from tensordict._nestedkey import NestedKey
from tensordict._td import _SubTensorDict, TensorDict
from tensordict._tensorcollection import TensorCollection
from tensordict._torch_func import TD_HANDLED_FUNCTIONS
from tensordict.base import (
    _default_is_leaf,
    _is_leaf_nontensor,
    _is_tensor_collection,
    _NESTED_TENSORS_AS_LISTS,
    _NESTED_TENSORS_AS_LISTS_NONTENSOR,
    _register_tensor_class,
    _SELF_NESTING_ERROR,
    CompatibleType,
    NO_DEFAULT,
    T,
    TensorDictBase,
)
from tensordict.utils import (
    _BufferLegacy,
    _erase_cache_first,
    _lock_blocked,
    _LOCK_ERROR,
    _unravel_key_to_tuple,
    _UPDATE_IGNORE_LOCK_WARNED,
    _warn_update_ignore_lock,
    _zip_strict,
    IndexType,
    is_batchedtensor,
    is_non_tensor,
    KEY_SEPARATOR,
)
from torch import multiprocessing as mp, nn, Tensor
from torch.compiler import is_compiling, is_dynamo_compiling
from torch.nn.parameter import Buffer
from torch.utils._pytree import tree_map

try:
    from functorch import dim as ftdim

    _has_funcdim = True
except ImportError:
    from tensordict.utils import _ftdim_mock as ftdim

    _has_funcdim = False

if TYPE_CHECKING:
    from typing_extensions import Self
else:
    Self = Any

__all__ = ["TensorDictParams"]


def _apply_leaves(data, fn, *others):
    # ``others`` have the same structure as ``data``: ``fn`` receives their
    # matching leaves after the leaf of ``data``.
    if isinstance(data, TensorDict):
        with data.unlock_():
            for key, val in list(data.items()):
                other_vals = [other._get_str(key, NO_DEFAULT) for other in others]
                data._set_str(
                    key,
                    _apply_leaves(val, fn, *other_vals),
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
        _apply_leaves(data._source, fn, *(other._source for other in others))
        return data
    elif isinstance(data, _SubTensorDict):
        raise RuntimeError(
            "Using a _SubTensorDict within a TensorDictParams isn't permitted."
        )
    else:
        return fn(data, *others)


def _get_args_dict(signature, args, kwargs):
    bound_arguments = signature.bind(*args, **kwargs)
    bound_arguments.apply_defaults()

    args_dict = dict(bound_arguments.arguments)
    return args_dict


def _maybe_make_param(tensor):
    if isinstance(tensor, (Tensor, ftdim.Tensor)) and not isinstance(
        tensor, (nn.Parameter, Buffer, _BufferLegacy)
    ):
        if tensor.dtype in (torch.float, torch.double, torch.half):
            tensor = nn.Parameter(tensor)
        elif not is_batchedtensor(tensor):
            # convert all non-parameters to buffers
            # dataptr = tensor.data.data_ptr()
            tensor = Buffer(tensor)
        else:
            # We want to keep the grad_fn of tensors, e.g. param.expand(10) should point to the original param
            tensor = _BufferLegacy(tensor)
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
            tensor = _BufferLegacy(tensor)

        # assert tensor.data.data_ptr() == dataptr
    return tensor


def _wrap_like(tensor, orig):
    # to(), cpu() and cuda() return plain tensors for the leaves they convert:
    # wrap each like the leaf it replaces. Parameters stay parameters with the
    # same requires_grad (a parameter with an integer or boolean dtype after
    # the call becomes a buffer) and buffers stay buffers.
    if tensor is orig:
        return tensor
    if isinstance(orig, nn.Parameter) and (
        tensor.is_floating_point() or tensor.is_complex()
    ):
        return nn.Parameter(tensor, requires_grad=orig.requires_grad)
    return _maybe_make_param_or_buffer(tensor)


# Override: the methods decorated with _unlock_and_set, _get_post_hook,
# _fallback, _fallback_property, _carry_over and _apply_on_data forward to
# _param_td, so that results keep their nn.Parameters.
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
        # p["x"] = p would write p into the tensordict that p wraps
        check_self = name in ("__setitem__", "set")
        # computed once: building it on every call cost more with each parameter
        signature = inspect.signature(func) if self.only_for_kwargs else None

        @wraps(func)
        def new_func(_self, *args, **kwargs):  # type: ignore[misc]
            if self.only_for_kwargs:
                # Dynamo can't bind a Signature built outside the traced
                # code (its parameters are a mappingproxy): build it there.
                sig = inspect.signature(func) if is_dynamo_compiling() else signature
                arg_dict = _get_args_dict(sig, (_self, *args), kwargs)
                for kwarg, exp_value in self.only_for_kwargs.items():
                    cur_val = arg_dict.get(kwarg, NO_DEFAULT)
                    if cur_val != exp_value:
                        # escape
                        meth = getattr(_self._param_td, name)
                        out = meth(*args, **kwargs)
                        return out
            if check_self:
                value = args[1] if len(args) > 1 else kwargs.get("item")
                if value is _self:
                    key = args[0] if args else kwargs.get("key")
                    raise ValueError(_SELF_NESTING_ERROR.format(key))
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


# items() and values() iterate over the wrapped tensordict with its default rule
# for these is_leaf values. The internal calls pass the two markers to read lazy
# stacks per member; TensorDictParams reads them stacked.
_DEFAULT_TRAVERSAL = (
    None,
    _default_is_leaf,
    _NESTED_TENSORS_AS_LISTS,
    _NESTED_TENSORS_AS_LISTS_NONTENSOR,
)


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


def _data_and_non_tensor(value):
    return value if is_non_tensor(value) else value.data


def _set_replaced_non_tensor(param_td, data):
    # set_at_ and update_at_ replace a non-tensor leaf of data with a stack
    # instead of writing into it: put the new leaf in param_td (#2740).
    for key, value in data.items(True, True, is_leaf=_is_leaf_nontensor):
        if not is_non_tensor(value) or value is param_td.get(key):
            continue
        key = _unravel_key_to_tuple(key)
        parent = param_td._get_tuple(key[:-1], NO_DEFAULT) if len(key) > 1 else param_td
        if isinstance(parent, TensorDictParams):
            # Its _set_str ignores ignore_lock: write into the tensordict it wraps.
            parent = parent._param_td
        parent._set_str(
            key[-1],
            value,
            inplace=False,
            validated=True,
            ignore_lock=True,
            non_blocking=False,
        )


def _apply_on_data_and_non_tensor(func):
    # As _apply_on_data, on a .data that holds the non-tensor leaves of the
    # wrapped tensordict, which .data rebuilds: the writes reach them (#2263).
    replaces_leaves = func.__name__ in ("set_at_", "update_at_")

    @wraps(func)
    def new_func(self, *args, **kwargs):  # type: ignore[misc]
        has_non_tensor = self._param_td._has_non_tensor
        if has_non_tensor:
            data = self._param_td._fast_apply(
                _data_and_non_tensor, is_leaf=_is_leaf_nontensor, propagate_lock=True
            )
        else:
            data = self.data
        getattr(data, func.__name__)(*args, **kwargs)
        if has_non_tensor and replaces_leaves:
            _set_replaced_non_tensor(self._param_td, data)
        return self

    return new_func


class TensorDictParams(TensorDictBase, nn.Module):  # type: ignore[override,misc,attr-defined]
    r"""A TensorDict whose entries are registered as the parameters and buffers of an :class:`~torch.nn.Module`.

    Use it to keep a TensorDict of parameters on a module, for example the
    stacked weights of an ensemble or a copy of the weights for a target
    network. A plain TensorDict attribute is not seen by ``module.parameters()``,
    ``module.state_dict()`` or ``module.to()``. A TensorDictParams attribute is
    seen by all three, so an optimizer can train its entries.

    A TensorDictParams behaves like the TensorDict that it wraps. You can index
    it, set entries, and call :meth:`~tensordict.TensorDictBase.to_module` to
    load the values into a module. :meth:`named_parameters` names each entry
    by its nested key joined with ``"."``, for example ``"0.weight"``, as
    :class:`~torch.nn.Module` does.

    When you set an entry, float16, float32 and float64 tensors become
    :class:`~torch.nn.Parameter`. Tensors of all other dtypes (including
    bfloat16, complex and bool) become buffers. An optimizer built from
    ``parameters()`` therefore does not see bfloat16 entries. Pass
    ``no_convert=True`` to register all tensors as buffers.

    .. warning:: TensorDictParams wraps the TensorDict that you pass,
        without a copy, and turns its leaves into parameters and
        buffers in place. It registers the leaves that are present at
        construction. If you later add or replace an entry of the
        original TensorDict, indexing the TensorDictParams shows the
        new value, but :meth:`.parameters` and :meth:`~.buffers` keep
        the old one. Write through the TensorDictParams, or pass
        ``lock=True``.

    Args:
        parameters (TensorDictBase or dict, optional): the tensordict to wrap.
            A ``dict`` is wrapped in a :class:`~tensordict.TensorDict` first.
            Exclusive with ``**kwargs``. Defaults to ``None``.

    Keyword Args:
        no_convert (bool, optional): if ``True``, tensors that are not
            parameters or buffers become :class:`~torch.nn.Buffer` instances.
            If ``False``, float16, float32 and float64 tensors become
            :class:`~torch.nn.Parameter`, and tensors of all other dtypes
            become buffers. Tensors that already are parameters or buffers
            stay as they are. Defaults to ``False``.
        lock (bool, optional): if ``True``, locks the wrapped tensordict (see
            :ref:`keys-locking`), so that entries can no longer be added to it
            or removed from it directly. Writes through the TensorDictParams
            still work. Defaults to ``False``.
        **kwargs: the entries of the tensordict, as ``key=value`` pairs, when
            ``parameters`` is not given. Exclusive with ``parameters``.

    Examples:
        >>> import torch
        >>> from torch import nn
        >>> from tensordict import TensorDict
        >>> from tensordict.nn import TensorDictParams
        >>> module = nn.Sequential(nn.Linear(3, 4), nn.Linear(4, 4))
        >>> params = TensorDict.from_module(module)
        >>> _ = params.lock_()
        >>> p = TensorDictParams(params)
        >>> print(p)
        TensorDictParams(params=TensorDict(batch_size=[], fields={
            0: {bias: f32[4], weight: f32[4,3]},
            1: {bias: f32[4], weight: f32[4,4]}}))
        >>> class CustomModule(nn.Module):
        ...     def __init__(self, params):
        ...         super().__init__()
        ...         self.params = params
        >>> m = CustomModule(p)
        >>> [name for name, _ in m.named_parameters()]
        ['params.0.weight', 'params.0.bias', 'params.1.weight', 'params.1.bias']
        >>> # The wrapper supports assignment, and values are converted to Parameters
        >>> m.params['other'] = torch.randn(3)
        >>> isinstance(m.params['other'], nn.Parameter)
        True
        >>> # A bfloat16 entry becomes a buffer
        >>> m.params['half'] = torch.zeros(3, dtype=torch.bfloat16)
        >>> [name for name, _ in m.named_buffers()]
        ['params.half']

    """

    if TYPE_CHECKING:
        # The comparisons return a TensorDict, not the Self of the
        # TensorDictBase declarations (#3305).
        def __eq__(self, other: object) -> TensorDict: ...  # type: ignore[override]

        def __ne__(self, other: object) -> TensorDict: ...  # type: ignore[override]

        # save, dumps and memmap return a TensorDict, as memmap_like below
        # does (#3320).
        def save(  # type: ignore[override]
            self,
            prefix: str | Path | None = None,
            copy_existing: bool = False,
            *,
            num_threads: int = 0,
            return_early: bool = False,
            share_non_tensor: bool = False,
            robust_key: bool | None = True,
            archive: bool | None = None,
            compression: str | int | None = None,
            atomic: bool = False,
        ) -> TensorDict: ...

        def dumps(  # type: ignore[override]
            self,
            prefix: str | Path | None = None,
            copy_existing: bool = False,
            *,
            num_threads: int = 0,
            return_early: bool = False,
            share_non_tensor: bool = False,
            robust_key: bool | None = True,
            archive: bool | None = None,
            compression: str | int | None = None,
            atomic: bool = False,
        ) -> TensorDict: ...

        def memmap(  # type: ignore[override]
            self,
            prefix: str | Path | None = None,
            copy_existing: bool = False,
            *,
            num_threads: int = 0,
            return_early: bool = False,
            share_non_tensor: bool = False,
            existsok: bool = True,
            robust_key: bool | None = True,
            archive: bool | None = None,
            compression: str | int | None = None,
            atomic: bool = False,
        ) -> TensorDict: ...

    def __init__(
        self,
        parameters: TensorDictBase | dict | None = None,
        *,
        no_convert=False,
        lock: bool = False,
        **kwargs,
    ):
        if is_compiling():
            # Dynamo does not trace the explicit nn.Module.__init__(self) call,
            # but traces it through super(). TensorCollection.__init__ is
            # skipped because it refuses direct instantiation.
            super(TensorCollection, self).__init__()
        else:
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
        parameters: TensorDictBase | dict | None = None,
        *,
        source: TensorDictBase | dict | None = None,
        no_convert=None,
        lock: bool = False,
        params: dict | None = None,
        buffers: dict | None = None,
        **kwargs,
    ):
        # ``source`` is the name that the base code uses for ``parameters``
        if source is not None:
            if parameters is not None:
                raise TypeError(
                    "TensorDictParams._new_unsafe() got both parameters and source."
                )
            parameters = source
        if parameters is None:
            parameters = kwargs

        if isinstance(parameters, dict):
            if no_convert is None:
                # Then _new_unsafe is called from somewhere that doesn't know
                #  that it's a TDParams and we return a TensorDict (eg, torch.gather)
                return TensorDict._new_unsafe(parameters, lock=lock, **kwargs)
            parameters = TensorDict._new_unsafe(parameters, **kwargs)
        if is_compiling():
            return TensorDictParams(parameters, no_convert="skip", lock=lock)
        if isinstance(parameters, TensorDictParams):
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

    def register_get_post_hook(self, hook):
        """Registers a hook that is called after each read of a leaf.

        Args:
            hook (Callable): called as ``hook(tensordict_params, value)``
                after each read of a leaf. If it returns a value other than
                ``None``, that value replaces the one read.
        """
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
        if func not in _TDPARAM_HANDLED_FUNCTIONS or not all(
            issubclass(t, (Tensor, ftdim.Tensor, TensorDictBase)) for t in types
        ):
            from torch._ops import HigherOrderOperator

            if isinstance(func, HigherOrderOperator):
                with torch._C.DisableTorchFunctionSubclass():
                    return func(*args, **kwargs)
            return NotImplemented
        return _TDPARAM_HANDLED_FUNCTIONS[func](*args, **kwargs)

    # Override: perf. Through the base front and _set_tuple, a key write is
    # 10-25% slower.
    @_lock_blocked
    @_unlock_and_set
    def __setitem__(  # type: ignore[misc]
        self,
        index: IndexType,
        value: Any,
    ) -> None: ...

    @_lock_blocked
    @_unlock_and_set
    def set(
        self,
        key: NestedKey,
        item: CompatibleType,
        inplace: bool = False,
        *,
        non_blocking: bool = False,
        **kwargs: Any,
    ) -> TensorDictBase: ...

    # Override: the base runs with _param_td unlocked, so that a lock=True
    # content takes nested entries, and the parameters are registered again
    # afterwards, since the base writes nested entries into the nested
    # tensordicts directly.
    @_lock_blocked
    def update(
        self,
        input_dict_or_td: Mapping[Any, Any] | TensorDictBase | None = None,
        clone: bool = False,
        inplace: bool = False,
        *,
        non_blocking: bool = False,
        keys_to_update: Sequence[NestedKey] | None = None,
        is_leaf: Callable[[Type], bool] | None = None,
        update_batch_size: bool = False,
        ignore_lock: bool = False,
        **kwargs,
    ) -> TensorDictBase:
        if ignore_lock and ignore_lock is not _UPDATE_IGNORE_LOCK_WARNED:
            _warn_update_ignore_lock(stacklevel=3)
        # Not ``with self._param_td.unlock_()``: Dynamo cannot enter a
        # tensordict passed into the compiled region as a context manager.
        relock = self._param_td.is_locked
        self._param_td.unlock_()
        try:
            TensorDictBase.update(
                self,
                input_dict_or_td,
                clone=clone,
                inplace=inplace,
                keys_to_update=keys_to_update,
                non_blocking=non_blocking,
                is_leaf=is_leaf,
                update_batch_size=update_batch_size,
                **kwargs,
            )
            self._reset_params()
        finally:
            if relock:
                self._param_td.lock_()
        return self

    @_lock_blocked
    @_unlock_and_set
    def pop(self, key: NestedKey, default: Any = NO_DEFAULT) -> CompatibleType: ...

    # Override: the base would read the entry through the get hooks and write
    # the hooked value.
    @_lock_blocked
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
        *,
        out: TensorCollection | None = None,
        index_with_generator: bool = False,
        pbar: bool = False,
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
        propagate_lock: bool = False,
        call_on_nested: bool = False,
        is_leaf: Callable[[type], bool] | None = None,
        out: TensorDictBase | None = None,
        **constructor_kwargs,
    ) -> TensorDictBase | None: ...

    @_unlock_and_set(inplace=True)
    def named_apply(
        self,
        fn: Callable,
        *others: TensorDictBase,
        nested_keys: bool = False,
        batch_size: Sequence[int] | None = None,
        device: torch.device | None = NO_DEFAULT,
        names: Sequence[str] | None = NO_DEFAULT,
        inplace: bool = False,
        default: Any = NO_DEFAULT,
        filter_empty: bool | None = None,
        propagate_lock: bool = False,
        call_on_nested: bool = False,
        is_leaf: Callable[[type], bool] | None = None,
        out: TensorDictBase | None = None,
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
    def get(self, key: NestedKey, *args, **kwargs) -> CompatibleType: ...

    @_get_post_hook
    @_fallback
    def __getitem__(
        self, index: IndexType
    ) -> Self | Tensor | TensorCollection | Any: ...

    @_fallback
    def _set_device(self, device: torch.device) -> Self: ...  # type: ignore[misc]

    @_fallback
    def _set_names(self, names: Sequence[str] | None) -> None: ...

    @_fallback
    def auto_device_(self) -> Self: ...

    __getitems__ = __getitem__

    def to(self, *args, **kwargs) -> TensorDictBase:
        params = self._param_td.to(*args, **kwargs)
        if params is self._param_td:
            return self
        return self._from_converted(params)

    def cpu(self, **kwargs):
        params = self._param_td.cpu(**kwargs)
        if params is self._param_td:
            return self
        return self._from_converted(params)

    def cuda(self, device: int | str | torch.device | None = None, **kwargs):
        params = self._param_td.cuda(device=device, **kwargs)
        if params is self._param_td:
            return self
        return self._from_converted(params)

    def _from_converted(self, params: TensorDictBase) -> TensorDictParams:
        # Wraps what to(), cpu() or cuda() returned for self._param_td, keeping
        # the parameters and buffers of self and its no_convert.
        params = _apply_leaves(params, _wrap_like, self._param_td)
        out = TensorDictParams(params, no_convert="skip")
        out.no_convert = self.no_convert
        return out

    def _clone(self, recurse: bool = True) -> TensorDictBase:
        """Clones the TensorDictParams.

        .. warning::
            The effect of this call is different from a regular torch.Tensor.clone call
            in that it will create a TensorDictParams instance with a new copy of the
            parameters and buffers __detached__ from the current graph. For a
            regular clone (ie, cloning leaf parameters onto a new tensor that
            is part of the graph), simply call::

                params.apply(torch.clone)

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

    # Override: TensorDict._chunk is faster than the generic split
    @_fallback
    def _chunk(self, chunks: int, dim: int) -> tuple[TensorDictBase, ...]: ...

    # Override: the base would read the entries through the get hooks.
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

    @_fallback
    def to_zarr(
        self,
        filename,
        *,
        allow_pickle: bool | None = None,
        **kwargs,
    ): ...

    def __hash__(self):
        return hash((id(self), id(self.__dict__.get("_param_td"))))

    @_fallback
    def _compare(self, other, op, reflected_op, default): ...

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

    # Override: the get hooks run on the indexed value, as get_at reads it;
    # the base would run them on the whole entry.
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

    # Override: the base would read the entries through the get hooks.
    @_fallback
    def contiguous(self, *args, **kwargs): ...

    # Override: perf, the base front and a _del hook take 13% longer.
    @_lock_blocked
    @_unlock_and_set
    def del_(self, *args, **kwargs): ...

    # Override: forwards to _param_td so that results keep their nn.Parameters.
    @_fallback
    def _detach_(self): ...

    @_fallback_property
    def device(self): ...

    @_fallback
    def entry_class(self, *args, **kwargs): ...

    # Override: forwards to _param_td.
    @_fallback
    def is_contiguous(self, *args, **kwargs): ...

    @_fallback
    def keys(self, *args, **kwargs): ...

    # Override: the base fills a clone in place, which a leaf parameter refuses.
    @_fallback
    def masked_fill(self, *args, **kwargs): ...

    @_fallback
    def _masked_fill_(self, mask, value): ...

    def memmap_(
        self,
        prefix: str | Path | None = None,
        copy_existing: bool = False,
        num_threads: int = 0,
        *,
        return_early: bool = False,
        share_non_tensor: bool = False,
        existsok: bool = True,
        robust_key: bool | None = True,
        atomic: bool = False,
    ) -> TensorDictBase:
        raise RuntimeError(
            "Cannot build a memmap TensorDict in-place. Use memmap or memmap_like instead."
        )

    _memmap_ = TensorDict._memmap_

    @_fallback_property
    def names(self): ...

    # Override: forwards to _param_td; in-place pinning would replace the parameters.
    def pin_memory(self, *args, **kwargs):
        if kwargs.get("inplace", False):
            raise RuntimeError(
                f"Cannot pin_memory in-place with {type(self).__name__}."
            )
        return _fallback(self.pin_memory)(self, *args, **kwargs)

    @_unlock_and_set
    def _select(self, *args, **kwargs): ...

    # Override: renames in _param_td. The result is a TensorDict, as for select and
    # exclude: building a TensorDictParams, an nn.Module, does not compile.
    @_fallback
    def _rename_keys(self, pairs, *, strict, prune_root):
        """Renames the keys of a copy of _param_td."""

    # Override: shares the parameters of _param_td in place.
    @_fallback
    def _share_memory_(self): ...

    # Override: setting the flag does not lock the parameters.
    @TensorDictBase.is_locked.setter
    def is_locked(self, value):
        self._is_locked = bool(value)

    # Override: the flag lives on _param_td.
    @_fallback_property
    def is_shared(self) -> bool: ...

    # Override: the flag lives on _param_td.
    @_fallback_property
    def is_memmap(self) -> bool: ...

    # Override: the flag lives on _param_td.
    @property
    def _is_shared(self) -> bool:
        return self._param_td._is_shared

    # Override: the flag lives on _param_td.
    @property
    def _is_memmap(self) -> bool:
        return self._param_td._is_memmap

    @_fallback_property
    def shape(self) -> torch.Size: ...

    # Override: locks _param_td only if it is not locked already.
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

    # Override: unlocks _param_td only if the parameters' lock is not kept.
    @_erase_cache_first
    def _propagate_unlock(self):
        # if we end up here, we can clear the graph associated with this td
        self._is_locked = False

        if not self._lock_content:
            return self._param_td._propagate_unlock()
        return []

    @property
    def data(self) -> Self:
        return self._param_td._data()

    @property
    def grad(self):
        return self._param_td._grad()

    @_unlock_and_set(inplace=True)
    def flatten_keys(
        self,
        separator: str = KEY_SEPARATOR,
        inplace: bool = False,
        is_leaf: Callable[[Type], bool] | None = None,
    ) -> TensorDictBase: ...

    @_unlock_and_set(inplace=True)
    def unflatten_keys(
        self, separator: str = KEY_SEPARATOR, inplace: bool = False
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
        auto_batch_size: bool | None = None,
        batch_size=None,
        device=None,
        batch_dims=None,
        names=None,
    ): ...

    @_carry_over
    def _legacy_transpose(self, dim0, dim1): ...

    @_fallback
    def _transpose(self, dim0, dim1): ...

    @_fallback
    def _where(
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

    @_fallback
    def _cast_reduction(self, **kwargs): ...

    # Override: the base reads the entries through the get hooks for a dim.
    @_fallback
    def all(self, dim: int | None = None) -> bool | TensorDictBase: ...

    # Override: as all.
    @_fallback
    def any(self, dim: int | None = None) -> bool | TensorDictBase: ...

    @_fallback
    def _expand(self, shape, names): ...

    def masked_select(self, mask: Tensor, *, inplace: bool = False) -> Self:
        if inplace:
            # the masked leaves would no longer be registered parameters
            raise NotImplementedError(
                "masked_select(inplace=True) is not supported on "
                "TensorDictParams; use inplace=False."
            )
        return self._param_td.masked_select(mask)

    @_fallback
    def memmap_like(  # type: ignore[override]
        self,
        prefix: str | Path | None = None,
        copy_existing: bool = False,
        *,
        existsok: bool = True,
        num_threads: int = 0,
        return_early: bool = False,
        share_non_tensor: bool = False,
        robust_key: bool | None = True,
        archive: bool | None = None,
    ) -> TensorDict: ...

    # Override: with inplace=True, the base would rewrite the leaves read
    # through the get hooks.
    @_fallback
    def reshape(self, *args, **kwargs): ...

    # Override: as reshape.
    @_fallback
    def repeat_interleave(
        self,
        repeats: torch.Tensor | int,
        dim: int | None = None,
        *,
        output_size: int | None = None,
        inplace: bool = False,
    ): ...

    @_fallback
    def _repeat(self, *repeats: int): ...

    @_fallback
    def _split(self, split_sizes, dim): ...

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
        preserve_module_state: bool = True,
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
        if leaves_only and is_leaf not in _DEFAULT_TRAVERSAL:
            for _, v in self._leaves_from_keys(include_nested, is_leaf, sort):
                yield v
            return
        if is_leaf is None:
            is_leaf = _default_is_leaf
        for v in self._param_td.values(include_nested, leaves_only, sort=sort):
            if not is_leaf(type(v)):
                yield v
                continue
            yield self._apply_get_post_hook(v)

    def _leaves_from_keys(self, include_nested, is_leaf, sort):
        # The leaves that keys() lists for is_leaf, with their values. The keys
        # view only recurses into tensor collections, while the fast path of
        # TensorDict.items calls items() on any value that is_leaf rejects, and
        # raises for a tensor.
        param_td = self._param_td
        for key in param_td.keys(include_nested, True, is_leaf=is_leaf, sort=sort):
            if isinstance(key, str):
                val = param_td._get_str(key, NO_DEFAULT)
            else:
                val = param_td._get_tuple(key, NO_DEFAULT)
            yield key, self._apply_get_post_hook(val)

    def state_dict(self, destination=None, prefix="", keep_vars=False, flatten=True):
        # flatten must be True by default to comply with module's state-dict API
        # since we want all params to be visible at root
        return self._param_td.state_dict(
            destination=destination,
            prefix=prefix,
            keep_vars=keep_vars,
            flatten=flatten,
        )

    def load_state_dict(
        self,
        state_dict: OrderedDict[str, Any],
        strict=True,
        assign=False,
        from_flatten=None,
    ):
        _metadata = getattr(state_dict, "_metadata", None)

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
        state_dict = collections.OrderedDict(state_dict)
        if _metadata is not None:
            state_dict._metadata = _metadata
        self.data.load_state_dict(
            state_dict, strict=strict, assign=assign, from_flatten=from_flatten
        )
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
        nested_prefix = tuple(key for key in prefix.split(".") if key)
        if nested_prefix:
            data = data.get(nested_prefix)
        if data is None:
            return nn.Module._load_from_state_dict(
                self,
                state_dict,
                prefix,
                local_metadata,
                strict,
                missing_keys,
                unexpected_keys,
                error_msgs,
            )
        self.data.load_state_dict(data)

    def items(
        self,
        include_nested: bool = False,
        leaves_only: bool = False,
        is_leaf: Callable[[Type], bool] | None = None,
        *,
        sort: bool = False,
    ) -> Iterator[CompatibleType]:
        if leaves_only and is_leaf not in _DEFAULT_TRAVERSAL:
            yield from self._leaves_from_keys(include_nested, is_leaf, sort)
            return
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

    @_apply_on_data_and_non_tensor
    def set_at_(
        self,
        key: NestedKey,
        value: CompatibleType,
        index: IndexType,
        *,
        non_blocking: bool = False,
    ) -> Self: ...

    @_apply_on_data_and_non_tensor
    def set_(
        self,
        key: NestedKey,
        item: CompatibleType,
        *,
        non_blocking: bool = False,
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
        list_item: list[CompatibleType],
        dim: int,
        idx: IndexType,
    ) -> Self: ...

    @_apply_on_data_and_non_tensor
    def update_(
        self,
        input_dict_or_td: dict[str, CompatibleType] | T | None = None,
        clone: bool = False,
        *,
        non_blocking: bool = False,
        keys_to_update: Sequence[NestedKey] | None = None,
        **kwargs,
    ) -> T: ...

    @_apply_on_data_and_non_tensor
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


_TDPARAM_HANDLED_FUNCTIONS = copy(TD_HANDLED_FUNCTIONS)


def _implements_for_tdparam(torch_function: Callable) -> Callable[[Callable], Callable]:
    """Register a torch function override for TensorDictParams."""

    @functools.wraps(torch_function)
    def decorator(func: Callable) -> Callable:
        _TDPARAM_HANDLED_FUNCTIONS[torch_function] = func
        return func

    return decorator


@_implements_for_tdparam(torch.empty_like)
def _empty_like(td: TensorDictBase, *args, **kwargs) -> TensorDictBase:
    return td.apply(
        lambda x: torch.empty_like(x, *args, **kwargs),
        device=kwargs.pop("device", NO_DEFAULT),
    )


_register_tensor_class(TensorDictParams)


__getattr__ = deprecated_attributes(
    __name__,
    {
        "TDPARAM_HANDLED_FUNCTIONS": (_TDPARAM_HANDLED_FUNCTIONS, None),
        "implements_for_tdparam": (_implements_for_tdparam, None),
    },
    removal="0.17",
)
