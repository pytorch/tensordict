# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import functools
import inspect
import warnings
from functools import wraps
from textwrap import indent
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    MutableSequence,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import torch
from cloudpickle import dumps as cloudpickle_dumps, loads as cloudpickle_loads
from tensordict._td import TensorDict

from tensordict.base import is_tensor_collection, NO_DEFAULT, TensorDictBase
from tensordict.functional import make_tensordict
from tensordict.nn.utils import _dispatch_td_nn_modules, _set_skip_existing_None
from tensordict.utils import (
    _unravel_key_to_tuple,
    _zip_strict,
    NestedKey,
    unravel_key_list,
)
from torch import nn, Tensor

try:
    from torch.compiler import is_compiling
except ImportError:  # torch 2.0
    from torch._dynamo import is_compiling

try:
    from functorch import FunctionalModule, FunctionalModuleWithBuffers

    _has_functorch = True
except ImportError:
    _has_functorch = False

    class FunctionalModule:
        pass

    class FunctionalModuleWithBuffers:
        pass


__all__ = [
    "TensorDictModule",
    "TensorDictModuleWrapper",
]


class dispatch:
    """Allows for a function expecting a TensorDict to be called using kwargs.

    :func:`dispatch` must be used within modules that have an ``in_keys`` (or
    another source of keys indicated by the ``source`` keyword argument) and
    ``out_keys`` (or another ``dest`` key list) attributes indicating what keys
    to be read and written from the tensordict. The wrapped function should
    also have a ``tensordict`` leading argument.

    The resulting function will return a single tensor (if there is a single
    element in out_keys), otherwise it will return a tuple sorted as the ``out_keys``
    of the module.

    :func:`dispatch` can be used either as a method or as a class when extra arguments
    need to be passed.

    Args:
        separator (str, optional): separator that combines sub-keys together
            for ``in_keys`` that are tuples of strings.
            Defaults to ``"_"``.
        source (str or list of keys, optional): if a string is provided,
            it points to the module attribute that contains the
            list of input keys to be used. If a list is provided instead, it
            will contain the keys used as input to the module.
            Defaults to ``"in_keys"`` which is the attribute name of
            :class:`~.TensorDictModule` list of input keys.
        dest (str or list of keys, optional): if a string is provided,
            it points to the module attribute that contains the
            list of output keys to be used. If a list is provided instead, it
            will contain the keys used as output to the module.
            Defaults to ``"out_keys"`` which is the attribute name of
            :class:`~.TensorDictModule` list of output keys.
        auto_batch_size (bool, optional): if ``True``, the batch-size of the
            input tensordict is determined automatically as the maximum number
            of common dimensions across all the input tensors.
            Defaults to ``True``.

    Examples:
        >>> class MyModule(nn.Module):
        ...     in_keys = ["a"]
        ...     out_keys = ["b"]
        ...
        ...     @dispatch
        ...     def forward(self, tensordict):
        ...         tensordict['b'] = tensordict['a'] + 1
        ...         return tensordict
        ...
        >>> module = MyModule()
        >>> b = module(a=torch.zeros(1, 2))
        >>> assert (b == 1).all()
        >>> # equivalently
        >>> class MyModule(nn.Module):
        ...     keys_in = ["a"]
        ...     keys_out = ["b"]
        ...
        ...     @dispatch(source="keys_in", dest="keys_out")
        ...     def forward(self, tensordict):
        ...         tensordict['b'] = tensordict['a'] + 1
        ...         return tensordict
        ...
        >>> module = MyModule()
        >>> b = module(a=torch.zeros(1, 2))
        >>> assert (b == 1).all()
        >>> # or this
        >>> class MyModule(nn.Module):
        ...     @dispatch(source=["a"], dest=["b"])
        ...     def forward(self, tensordict):
        ...         tensordict['b'] = tensordict['a'] + 1
        ...         return tensordict
        ...
        >>> module = MyModule()
        >>> b = module(a=torch.zeros(1, 2))
        >>> assert (b == 1).all()

    :func:`dispatch_kwargs` will also work with nested keys with the default
    ``"_"`` separator.

    Examples:
        >>> class MyModuleNest(nn.Module):
        ...     in_keys = [("a", "c")]
        ...     out_keys = ["b"]
        ...
        ...     @dispatch
        ...     def forward(self, tensordict):
        ...         tensordict['b'] = tensordict['a', 'c'] + 1
        ...         return tensordict
        ...
        >>> module = MyModuleNest()
        >>> b, = module(a_c=torch.zeros(1, 2))
        >>> assert (b == 1).all()

    If another separator is wanted, it can be indicated with the ``separator``
    argument in the constructor:

    Examples:
        >>> class MyModuleNest(nn.Module):
        ...     in_keys = [("a", "c")]
        ...     out_keys = ["b"]
        ...
        ...     @dispatch(separator="sep")
        ...     def forward(self, tensordict):
        ...         tensordict['b'] = tensordict['a', 'c'] + 1
        ...         return tensordict
        ...
        >>> module = MyModuleNest()
        >>> b, = module(asepc=torch.zeros(1, 2))
        >>> assert (b == 1).all()


    Since the input keys is a sorted sequence of strings,
    :func:`dispatch` can also be used with unnamed arguments where the order
    must match the order of the input keys.

    .. note::
        If the first argument is a :class:`~.TensorDictBase` instance, it is
        assumed that dispatch is __not__ being used and that this tensordict
        contains all the necessary information to be run through the module.
        In other words, one cannot decompose a tensordict with the first key
        of the module inputs pointing to a tensordict instance.
        In general, it is preferred to use :func:`dispatch` with tensordict
        leaves only.

    Examples:
        >>> class MyModuleNest(nn.Module):
        ...     in_keys = [("a", "c"), "d"]
        ...     out_keys = ["b"]
        ...
        ...     @dispatch
        ...     def forward(self, tensordict):
        ...         tensordict['b'] = tensordict['a', 'c'] + tensordict["d"]
        ...         return tensordict
        ...
        >>> module = MyModuleNest()
        >>> b, = module(torch.zeros(1, 2), d=torch.ones(1, 2))  # works
        >>> assert (b == 1).all()
        >>> b, = module(torch.zeros(1, 2), torch.ones(1, 2))  # works
        >>> assert (b == 1).all()
        >>> try:
        ...     b, = module(torch.zeros(1, 2), a_c=torch.ones(1, 2))  # fails
        ... except:
        ...     print("oopsy!")
        ...

    """

    DEFAULT_SEPARATOR = "_"
    DEFAULT_SOURCE = "in_keys"
    DEFAULT_DEST = "out_keys"

    def __new__(
        cls,
        separator=DEFAULT_SEPARATOR,
        source=DEFAULT_SOURCE,
        dest=DEFAULT_DEST,
        auto_batch_size: bool = True,
    ):
        if callable(separator):
            func = separator
            separator = dispatch.DEFAULT_SEPARATOR
            self = super().__new__(cls)
            self.__init__(separator, source, dest)
            return self.__call__(func)
        return super().__new__(cls)

    def __init__(
        self,
        separator=DEFAULT_SEPARATOR,
        source=DEFAULT_SOURCE,
        dest=DEFAULT_DEST,
        auto_batch_size: bool = True,
    ):
        self.separator = separator
        self.source = source
        self.dest = dest
        self.auto_batch_size = auto_batch_size

    def __call__(self, func: Callable) -> Callable:

        is_method = inspect.ismethod(func) or (
            inspect.isfunction(func)
            and func.__code__.co_argcount > 0
            and func.__code__.co_varnames[0] == "self"
        )
        # sanity check
        for i, key in enumerate(inspect.signature(func).parameters):
            if (is_method or (key == "self")) and (i == 0):
                is_method = True
                # skip self
                continue
            if key != "tensordict":
                raise RuntimeError(
                    "the first argument of the wrapped function must be "
                    f"named 'tensordict'. Got {key} instead."
                )
            break
        # if the env variable was used, we can skip the wrapper altogether
        if not _dispatch_td_nn_modules():
            return func

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if is_method:
                _self = args[0]
                args = args[1:]
            else:
                _self = None
            if not _dispatch_td_nn_modules():
                return func(_self, *args, **kwargs)

            source = self.source
            if isinstance(source, str):
                if _self is None:
                    raise RuntimeError(
                        "The in keys must be passed to dispatch when func is not a method but a function."
                    )
                source = getattr(_self, source)
            tensordict = None
            if len(args):
                if is_tensor_collection(args[0]):
                    tensordict, args = args[0], args[1:]
            if tensordict is None:
                tensordict_values = {}
                dest = self.dest
                if isinstance(dest, str):
                    if _self is None:
                        raise RuntimeError(
                            "The in keys must be passed to dispatch when func is not a method but a function."
                        )
                    dest = getattr(_self, dest)
                for key in source:
                    expected_key = self.separator.join(_unravel_key_to_tuple(key))
                    if len(args):
                        tensordict_values[key] = args[0]
                        args = args[1:]
                        if expected_key in kwargs:
                            raise RuntimeError(
                                "Duplicated argument in args and kwargs."
                            )
                    elif expected_key in kwargs:
                        try:
                            tensordict_values[key] = kwargs.pop(expected_key)
                        except KeyError:
                            raise KeyError(
                                f"The key {expected_key} wasn't found in the keyword arguments "
                                f"but is expected to execute that function."
                            )
                batch_size = torch.Size([]) if not self.auto_batch_size else None
                tensordict = make_tensordict(
                    tensordict_values,
                    batch_size=batch_size,
                    auto_batch_size=self.auto_batch_size,
                )
                if _self is not None:
                    out = func(_self, tensordict, *args, **kwargs)
                else:
                    out = func(tensordict, *args, **kwargs)

                # This makes dispatch responsible of handling partial outputs (such as selected through select_out_keys)
                out = tuple(out[key] for key in dest)
                return out[0] if len(out) == 1 else out

            if _self is not None:
                return func(_self, tensordict, *args, **kwargs)
            return func(tensordict, *args, **kwargs)

        return self._update_func_signature(func, wrapper)

    def _update_func_signature(self, func, wrapper):
        # Create a new signature with the desired parameters
        # Get the original function's signature
        orig_signature = inspect.signature(func)

        # params = [inspect.Parameter(name='', kind=inspect.Parameter.VAR_POSITIONAL)]
        params = []
        i = -1
        for i, param in enumerate(orig_signature.parameters.values()):
            if param.kind in (
                inspect.Parameter.VAR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            ):
                i = i - 1
                break
            if param.default is inspect._empty:
                params.append(
                    inspect.Parameter(
                        name=param.name,
                        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        default=None,
                    )
                )
            else:
                params.append(param)

        # Add the **kwargs parameter

        # for key in self.get_source(func, self_func):
        if i >= 0:
            params.extend(list(orig_signature.parameters.values())[i + 1 :])
        elif i == -1:
            params.extend(list(orig_signature.parameters.values()))

        # Update the wrapper's signature
        wrapper.__signature__ = inspect.Signature(params)

        return wrapper

    def get_source(self, func, self_func):
        source = self.source
        if isinstance(source, str):
            return getattr(self_func, source)
        return source


class _OutKeysSelect:
    module: nn.Module | None = None

    def __init__(self, out_keys):
        self.out_keys = list(out_keys)
        self._initialized = None
        self._is_dispatched = None

    def _init(self, module):
        if self._initialized:
            return
        self._initialized = True
        self.module = module
        if not all(key in module.out_keys for key in self.out_keys):
            raise RuntimeError("Some keys are not part of the module out_keys.")
        module.out_keys = self.out_keys

    def __call__(  # noqa: F811
        self,
        module: TensorDictModuleBase,
        tensordict_in: TensorDictBase,
        kwargs: Dict,
        tensordict_out: TensorDictBase,
    ):
        if not self._initialized:
            raise RuntimeError(
                "_OutKeysSelect must be initialized before being called."
            )
        # detect dispatch calls
        in_keys = module.in_keys
        if not tensordict_in and kwargs.get("tensordict") is not None:
            tensordict_in = kwargs.pop("tensordict")
        is_dispatched = self._detect_dispatch(tensordict_in, kwargs, in_keys)
        out_keys = self.out_keys
        # if dispatch filtered the out keys as they should we're happy
        if is_dispatched:
            if (not isinstance(tensordict_out, tuple) and len(out_keys) == 1) or (
                isinstance(tensordict_out, tuple)
                and len(out_keys) == len(tensordict_out)
            ):
                return tensordict_out
        if is_dispatched:
            # it might be the case that dispatch was not aware of what the out-keys were.
            if isinstance(tensordict_out, tuple):
                out = tuple(
                    item
                    for i, item in enumerate(tensordict_out)
                    if module._out_keys[i] in module.out_keys
                )
                if len(out) == 1:
                    return out[0]
                return out
            elif module._out_keys[0] in module.out_keys and len(module._out_keys) == 1:
                return tensordict_out
            elif (
                module._out_keys[0] not in module.out_keys
                and len(module._out_keys) == 1
            ):
                return ()
            else:
                raise RuntimeError(
                    f"Selecting out-keys failed. Original out_keys: {module._out_keys}, selected: {module.out_keys}."
                )
        return tensordict_out.select(
            *in_keys, *out_keys, inplace=True, strict=tensordict_out is tensordict_in
        )

    def _detect_dispatch(self, tensordict_in, kwargs, in_keys):  # noqa: F811
        if isinstance(tensordict_in, TensorDictBase) and all(
            key in tensordict_in.keys(include_nested=True) for key in in_keys
        ):
            return False
        elif isinstance(tensordict_in, tuple):
            if len(tensordict_in) or len(kwargs):
                if len(tensordict_in) and isinstance(tensordict_in[0], TensorDictBase):
                    return self._detect_dispatch(tensordict_in[0], kwargs, in_keys)
                elif (
                    not len(tensordict_in)
                    and len(kwargs)
                    and isinstance(kwargs.get("tensordict"), TensorDictBase)
                ):
                    return self._detect_dispatch(kwargs["tensordict"], in_keys)
                return True
            return not len(in_keys)
        # not a TDBase: must be True
        return True

    def remove(self):
        # reset ground truth
        if self.module is None:
            return
        if self.module._out_keys is not None:
            self.module.out_keys = self.module._out_keys

    def __del__(self):
        self.remove()


class TensorDictModuleBase(nn.Module):
    """Base class to TensorDict modules.

    TensorDictModule subclasses are characterized by ``in_keys`` and ``out_keys``
    key-lists that indicate what input entries are to be read and what output
    entries should be expected to be written.

    The forward method input/output signature should always follow the
    convention:

        >>> tensordict_out = module.forward(tensordict_in)

    Unlike :class:`~tensordict.nn.TensorDictModule`, `TensorDictModuleBase` is typically used via subclassing:
    you can wrap any python function in a `TensorDictModuleBase` subclass, as long as the subclass forward reads and
    writes tensordict (or related types) instances.

    The `in_keys` and `out_keys` should be properly specified. For example, `out_keys` can be dynamically reduced using
    :meth:`~tensordict.nn.TensorDictBase.select_out_keys`.

    Examples:
        >>> from tensordict import TensorDict
        >>> from tensordict.nn import TensorDictModuleBase
        >>> class Mod(TensorDictModuleBase):
        ...     in_keys = ["a"] # can also be specified during __init__
        ...     out_keys = ["b", "c"]
        ...     def forward(self, tensordict):
        ...         b = tensordict["a"].clone()
        ...         c = b + 1
        ...         return tensordict.replace({"b": b, "c": c})
        >>> mod = Mod()
        >>> td = mod(TensorDict(a=0))
        >>> td["b"]
        tensor(0)
        >>> td["c"]
        tensor(1)
        >>> mod.select_out_keys("c")
        >>> td = mod(TensorDict(a=0))
        >>> td["c"]
        tensor(1)
        >>> assert "b" not in td

    """

    def __new__(cls, *args, **kwargs):
        # check the out_keys and in_keys in the dict
        if "in_keys" in cls.__dict__ and not isinstance(
            cls.__dict__.get("in_keys"), property
        ):
            in_keys = cls.__dict__.get("in_keys")
            # now let's remove it
            delattr(cls, "in_keys")
            cls._in_keys = unravel_key_list(in_keys)
            cls.in_keys = TensorDictModuleBase.in_keys
        if "out_keys" in cls.__dict__ and not isinstance(
            cls.__dict__.get("out_keys"), property
        ):
            out_keys = cls.__dict__.get("out_keys")
            # now let's remove it
            delattr(cls, "out_keys")
            out_keys = unravel_key_list(out_keys)
            cls._out_keys = out_keys
            cls._out_keys_apparent = out_keys
            cls.out_keys = TensorDictModuleBase.out_keys
        out = super().__new__(cls)
        return out

    @staticmethod
    def is_tdmodule_compatible(module):
        """Checks if a module is compatible with TensorDictModule API."""
        return hasattr(module, "in_keys") and hasattr(module, "out_keys")

    @property
    def in_keys(self):
        return self._in_keys

    @in_keys.setter
    def in_keys(self, value: List[Union[str, Tuple[str]]]):
        self._in_keys = unravel_key_list(value)

    @property
    def out_keys(self):
        return self._out_keys_apparent

    @property
    def out_keys_source(self):
        return self._out_keys

    @out_keys.setter
    def out_keys(self, value: List[Union[str, Tuple[str]]]):
        # the first time out_keys are set, they are marked as ground truth
        value = unravel_key_list(list(value))
        if not hasattr(self, "_out_keys"):
            self._out_keys = value
        self._out_keys_apparent = value

    def select_out_keys(self, *out_keys) -> TensorDictModuleBase:  # noqa: F811
        """Selects the keys that will be found in the output tensordict.

        This is useful whenever one wants to get rid of intermediate keys in a
        complicated graph, or when the presence of these keys may trigger unexpected
        behaviours.

        The original ``out_keys`` can still be accessed via ``module.out_keys_source``.

        Args:
            *out_keys (a sequence of strings or tuples of strings): the
                out_keys that should be found in the output tensordict.

        Returns: the same module, modified in-place with updated ``out_keys``.

        The simplest usage is with :class:`~.TensorDictModule`:

        Examples:
            >>> from tensordict import TensorDict
            >>> from tensordict.nn import TensorDictModule, TensorDictSequential
            >>> import torch
            >>> mod = TensorDictModule(lambda x, y: (x+2, y+2), in_keys=["a", "b"], out_keys=["c", "d"])
            >>> td = TensorDict({"a": torch.zeros(()), "b": torch.ones(())}, [])
            >>> mod(td)
            TensorDict(
                fields={
                    a: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
                    b: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
                    c: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
                    d: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False)},
                batch_size=torch.Size([]),
                device=None,
                is_shared=False)
            >>> mod.select_out_keys("d")
            >>> td = TensorDict({"a": torch.zeros(()), "b": torch.ones(())}, [])
            >>> mod(td)
            TensorDict(
                fields={
                    a: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
                    b: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
                    d: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False)},
                batch_size=torch.Size([]),
                device=None,
                is_shared=False)

        This feature will also work with dispatched arguments:
        Examples:
            >>> mod(torch.zeros(()), torch.ones(()))
            tensor(2.)

        This change will occur in-place (ie the same module will be returned
        with an updated list of out_keys). It can be reverted using the
        :meth:`TensorDictModuleBase.reset_out_keys` method.

        Examples:
            >>> mod.reset_out_keys()
            >>> mod(TensorDict({"a": torch.zeros(()), "b": torch.ones(())}, []))
            TensorDict(
                fields={
                    a: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
                    b: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
                    c: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
                    d: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False)},
                batch_size=torch.Size([]),
                device=None,
                is_shared=False)

        This will work with other classes too, such as Sequential:
        Examples:
            >>> from tensordict.nn import TensorDictSequential
            >>> seq = TensorDictSequential(
            ...     TensorDictModule(lambda x: x+1, in_keys=["x"], out_keys=["y"]),
            ...     TensorDictModule(lambda x: x+1, in_keys=["y"], out_keys=["z"]),
            ... )
            >>> td = TensorDict({"x": torch.zeros(())}, [])
            >>> seq(td)
            TensorDict(
                fields={
                    x: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
                    y: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
                    z: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False)},
                batch_size=torch.Size([]),
                device=None,
                is_shared=False)
            >>> seq.select_out_keys("z")
            >>> td = TensorDict({"x": torch.zeros(())}, [])
            >>> seq(td)
            TensorDict(
                fields={
                    x: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
                    z: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False)},
                batch_size=torch.Size([]),
                device=None,
                is_shared=False)

        """
        out_keys = unravel_key_list(list(out_keys))
        if len(out_keys) == 1:
            if out_keys[0] not in self.out_keys:
                err_msg = f"Can't select non existent key: {out_keys[0]}. "
                if (
                    out_keys[0]
                    and isinstance(out_keys[0], (tuple, list))
                    and out_keys[0][0] in self.out_keys
                ):
                    err_msg += f"Are you passing the keys in a list? Try unpacking as: `{', '.join(out_keys[0])}`"
                raise ValueError(err_msg)
        self.register_forward_hook(_OutKeysSelect(out_keys), with_kwargs=True)
        for hook in self._forward_hooks.values():
            if isinstance(hook, _OutKeysSelect):
                hook._init(self)
        return self

    def reset_out_keys(self):
        """Resets the ``out_keys`` attribute to its orignal value.

        Returns: the same module, with its original ``out_keys`` values.

        Examples:
            >>> from tensordict import TensorDict
            >>> from tensordict.nn import TensorDictModule, TensorDictSequential
            >>> import torch
            >>> mod = TensorDictModule(lambda x, y: (x+2, y+2), in_keys=["a", "b"], out_keys=["c", "d"])
            >>> mod.select_out_keys("d")
            >>> td = TensorDict({"a": torch.zeros(()), "b": torch.ones(())}, [])
            >>> mod(td)
            TensorDict(
                fields={
                    a: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
                    b: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
                    d: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False)},
                batch_size=torch.Size([]),
                device=None,
                is_shared=False)
            >>> mod.reset_out_keys()
            >>> mod(td)
            TensorDict(
                fields={
                    a: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
                    b: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
                    c: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
                    d: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False)},
                batch_size=torch.Size([]),
                device=None,
                is_shared=False)
        """
        for i, hook in list(self._forward_hooks.items()):
            if isinstance(hook, _OutKeysSelect):
                hook.remove()
                del self._forward_hooks[i]
        return self

    def reset_parameters_recursive(
        self, parameters: Optional[TensorDictBase] = None
    ) -> Optional[TensorDictBase]:
        """Recursively reset the parameters of the module and its children.

        Args:
            parameters (TensorDict of parameters, optional): If set to None, the module will reset using self.parameters().
                Otherwise, we will reset the parameters in the tensordict in-place. This is
                useful for functional modules where the parameters are not stored in the module itself.

        Returns:
            A tensordict of the new parameters, only if parameters was not None.

        Examples:
            >>> from tensordict.nn import TensorDictModule
            >>> from torch import nn
            >>> net = nn.Sequential(nn.Linear(2,3), nn.ReLU())
            >>> old_param = net[0].weight.clone()
            >>> module = TensorDictModule(net, in_keys=['bork'], out_keys=['dork'])
            >>> module.reset_parameters()
            >>> (old_param == net[0].weight).any()
            tensor(False)

        This method also supports functional parameter sampling:

            >>> from tensordict import TensorDict
            >>> from tensordict.nn import TensorDictModule
            >>> from torch import nn
            >>> net = nn.Sequential(nn.Linear(2,3), nn.ReLU())
            >>> module = TensorDictModule(net, in_keys=['bork'], out_keys=['dork'])
            >>> params = TensorDict.from_module(module)
            >>> old_params = params.clone(recurse=True)
            >>> module.reset_parameters(params)
            >>> (old_params == params).any()
            False
        """
        if parameters is None:
            any_reset = self._reset_parameters(self)
            if not any_reset:
                warnings.warn(
                    "reset_parameters_recursive was called without the parameters argument and did not find any parameters to reset"
                )
            return
        elif parameters.ndim:
            raise RuntimeError(
                "reset_parameters_recursive does not support batched TensorDicts, ensure `batch_size` is empty and the parameters shape match their original shape."
            )

        sanitized_parameters = parameters.apply(
            lambda x: x.detach().requires_grad_(), inplace=False
        )

        with sanitized_parameters.to_module(self):
            self._reset_parameters(self)
        return sanitized_parameters

    def _reset_parameters(self, module: nn.Module) -> bool:
        any_reset = False
        for child in module.children():
            if isinstance(child, nn.Module):
                any_reset |= self._reset_parameters(child)

            if hasattr(child, "reset_parameters"):
                child.reset_parameters()
                any_reset |= True
        return any_reset

    @property
    def __name__(self):
        # This is necessary to make compiled vmap over TDModule happy
        return self.__class__.__name__

    def __repr__(self):
        return f"{self.__class__.__name__}()"


class TensorDictModule(TensorDictModuleBase):
    """A TensorDictModule, is a python wrapper around a :obj:`nn.Module` that reads and writes to a TensorDict.

    Args:
        module (Callable[[Any], Any]): a callable, typically a :class:`torch.nn.Module`,
            used to map the input to the output parameter space. Its forward method
            can return a single tensor, a tuple of tensors or even a dictionary.
            In the latter case, the output keys of the :class:`TensorDictModule`
            will be used to populate the output tensordict (ie. the keys present
            in ``out_keys`` should be present in the dictionary returned by the
            ``module`` forward method).
        in_keys (iterable of NestedKeys, Dict[NestedStr, str]): keys to be read
            from input tensordict and passed to the module. If it
            contains more than one element, the values will be passed in the
            order given by the in_keys iterable.
            If ``in_keys`` is a dictionary, its keys must correspond to the key
            to be read in the tensordict and its values must match the name of
            the keyword argument in the function signature. If `out_to_in_map` is ``True``,
            the mapping gets inverted so that the keys correspond to the keyword
            arguments in the function signature.
        out_keys (iterable of str): keys to be written to the input tensordict. The length of out_keys must match the
            number of tensors returned by the embedded module. Using "_" as a key avoid writing tensor to output.

    Keyword Args:
        out_to_in_map (bool, optional): if ``True`` (default), `in_keys` is read as if the keys are the arguments keys of
            the :meth:`~.forward` method and the values are the keys in the input :class:`~tensordict.TensorDict`. If
            ``False``, keys are considered to be the input keys and values the method's arguments keys.
        inplace (bool or string, optional): if ``True`` (default), the output of the module are written in the tensordict
            provided to the :meth:`~.forward` method. If ``False``, a new :class:`~tensordict.TensorDict` with and empty
            batch-size and no device is created. if ``"empty"``, :meth:`~tensordict.TensorDict.empty` will be used to
            create the output tensordict.

            .. note::
                If ``inplace=False`` and the tensordict passed to the module is another
                :class:`~tensordict.TensorDictBase` subclass than :class:`~tensordict.TensorDict`, the output will still
                be a :class:`~tensordict.TensorDict` instance. Its batch-size will be empty, and it will have no device.
                Set to ``"empty"`` to get the same :class:`~tensordict.TensorDictBase` subtype, an identical batch-size
                and device. Use ``tensordict_out`` at runtime (see below) to have a more fine-grained control over the
                output.

            .. note::
                If ``inplace=False`` and a `tensordict_out` is passed to the :meth:`~.forward` method,
                the ``tensordict_out`` will prevail. This is the way one can get a tensordict_out taensordict passed to the module is another
                :class:`~tensordict.TensorDictBase` subclass than :class:`~tensordict.TensorDict`, the output will still
                be a :class:`~tensordict.TensorDict` instance.

        method (str, optional): the method to be called in the module, if any. Defaults to `__call__`.
        method_kwargs (Dict[str, Any], optional): additional keyword arguments to be passed to the module's method being called.
        strict (bool, optional): if ``True``, the module will raise an exception if any of the inputs is missing from
            the input tensordict. Otherwise, a `None` value will be used as placeholder. Defaults to ``False``.
        get_kwargs (dict[str, Any], optional): additional keyword arguments to be passed to the :meth:`~tensordict.TensorDictBase.get`
            method. This is particularily useful when dealing with ragged tensors (see :meth:`~tensordict.LazyStackedTensorDict.get`).
            Defaults to ``{}``.

    Embedding a neural network in a TensorDictModule only requires to specify the input
    and output keys. TensorDictModule support functional and regular :obj:`nn.Module`
    objects. In the functional case, the 'params' (and 'buffers') keyword argument must
    be specified:

    Examples:
        >>> from tensordict import TensorDict
        >>> # one can wrap regular nn.Module
        >>> module = TensorDictModule(nn.Transformer(128), in_keys=["input", "tgt"], out_keys=["out"])
        >>> input = torch.ones(2, 3, 128)
        >>> tgt = torch.zeros(2, 3, 128)
        >>> data = TensorDict({"input": input, "tgt": tgt}, batch_size=[2, 3])
        >>> data = module(data)
        >>> print(data)
        TensorDict(
            fields={
                input: Tensor(shape=torch.Size([2, 3, 128]), device=cpu, dtype=torch.float32, is_shared=False),
                out: Tensor(shape=torch.Size([2, 3, 128]), device=cpu, dtype=torch.float32, is_shared=False),
                tgt: Tensor(shape=torch.Size([2, 3, 128]), device=cpu, dtype=torch.float32, is_shared=False)},
            batch_size=torch.Size([2, 3]),
            device=None,
            is_shared=False)

    We can also pass directly the tensors

    Examples:
        >>> out = module(input, tgt)
        >>> assert out.shape == input.shape
        >>> # we can also wrap regular functions
        >>> module = TensorDictModule(lambda x: (x-1, x+1), in_keys=[("input", "x")], out_keys=[("output", "x-1"), ("output", "x+1")])
        >>> module(TensorDict({("input", "x"): torch.zeros(())}, batch_size=[]))
        TensorDict(
            fields={
                input: TensorDict(
                    fields={
                        x: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False)},
                    batch_size=torch.Size([]),
                    device=None,
                    is_shared=False),
                output: TensorDict(
                    fields={
                        x+1: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
                        x-1: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False)},
                    batch_size=torch.Size([]),
                    device=None,
                    is_shared=False)},
            batch_size=torch.Size([]),
            device=None,
            is_shared=False)

    We can use TensorDictModule to populate a tensordict:

    Examples:
        >>> module = TensorDictModule(lambda: torch.randn(3), in_keys=[], out_keys=["x"])
        >>> print(module(TensorDict({}, batch_size=[])))
        TensorDict(
            fields={
                x: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False)},
            batch_size=torch.Size([]),
            device=None,
            is_shared=False)

    Another feature is passing a dictionary as input keys, to control the
    dispatching of values to specific keyword arguments.

    Examples:
        >>> module = TensorDictModule(lambda x, *, y: x+y,
        ...     in_keys={'1': 'x', '2': 'y'}, out_keys=['z'], out_to_in_map=False
        ...     )
        >>> td = module(TensorDict({'1': torch.ones(()), '2': torch.ones(())*2}, []))
        >>> td['z']
        tensor(3.)

    If `out_to_in_map` is set to ``True``, then the `in_keys` mapping is reversed. This way,
    one can use the same input key for different keyword arguments.

    Examples:
        >>> module = TensorDictModule(lambda x, *, y, z: x+y+z,
        ...     in_keys={'x': '1', 'y': '2', z: '2'}, out_keys=['t'], out_to_in_map=True
        ...     )
        >>> td = module(TensorDict({'1': torch.ones(()), '2': torch.ones(())*2}, []))
        >>> td['t']
        tensor(5.)

    We can specify the method to be called within a module. Compared to using a lambda function or similar around the
    module's method, this has the advantage that the module attributes (params, buffers, submodules) will be exposed.

    Examples:
        >>> from tensordict import TensorDict
        >>> from tensordict.nn import TensorDictSequential as Seq, TensorDictModule as Mod
        >>> from torch import nn
        >>> import torch
        >>>
        >>> class MyNet(nn.Module):
        ...     def my_func(self, tensor: torch.Tensor, *, an_integer: int):
        ...         return tensor + an_integer
        ...
        >>> s = Seq(
        ...     {
        ...         "a": lambda td: td+1,
        ...         "b": lambda td: td * 2,
        ...         "c": Mod(MyNet(), in_keys=["a"], out_keys=["b"], method="my_func", method_kwargs={"an_integer": 2}),
        ...     }
        ... )
        >>> td = s(TensorDict(a=0))
        >>> print(td)
        >>>
        >>> assert td["b"] == 4

    Functional calls to a tensordict module is easy:

    Examples:
        >>> import torch
        >>> from tensordict import TensorDict
        >>> from tensordict.nn import TensorDictModule
        >>> td = TensorDict({"input": torch.randn(3, 4), "hidden": torch.randn(3, 8)}, [3,])
        >>> module = torch.nn.GRUCell(4, 8)
        >>> td_module = TensorDictModule(
        ...    module=module, in_keys=["input", "hidden"], out_keys=["output"]
        ... )
        >>> params = TensorDict.from_module(td_module)
        >>> # functional API
        >>> with params.to_module(td_module):
        ...     td_functional = td_module(td.clone())
        >>> print(td_functional)
        TensorDict(
            fields={
                hidden: Tensor(shape=torch.Size([3, 8]), device=cpu, dtype=torch.float32, is_shared=False),
                input: Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.float32, is_shared=False),
                output: Tensor(shape=torch.Size([3, 8]), device=cpu, dtype=torch.float32, is_shared=False)},
            batch_size=torch.Size([3]),
            device=None,
            is_shared=False)

    In the stateful case:
        >>> module = torch.nn.GRUCell(4, 8)
        >>> td_module = TensorDictModule(
        ...    module=module, in_keys=["input", "hidden"], out_keys=["output"]
        ... )
        >>> td_stateful = td_module(td.clone())
        >>> print(td_stateful)
        TensorDict(
            fields={
                hidden: Tensor(shape=torch.Size([3, 8]), device=cpu, dtype=torch.float32, is_shared=False),
                input: Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.float32, is_shared=False),
                output: Tensor(shape=torch.Size([3, 8]), device=cpu, dtype=torch.float32, is_shared=False)},
            batch_size=torch.Size([3]),
            device=None,
            is_shared=False)

    """

    _IN_KEY_ERR = "in_keys must be of type list, str or tuples of str, or dict."
    _OUT_KEY_ERR = "out_keys must be of type list, str or tuples of str."

    def __init__(
        self,
        module: Callable,
        in_keys: NestedKey | List[NestedKey] | Dict[NestedKey:str],
        out_keys: NestedKey | List[NestedKey],
        *,
        out_to_in_map: bool | None = None,
        inplace: bool | str = True,
        method: str | None = None,
        method_kwargs: dict | None = None,
        strict: bool = False,
        get_kwargs: dict | None = None,
    ) -> None:
        super().__init__()

        if out_to_in_map is not None and not isinstance(in_keys, dict):
            warnings.warn(
                "out_to_in_map is not None but is only used when in_key is a dictionary."
            )

        if isinstance(in_keys, dict):
            if out_to_in_map is None:
                out_to_in_map = True

            # write the kwargs and create a list instead
            _in_keys = []
            self._kwargs = []
            for key, value in in_keys.items():
                if out_to_in_map:  # arg: td_key
                    self._kwargs.append(key)
                    _in_keys.append(value)
                else:  # td_key: arg
                    self._kwargs.append(value)
                    _in_keys.append(key)
            in_keys = _in_keys
        else:
            if isinstance(in_keys, (str, tuple)):
                in_keys = [in_keys]
            elif not isinstance(in_keys, MutableSequence):
                raise ValueError(self._IN_KEY_ERR)
            self._kwargs = None

        if isinstance(out_keys, (str, tuple)):
            out_keys = [out_keys]
        elif not isinstance(out_keys, MutableSequence):
            raise ValueError(self._OUT_KEY_ERR)
        try:
            in_keys = unravel_key_list(list(in_keys))
        except Exception:
            raise ValueError(self._IN_KEY_ERR)
        try:
            out_keys = unravel_key_list(list(out_keys))
        except Exception:
            raise ValueError(self._OUT_KEY_ERR)

        if type(module) is type or (method is None and not callable(module)):
            raise ValueError(
                f"Module {module} if type {type(module)} is not callable. "
                f"Typical accepted types are nn.Module or TensorDictModule. "
                f"If you need to call a specific method from your module, pass the "
                f"`method` keyword argument to the TensorDictModule constructor."
            )
        self.out_keys = out_keys
        self.in_keys = in_keys

        self.strict = strict

        if "_" in self.in_keys:
            warnings.warn(
                'key "_" is for ignoring output, it should not be used in input keys',
                stacklevel=2,
            )

        self.module = module
        if inplace not in (None, True, False, "empty"):
            raise ValueError(
                f"The only accepted valued for inplace is `None`, `True`, `False`, or `'empty'`. Got inplace={inplace} "
                "instead."
            )
        self.inplace = inplace
        self.method = method
        self.method_kwargs = method_kwargs if method_kwargs is not None else {}
        self._get_kwargs = get_kwargs if get_kwargs is not None else {}

    @property
    def is_functional(self) -> bool:
        return _has_functorch and isinstance(
            self.module,
            (FunctionalModule, FunctionalModuleWithBuffers),
        )

    def _write_to_tensordict(
        self,
        tensordict: TensorDictBase,
        tensors: list[Tensor],
        tensordict_out: TensorDictBase | None = None,
        out_keys: Iterable[NestedKey] | None = None,
    ) -> TensorDictBase:
        if out_keys is None:
            out_keys = self.out_keys_source
        if tensordict_out is None:
            if self.inplace is not True:
                if self.inplace == "empty":
                    tensordict_out = tensordict.empty()
                else:
                    tensordict_out = TensorDict()
            else:
                tensordict_out = tensordict
        if len(tensors) > len(out_keys):
            raise RuntimeError(
                f"There are more tensors ({len(tensors)=}) than out_keys ({out_keys=})."
            )
        elif len(out_keys) > len(tensors):
            raise RuntimeError("There are more out_keys than tensors.")
        for _out_key, _tensor in zip(out_keys, tensors):
            if _out_key != "_":
                tensordict_out.set(_out_key, TensorDict.from_any(_tensor))
        return tensordict_out

    def _call_module(
        self, tensors: Sequence[Tensor], **kwargs: Any
    ) -> Tensor | Sequence[Tensor]:
        kwargs.update(self.method_kwargs)
        if self.method is None:
            out = self.module(*tensors, **kwargs)
        else:
            out = getattr(self.module, self.method)(*tensors, **kwargs)
        return out

    @dispatch(auto_batch_size=False)
    @_set_skip_existing_None()
    def forward(
        self,
        tensordict: TensorDictBase,
        *args,
        tensordict_out: TensorDictBase | None = None,
        **kwargs: Any,
    ) -> TensorDictBase:
        """When the tensordict parameter is not set, kwargs are used to create an instance of TensorDict."""
        try:
            if len(args):
                raise ValueError(
                    "Got a non-empty list of extra agruments, when none was expected."
                )
            default = None if not self.strict else NO_DEFAULT
            if self._kwargs is not None:
                kwargs.update(
                    {
                        kwarg: tensordict._get_tuple_maybe_non_tensor(
                            _unravel_key_to_tuple(in_key), default=default
                        )
                        for kwarg, in_key in _zip_strict(self._kwargs, self.in_keys)
                    }
                )
                tensors = ()
            else:
                tensors = tuple(
                    tensordict._get_tuple_maybe_non_tensor(
                        _unravel_key_to_tuple(in_key),
                        default,
                        **self._get_kwargs,
                    )
                    for in_key in self.in_keys
                )
            try:
                tensors_out = self._call_module(tensors, **kwargs)
                if tensors_out is None:
                    tensors_out = ()
            except Exception as err:
                if any(tensor is None for tensor in tensors) and "None" in str(err):
                    none_set = {
                        key
                        for key, tensor in _zip_strict(self.in_keys, tensors)
                        if tensor is None
                    }
                    raise KeyError(
                        "Some tensors that are necessary for the module call may "
                        "not have not been found in the input tensordict: "
                        f"the following inputs are None: {none_set}."
                    ) from err
                else:
                    raise err
            if isinstance(tensors_out, (dict, TensorDictBase)) and all(
                key in tensors_out for key in self.out_keys
            ):
                if isinstance(tensors_out, dict):
                    keys = unravel_key_list(list(tensors_out.keys()))
                    values = tensors_out.values()
                    tensors_out = dict(_zip_strict(keys, values))
                tensors_out = tuple(tensors_out.get(key) for key in self.out_keys)
            if not isinstance(tensors_out, tuple):
                tensors_out = (tensors_out,)
            tensordict_out = self._write_to_tensordict(
                tensordict, tensors_out, tensordict_out
            )
            return tensordict_out
        except Exception as err:
            module = self.module
            if not isinstance(module, nn.Module):
                try:
                    import inspect

                    module = inspect.getsource(module)
                except Exception:
                    # then we can't print the source code
                    pass
            module = indent(str(module), 4 * " ")
            in_keys = indent(f"in_keys={self.in_keys}", 4 * " ")
            out_keys = indent(f"out_keys={self.out_keys}", 4 * " ")
            raise err from RuntimeError(
                f"TensorDictModule failed with operation\n{module}\n{in_keys}\n{out_keys}."
            )

    @property
    def device(self) -> torch.device:
        for p in self.parameters():
            return p.device
        return torch.device("cpu")

    def __repr__(self) -> str:
        fields = indent(
            f"module={self.module},\n"
            f"device={self.device},\n"
            f"in_keys={self.in_keys},\n"
            f"out_keys={self.out_keys}",
            4 * " ",
        )

        return f"{type(self).__name__}(\n{fields})"

    def __getattr__(self, name: str) -> Any:
        if not is_compiling():
            __dict__ = self.__dict__
            _parameters = __dict__.get("_parameters")
            if _parameters:
                # A param can be None so we use False instead to check if the key
                # is in the _parameters dict once and only once.
                # We use False but any non-None, non-Parameter value would do.
                # The alternative `if name in _parameters: return _parameters[name]`
                # accesses the value of `name` twice when only one is required
                result = _parameters.get(name, False)
                if result is not False:
                    return result
            _buffers = __dict__.get("_buffers")
            if _buffers:
                result = _buffers.get(name, False)
                if result is not False:
                    return result
            _modules = __dict__.get("_modules")
            if _modules:
                result = _modules.get(name, False)
                if result is not False:
                    return result
        # TODO: find a way to make this check work with dynamo
        # elif hasattr(self, "_parameters"):
        else:
            _parameters = self._parameters
            result = _parameters.get(name, False)
            if result is not False:
                return result
            _buffers = self._buffers
            result = _buffers.get(name, False)
            if result is not False:
                return result
            _modules = self._modules
            result = _modules.get(name, False)
            if result is not False:
                return result

        if not name.startswith("_"):
            # no fallback for private attributes
            return getattr(super().__getattr__("module"), name)
        raise AttributeError(
            f"module {type(self).__name__} has no attribute named {name}."
        )

    def __getstate__(self):
        state = self.__dict__.copy()
        if not isinstance(self.module, nn.Module):
            state["module"] = cloudpickle_dumps(state["module"])
        return state

    def __setstate__(self, state):
        if "module" in state:
            state["module"] = cloudpickle_loads(state["module"])
        self.__dict__ = state


class TensorDictModuleWrapper(TensorDictModuleBase):
    """Wrapper class for TensorDictModule objects.

    Once created, a TensorDictModuleWrapper will behave exactly as the
    TensorDictModule it contains except for the methods that are
    overwritten.

    Args:
        td_module (TensorDictModule): operator to be wrapped.

    """

    def __init__(self, td_module: TensorDictModuleBase) -> None:
        super().__init__()
        self.td_module = td_module
        if len(self.td_module._forward_hooks):
            for pre_hook in self.td_module._forward_hooks:
                self.register_forward_hook(self.td_module._forward_hooks[pre_hook])

    def __getattr__(self, name: str) -> Any:
        if not is_compiling():
            __dict__ = self.__dict__
            _parameters = __dict__.get("_parameters")
            if _parameters:
                # A param can be None so we use False instead to check if the key
                # is in the _parameters dict once and only once.
                # We use False but any non-None, non-Parameter value would do.
                # The alternative `if name in _parameters: return _parameters[name]`
                # accesses the value of `name` twice when only one is required
                result = _parameters.get(name, False)
                if result is not False:
                    return result
            _buffers = __dict__.get("_buffers")
            if _buffers:
                result = _buffers.get(name, False)
                if result is not False:
                    return result
            _modules = __dict__.get("_modules")
            if _modules:
                result = _modules.get(name, False)
                if result is not False:
                    return result
        # TODO: find a way to make this check work with dynamo
        # elif hasattr(self, "_parameters"):
        else:
            _parameters = self._parameters
            result = _parameters.get(name, False)
            if result is not False:
                return result
            _buffers = self._buffers
            result = _buffers.get(name, False)
            if result is not False:
                return result
            _modules = self._modules
            result = _modules.get(name, False)
            if result is not False:
                return result
        if name not in self.__dict__ and not name.startswith("__"):
            return getattr(self._modules["td_module"], name)
        raise AttributeError(
            f"attribute {name} not recognised in {type(self).__name__}"
        )

    def forward(self, *args: Any, **kwargs: Any) -> TensorDictBase:
        return self.td_module.forward(*args, **kwargs)


class WrapModule(TensorDictModuleBase):
    """A wrapper around any callable that processes TensorDict instances.

    This wrapper is useful when building :class:`~tensordict.nn.TensorDictSequential` stacks and when a transform
    requires the entire TensorDict instance to be visible.

    Args:
        func (Callable[[TensorDictBase], TensorDictBase]): A callable function that takes in a TensorDictBase instance
            and returns a transformed TensorDictBase instance.

    Keyword Args:
        inplace (bool, optional): If ``True``, the input TensorDict will be modified in-place. Otherwise, a new TensorDict
            will be returned (if the function does not modify it in-place and returns it). Defaults to ``False``.
        in_keys (list of NestedKey, optional): if provided, indicates what entries are read by the module.
            This will not be checked and is provided just for the purpose of informing :class:`~tensordict.nn.TensorDictSequential`
            about the input keys of the wrapped module. Defaults to `[]`.
        out_keys (list of NestedKey, optional): if provided, indicates what entries are written by the module.
            This will not be checked and is provided just for the purpose of informing :class:`~tensordict.nn.TensorDictSequential`
            about the output keys of the wrapped module. Defaults to `[]`.

    Examples:
        >>> from tensordict.nn import TensorDictSequential as Seq, TensorDictModule as Mod, WrapModule
        >>> seq = Seq(
        ...     Mod(lambda x: x * 2, in_keys=["x"], out_keys=["y"]),
        ...     WrapModule(lambda td: td.reshape(-1)),
        ... )
        >>> td = TensorDict(x=torch.ones(3, 4, 5), batch_size=[3, 4])
        >>> td = Seq(td)
        >>> assert td.shape == (12,)
        >>> assert (td["y"] == 2).all()
        >>> assert td["y"].shape == (12, 5)

    """

    in_keys = []
    out_keys = []

    def __init__(
        self,
        func: Callable[[TensorDictBase], TensorDictBase],
        *,
        inplace: bool = False,
        in_keys: List[NestedKey] | None = None,
        out_keys: List[NestedKey] | None = None,
    ) -> None:
        super().__init__()
        self.func = func
        self.inplace = inplace
        if in_keys is not None:
            self.in_keys = in_keys
        if out_keys is not None:
            self.out_keys = out_keys

    def forward(self, data: TensorDictBase) -> TensorDictBase:
        result = self.func(data)
        if self.inplace and result is not data:
            return data.update(result)
        return result


class as_tensordict_module:
    """A decorator that converts a function into a TensorDictModule.

    Args:
        in_keys (List[NestedKey] | NestedKey | None, optional): The input keys of the resulting TensorDictModule.
        out_keys (List[NestedKey] | NestedKey | None, optional): The output keys of the resulting TensorDictModule.

    Returns:
        Callable: A decorator that can be applied to a function to convert it into a TensorDictModule.

    Examples:
        >>> class MyClass:
        ...     @as_tensordict_module(in_keys="c", out_keys="d")
        ...     def my_method(self, c):
        ...         return c + 1
        >>> obj = MyClass()
        >>> result = obj.my_method(TensorDict(c=0))
        >>> print(result["d"])  # prints: 1
        >>> @as_tensordict_module(in_keys="c", out_keys="d")
        ... def my_function(c):
        ...     return c + 1
        >>> result = my_function(TensorDict(c=0))
        >>> print(result["d"])  # prints: 1
    """

    def __init__(
        self,
        *,
        in_keys: List[NestedKey] | NestedKey,
        out_keys: List[NestedKey] | NestedKey,
    ) -> None:
        if isinstance(in_keys, NestedKey):
            in_keys = [in_keys]
        if isinstance(out_keys, NestedKey):
            out_keys = [out_keys]
        self.in_keys = in_keys
        self.out_keys = out_keys

    def __call__(self, func):
        tdmodule = None
        is_bound_method = inspect.ismethod(func) or (
            inspect.isfunction(func) and "self" in inspect.signature(func).parameters
        )

        if is_bound_method:

            @wraps(func)
            def wrapped(_self, *args, **kwargs):
                nonlocal tdmodule
                if tdmodule is None:

                    def newfunc(*args, **kwargs):
                        return func(_self, *args, **kwargs)

                    tdmodule = TensorDictModule(
                        newfunc, in_keys=self.in_keys, out_keys=self.out_keys
                    )
                return tdmodule(*args, **kwargs)

        else:

            @wraps(func)
            def wrapped(*args, **kwargs):
                nonlocal tdmodule
                if tdmodule is None:
                    tdmodule = TensorDictModule(
                        func, in_keys=self.in_keys, out_keys=self.out_keys
                    )
                return tdmodule(*args, **kwargs)

        return wrapped
