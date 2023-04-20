# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import functools
import inspect
import warnings
from textwrap import indent
from typing import Any, Callable, Iterable, List, Sequence, Tuple, Union

import torch
from tensordict.nn.functional_modules import make_functional

from tensordict.nn.utils import set_skip_existing
from tensordict.tensordict import make_tensordict, TensorDictBase
from tensordict.utils import _normalize_key, _seq_of_nested_key_check, NestedKey
from torch import nn, Tensor

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
        # sanity check
        for i, key in enumerate(inspect.signature(func).parameters):
            if i == 0:
                # skip self
                continue
            if key != "tensordict":
                raise RuntimeError(
                    "the first argument of the wrapped function must be "
                    "named 'tensordict'."
                )
            break

        @functools.wraps(func)
        def wrapper(_self, *args: Any, **kwargs: Any) -> Any:
            from tensordict.tensorclass import is_tensorclass

            source = self.source
            if isinstance(source, str):
                source = getattr(_self, source)
            tensordict = None
            if len(args):
                if isinstance(args[0], TensorDictBase) or is_tensorclass(args[0]):
                    tensordict, args = args[0], args[1:]
            if tensordict is None:
                tensordict_values = {}
                dest = self.dest
                if isinstance(dest, str):
                    dest = getattr(_self, dest)
                for key in source:
                    expected_key = (
                        self.separator.join(key) if isinstance(key, tuple) else key
                    )
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
                tensordict = make_tensordict(
                    tensordict_values,
                    batch_size=torch.Size([]) if not self.auto_batch_size else None,
                )
                out = func(_self, tensordict, *args, **kwargs)
                out = tuple(out[key] for key in dest)
                return out[0] if len(out) == 1 else out
            return func(_self, tensordict, *args, **kwargs)

        return wrapper


class _OutKeysSelect:
    def __init__(self, out_keys):
        self.out_keys = out_keys
        self._initialized = False

    def _init(self, module):
        if self._initialized:
            return
        self._initialized = True
        self.module = module
        module.out_keys = self.out_keys

    def __call__(
        self,
        module: TensorDictModuleBase,
        tensordict_in: TensorDictBase,
        tensordict_out: TensorDictBase,
    ):
        # detect dispatch calls
        in_keys = module.in_keys
        is_dispatched = self._detech_dispatch(tensordict_in, in_keys)
        out_keys = self.out_keys
        if is_dispatched:
            # if dispatch filtered the out keys as they should we're happy
            if (not isinstance(tensordict_out, tuple) and len(out_keys) == 1) or (
                len(out_keys) == len(tensordict_out)
            ):
                return tensordict_out
        self._init(module)
        if is_dispatched:
            # it might be the case that dispatch was not aware of what the out-keys were.
            return (
                item
                for i, item in tensordict_out
                if module._out_keys[i] in module.out_keys
            )
        return tensordict_out.select(
            *in_keys, *out_keys, inplace=tensordict_out is tensordict_in
        )

    def _detech_dispatch(self, tensordict_in, in_keys):
        if isinstance(tensordict_in, TensorDictBase) and all(
            key in tensordict_in.keys() for key in in_keys
        ):
            return False
        elif isinstance(tensordict_in, tuple):
            if len(tensordict_in):
                if isinstance(tensordict_in[0], TensorDictBase):
                    return self._detech_dispatch(tensordict_in[0], in_keys)
                return True
            return not len(in_keys)

    def remove(self):
        # reset ground truth
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

    """

    @property
    def in_keys(self):
        return self._in_keys

    @in_keys.setter
    def in_keys(self, value: List[Union[str, Tuple[str]]]):
        self._in_keys = value

    @property
    def out_keys(self):
        return self._out_keys_apparent

    @out_keys.setter
    def out_keys(self, value: List[Union[str, Tuple[str]]]):
        # the first time out_keys are set, they are marked as ground truth
        if not hasattr(self, "_out_keys"):
            self._out_keys = value
        self._out_keys_apparent = value

    def select_out_keys(self, out_keys):
        self.register_forward_hook(_OutKeysSelect(out_keys))


class TensorDictModule(TensorDictModuleBase):
    """A TensorDictModule, is a python wrapper around a :obj:`nn.Module` that reads and writes to a TensorDict.

    By default, :class:`TensorDictModule` subclasses are always functional,
    meaning that they support the ``td_module(input, params=params)`` function
    call signature.

    Args:
        module (Callable): a callable, typically a :class:`torch.nn.Module`, used to map the input to the output parameter space.
        in_keys (iterable of str): keys to be read from input tensordict and passed to the module. If it
            contains more than one element, the values will be passed in the order given by the in_keys iterable.
        out_keys (iterable of str): keys to be written to the input tensordict. The length of out_keys must match the
            number of tensors returned by the embedded module. Using "_" as a key avoid writing tensor to output.

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
        >>> # we can also pass directly the tensors
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
        >>> # we can use TensorDictModule to populate a tensordict
        >>> module = TensorDictModule(lambda: torch.randn(3), in_keys=[], out_keys=["x"])
        >>> print(module(TensorDict({}, batch_size=[])))
        TensorDict(
            fields={
                x: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False)},
            batch_size=torch.Size([]),
            device=None,
            is_shared=False)

    Functional calls to a tensordict module is easy:
    Examples:
        >>> import torch
        >>> from tensordict import TensorDict
        >>> from tensordict.nn import TensorDictModule
        >>> from tensordict.nn.functional_modules import make_functional
        >>> td = TensorDict({"input": torch.randn(3, 4), "hidden": torch.randn(3, 8)}, [3,])
        >>> module = torch.nn.GRUCell(4, 8)
        >>> td_module = TensorDictModule(
        ...    module=module, in_keys=["input", "hidden"], out_keys=["output"]
        ... )
        >>> params = make_functional(td_module)
        >>> td_functional = td_module(td.clone(), params=params)
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

    One can use a vmap operator to call the functional module.
        >>> from functorch import vmap
        >>> from tensordict.nn.functional_modules import extract_weights_and_buffers
        >>> params = extract_weights_and_buffers(td_module)
        >>> params_repeat = params.expand(4)
        >>> print(params_repeat)
        TensorDict(
            fields={
                module: TensorDict(
                    fields={
                        bias_hh: Tensor(shape=torch.Size([4, 24]), device=cpu, dtype=torch.float32, is_shared=False),
                        bias_ih: Tensor(shape=torch.Size([4, 24]), device=cpu, dtype=torch.float32, is_shared=False),
                        weight_hh: Tensor(shape=torch.Size([4, 24, 8]), device=cpu, dtype=torch.float32, is_shared=False),
                        weight_ih: Tensor(shape=torch.Size([4, 24, 4]), device=cpu, dtype=torch.float32, is_shared=False)},
                    batch_size=torch.Size([4]),
                    device=None,
                    is_shared=False)},
            batch_size=torch.Size([4]),
            device=None,
            is_shared=False)
        >>> td_vmap = vmap(td_module, (None, 0))(td.clone(), params_repeat)
        >>> print(td_vmap)
        TensorDict(
            fields={
                hidden: Tensor(shape=torch.Size([4, 3, 8]), device=cpu, dtype=torch.float32, is_shared=False),
                input: Tensor(shape=torch.Size([4, 3, 4]), device=cpu, dtype=torch.float32, is_shared=False),
                output: Tensor(shape=torch.Size([4, 3, 8]), device=cpu, dtype=torch.float32, is_shared=False)},
            batch_size=torch.Size([4, 3]),
            device=None,
            is_shared=False)

    """

    def __init__(
        self,
        module: Callable,
        in_keys: Sequence[NestedKey],
        out_keys: Sequence[NestedKey],
    ) -> None:
        super().__init__()

        _seq_of_nested_key_check(out_keys)
        _seq_of_nested_key_check(in_keys)

        if type(module) is type or not callable(module):
            raise ValueError(
                f"Module {module} if type {type(module)} is not callable. "
                f"Typical accepted types are nn.Module or TensorDictModule."
            )
        self.out_keys = [_normalize_key(key) for key in out_keys]
        self.in_keys = [_normalize_key(key) for key in in_keys]

        if "_" in in_keys:
            warnings.warn(
                'key "_" is for ignoring output, it should not be used in input keys',
                stacklevel=2,
            )

        self.module = module
        make_functional(self, keep_params=True, return_params=False)

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
            out_keys = self.out_keys
        if tensordict_out is None:
            tensordict_out = tensordict
        for _out_key, _tensor in zip(out_keys, tensors):
            if _out_key != "_":
                tensordict_out.set(_out_key, _tensor)
        return tensordict_out

    def _call_module(
        self, tensors: Sequence[Tensor], **kwargs: Any
    ) -> Tensor | Sequence[Tensor]:
        out = self.module(*tensors, **kwargs)
        return out

    @dispatch(auto_batch_size=False)
    @set_skip_existing(None)
    def forward(
        self,
        tensordict: TensorDictBase,
        *args,
        tensordict_out: TensorDictBase | None = None,
        **kwargs: Any,
    ) -> TensorDictBase:
        """When the tensordict parameter is not set, kwargs are used to create an instance of TensorDict."""
        if len(args):
            tensordict_out = args[0]
            args = args[1:]
            # we will get rid of tensordict_out as a regular arg, because it
            # blocks us when using vmap
            # with stateful but functional modules: the functional module checks if
            # it still contains parameters. If so it considers that only a "params" kwarg
            # is indicative of what the params are, when we could potentially make a
            # special rule for TensorDictModule that states that the second arg is
            # likely to be the module params.
            warnings.warn(
                "tensordict_out will be deprecated soon.", category=DeprecationWarning
            )
        if len(args):
            raise ValueError(
                "Got a non-empty list of extra agruments, when none was expected."
            )
        tensors = tuple(tensordict.get(in_key, None) for in_key in self.in_keys)
        try:
            tensors = self._call_module(tensors, **kwargs)
        except Exception as err:
            if any(tensor is None for tensor in tensors) and "None" in str(err):
                none_set = {
                    key for key, tensor in zip(self.in_keys, tensors) if tensor is None
                }
                raise KeyError(
                    "Some tensors that are necessary for the module call may "
                    "not have not been found in the input tensordict: "
                    f"the following inputs are None: {none_set}."
                ) from err
            else:
                raise err
        if not isinstance(tensors, tuple):
            tensors = (tensors,)
        tensordict_out = self._write_to_tensordict(tensordict, tensors, tensordict_out)
        return tensordict_out

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

        return f"{self.__class__.__name__}(\n{fields})"

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError as err1:
            try:
                return getattr(super().__getattr__("module"), name)
            except Exception as err2:
                raise err2 from err1


class TensorDictModuleWrapper(TensorDictModuleBase):
    """Wrapper class for TensorDictModule objects.

    Once created, a TensorDictModuleWrapper will behave exactly as the
    TensorDictModule it contains except for the methods that are
    overwritten.

    Args:
        td_module (TensorDictModule): operator to be wrapped.

    """

    def __init__(self, td_module: TensorDictModule) -> None:
        super().__init__()
        self.td_module = td_module
        if len(self.td_module._forward_hooks):
            for pre_hook in self.td_module._forward_hooks:
                self.register_forward_hook(self.td_module._forward_hooks[pre_hook])

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            if name not in self.__dict__ and not name.startswith("__"):
                return getattr(self._modules["td_module"], name)
            else:
                raise AttributeError(
                    f"attribute {name} not recognised in {type(self).__name__}"
                )

    def forward(self, *args: Any, **kwargs: Any) -> TensorDictBase:
        return self.td_module.forward(*args, **kwargs)

    @property
    def in_keys(self):
        if hasattr(self, "_in_keys"):
            return self._in_keys
        return self.td_module.in_keys

    @property
    def out_keys(self):
        if hasattr(self, "_out_keys"):
            return self._out_keys
        return self.td_module.out_keys

    @in_keys.setter
    def in_keys(self, value):
        self._in_keys = value

    @out_keys.setter
    def out_keys(self, value):
        self._out_keys = value
