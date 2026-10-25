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
from functorch import (
    FunctionalModule as _FunctionalModule,
    FunctionalModuleWithBuffers as _FunctionalModuleWithBuffers,
)
from tensordict._deprecation import deprecated, deprecated_attributes
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
from torch.compiler import is_compiling

__all__ = [
    "TensorDictModule",
    "TensorDictModuleBase",
    "TensorDictModuleWrapper",
    "WrapModule",
    "as_tensordict_module",
    "dispatch",
]


class dispatch:
    """Lets a function that takes a tensordict be called with tensors as arguments.

    Decorate a function or a method whose first argument (after ``self``) is
    named ``tensordict``. When you call it with tensors, the wrapper puts them
    in a new tensordict at the keys of ``source``, calls the function, and
    returns the entries at the keys of ``dest``: one tensor if there is one
    key, else a tuple in the order of the keys. An output key ``"_"`` is
    skipped. When you call it with a tensordict as first argument, or as the
    keyword argument ``tensordict``, the wrapper passes it on unchanged.

    Tensors are matched to the input keys in two ways:

    - Positional arguments, in the order of the keys.
    - Keyword arguments, named after the key. A nested key such as
      ``("a", "c")`` is joined with ``separator``: ``a_c=...``.

    Other keyword arguments are passed on to the function. Use the class as
    ``@dispatch`` or, with arguments, as ``@dispatch(source=..., dest=...)``.
    :class:`~tensordict.nn.TensorDictModule` and the other modules of
    ``tensordict.nn`` decorate their ``forward`` with it.

    Args:
        separator (str, optional): the string that joins the parts of a
            nested key into a keyword argument name. Defaults to ``"_"``.
        source (str or list of NestedKey, optional): the input keys. A
            string is the name of the attribute of the module that holds
            them. Defaults to ``"in_keys"``.
        dest (str or list of NestedKey, optional): the output keys. A string
            is the name of the attribute of the module that holds them.
            Defaults to ``"out_keys"``.
        auto_batch_size (bool, optional): if ``True``, the batch size of the
            tensordict built from the arguments is the largest leading shape
            that all the inputs share. If ``False``, it is empty. Defaults to
            ``True``.

    Raises:
        RuntimeError: when the function is decorated, if its first argument
            is not named ``tensordict``; when it is called, if one input is
            given both as a positional and as a keyword argument.

    .. note::
        An input key cannot hold a tensordict passed as the first positional
        argument: that tensordict is taken as the input. The keyword
        argument ``tensordict`` is taken as the input too, unless an input key
        is named ``"tensordict"``.

    .. note::
        To turn keyword calls off for every module, set the environment
        variable ``DISPATCH_TDNN_MODULES=0`` before you import tensordict.
        The decorated functions then accept a tensordict only, and the call
        does not go through the wrapper of :func:`dispatch`. See
        :ref:`compile` for how modules behave under :func:`torch.compile`.

    Examples:
        >>> import torch
        >>> from torch import nn
        >>> from tensordict.nn import dispatch
        >>> class MyModule(nn.Module):
        ...     in_keys = ["a"]
        ...     out_keys = ["b"]
        ...
        ...     @dispatch
        ...     def forward(self, tensordict):
        ...         tensordict["b"] = tensordict["a"] + 1
        ...         return tensordict
        ...
        >>> module = MyModule()
        >>> module(a=torch.zeros(2))
        tensor([1., 1.])

    The keys can also be given to the decorator, as a list or as the names
    of other attributes (``@dispatch(source="keys_in", dest="keys_out")``):

        >>> class MyModule(nn.Module):
        ...     @dispatch(source=["a"], dest=["b"])
        ...     def forward(self, tensordict):
        ...         tensordict["b"] = tensordict["a"] + 1
        ...         return tensordict
        ...
        >>> MyModule()(a=torch.zeros(2))
        tensor([1., 1.])

    A nested key is passed by joining its parts with ``separator``. Inputs
    can also be positional; one input given both ways raises an error:

        >>> class MyModuleNest(nn.Module):
        ...     in_keys = [("a", "c"), "d"]
        ...     out_keys = ["b"]
        ...
        ...     @dispatch(separator="__")
        ...     def forward(self, tensordict):
        ...         tensordict["b"] = tensordict["a", "c"] + tensordict["d"]
        ...         return tensordict
        ...
        >>> module = MyModuleNest()
        >>> module(a__c=torch.zeros(2), d=torch.ones(2))
        tensor([1., 1.])
        >>> module(torch.zeros(2), torch.ones(2))
        tensor([1., 1.])
        >>> module(torch.zeros(2), a__c=torch.ones(2))
        Traceback (most recent call last):
            ...
        RuntimeError: Duplicated argument in args and kwargs.

    """

    DEFAULT_SEPARATOR = "_"
    DEFAULT_SOURCE = "in_keys"
    DEFAULT_DEST = "out_keys"

    def __new__(
        cls,
        separator=DEFAULT_SEPARATOR,
        source: str | Sequence[NestedKey] = DEFAULT_SOURCE,
        dest: str | Sequence[NestedKey] = DEFAULT_DEST,
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
        source: str | Sequence[NestedKey] = DEFAULT_SOURCE,
        dest: str | Sequence[NestedKey] = DEFAULT_DEST,
        auto_batch_size: bool = True,
    ):
        self.separator = separator
        self.source = source
        self.dest = dest
        self.auto_batch_size = auto_batch_size

    def __call__(self, func: Callable) -> Callable:
        # A bound method already carries its instance: string ``source`` and
        # ``dest`` are read from it, and all positional arguments are inputs.
        bound_self = func.__self__ if inspect.ismethod(func) else None
        is_method = bound_self is None and (
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
            # A tensordict input is passed on as is whether or not dispatch is
            # on: test for it first, so that ``module(td)`` skips the rest
            if is_method:
                # (an unbound call with self=None raises below, as before)
                if (
                    len(args) > 1
                    and args[0] is not None
                    and is_tensor_collection(args[1])
                ):
                    # Read the source as below when dispatch is on, so that a
                    # missing attribute raises and a lazy property resolves
                    if isinstance(self.source, str) and _dispatch_td_nn_modules():
                        getattr(args[0], self.source)
                    return func(*args, **kwargs)
            elif bound_self is not None and len(args) and is_tensor_collection(args[0]):
                if isinstance(self.source, str) and _dispatch_td_nn_modules():
                    getattr(bound_self, self.source)
                return func(*args, **kwargs)
            if not _dispatch_td_nn_modules():
                return func(*args, **kwargs)
            if is_method:
                _self = args[0]
                args = args[1:]
            else:
                _self = bound_self

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
            elif (
                "tensordict" in kwargs
                and is_tensor_collection(kwargs["tensordict"])
                and all(
                    self.separator.join(_unravel_key_to_tuple(key)) != "tensordict"
                    for key in source
                )
            ):
                # ``tensordict=td`` is the positional call by keyword, unless an
                # input key is named "tensordict"
                tensordict = kwargs.pop("tensordict")
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
                if is_method:
                    out = func(_self, tensordict, *args, **kwargs)
                else:
                    out = func(tensordict, *args, **kwargs)

                # This makes dispatch responsible of handling partial outputs (such as selected through select_out_keys)
                # The out-key "_" marks an output that is not written.
                out = tuple(out[key] for key in dest if key != "_")
                return out[0] if len(out) == 1 else out

            if is_method:
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

    @deprecated("dispatch.get_source()", removal="0.17")
    def get_source(self, func, self_func):
        """Returns the input keys that ``source`` points to.

        A string ``source`` is read as an attribute of ``self_func``. ``func``
        is not used.

        .. deprecated:: 0.15
            This method will be removed in TensorDict 0.17. It has no
            replacement.
        """
        source = self.source
        if isinstance(source, str):
            return getattr(self_func, source)
        return source


class _OutKeyNotFoundError(ValueError, RuntimeError):
    """The error raised when :meth:`~TensorDictModuleBase.select_out_keys` gets a key that is not in ``out_keys``.

    It subclasses both ``ValueError`` and ``RuntimeError``, the types that
    a single missing key and several keys raised before.
    """


class _OutKeysSelect:
    module: nn.Module | None = None

    def __init__(self, out_keys):
        self.out_keys = list(out_keys)
        self._initialized = None
        self._is_dispatched = None

    def _init(self, module):
        if self._initialized:
            return
        missing = [key for key in self.out_keys if key not in module.out_keys]
        if missing:
            raise _OutKeyNotFoundError(
                f"Can't select non existent keys: {missing}. "
                f"The module out_keys are {module.out_keys}."
            )
        self.source = list(module.out_keys)
        had_out_keys_apparent = "_out_keys_apparent" in module.__dict__
        out_keys_apparent = module.__dict__.get("_out_keys_apparent")
        module._out_keys_apparent = self.out_keys
        if module.out_keys != self.out_keys:
            if had_out_keys_apparent:
                module._out_keys_apparent = out_keys_apparent
            else:
                del module._out_keys_apparent
            out_keys_property = type(module).out_keys
            if out_keys_property.fset is not None:
                module.out_keys = self.out_keys
        if module.out_keys != self.out_keys:
            if out_keys_property.fset is not None:
                module.out_keys = self.source
            raise RuntimeError(
                f"{type(module).__name__} does not support select_out_keys."
            )
        self._initialized = True
        self.module = module

    def __call__(
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
        out_keys = module.out_keys
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
        # Remove only the out-keys that are not selected: tensordict_out may be
        # the input, whose other entries must survive the call.
        excluded_keys = [
            key for key in self.source if key not in out_keys and key not in in_keys
        ]
        tensordict_out.exclude(*excluded_keys, inplace=True)
        # Drop the sub-tensordicts that only held excluded keys.
        for key in excluded_keys:
            if isinstance(key, str):
                continue
            for depth in range(len(key) - 1, 0, -1):
                parent = key[:depth] if depth > 1 else key[0]
                if parent in out_keys or parent in in_keys:
                    break
                value = tensordict_out.get(parent, None)
                if not is_tensor_collection(value) or not value.is_empty():
                    break
                del tensordict_out[parent]
        return tensordict_out

    def _detect_dispatch(self, tensordict_in, kwargs, in_keys):
        if isinstance(tensordict_in, TensorDictBase):
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
        if self.module is None:
            return
        if self.module.out_keys == self.out_keys:
            self.module.out_keys = self.source

    def __del__(self):
        self.remove()


class TensorDictModuleBase(nn.Module):
    """Base class of the tensordict modules, which read ``in_keys`` from a tensordict and write ``out_keys``.

    Subclass it to write a module whose ``forward`` takes a tensordict and
    returns a tensordict (``tensordict_out = module(tensordict_in)``). Set
    ``in_keys`` and ``out_keys`` as class attributes or in ``__init__``:
    :class:`~tensordict.nn.TensorDictSequential` and the other modules read
    them to know which entries the module reads and writes. To wrap a
    module that takes and returns tensors, use
    :class:`~tensordict.nn.TensorDictModule` instead. See
    :ref:`modules-guide`.

    The base class gives every module :meth:`select_out_keys` and
    :meth:`reset_out_keys`, to choose which outputs a call keeps, and
    :meth:`reset_parameters_recursive`.

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
        >>> _ = mod.select_out_keys("c")
        >>> td = mod(TensorDict(a=0))
        >>> td["c"]
        tensor(1)
        >>> "b" in td
        False

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
        """Returns ``True`` if ``module`` has both an ``in_keys`` and an ``out_keys`` attribute.

        :class:`~tensordict.nn.TensorDictSequential` wraps the callables that
        fail this check in a :class:`~tensordict.nn.WrapModule`.

        Args:
            module (object): the object to check.
        """
        return hasattr(module, "in_keys") and hasattr(module, "out_keys")

    @property
    def in_keys(self):
        """The keys that the module reads from the input tensordict."""
        return self._in_keys

    @in_keys.setter
    def in_keys(self, value: List[Union[str, Tuple[str]]]):
        self._in_keys = unravel_key_list(value)

    @property
    def out_keys(self):
        """The keys that the module writes to the output tensordict.

        After :meth:`select_out_keys`, only the selected keys. The full list
        is in :attr:`out_keys_source`.
        """
        return self._out_keys_apparent

    @out_keys.setter
    def out_keys(self, value: List[Union[str, Tuple[str]]]):
        self._out_keys = self._out_keys_apparent = unravel_key_list(list(value))

    # Below the setter: type checkers drop a setter that does not follow its
    # getter, and then see out_keys as read-only (#3237).
    @property
    def out_keys_source(self):
        """All the keys that the module writes, before :meth:`select_out_keys`."""
        return self._out_keys

    def select_out_keys(self, *out_keys) -> TensorDictModuleBase:
        """Keeps only some of the ``out_keys`` in the output of the module, and returns the module.

        Use it to drop intermediate outputs, for example those of a
        :class:`~tensordict.nn.TensorDictSequential`. The module changes in
        place: ``out_keys`` becomes the selected keys, and
        :attr:`out_keys_source` keeps the full list. A call still computes all
        the outputs, and then removes the ones that are not selected. It
        removes nothing else, so the input entries stay in the output. When
        you call the module with tensors (see :func:`~tensordict.nn.dispatch`),
        it returns only the selected outputs. :meth:`reset_out_keys` undoes
        the selection.

        Args:
            *out_keys (NestedKey): the keys to keep. Each one must be in
                ``out_keys``.

        Returns:
            ``self``.

        Raises:
            ValueError: if a key is not in ``out_keys``. The error names the
                missing keys, and is also a ``RuntimeError``.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> from tensordict.nn import TensorDictModule
            >>> mod = TensorDictModule(
            ...     lambda x, y: (x + 2, y + 2), in_keys=["a", "b"], out_keys=["c", "d"]
            ... )
            >>> _ = mod.select_out_keys("d")
            >>> mod.out_keys, mod.out_keys_source
            (['d'], ['c', 'd'])
            >>> td = mod(TensorDict(a=torch.zeros(()), b=torch.ones(())))
            >>> sorted(td.keys())
            ['a', 'b', 'd']
            >>> mod(torch.zeros(()), torch.ones(()))
            tensor(3.)
            >>> _ = mod.reset_out_keys()
            >>> sorted(mod(TensorDict(a=torch.zeros(()), b=torch.ones(()))).keys())
            ['a', 'b', 'c', 'd']

        It works on a :class:`~tensordict.nn.TensorDictSequential` too:

            >>> from tensordict.nn import TensorDictSequential
            >>> seq = TensorDictSequential(
            ...     TensorDictModule(lambda x: x + 1, in_keys=["x"], out_keys=["y"]),
            ...     TensorDictModule(lambda y: y + 1, in_keys=["y"], out_keys=["z"]),
            ... )
            >>> _ = seq.select_out_keys("z")
            >>> sorted(seq(TensorDict(x=torch.zeros(()))).keys())
            ['x', 'z']

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
                raise _OutKeyNotFoundError(err_msg)
        hook = _OutKeysSelect(out_keys)
        hook._init(self)
        self.register_forward_hook(hook, with_kwargs=True)
        return self

    def reset_out_keys(self):
        """Undoes :meth:`select_out_keys`: the module writes all its ``out_keys`` again.

        Returns:
            ``self``.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> from tensordict.nn import TensorDictModule
            >>> mod = TensorDictModule(
            ...     lambda x, y: (x + 2, y + 2), in_keys=["a", "b"], out_keys=["c", "d"]
            ... )
            >>> _ = mod.select_out_keys("d")
            >>> mod.out_keys
            ['d']
            >>> _ = mod.reset_out_keys()
            >>> mod.out_keys
            ['c', 'd']
            >>> td = mod(TensorDict(a=torch.zeros(()), b=torch.ones(())))
            >>> sorted(td.keys())
            ['a', 'b', 'c', 'd']
        """
        for i, hook in reversed(list(self._forward_hooks.items())):
            if isinstance(hook, _OutKeysSelect):
                hook.remove()
                del self._forward_hooks[i]
        return self

    def reset_parameters_recursive(
        self, parameters: Optional[TensorDictBase] = None
    ) -> Optional[TensorDictBase]:
        """Resets the parameters of every submodule that has a ``reset_parameters`` method.

        Without ``parameters``, the parameters of the module are reset in
        place. With ``parameters``, the module is left as it is: the values in
        ``parameters`` are reset in place, as if they were loaded into the
        module (see :meth:`~tensordict.TensorDictBase.to_module`). Use this
        for parameters kept outside the module, for example in a functional
        call. Values that share their storage with the module parameters,
        such as those of :meth:`~tensordict.TensorDict.from_module`, reset
        the module too.

        Args:
            parameters (TensorDictBase, optional): the parameters to reset,
                with the structure of
                :meth:`~tensordict.TensorDict.from_module` and an empty batch
                size. Defaults to ``None``, which resets the parameters of the
                module.

        Returns:
            ``None`` without ``parameters``. Otherwise a new tensordict of
            tensors that share their storage with ``parameters`` and require
            gradients.

        Raises:
            RuntimeError: if ``parameters`` has batch dimensions.

        Without ``parameters``, the call warns if no submodule has a
        ``reset_parameters`` method.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> from tensordict.nn import TensorDictModule
            >>> from torch import nn
            >>> net = nn.Sequential(nn.Linear(2, 3), nn.ReLU())
            >>> old_weight = net[0].weight.clone()
            >>> module = TensorDictModule(net, in_keys=["x"], out_keys=["y"])
            >>> module.reset_parameters_recursive()
            >>> torch.equal(old_weight, net[0].weight)
            False

        The parameters can also live outside the module:

            >>> params = TensorDict.from_module(module).clone()
            >>> old_params = params.clone()
            >>> _ = module.reset_parameters_recursive(params)
            >>> (old_params == params).any()
            False
            >>> torch.equal(net[0].weight, old_params["module", "0", "weight"])
            True
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

        with sanitized_parameters.to_module(self, preserve_module_state=False):
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

    def __repr__(self) -> str:
        entries = [f"{name}={module}" for name, module in self.named_children()]
        # a wrapped callable that is not an nn.Module (a plain function, a lambda, ...)
        # is stored as a regular attribute, hence missing from the children
        module = self.__dict__.get("module")
        if module is not None:
            entries.insert(0, f"module={module}")
        entries.append(f"in_keys={self.in_keys}")
        entries.append(f"out_keys={self.out_keys}")
        fields = indent(",\n".join(entries), 4 * " ")
        return f"{type(self).__name__}(\n{fields})"


class TensorDictModule(TensorDictModuleBase):
    """Wraps a callable so that it reads its inputs from a tensordict and writes its outputs to it.

    When you call the module with a tensordict, it reads the entries at
    ``in_keys``, passes them to ``module``, and writes the results at
    ``out_keys``. By default it writes into the input tensordict and returns
    it. You can also call it with tensors as positional or keyword arguments
    (see :func:`~tensordict.nn.dispatch`). It then returns the output tensors.
    See :ref:`modules-guide` for a guide.

    Args:
        module (Callable): the callable to wrap, typically a
            :class:`torch.nn.Module`. It can return a tensor, a tuple of
            tensors, or a dict or a tensordict. A dict or a tensordict that
            holds all the ``out_keys`` is read by key, so its order does not
            matter.
        in_keys (NestedKey, list of NestedKey, or dict): the entries that the
            module reads from the input tensordict. If a list, the values are
            passed as positional arguments, in this order. If a dict, the values
            are passed as keyword arguments. By default, the dict maps an
            argument name to a tensordict key: ``in_keys={"x": "obs"}`` calls
            ``module(x=td["obs"])``. One tensordict key can feed several
            arguments. ``out_to_in_map=False`` reverses the direction.
        out_keys (NestedKey or list of NestedKey): the keys where the outputs
            of ``module`` are written, one key per output, in order. An output
            whose key is ``"_"`` is not written.

    Keyword Args:
        out_to_in_map (bool, optional): the direction of a dict ``in_keys``.
            ``True`` reads it as ``{argument: tensordict_key}``, ``False`` as
            ``{tensordict_key: argument}``. Only used when ``in_keys`` is a
            dict. If you pass it with another type of ``in_keys``, the
            constructor warns. Defaults to ``True`` for a dict ``in_keys``.
        inplace (bool or str, optional): where the module writes its outputs.

            - ``True``: into the input tensordict, which ``forward`` returns.
            - ``False``: into a new :class:`~tensordict.TensorDict` that holds only
              the outputs, with an empty batch size and no device, whatever
              the type of the input.
            - ``"empty"``: into ``tensordict.empty()``, which keeps the type,
              batch size and device of the input.

            If you pass ``tensordict_out`` to :meth:`forward`, the outputs are
            written into it, whatever the value of ``inplace``. Defaults to
            ``True``.

        method (str, optional): the name of the method of ``module`` to call.
            Unlike a lambda around the method, this keeps the parameters and
            buffers of ``module`` registered in the wrapper. Defaults to
            ``None``, which calls ``module`` itself.
        method_kwargs (dict, optional): keyword arguments passed to
            ``module`` (or to ``method``) at each call. Defaults to ``None``.
        strict (bool, optional): if ``True``, a missing input raises a
            ``KeyError``. If ``False``, ``None`` is passed in its place, and if
            ``module`` then fails, the error names the missing keys. Defaults to
            ``False``.
        get_kwargs (dict, optional): keyword arguments passed to
            :meth:`~tensordict.TensorDictBase.get` when the inputs are read,
            for example ``{"as_padded_tensor": True}`` for ragged inputs.
            Defaults to ``None``.

    Attributes that the wrapper does not have, such as ``weight`` for a
    :class:`~torch.nn.Linear`, are read from ``module``. Inside a
    :class:`~tensordict.nn.set_skip_existing` block, the module does not run
    when all its ``out_keys`` are already in the input.

    Examples:
        >>> import torch
        >>> from torch import nn
        >>> from tensordict import TensorDict
        >>> from tensordict.nn import TensorDictModule
        >>> module = TensorDictModule(nn.Linear(3, 4), in_keys=["x"], out_keys=["y"])
        >>> data = TensorDict(x=torch.ones(2, 3), batch_size=[2])
        >>> data = module(data)
        >>> print(data)
        TensorDict(batch_size=[2], fields={x: f32[3], y: f32[4]})
        >>> module(torch.ones(2, 3)).shape
        torch.Size([2, 4])

    A plain function works too. This one reads a nested key and writes two
    nested keys:

        >>> module = TensorDictModule(
        ...     lambda x: (x - 1, x + 1),
        ...     in_keys=[("input", "x")],
        ...     out_keys=[("output", "x-1"), ("output", "x+1")],
        ... )
        >>> td = module(TensorDict({("input", "x"): torch.zeros(())}))
        >>> td["output", "x+1"]
        tensor(1.)

    With empty ``in_keys``, the module only writes entries:

        >>> module = TensorDictModule(lambda: torch.zeros(3), in_keys=[], out_keys=["x"])
        >>> module(TensorDict())["x"]
        tensor([0., 0., 0.])

    A dict ``in_keys`` passes the values as keyword arguments. By default
    (``out_to_in_map=True``), its keys are the argument names and its values
    are the tensordict keys. One tensordict key can feed several arguments:

        >>> module = TensorDictModule(lambda x, *, y, z: x+y+z,
        ...     in_keys={'x': '1', 'y': '2', 'z': '2'}, out_keys=['t']
        ...     )
        >>> td = module(TensorDict({'1': torch.ones(()), '2': torch.ones(())*2}, []))
        >>> td['t']
        tensor(5.)

    With ``out_to_in_map=False``, the keys of the dict are the tensordict keys
    and its values are the argument names:

        >>> module = TensorDictModule(lambda x, *, y: x+y,
        ...     in_keys={'1': 'x', '2': 'y'}, out_keys=['z'], out_to_in_map=False
        ...     )
        >>> td = module(TensorDict({'1': torch.ones(()), '2': torch.ones(())*2}, []))
        >>> td['z']
        tensor(3.)

    ``method`` calls another method of the module, here with a keyword
    argument from ``method_kwargs``:

        >>> class MyNet(nn.Module):
        ...     def my_func(self, tensor, *, an_integer):
        ...         return tensor + an_integer
        ...
        >>> module = TensorDictModule(
        ...     MyNet(),
        ...     in_keys=["a"],
        ...     out_keys=["b"],
        ...     method="my_func",
        ...     method_kwargs={"an_integer": 2},
        ... )
        >>> module(TensorDict(a=0))["b"]
        tensor(2)

    To call the module with other parameters, put them in a tensordict and
    use :meth:`~tensordict.TensorDictBase.to_module` as a context manager:

        >>> module = TensorDictModule(
        ...     nn.GRUCell(4, 8), in_keys=["input", "hidden"], out_keys=["output"]
        ... )
        >>> params = TensorDict.from_module(module)
        >>> params = params.apply(torch.zeros_like)
        >>> td = TensorDict(input=torch.ones(3, 4), hidden=torch.ones(3, 8), batch_size=[3])
        >>> with params.to_module(module):
        ...     td = module(td)
        >>> # with zero weights, the output is half the hidden state
        >>> (td["output"] == 0.5).all()
        tensor(True)

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
    @deprecated("TensorDictModule.is_functional", removal="0.17")
    def is_functional(self) -> bool:
        """Whether the wrapped module is a legacy functorch functional module.

        .. deprecated:: 0.15
            This property will be removed in TensorDict 0.17. It only checks
            for the ``FunctionalModule`` classes of the legacy
            ``functorch.make_functional`` API and has no replacement.
        """
        return isinstance(
            self.module,
            (_FunctionalModule, _FunctionalModuleWithBuffers),
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
                if type(_tensor) is not Tensor:
                    _tensor = TensorDict.from_any(_tensor)
                tensordict_out.set(_out_key, _tensor)
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
        """Reads ``in_keys`` from ``tensordict``, calls the wrapped module, and writes ``out_keys``.

        If you call the module with tensors instead of a tensordict, the
        tensors are put in a new tensordict first, and the call returns the
        output tensors (see :func:`~tensordict.nn.dispatch`).

        Args:
            tensordict (TensorDictBase): the input tensordict.

        Keyword Args:
            tensordict_out (TensorDictBase, optional): if given, the outputs
                are written into it and it is returned, whatever the value of
                ``inplace``. Defaults to ``None``.
            **kwargs: passed to the wrapped module.

        Returns:
            The tensordict that holds the outputs (see ``inplace``).
        """
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
                            _unravel_key_to_tuple(in_key),
                            default,
                            **self._get_kwargs,
                        )
                        for kwarg, in_key in _zip_strict(self._kwargs, self.in_keys)
                    }
                )
                tensors = ()
            else:
                # The in_keys setter unravels the keys: a str key needs no
                # unravel call, and a list comprehension makes no generator
                tensors = tuple(  # type: ignore[unreachable]
                    [
                        tensordict._get_tuple_maybe_non_tensor(
                            (in_key,)
                            if type(in_key) is str
                            else _unravel_key_to_tuple(in_key),
                            default,
                            **self._get_kwargs,
                        )
                        for in_key in self.in_keys
                    ]
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
            # An exact Tensor or tuple skips the ABC check of TensorDictBase
            if (
                type(tensors_out) is not Tensor
                and type(tensors_out) is not tuple
                and isinstance(tensors_out, (dict, TensorDictBase))
                and all(key in tensors_out for key in self.out_keys_source)
            ):
                if isinstance(tensors_out, dict):
                    keys = unravel_key_list(list(tensors_out.keys()))
                    values = tensors_out.values()
                    tensors_out = dict(_zip_strict(keys, values))
                tensors_out = tuple(
                    tensors_out.get(key) for key in self.out_keys_source
                )
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
        """The device of the first parameter of the module, or ``cpu`` if it has no parameter.

        .. deprecated:: 0.15
            This property will be removed in TensorDict 0.17 because modules may
            contain parameters and buffers on multiple devices.
        """
        warnings.warn(
            "TensorDictModule.device is deprecated and will be removed in "
            "TensorDict 0.17. Modules may contain parameters and buffers on "
            "multiple devices; inspect the relevant module state explicitly instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        for p in self.parameters():
            return p.device
        return torch.device("cpu")

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
    """Wraps a tensordict module and behaves like it, except for the methods that a subclass overrides.

    Subclass it to change :meth:`forward` and keep the ``in_keys``,
    ``out_keys``, parameters and attributes of the wrapped module. An
    attribute that the wrapper does not have is read from ``td_module``.

    Args:
        td_module (TensorDictModuleBase): the module to wrap.

    Examples:
        >>> import torch
        >>> from torch import nn
        >>> from tensordict import TensorDict
        >>> from tensordict.nn import TensorDictModule, TensorDictModuleWrapper
        >>> class TimesTen(TensorDictModuleWrapper):
        ...     def forward(self, tensordict):
        ...         tensordict = self.td_module(tensordict)
        ...         tensordict["y"] = tensordict["y"] * 10
        ...         return tensordict
        >>> inner = TensorDictModule(lambda x: x + 1, in_keys=["x"], out_keys=["y"])
        >>> wrapped = TimesTen(inner)
        >>> wrapped.in_keys, wrapped.out_keys
        (['x'], ['y'])
        >>> wrapped(TensorDict(x=torch.zeros(2)))["y"]
        tensor([10., 10.])

    """

    def __init__(self, td_module: TensorDictModuleBase) -> None:
        super().__init__()
        # The wrapped module's forward hooks run when forward calls it, so
        # they are not copied onto the wrapper, which would run them twice
        # (gh-2942).
        self.td_module = td_module

    @property
    def out_keys(self):
        """The keys that the module writes, by default those of ``td_module``."""
        return self.__dict__.get("_out_keys_apparent", self.td_module.out_keys)

    @out_keys.setter
    def out_keys(self, value: List[Union[str, Tuple[str]]]):
        self.td_module.out_keys = value
        self._out_keys_apparent = self.td_module.out_keys

    @property
    def out_keys_source(self):
        """All the keys that ``td_module`` writes, before :meth:`select_out_keys`."""
        return self.td_module.out_keys_source

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
        """Calls ``td_module`` with the same arguments and returns its result."""
        return self.td_module(*args, **kwargs)


class WrapModule(TensorDictModuleBase):
    """Wraps a function that takes and returns a tensordict as a tensordict module.

    Use it to put a function that works on the whole tensordict, such as a
    reshape, in a :class:`~tensordict.nn.TensorDictSequential`. A sequence
    wraps the callables that have no ``in_keys`` and ``out_keys`` in a
    ``WrapModule`` itself.

    Args:
        func (Callable[[TensorDictBase], TensorDictBase]): the function to
            wrap. It takes a tensordict and returns a tensordict.

    Keyword Args:
        inplace (bool, optional): if ``True``, the call writes the entries of
            the result of ``func`` into the input with
            :meth:`~tensordict.TensorDictBase.update` and returns the input.
            If ``False``, the call returns what ``func`` returns. Defaults to
            ``False``.
        in_keys (list of NestedKey, optional): the keys that ``func`` reads.
            They are not checked: they tell a
            :class:`~tensordict.nn.TensorDictSequential` what the module
            reads. Defaults to ``None``, which sets ``[]``.
        out_keys (list of NestedKey, optional): the keys that ``func``
            writes. They are not checked, like ``in_keys``. Defaults to
            ``None``, which sets ``[]``.

    Examples:
        >>> import torch
        >>> from tensordict import TensorDict
        >>> from tensordict.nn import TensorDictSequential as Seq, TensorDictModule as Mod, WrapModule
        >>> seq = Seq(
        ...     Mod(lambda x: x * 2, in_keys=["x"], out_keys=["y"]),
        ...     WrapModule(lambda td: td.reshape(-1)),
        ... )
        >>> td = TensorDict(x=torch.ones(3, 4, 5), batch_size=[3, 4])
        >>> td = seq(td)
        >>> td.batch_size
        torch.Size([12])
        >>> td["y"].shape
        torch.Size([12, 5])

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
        """Calls ``func`` on ``data`` and returns the result, or ``data`` updated with it if ``inplace=True``."""
        result = self.func(data)
        if self.inplace and result is not data:
            return data.update(result)
        return result


class as_tensordict_module:
    """Turns a function or a method into a tensordict module that reads ``in_keys`` and writes ``out_keys``.

    The decorated function is wrapped in a
    :class:`~tensordict.nn.TensorDictModule`, built at the first call.
    Called with a tensordict, it returns the tensordict with the outputs
    written at ``out_keys``. Called with tensors (see
    :func:`~tensordict.nn.dispatch`), it returns the output tensors. On a
    method, one module serves all the instances of the class, and the
    instance is passed as ``self`` at each call.

    Keyword Args:
        in_keys (NestedKey or list of NestedKey): the keys of the entries
            passed to the function, in the order of its arguments.
        out_keys (NestedKey or list of NestedKey): the keys where the outputs
            of the function are written, one key per output.

    Examples:
        >>> import torch
        >>> from tensordict import TensorDict
        >>> from tensordict.nn import as_tensordict_module
        >>> @as_tensordict_module(in_keys="c", out_keys="d")
        ... def my_function(c):
        ...     return c + 1
        >>> my_function(TensorDict(c=0))["d"]
        tensor(1)
        >>> my_function(c=torch.zeros(2))
        tensor([1., 1.])

    It works on methods too:

        >>> class MyClass:
        ...     @as_tensordict_module(in_keys="c", out_keys="d")
        ...     def my_method(self, c):
        ...         return c + 1
        >>> MyClass().my_method(TensorDict(c=0))["d"]
        tensor(1)
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
                    # The module is shared by all instances, so the instance
                    # is passed with each call rather than kept in a closure.
                    def newfunc(*args, _as_tensordict_module_self, **kwargs):
                        return func(_as_tensordict_module_self, *args, **kwargs)

                    tdmodule = TensorDictModule(
                        newfunc, in_keys=self.in_keys, out_keys=self.out_keys
                    )
                return tdmodule(*args, _as_tensordict_module_self=_self, **kwargs)

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


__getattr__ = deprecated_attributes(
    __name__,
    {
        "FunctionalModule": (_FunctionalModule, "functorch.FunctionalModule"),
        "FunctionalModuleWithBuffers": (
            _FunctionalModuleWithBuffers,
            "functorch.FunctionalModuleWithBuffers",
        ),
    },
    removal="0.17",
)
