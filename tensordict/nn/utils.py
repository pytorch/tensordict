# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import functools
import inspect
import os
from enum import Enum
from typing import Any, Callable

import torch
from tensordict._deprecation import deprecated_attributes
from tensordict._nestedkey import NestedKey
from tensordict.utils import _ContextManager, _strtobool, unravel_key_list
from torch import nn
from torch.compiler import is_compiling
from torch.utils._contextlib import _DecoratorContextManager

_dispatch_tdnn_modules = _ContextManager(
    default=_strtobool(os.environ.get("DISPATCH_TDNN_MODULES", "True"))
)

__all__ = [
    "mappings",
    "inv_softplus",
    "biased_softplus",
    "expln",
    "add_custom_mapping",
    "set_skip_existing",
    "skip_existing",
    "composite_lp_aggregate",
    "set_composite_lp_aggregate",
]

_skip_existing = _ContextManager(default=False)


def inv_softplus(bias: float | torch.Tensor) -> float | torch.Tensor:
    """Inverse softplus function.

    Args:
        bias (float or tensor): the value to be softplus-inverted.
    """
    is_tensor = True
    if not isinstance(bias, torch.Tensor):
        is_tensor = False
        bias = torch.tensor(bias)
    out = bias.expm1().clamp_min(1e-6).log()
    if not is_tensor and out.numel() == 1:
        return out.item()
    return out


class biased_softplus(nn.Module):
    """A biased softplus module.

    The bias indicates the value that is to be returned when a zero-tensor is
    passed through the transform. The module computes
    ``softplus(x + inv_softplus(bias - min_val)) + min_val``.

    Args:
        bias (float): the output for a zero input.
        min_val (float, optional): minimum value of the output. Defaults to ``0.01``.

    Examples:
        >>> import torch
        >>> from tensordict.nn import biased_softplus
        >>> f = biased_softplus(1.0)
        >>> f(torch.zeros(2))
        tensor([1., 1.])
    """

    def __init__(self, bias: float, min_val: float = 0.01) -> None:
        super().__init__()
        self.bias = inv_softplus(bias - min_val)
        self.min_val = min_val

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.softplus(x + self.bias) + self.min_val


def expln(x):
    """A smooth, continuous positive mapping presented in "State-Dependent Exploration for Policy Gradient Methods".

    https://people.idsia.ch/~juergen/ecml2008rueckstiess.pdf

    """
    out = torch.empty_like(x)
    idx_neg = x <= 0
    out[idx_neg] = x[idx_neg].exp()
    out[~idx_neg] = x[~idx_neg].log1p() + 1
    return out


_MAPPINGS: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "softplus": torch.nn.functional.softplus,
    "exp": torch.exp,
    "relu": torch.relu,
    "biased_softplus": biased_softplus(1.0),
    "none": lambda x: x,
    "expln": expln,
}


def mappings(key: str) -> Callable:
    """Returns the function named by ``key``.

    All the functions except ``"none"`` (the identity) map real numbers to
    non-negative numbers. ``"relu"`` can return 0, the others return
    positive numbers.

    Args:
        key (str): one of `"softplus"`, `"exp"`, `"relu"`, `"expln"`,
            `"biased_softplus"` or `"none"` (no mapping).

    .. note::
        If the key begins with ``"biased_softplus"``, then it needs to take the following form:
        ``"biased_softplus_{bias}"`` where ``bias`` can be converted to a floating point number.
        Alternatively, the ``"biased_softplus_{bias}_{min_val}"`` syntax can be used.
        The function is then :class:`biased_softplus` with these arguments:
        ``softplus(x + inv_softplus(bias - min_val)) + min_val``, which returns
        ``bias`` for a zero input and is always greater than ``min_val``.

    .. note::
        Custom mappings can be added through ``tensordict.nn.add_custom_mapping``.

    Returns:
         a Callable

    """
    if key in _MAPPINGS:
        return _MAPPINGS[key]
    elif key.startswith("biased_softplus"):
        stripped_key = key.split("_")
        if len(stripped_key) == 3:
            return biased_softplus(float(stripped_key[-1]))
        elif len(stripped_key) == 4:
            return biased_softplus(
                float(stripped_key[-2]), min_val=float(stripped_key[-1])
            )
        else:
            raise ValueError(f"Invalid number of args in  {key}")

    else:
        raise NotImplementedError(f"Unknown mapping {key}")


def add_custom_mapping(name: str, mapping: Callable[[torch.Tensor], torch.Tensor]):
    """Adds a custom mapping to be used in mapping classes.

    Args:
        name (str): a mapping name.
        mapping (callable): a callable that takes a tensor as input and outputs a tensor
            with the same shape.

    Examples:
        >>> import torch
        >>> from tensordict.nn import add_custom_mapping, NormalParamExtractor
        >>> add_custom_mapping("my_mapping", lambda x: torch.zeros_like(x))
        >>> npe = NormalParamExtractor(scale_mapping="my_mapping", scale_lb=0.0)
        >>> assert (npe(torch.randn(10))[1] == torch.zeros(5)).all()
    """
    _MAPPINGS[name] = mapping


class set_skip_existing(_DecoratorContextManager):
    """A context manager for skipping existing nodes in a TensorDict graph.

    When used as a context manager, it will set the `skip_existing()` value
    to the ``mode`` indicated, leaving the user able to code up methods that
    will check the global value and execute the code accordingly.

    When used as a method decorator, it will check the tensordict input keys
    and skip the method if the current mode applies to all output keys and they
    are already present.
    This is not expected to be used as a decorator for methods that do not
    respect the following signature: ``def fun(self, tensordict, *args, **kwargs)``.

    Args:
        mode (bool, list of NestedKey, optional):
            If ``True``, it indicates that existing entries in the graph
            won't be overwritten, unless they are only partially present. :func:`~.skip_existing`
            will return ``True``.
            If ``False``, no check will be performed.
            If a list of keys, only modules whose output keys are all in the
            list can be skipped. The keys follow TensorDict's nested-key
            conventions. If any output key is not in the list, the module runs
            and recomputes all of its outputs.
            If ``None``, the value of :func:`~.skip_existing` will not be
            changed. This is intended to be used exclusively for decorating
            methods and allow their behaviour to depend on the same class
            when used as a context manager (see example below).
            Defaults to ``True``.
        in_key_attr (str, optional): the name of the input key list attribute
            in the module's method being decorated. Defaults to ``in_keys``.
        out_key_attr (str, optional): the name of the output key list attribute
            in the module's method being decorated. Defaults to ``out_keys``.

    Keyword Args:
        thread_local (bool, optional): if ``True``, the mode applies to the
            calling thread only, and blocks that other threads enter do not
            change it. Inside the block, blocks without this flag that the
            same thread enters stay in that thread too. Threads started inside
            the block see the process-wide mode. If ``False``, the mode applies
            to the whole process. Defaults to ``False``.

    .. note:: Threads. By default the mode is process-wide: a ``with`` block
        in one thread also changes what :func:`~.skip_existing` returns in
        other threads at that time. Pass ``thread_local=True`` to keep a
        block to its own thread. A thread started inside such a block reads
        the process-wide value, not the block's.

    Examples:
        >>> import torch
        >>> from tensordict.nn import set_skip_existing, skip_existing
        >>> with set_skip_existing():
        ...     if skip_existing():
        ...         print("True")
        ...     else:
        ...         print("False")
        ...
        True
        >>> print("calling from outside:", skip_existing())
        calling from outside: False

    This class can also be used as a decorator:

    Examples:
        >>> from tensordict import TensorDict
        >>> from tensordict.nn import set_skip_existing, skip_existing, TensorDictModuleBase
        >>> class MyModule(TensorDictModuleBase):
        ...     in_keys = []
        ...     out_keys = ["out"]
        ...     @set_skip_existing()
        ...     def forward(self, tensordict):
        ...         print("hello")
        ...         tensordict.set("out", torch.zeros(()))
        ...         return tensordict
        >>> module = MyModule()
        >>> module(TensorDict({"out": torch.zeros(())}, []))  # does not print anything
        TensorDict(
            fields={
                out: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False)},
            batch_size=torch.Size([]),
            device=None,
            is_shared=False)
        >>> module(TensorDict())  # prints hello
        hello
        TensorDict(
            fields={
                out: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False)},
            batch_size=torch.Size([]),
            device=None,
            is_shared=False)

    Decorating a method with the mode set to ``None`` is useful whenever one
    wants to let the context manager take care of skipping things from the outside:

    Examples:
        >>> from tensordict import TensorDict
        >>> from tensordict.nn import set_skip_existing, skip_existing, TensorDictModuleBase
        >>> class MyModule(TensorDictModuleBase):
        ...     in_keys = []
        ...     out_keys = ["out"]
        ...     @set_skip_existing(None)
        ...     def forward(self, tensordict):
        ...         print("hello")
        ...         tensordict.set("out", torch.zeros(()))
        ...         return tensordict
        >>> module = MyModule()
        >>> _ = module(TensorDict({"out": torch.zeros(())}, []))  # prints "hello"
        hello
        >>> with set_skip_existing(True):
        ...     _ = module(TensorDict({"out": torch.zeros(())}, []))  # no print


    .. note::
        To allow for modules to have the same input and output keys and not
        mistakenly ignoring subgraphs, ``@set_skip_existing(True)`` will be
        deactivated whenever the output keys are also the input keys:

            >>> class MyModule(TensorDictModuleBase):
            ...     in_keys = ["out"]
            ...     out_keys = ["out"]
            ...     @set_skip_existing()
            ...     def forward(self, tensordict):
            ...         print("calling the method!")
            ...         return tensordict
            ...
            >>> module = MyModule()
            >>> module(TensorDict({"out": torch.zeros(())}, []))  # runs: in_keys and out_keys overlap
            calling the method!
            TensorDict(
                fields={
                    out: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False)},
                batch_size=torch.Size([]),
                device=None,
                is_shared=False)


    """

    def __init__(
        self,
        mode: bool | list[NestedKey] | None = True,
        in_key_attr="in_keys",
        out_key_attr="out_keys",
        *,
        thread_local: bool = False,
    ):
        self.mode = unravel_key_list(mode) if isinstance(mode, list) else mode
        self.in_key_attr = in_key_attr
        self.out_key_attr = out_key_attr
        self.thread_local = thread_local
        self._called = False

    def clone(self) -> set_skip_existing:
        # override this method if your children class takes __init__ parameters
        out = type(self)(self.mode, thread_local=self.thread_local)
        out._called = self._called
        return out

    def __call__(self, func: Callable):

        self._called = True

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
        def wrapper(_self, tensordict, *args: Any, **kwargs: Any) -> Any:
            in_keys = getattr(_self, self.in_key_attr)
            out_keys = getattr(_self, self.out_key_attr)
            # we use skip_existing to allow users to override the mode internally
            skip_mode = skip_existing()
            if (
                skip_mode
                and (skip_mode is True or all(key in skip_mode for key in out_keys))
                and all(key in tensordict.keys(True) for key in out_keys)
                and not any(key in out_keys for key in in_keys)
            ):
                return tensordict
            return func(_self, tensordict, *args, **kwargs)

        return super().__call__(wrapper)

    def __enter__(self) -> None:
        if self.mode and is_compiling():
            raise RuntimeError("skip_existing is not compatible with TorchDynamo.")
        if self.thread_local:
            self.prev = _skip_existing.get_thread_mode()
        else:
            self.prev = _skip_existing.get_mode()
        if self.mode is not None:
            if self.thread_local:
                _skip_existing.set_thread_mode(self.mode)
            else:
                _skip_existing.set_mode(self.mode)
        elif not self._called:
            raise RuntimeError(
                f"It seems you are using {type(self).__name__} as a context manager with ``None`` input. "
                f"This behaviour is not allowed."
            )

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        if self.thread_local:
            _skip_existing.set_thread_mode(self.prev)
        else:
            _skip_existing.set_mode(self.prev)


class _set_skip_existing_None(set_skip_existing):
    """A version of skip_existing that is constant wrt init inputs (for torch.compile compatibility).

    This class should only be used as a decorator, not a context manager.
    """

    def __call__(self, func: Callable):
        self._called = True

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
        def wrapper(_self, tensordict, *args: Any, **kwargs: Any) -> Any:
            # we use skip_existing to allow users to override the mode internally
            skip_mode = _skip_existing.get_mode()
            if skip_mode:
                if is_compiling():
                    raise RuntimeError(
                        "skip_existing is not compatible with torch.compile."
                    )
                in_keys = getattr(_self, self.in_key_attr)
                out_keys = getattr(_self, self.out_key_attr)
                if (
                    (skip_mode is True or all(key in skip_mode for key in out_keys))
                    and all(key in tensordict.keys(True) for key in out_keys)
                    and not any(key in out_keys for key in in_keys)
                ):
                    return tensordict
            try:
                return func(_self, tensordict, *args, **kwargs)
            finally:
                # Writing back an unchanged mode does nothing: skip that call.
                # A mode that func changed is not restored when compiling.
                if _skip_existing.get_mode() is not skip_mode and not is_compiling():
                    _skip_existing.set_mode(skip_mode)

        return wrapper

    in_key_attr = "in_keys"
    out_key_attr = "out_keys"
    __init__ = object.__init__

    def clone(self) -> _set_skip_existing_None:
        # override this method if your children class takes __init__ parameters
        out = type(self)()
        return out


def skip_existing() -> bool | list[NestedKey]:
    """Returns which existing entries should not be re-computed by a module."""
    return _skip_existing.get_mode()


# For backward compatibility in imports
from torch.nn.parameter import Buffer  # noqa


def _dispatch_td_nn_modules():
    """Returns ``True`` if @dispatch should be used. Not using dispatch is faster and also better compatible with torch.compile."""
    return _dispatch_tdnn_modules.get_mode()


_NO_RESTORE = object()


class _set_dispatch_td_nn_modules(_DecoratorContextManager):
    """Controls whether @dispatch should be used. Not using dispatch is faster and also better compatible with torch.compile."""

    def __init__(self, mode):
        self.mode = mode
        # One saved mode per open block, so that an instance can be reused
        # in sequence or nested in itself.
        self._saved_modes = []

    def clone(self):
        return type(self)(self.mode)

    def __enter__(self):
        # We want to avoid changing global variables because compile puts guards on them
        prev = _dispatch_tdnn_modules.get_mode()
        if prev != self.mode:
            self._saved_modes.append(prev)
            _dispatch_tdnn_modules.set_mode(self.mode)
        else:
            self._saved_modes.append(_NO_RESTORE)

    def __exit__(self, exc_type, exc_val, exc_tb):
        prev = self._saved_modes.pop()
        if prev is not _NO_RESTORE:
            _dispatch_tdnn_modules.set_mode(prev)


# Reproduce StrEnum for python<3.11


class _StrEnum(str, Enum):  # noqa
    def __new__(cls, *values):
        if len(values) > 3:
            raise TypeError("too many arguments for str(): %r" % (values,))
        if len(values) == 1:
            # it must be a string
            if not isinstance(values[0], str):
                raise TypeError("%r is not a string" % (values[0],))
        if len(values) >= 2:
            # check that encoding argument is a string
            if not isinstance(values[1], str):
                raise TypeError("encoding must be a string, not %r" % (values[1],))
        if len(values) == 3:
            # check that errors argument is a string
            if not isinstance(values[2], str):
                raise TypeError("errors must be a string, not %r" % (values[2]))
        value = str(*values)
        member = str.__new__(cls, value)
        member._value_ = value
        return member

    def _generate_next_value_(name, start, count, last_values):
        return name.lower()


_composite_lp_aggregate = _ContextManager(
    default=(
        _strtobool(os.getenv("COMPOSITE_LP_AGGREGATE"))
        if os.getenv("COMPOSITE_LP_AGGREGATE") is not None
        else None
    )
)


def composite_lp_aggregate(nowarn: bool = False) -> bool | None:
    """Returns whether a :class:`~tensordict.nn.distributions.CompositeDistribution` log-probabilities and entropies will be aggregated in a single tensor.

    Returns ``False`` unless :class:`~tensordict.nn.set_composite_lp_aggregate`
    or the ``COMPOSITE_LP_AGGREGATE`` environment variable set it to ``True``.
    See the :ref:`log-probability key table <log-prob-keys>` of
    :class:`~tensordict.nn.ProbabilisticTensorDictModule` for what each value
    changes.

    Args:
        nowarn (bool, optional): has no effect. Defaults to False.

    .. seealso:: :func:`~tensordict.nn.set_composite_lp_aggregate`

    """
    mode = _composite_lp_aggregate.get_mode()
    return bool(mode)


class set_composite_lp_aggregate(_DecoratorContextManager):
    """Controls whether :class:`~tensordict.nn.distributions.CompositeDistribution` log-probabilities and entropies will be aggregated in a single tensor.

    When :func:`~tensordict.nn.composite_lp_aggregate` returns ``True``, the log-probs / entropies of :class:`~tensordict.nn.distributions.CompositeDistribution`
    will be summed into a single tensor with the shape of the root tensordict.
    When it returns ``False`` (the default), they are returned as a
    TensorDict with one ``<name>_log_prob`` (or ``<name>_entropy``) entry per
    sampled key. The value also decides the keys where probabilistic modules
    write their log-probabilities: see the :ref:`log-probability key table <log-prob-keys>` of
    :class:`~tensordict.nn.ProbabilisticTensorDictModule`, and
    :ref:`modules-guide`.

    The value of composite_lp_aggregate can also be controlled through the `COMPOSITE_LP_AGGREGATE` environment variable.
    It works as a context manager and as a decorator, and :meth:`set` and
    :meth:`unset` change it outside of a block.

    Args:
        mode (bool, optional): the value that
            :func:`~tensordict.nn.composite_lp_aggregate` returns inside the
            block. Defaults to ``True``.

    Keyword Args:
        thread_local (bool, optional): if ``True``, the mode applies to the
            calling thread only, and blocks that other threads enter do not
            change it. Inside the block, blocks without this flag that the
            same thread enters stay in that thread too. Threads started inside
            the block see the process-wide mode. If ``False``, the mode applies
            to the whole process. Defaults to ``False``.

    .. note:: Threads. The mode is process-wide by default, and is meant to
        be set once for the whole process, for example with :meth:`set`
        before you build the modules: it decides the keys of the
        log-probabilities, which the threads that write them and the threads
        that read them must agree on. A ``with`` block in one thread also
        changes the value in other threads at that time, unless it passes
        ``thread_local=True``. A thread started inside a ``thread_local``
        block reads the process-wide value, not the block's.

    Example:
        >>> import torch
        >>> from tensordict import TensorDict
        >>> from tensordict.nn import CompositeDistribution, set_composite_lp_aggregate
        >>> from torch import distributions as d
        >>> _ = torch.manual_seed(0)
        >>> params = TensorDict({
        ...     "cont": {"loc": torch.randn(3, 4), "scale": torch.rand(3, 4)},
        ...     ("nested", "disc"): {"logits": torch.randn(3, 10)}
        ... }, [3])
        >>> dist = CompositeDistribution(params,
        ...     distribution_map={"cont": d.Normal, ("nested", "disc"): d.Categorical})
        >>> sample = dist.sample((4,))
        >>> with set_composite_lp_aggregate(False):
        ...     lp = dist.log_prob(sample)
        ...     print(lp)
        TensorDict(
            fields={
                cont_log_prob: Tensor(shape=torch.Size([4, 3, 4]), device=cpu, dtype=torch.float32, is_shared=False),
                nested: TensorDict(
                    fields={
                        disc_log_prob: Tensor(shape=torch.Size([4, 3]), device=cpu, dtype=torch.float32, is_shared=False)},
                    batch_size=torch.Size([4, 3]),
                    device=None,
                    is_shared=False)},
            batch_size=torch.Size([4, 3]),
            device=None,
            is_shared=False)
        >>> with set_composite_lp_aggregate(True):
        ...     lp = dist.log_prob(sample)
        ...     print(lp)
        tensor([[-2.0886, -1.2155, -0.0414],
                [-2.8973, -5.5165,  2.4402],
                [-0.2806, -1.2799,  3.1733],
                [-3.0407, -4.3593,  0.5763]])
    """

    def __init__(
        self,
        mode: bool = True,
        *,
        thread_local: bool = False,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.thread_local = thread_local

    def clone(self) -> set_composite_lp_aggregate:
        # override this method if your children class takes __init__ parameters
        return type(self)(self.mode, thread_local=self.thread_local)

    def __enter__(self) -> None:
        if self.thread_local:
            self.prev = _composite_lp_aggregate.get_thread_mode()
            _composite_lp_aggregate.set_thread_mode(self.mode)
        else:
            self.prev = _composite_lp_aggregate.get_mode()
            _composite_lp_aggregate.set_mode(self.mode)

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        if self.thread_local:
            _composite_lp_aggregate.set_thread_mode(self.prev)
        else:
            _composite_lp_aggregate.set_mode(self.prev)

    def set(self):
        self.__enter__()

    def unset(self):
        return self.__exit__(None, None, None)


__getattr__ = deprecated_attributes(
    __name__, {"StrEnum": (_StrEnum, None)}, removal="0.17"
)
