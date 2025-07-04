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
from tensordict.utils import _ContextManager, strtobool
from torch import nn

from torch.utils._contextlib import _DecoratorContextManager

try:
    from torch.compiler import is_compiling
except ImportError:  # torch 2.0
    from torch._dynamo import is_compiling


_dispatch_tdnn_modules = _ContextManager(
    default=strtobool(os.environ.get("DISPATCH_TDNN_MODULES", "True"))
)

__all__ = ["mappings", "inv_softplus", "biased_softplus"]

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
    passed through the transform.

    Args:
        bias (scalar): 'bias' of the softplus transform. If bias=1.0, then a _bias shift will be computed such that
            softplus(0.0 + _bias) = bias.
        min_val (scalar): minimum value of the transform.
            default: 0.1
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
    """Given an input string, returns a surjective function f(x): R -> R^+.

    Args:
        key (str): one of `"softplus"`, `"exp"`, `"relu"`, `"expln"`,
            `"biased_softplus"` or `"none"` (no mapping).

    .. note::
        If the key begins with `"biased_softplus"`, then it needs to take the following form:
        ```"biased_softplus_{bias}"``` where ```bias``` can be converted to a floating point number that will be
        used to bias the softplus function.
        Alternatively, the ```"biased_softplus_{bias}_{min_val}"``` syntax can be used.
        In that case, the additional ```min_val``` term is a floating point
        number that will be used to encode the minimum value of the softplus transform.
        In practice, the equation used is `softplus(x + bias) + min_val`, where bias and min_val are values computed
        such that the conditions above are met.

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
    and if the ``skip_existing()`` call returns ``True``, it will skip the method
    if all the output keys are already present.
    This not not expected to be used as a decorator for methods that do not
    respect the following signature: ``def fun(self, tensordict, *args, **kwargs)``.

    Args:
        mode (bool, optional):
            If ``True``, it indicates that existing entries in the graph
            won't be overwritten, unless they are only partially present. :func:`~.skip_existing`
            will return ``True``.
            If ``False``, no check will be performed.
            If ``None``, the value of :func:`~.skip_existing` will not be
            changed. This is intended to be used exclusively for decorating
            methods and allow their behaviour to depend on the same class
            when used as a context manager (see example below).
            Defaults to ``True``.
        in_key_attr (str, optional): the name of the input key list attribute
            in the module's method being decorated. Defaults to ``in_keys``.
        out_key_attr (str, optional): the name of the output key list attribute
            in the module's method being decorated. Defaults to ``out_keys``.

    Examples:
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
    wants ot let the context manager take care of skipping things from the outside:

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
            >>> module(TensorDict({"out": torch.zeros(())}, []))  # does not print anything
            calling the method!
            TensorDict(
                fields={
                    out: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False)},
                batch_size=torch.Size([]),
                device=None,
                is_shared=False)


    """

    def __init__(
        self, mode: bool | None = True, in_key_attr="in_keys", out_key_attr="out_keys"
    ):
        self.mode = mode
        self.in_key_attr = in_key_attr
        self.out_key_attr = out_key_attr
        self._called = False

    def clone(self) -> set_skip_existing:
        # override this method if your children class takes __init__ parameters
        out = type(self)(self.mode)
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
            if (
                skip_existing()
                and all(key in tensordict.keys(True) for key in out_keys)
                and not any(key in out_keys for key in in_keys)
            ):
                return tensordict
            return func(_self, tensordict, *args, **kwargs)

        return super().__call__(wrapper)

    def __enter__(self) -> None:
        if self.mode and is_compiling():
            raise RuntimeError("skip_existing is not compatible with TorchDynamo.")
        self.prev = _skip_existing.get_mode()
        if self.mode is not None:
            _skip_existing.set_mode(self.mode)
        elif not self._called:
            raise RuntimeError(
                f"It seems you are using {type(self).__name__} as a context manager with ``None`` input. "
                f"This behaviour is not allowed."
            )

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
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
            if skip_existing() and is_compiling():
                raise RuntimeError(
                    "skip_existing is not compatible with torch.compile."
                )
            in_keys = getattr(_self, self.in_key_attr)
            out_keys = getattr(_self, self.out_key_attr)
            # we use skip_existing to allow users to override the mode internally
            if (
                skip_existing()
                and all(key in tensordict.keys(True) for key in out_keys)
                and not any(key in out_keys for key in in_keys)
            ):
                return tensordict
            if is_compiling():
                return func(_self, tensordict, *args, **kwargs)
            self.prev = _skip_existing.get_mode()
            try:
                result = func(_self, tensordict, *args, **kwargs)
            finally:
                _skip_existing.set_mode(self.prev)
            return result

        return wrapper

    in_key_attr = "in_keys"
    out_key_attr = "out_keys"
    __init__ = object.__init__

    def clone(self) -> _set_skip_existing_None:
        # override this method if your children class takes __init__ parameters
        out = type(self)()
        return out


def skip_existing():
    """Returns whether or not existing entries in a tensordict should be re-computed by a module."""
    return _skip_existing.get_mode()


def _rebuild_buffer(data, requires_grad, backward_hooks):
    buffer = Buffer(data, requires_grad)
    # NB: This line exists only for backwards compatibility; the
    # general expectation is that backward_hooks is an empty
    # OrderedDict.  See Note [Don't serialize hooks]
    buffer._backward_hooks = backward_hooks

    return buffer


# For backward compatibility in imports
try:
    from torch.nn.parameter import Buffer  # noqa
except ImportError:
    from tensordict.utils import Buffer  # noqa


def _dispatch_td_nn_modules():
    """Returns ``True`` if @dispatch should be used. Not using dispatch is faster and also better compatible with torch.compile."""
    return _dispatch_tdnn_modules.get_mode()


class _set_dispatch_td_nn_modules(_DecoratorContextManager):
    """Controls whether @dispatch should be used. Not using dispatch is faster and also better compatible with torch.compile."""

    def __init__(self, mode):
        self.mode = mode
        self._saved_mode = None

    def clone(self):
        return type(self)(self.mode)

    def __enter__(self):
        # We want to avoid changing global variables because compile puts guards on them
        if _dispatch_tdnn_modules.get_mode() != self.mode:
            self._saved_mode = _dispatch_tdnn_modules
            _dispatch_tdnn_modules.set_mode(self.mode)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._saved_mode is None:
            return
        _dispatch_tdnn_modules.set_mode(self._saved_mode)


# Reproduce StrEnum for python<3.11


class StrEnum(str, Enum):  # noqa
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
        strtobool(os.getenv("COMPOSITE_LP_AGGREGATE"))
        if os.getenv("COMPOSITE_LP_AGGREGATE") is not None
        else None
    )
)


def composite_lp_aggregate(nowarn: bool = False) -> bool | None:
    """Returns whether a :class:`~tensordict.nn.CompositeDistribution` log-probabilities and entropies will be aggregated in a single tensor.

    Args:
        nowarn (bool, optional): whether to ignore warnings. Defaults to False.

    .. seealso:: :func:`~tensordict.nn.set_composite_lp_aggregate`

    """
    mode = _composite_lp_aggregate.get_mode()
    return bool(mode)


class set_composite_lp_aggregate(_DecoratorContextManager):
    """Controls whether :class:`~tensordict.nn.CompositeDistribution` log-probabilities and entropies will be aggregated in a single tensor.

    When :func:`~tensordict.nn.composite_lp_aggregate` returns ``True``, the log-probs / entropies of :class:`~tensordict.nn.CompositeDistribution`
    will be summed into a single tensor with the shape of the root tensordict. This behaviour is being deprecated in favor of
    non-aggregated log-probs, which offer more flexibility and a somewhat more natural API (tensordict samples, tensordict log-probs, tensordict entropies).

    The value of composite_lp_aggregate can also be controlled through the `COMPOSITE_LP_AGGREGATE` environment variable.

    Example:
        >>> _ = torch.manual_seed(0)
        >>> from tensordict import TensorDict
        >>> from tensordict.nn import CompositeDistribution, set_composite_lp_aggregate
        >>> import torch
        >>> from torch import distributions as d
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
    ) -> None:
        super().__init__()
        self.mode = mode

    def clone(self) -> set_composite_lp_aggregate:
        # override this method if your children class takes __init__ parameters
        return type(self)(self.mode)

    def __enter__(self) -> None:
        self.prev = _composite_lp_aggregate.get_mode()
        _composite_lp_aggregate.set_mode(self.mode)

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        _composite_lp_aggregate.set_mode(self.prev)

    def set(self):
        self.__enter__()

    def unset(self):
        return self.__exit__(None, None, None)
