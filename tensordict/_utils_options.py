# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import os
import warnings
from typing import Any

from tensordict._deprecation import warn_deprecated, warn_deprecated_env_var
from torch.utils._contextlib import _DecoratorContextManager

__all__ = [
    "_REPR_OPTIONS",
    "_legacy_lazy",
    "capture_non_tensor_stack",
    "get_printoptions",
    "lazy_legacy",
    "list_to_stack",
    "set_capture_non_tensor_stack",
    "set_lazy_legacy",
    "set_list_to_stack",
    "set_printoptions",
]


def _strtobool(val: str) -> bool:
    val = val.lower()
    if val in ("y", "yes", "t", "true", "on", "1"):
        return True
    if val in ("n", "no", "f", "false", "off", "0"):
        return False
    raise ValueError(f"invalid truth value {val!r}")


_REPR_OPTIONS = {
    "show_batch_size": True,
    "show_device": True,
    "show_is_shared": True,
    "show_shape": True,
    "show_field_device": True,
    "show_dtype": True,
    "show_field_is_shared": True,
    "show_grad": False,
    "show_is_contiguous": False,
    "show_is_view": False,
    "show_storage_size": False,
    "plain": False,
    "sort_keys": "alphabetical",
}

_REPR_OPTIONS_KEYS = frozenset(_REPR_OPTIONS)

_VERBOSE_FALSE_OVERRIDES = {
    "show_device": False,
    "show_is_shared": False,
    "show_field_device": False,
    "show_dtype": False,
    "show_field_is_shared": False,
}


class set_printoptions(_DecoratorContextManager):
    """Controls which attributes appear in TensorDict's ``__repr__`` output.

    Use it as a context manager or a decorator to change the options for a
    block of code, or call ``set_printoptions(...).set()`` to change them
    for the rest of the program. :ref:`printing` describes every option
    with examples.

    Args:
        verbose (bool, optional): if ``False``, the device, dtype and
            ``is_shared`` attributes are hidden, unless a keyword argument
            sets them. Defaults to ``True``.
        **kwargs: the options to set: ``show_batch_size``, ``show_device``,
            ``show_is_shared``, ``show_shape``, ``show_field_device``,
            ``show_dtype``, ``show_field_is_shared``, ``show_grad``,
            ``show_is_contiguous``, ``show_is_view``, ``show_storage_size``,
            ``plain`` and ``sort_keys``. An unknown option raises a
            ``TypeError``.

    Examples:
        >>> import torch
        >>> from tensordict import TensorDict, set_printoptions
        >>> td = TensorDict(a=torch.zeros(3), batch_size=[3])
        >>> with set_printoptions(verbose=False):
        ...     print(td)
        TensorDict(
            fields={
                a: Tensor(shape=torch.Size([3]))},
            batch_size=torch.Size([3]))

    """

    def __init__(self, *, verbose: bool = True, **kwargs) -> None:
        super().__init__()
        unknown = set(kwargs) - _REPR_OPTIONS_KEYS
        if unknown:
            raise TypeError(
                f"Unknown printoptions: {unknown}. Valid options: {sorted(_REPR_OPTIONS_KEYS)}"
            )
        if not verbose:
            merged = dict(_VERBOSE_FALSE_OVERRIDES)
            merged.update(kwargs)
            kwargs = merged
        self._kwargs = kwargs

    def clone(self) -> set_printoptions:
        return type(self)(**self._kwargs)

    def __enter__(self) -> None:
        self.set()

    def set(self) -> None:
        self._old = dict(_REPR_OPTIONS)
        _REPR_OPTIONS.update(self._kwargs)

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        _REPR_OPTIONS.update(self._old)


def get_printoptions() -> dict:
    """Returns the current TensorDict print options as a dict.

    The keys are the options of :class:`~tensordict.set_printoptions`. See
    :ref:`printing`.

    Examples:
        >>> from tensordict import get_printoptions
        >>> get_printoptions()["show_dtype"]
        True

    """
    return dict(_REPR_OPTIONS)


def _warn_lazy_legacy(
    stacklevel: int, category: type[Warning] = DeprecationWarning
) -> None:
    warnings.warn(
        "The legacy lazy mode, turned on with set_lazy_legacy(True) or "
        "LAZY_LEGACY_OP=1, is deprecated and will be removed in TensorDict 0.17. "
        "Shape operations such as unsqueeze and permute will then always return "
        "a tensordict, and torch.stack will stack densely: use lazy_stack to "
        "build a lazy stack.",
        category,
        stacklevel=stacklevel + 1,
    )


_DEFAULT_LAZY_OP = False
_LAZY_OP = os.environ.get("LAZY_LEGACY_OP")
if _LAZY_OP is not None and _LAZY_OP.lower() in ("y", "yes", "t", "true", "on", "1"):
    # A FutureWarning, as warn_deprecated_env_var emits for the other variables.
    _warn_lazy_legacy(stacklevel=1, category=FutureWarning)


class set_lazy_legacy(_DecoratorContextManager):
    """Sets the behaviour of some methods to a lazy transform.

    .. deprecated:: 0.15
        The legacy lazy mode (``set_lazy_legacy(True)``) is deprecated and will
        be removed in TensorDict 0.17. ``set_lazy_legacy(False)`` will then do
        nothing.
    """

    def __init__(self, mode: bool) -> None:
        super().__init__()
        if mode:
            _warn_lazy_legacy(stacklevel=2)
        self.mode = mode

    def clone(self) -> set_lazy_legacy:
        # used for each call of a decorated function: only the decorator warns
        clone = type(self).__new__(type(self))
        clone.mode = self.mode
        return clone

    def __enter__(self) -> None:
        self.set()

    def set(self) -> None:
        global _LAZY_OP
        self._old_mode = _LAZY_OP
        self._old_env = os.environ.get("LAZY_LEGACY_OP")
        _LAZY_OP = bool(self.mode)
        os.environ["LAZY_LEGACY_OP"] = str(_LAZY_OP)

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        global _LAZY_OP
        _LAZY_OP = self._old_mode
        if self._old_env is None:
            os.environ.pop("LAZY_LEGACY_OP", None)
        else:
            os.environ["LAZY_LEGACY_OP"] = self._old_env


def lazy_legacy(allow_none=False):
    """Returns `True` if lazy representations will be used for selected methods.

    Args:
        allow_none (bool, optional): if ``True`` and the option was never
            set, ``None`` is returned instead of the default (``False``).
            Defaults to ``False``.

    .. deprecated:: 0.15
        The legacy lazy mode is deprecated and will be removed in TensorDict 0.17,
        see :class:`~tensordict.set_lazy_legacy`.
    """
    if _LAZY_OP is None and allow_none:
        return None
    if _LAZY_OP is None:
        return _DEFAULT_LAZY_OP
    return _strtobool(_LAZY_OP) if isinstance(_LAZY_OP, str) else _LAZY_OP


def _legacy_lazy(func):
    if not func.__name__.startswith("_legacy_"):
        raise NameError(
            f"The function name {func.__name__} must start with _legacy_ if it's decorated with _legacy_lazy."
        )
    func.LEGACY = True
    return func


_CAPTURE_NONTENSOR_STACK_REPLACEMENT = (
    "NonTensorStack.data to get the single value of a stack of identical values"
)


def _warn_capture_non_tensor_stack(what: str, stacklevel: int) -> None:
    warn_deprecated(
        what,
        removal="0.17",
        replacement=_CAPTURE_NONTENSOR_STACK_REPLACEMENT,
        stacklevel=stacklevel + 1,
    )


_DEFAULT_CAPTURE_NONTENSOR_STACK = False
_CAPTURE_NONTENSOR_STACK = os.environ.get("CAPTURE_NONTENSOR_STACK")
if _CAPTURE_NONTENSOR_STACK is not None and _CAPTURE_NONTENSOR_STACK.lower() in (
    "y",
    "yes",
    "t",
    "true",
    "on",
    "1",
):
    warn_deprecated_env_var(
        "CAPTURE_NONTENSOR_STACK",
        _CAPTURE_NONTENSOR_STACK,
        removal="0.17",
        replacement=_CAPTURE_NONTENSOR_STACK_REPLACEMENT,
    )


class _set_capture_non_tensor_stack(_DecoratorContextManager):
    # Sets the capture mode without warning: tensordict uses it internally.

    def __init__(self, mode: bool) -> None:
        super().__init__()
        self.mode = mode

    def clone(self) -> _set_capture_non_tensor_stack:
        return type(self)(self.mode)

    def __enter__(self) -> None:
        self.set()

    def set(self) -> None:
        global _CAPTURE_NONTENSOR_STACK
        self._old_mode = _CAPTURE_NONTENSOR_STACK
        self._old_env = os.environ.get("CAPTURE_NONTENSOR_STACK")
        _CAPTURE_NONTENSOR_STACK = bool(self.mode)
        os.environ["CAPTURE_NONTENSOR_STACK"] = str(_CAPTURE_NONTENSOR_STACK)

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        global _CAPTURE_NONTENSOR_STACK
        _CAPTURE_NONTENSOR_STACK = self._old_mode
        if self._old_env is None:
            os.environ.pop("CAPTURE_NONTENSOR_STACK", None)
        else:
            os.environ["CAPTURE_NONTENSOR_STACK"] = self._old_env


class set_capture_non_tensor_stack(_set_capture_non_tensor_stack):
    """Controls whether identical non-tensor data should be captured when stacked.

    .. deprecated:: 0.15
        Capturing identical non-tensor data (``set_capture_non_tensor_stack(True)``
        or ``CAPTURE_NONTENSOR_STACK=1``) is deprecated and will be removed in
        TensorDict 0.17. Use :attr:`NonTensorStack.data <tensordict.NonTensorStack.data>`
        to get the single value of a stack of identical values.
        ``set_capture_non_tensor_stack(False)``, the default, does not warn.
    """

    def __init__(self, mode: bool) -> None:
        super().__init__(mode)
        if mode:
            _warn_capture_non_tensor_stack(
                "set_capture_non_tensor_stack(True)", stacklevel=2
            )

    def clone(self) -> set_capture_non_tensor_stack:
        # used for each call of a decorated function: only the decorator warns
        clone = type(self).__new__(type(self))
        clone.mode = self.mode
        return clone


def capture_non_tensor_stack(allow_none=False):
    """Returns whether stacking identical non-tensor data gives one value instead of a stack.

    See :class:`~tensordict.set_capture_non_tensor_stack`.

    Args:
        allow_none (bool, optional): if ``True`` and the option was never
            set, ``None`` is returned instead of the default (``False``).
            Defaults to ``False``.
    """
    if _CAPTURE_NONTENSOR_STACK is None and allow_none:
        return None
    if _CAPTURE_NONTENSOR_STACK is None:
        return _DEFAULT_CAPTURE_NONTENSOR_STACK
    if (
        isinstance(_CAPTURE_NONTENSOR_STACK, str)
        and _CAPTURE_NONTENSOR_STACK.lower() == "none"
    ):
        return _DEFAULT_CAPTURE_NONTENSOR_STACK
    return (
        _strtobool(_CAPTURE_NONTENSOR_STACK)
        if isinstance(_CAPTURE_NONTENSOR_STACK, str)
        else _CAPTURE_NONTENSOR_STACK
    )


_LIST_TO_STACK_REPLACEMENT = (
    "td.set_non_tensor(key, value) to store a list as one value"
)


def _warn_list_to_stack(what: str, stacklevel: int) -> None:
    warn_deprecated(
        what,
        removal="0.17",
        replacement=_LIST_TO_STACK_REPLACEMENT,
        stacklevel=stacklevel + 1,
    )


_DEFAULT_LIST_TO_STACK = True
_LIST_TO_STACK = os.environ.get("LIST_TO_STACK")
if _LIST_TO_STACK is not None and _LIST_TO_STACK.lower() in (
    "n",
    "no",
    "f",
    "false",
    "off",
    "0",
):
    warn_deprecated_env_var(
        "LIST_TO_STACK",
        _LIST_TO_STACK,
        removal="0.17",
        replacement=_LIST_TO_STACK_REPLACEMENT,
    )


class _set_list_to_stack(_DecoratorContextManager):
    # Sets the list-to-stack mode without warning.

    def __init__(self, mode: bool) -> None:
        super().__init__()
        self.mode = mode

    def clone(self) -> _set_list_to_stack:
        return type(self)(self.mode)

    def __enter__(self) -> None:
        self.set()

    def set(self) -> None:
        global _LIST_TO_STACK
        self._old_mode = _LIST_TO_STACK
        self._old_env = os.environ.get("LIST_TO_STACK")
        _LIST_TO_STACK = bool(self.mode)
        os.environ["LIST_TO_STACK"] = str(_LIST_TO_STACK)

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        global _LIST_TO_STACK
        _LIST_TO_STACK = self._old_mode
        if self._old_env is None:
            os.environ.pop("LIST_TO_STACK", None)
        else:
            os.environ["LIST_TO_STACK"] = self._old_env


class set_list_to_stack(_set_list_to_stack):
    """Context manager and decorator to control list handling in TensorDict.

    .. deprecated:: 0.15
        Turning list-to-stack conversion off (``set_list_to_stack(False)`` or
        ``LIST_TO_STACK=0``) is deprecated and will be removed in TensorDict 0.17.
        Use :meth:`td.set_non_tensor(key, value) <tensordict.TensorDictBase.set_non_tensor>`
        to store a list as one value. ``set_list_to_stack(True)``, the default,
        does not warn.
    """

    def __init__(self, mode: bool) -> None:
        super().__init__(mode)
        if not mode:
            _warn_list_to_stack("set_list_to_stack(False)", stacklevel=2)

    def clone(self) -> set_list_to_stack:
        # used for each call of a decorated function: only the decorator warns
        clone = type(self).__new__(type(self))
        clone.mode = self.mode
        return clone


def list_to_stack(allow_none=False):
    """Returns whether assigning a list to a key stores a :class:`~tensordict.NonTensorStack`.

    See :class:`~tensordict.set_list_to_stack`.

    Args:
        allow_none (bool, optional): if ``True`` and the option was never
            set, ``None`` is returned instead of the default. Defaults to
            ``False``.
    """
    if _LIST_TO_STACK is None and allow_none:
        return None
    if _LIST_TO_STACK is None:
        return _DEFAULT_LIST_TO_STACK
    if isinstance(_LIST_TO_STACK, str) and _LIST_TO_STACK.lower() == "none":
        return _DEFAULT_LIST_TO_STACK
    return (
        _strtobool(_LIST_TO_STACK)
        if isinstance(_LIST_TO_STACK, str)
        else _LIST_TO_STACK
    )


for _name in __all__:
    _obj = globals()[_name]
    if hasattr(_obj, "__module__"):
        _obj.__module__ = "tensordict.utils"
