# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import os
from typing import Any

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


def _strtobool(val):
    val = val.lower()
    if val in ("y", "yes", "t", "true", "on", "1"):
        return 1
    if val in ("n", "no", "f", "false", "off", "0"):
        return 0
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
    """Controls which attributes appear in TensorDict's ``__repr__`` output."""

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
    """Returns the current TensorDict print options as a dict."""
    return dict(_REPR_OPTIONS)


_DEFAULT_LAZY_OP = False
_LAZY_OP = os.environ.get("LAZY_LEGACY_OP")


class set_lazy_legacy(_DecoratorContextManager):
    """Sets the behaviour of some methods to a lazy transform."""

    def __init__(self, mode: bool) -> None:
        super().__init__()
        self.mode = mode

    def clone(self) -> set_lazy_legacy:
        return type(self)(self.mode)

    def __enter__(self) -> None:
        self.set()

    def set(self) -> None:
        global _LAZY_OP
        self._old_mode = _LAZY_OP
        _LAZY_OP = bool(self.mode)
        os.environ["LAZY_LEGACY_OP"] = str(_LAZY_OP)

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        global _LAZY_OP
        _LAZY_OP = self._old_mode
        os.environ["LAZY_LEGACY_OP"] = str(_LAZY_OP)


def lazy_legacy(allow_none=False):
    """Returns `True` if lazy representations will be used for selected methods."""
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


_DEFAULT_CAPTURE_NONTENSOR_STACK = False
_CAPTURE_NONTENSOR_STACK = os.environ.get("CAPTURE_NONTENSOR_STACK")


class set_capture_non_tensor_stack(_DecoratorContextManager):
    """Controls whether identical non-tensor data should be captured when stacked."""

    def __init__(self, mode: bool) -> None:
        super().__init__()
        self.mode = mode

    def clone(self) -> set_capture_non_tensor_stack:
        return type(self)(self.mode)

    def __enter__(self) -> None:
        self.set()

    def set(self) -> None:
        global _CAPTURE_NONTENSOR_STACK
        self._old_mode = _CAPTURE_NONTENSOR_STACK
        _CAPTURE_NONTENSOR_STACK = bool(self.mode)
        os.environ["CAPTURE_NONTENSOR_STACK"] = str(_CAPTURE_NONTENSOR_STACK)

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        global _CAPTURE_NONTENSOR_STACK
        _CAPTURE_NONTENSOR_STACK = self._old_mode
        os.environ["CAPTURE_NONTENSOR_STACK"] = str(_CAPTURE_NONTENSOR_STACK)


def capture_non_tensor_stack(allow_none=False):
    """Get the current setting for capturing non-tensor stacks."""
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


_DEFAULT_LIST_TO_STACK = "1"
_LIST_TO_STACK = os.environ.get("LIST_TO_STACK")


class set_list_to_stack(_DecoratorContextManager):
    """Context manager and decorator to control list handling in TensorDict."""

    def __init__(self, mode: bool) -> None:
        super().__init__()
        self.mode = mode

    def clone(self) -> set_list_to_stack:
        return type(self)(self.mode)

    def __enter__(self) -> None:
        self.set()

    def set(self) -> None:
        global _LIST_TO_STACK
        self._old_mode = _LIST_TO_STACK
        _LIST_TO_STACK = bool(self.mode)
        os.environ["LIST_TO_STACK"] = str(_LIST_TO_STACK)

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        global _LIST_TO_STACK
        _LIST_TO_STACK = self._old_mode
        os.environ["LIST_TO_STACK"] = str(_LIST_TO_STACK)


def list_to_stack(allow_none=False):
    """Retrieves the current setting for list-to-stack conversion in TensorDict."""
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
