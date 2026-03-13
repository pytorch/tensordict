# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import typing
from collections.abc import Sequence
from typing import Any, ClassVar

import torch
from tensordict._td import TensorDict
from tensordict.base import NO_DEFAULT

try:
    from typing import dataclass_transform
except ImportError:

    def dataclass_transform(*args, **kwargs):  # noqa: D103
        def _identity(cls):
            return cls

        return _identity


try:
    from typing import NotRequired
except ImportError:
    from typing_extensions import NotRequired  # noqa: F401

# Fields on TensorDict that are safe to shadow without the "shadow" option.
_SAFE_SHADOW_NAMES = frozenset({"_is_non_tensor", "data"})

# Annotation names that are class-level metadata, not user fields.
_META_FIELDS = frozenset(
    {
        "__expected_keys__",
        "__required_keys__",
        "__optional_keys__",
        "_shadow",
        "_frozen",
        "_autocast",
        "_nocast",
        "_tensor_only",
    }
)

_OPTIONS = ("shadow", "frozen", "autocast", "nocast", "tensor_only")

_TD_DIR: frozenset[str] | None = None


def _get_td_dir() -> frozenset[str]:
    global _TD_DIR
    if _TD_DIR is None:
        _TD_DIR = frozenset(dir(TensorDict))
    return _TD_DIR


def _is_not_required(tp: Any) -> bool:
    """Check whether a type annotation is NotRequired[T]."""
    origin = getattr(tp, "__origin__", None)
    if origin is typing.NotRequired:
        return True
    origin = getattr(tp, "__class__", None)
    if origin is not None and getattr(origin, "__name__", None) == "_SpecialForm":
        return str(tp).startswith("typing.NotRequired") or str(tp).startswith(
            "typing_extensions.NotRequired"
        )
    return False


def _is_classvar(tp: Any) -> bool:
    """Check whether a type annotation is ClassVar[T]."""
    origin = getattr(tp, "__origin__", None)
    return origin is ClassVar


def _resolve_own_hints(cls: type) -> dict[str, Any]:
    """Resolve annotations declared directly on *cls* (not inherited).

    Manually resolves string annotations (from ``from __future__ import
    annotations``) using the module's global namespace instead of calling
    ``get_type_hints``, which traverses the full MRO and can fail on
    TensorDict's internal annotations.
    """
    import sys

    own_raw = cls.__dict__.get("__annotations__", {})
    if not own_raw:
        return {}

    module = getattr(cls, "__module__", None)
    globalns = vars(sys.modules[module]) if module and module in sys.modules else {}
    localns = {cls.__name__: cls}

    resolved: dict[str, Any] = {}
    for name, ann in own_raw.items():
        if isinstance(ann, str):
            try:
                resolved[name] = eval(ann, globalns, localns)  # noqa: S307
            except Exception:
                resolved[name] = ann
        else:
            resolved[name] = ann
    return resolved


def _collect_fields(cls: type) -> tuple[frozenset[str], frozenset[str], frozenset[str]]:
    """Collect field annotations from the class, separating required and optional.

    Walks the MRO but only collects annotations from TypedTensorDict
    subclasses (skipping TensorDict/TensorDictBase whose annotations
    use syntax that may not resolve at runtime).

    Returns (expected, required, optional) as frozensets of field names.
    """
    # Merge resolved annotations from all TypedTensorDict classes in the
    # MRO (reverse order so child annotations override parent ones).
    merged: dict[str, Any] = {}
    for base in reversed(cls.__mro__):
        if base is TypedTensorDict or not issubclass(base, TypedTensorDict):
            continue
        merged.update(_resolve_own_hints(base))

    expected: set[str] = set()
    required: set[str] = set()
    optional: set[str] = set()

    for field_name, field_type in merged.items():
        if field_name.startswith("_"):
            continue
        if field_name in _META_FIELDS:
            continue
        if _is_classvar(field_type):
            continue
        expected.add(field_name)
        if _is_not_required(field_type):
            optional.add(field_name)
        else:
            required.add(field_name)

    return frozenset(expected), frozenset(required), frozenset(optional)


def _make_init(cls: type) -> callable:
    """Generate an __init__ for a TypedTensorDict subclass.

    The generated __init__ accepts declared field names as keyword arguments,
    plus the standard TensorDict keyword arguments (batch_size, device, names,
    non_blocking, lock).
    """
    required_keys = cls.__required_keys__
    expected_keys = cls.__expected_keys__

    def __init__(
        self,
        *,
        batch_size: Sequence[int] | torch.Size | int | None = None,
        device: torch.device | str | int | None = None,
        names: Sequence[str] | None = None,
        non_blocking: bool | None = None,
        lock: bool = False,
        **kwargs: Any,
    ) -> None:
        missing = required_keys - kwargs.keys()
        if missing:
            missing_str = ", ".join(sorted(missing))
            raise TypeError(
                f"{type(self).__name__}() missing required field(s): {missing_str}"
            )
        source = {k: v for k, v in kwargs.items() if k in expected_keys}
        extra = {k: v for k, v in kwargs.items() if k not in expected_keys}
        if extra:
            extra_str = ", ".join(sorted(extra))
            raise TypeError(
                f"{type(self).__name__}() got unexpected field(s): {extra_str}"
            )
        TensorDict.__init__(
            self,
            source=source,
            batch_size=batch_size,
            device=device,
            names=names,
            non_blocking=non_blocking,
            lock=lock,
        )
        if getattr(cls, "_frozen", False):
            self.lock_()

    __init__.__qualname__ = f"{cls.__qualname__}.__init__"
    __init__.__name__ = "__init__"
    return __init__


def _install_shadow_property(cls: type, field_name: str) -> None:
    """Install a property on *cls* that routes attribute access to the TensorDict entry."""

    def _getter(self, _key=field_name):
        return self._get_str(_key, NO_DEFAULT)

    def _setter(self, value, _key=field_name):
        self[_key] = value

    setattr(cls, field_name, property(_getter, _setter))


@dataclass_transform()
class _TypedTensorDictMeta(type(TensorDict)):
    def __new__(
        mcs,
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, Any],
        shadow: bool | None = None,
        frozen: bool | None = None,
        autocast: bool | None = None,
        nocast: bool | None = None,
        tensor_only: bool | None = None,
        **kwargs: Any,
    ):
        cls = super().__new__(mcs, name, bases, namespace, **kwargs)

        # Inherit options from parent classes, override with explicit values
        option_vals = {
            "shadow": shadow,
            "frozen": frozen,
            "autocast": autocast,
            "nocast": nocast,
            "tensor_only": tensor_only,
        }
        for opt, val in option_vals.items():
            if val is None:
                val = getattr(cls, f"_{opt}", False)
            setattr(cls, f"_{opt}", bool(val))

        if name == "TypedTensorDict":
            cls.__expected_keys__ = frozenset()
            cls.__required_keys__ = frozenset()
            cls.__optional_keys__ = frozenset()
            return cls

        # Intermediate option classes (created by __getitem__) have no
        # user-declared annotations to process.
        if name.startswith("TypedTensorDict_"):
            cls.__expected_keys__ = frozenset()
            cls.__required_keys__ = frozenset()
            cls.__optional_keys__ = frozenset()
            return cls

        expected, required, optional = _collect_fields(cls)

        if not cls._shadow:
            td_dir = _get_td_dir()
            for attr in expected:
                if attr in td_dir and attr not in _SAFE_SHADOW_NAMES:
                    raise AttributeError(
                        f"Field '{attr}' shadows a TensorDict attribute. "
                        f"Use TypedTensorDict['shadow'] to allow this."
                    )

        cls.__expected_keys__ = expected
        cls.__required_keys__ = required
        cls.__optional_keys__ = optional

        # Generate properties for fields that clash with TensorDict attributes
        # so they override the parent's version. When shadow=False, only
        # _SAFE_SHADOW_NAMES fields (like "data") get properties; the rest
        # were already rejected above.
        td_dir = _get_td_dir()
        for attr in expected:
            if attr in td_dir:
                _install_shadow_property(cls, attr)

        if "__init__" not in namespace:
            cls.__init__ = _make_init(cls)

        return cls

    def __getitem__(cls, item):
        """Create an intermediate base class with options enabled.

        Usage: TypedTensorDict["shadow"] or TypedTensorDict["shadow", "frozen"]
        """
        if not isinstance(item, tuple):
            item = (item,)
        for opt in item:
            if opt not in _OPTIONS:
                raise ValueError(
                    f"Unknown TypedTensorDict option: {opt!r}. "
                    f"Valid options: {', '.join(_OPTIONS)}"
                )
        suffix = "_".join(item)
        return _TypedTensorDictMeta(
            f"TypedTensorDict_{suffix}",
            (cls,),
            {},
            **{opt: True for opt in item},
        )


class TypedTensorDict(TensorDict, metaclass=_TypedTensorDictMeta):
    """A TensorDict subclass with typed field declarations.

    TypedTensorDict combines TensorDict's tensor operations with TypedDict-style
    field declarations.  Subclasses declare fields as class annotations and get:

    - Typed attribute access (``state.eta``)
    - Typed construction (``PredictorState(eta=..., X=..., beta=..., batch_size=...)``)
    - Inheritance (``class ObservedState(PredictorState): ...``)
    - ``NotRequired`` fields for optional pipeline branches
    - Full TensorDict interop (``.to()``, ``.clone()``, slicing, ``**state`` spreading)

    Examples:
        >>> import torch
        >>> from tensordict import TypedTensorDict
        >>> from torch import Tensor
        >>>
        >>> class PredictorState(TypedTensorDict):
        ...     eta: Tensor
        ...     X: Tensor
        ...
        >>> state = PredictorState(eta=torch.randn(5, 3), X=torch.randn(5, 4), batch_size=[5])
        >>> state.eta.shape
        torch.Size([5, 3])
        >>> state["X"].shape
        torch.Size([5, 4])
    """

    def __getattr__(self, name: str) -> Any:
        # Only intercept declared field names; everything else goes through
        # the normal MRO (and eventually raises AttributeError).
        expected = type(self).__expected_keys__
        if expected and name in expected:
            try:
                return self._get_str(name, NO_DEFAULT)
            except KeyError:
                raise AttributeError(
                    f"'{type(self).__name__}' has field '{name}' declared but not set"
                ) from None
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute {name!r}"
        )

    def __setattr__(self, name: str, value: Any) -> None:
        expected = type(self).__dict__.get("__expected_keys__", None)
        if expected is not None and name in expected:
            self[name] = value
            return
        object.__setattr__(self, name, value)
