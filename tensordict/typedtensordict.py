# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import sys
from collections.abc import Callable, Sequence
from typing import Any, ClassVar

import torch
from tensordict._td import TensorDict
from tensordict.base import NO_DEFAULT, TensorDictBase
from tensordict.utils import _as_context_manager, cache

try:
    # Python 3.11+ (PEP 681)
    from typing import dataclass_transform
except ImportError:

    def dataclass_transform(*args, **kwargs):  # noqa: D103
        def _identity(cls):
            return cls

        return _identity


try:
    # Python 3.11+ (PEP 655)
    from typing import NotRequired
except ImportError:
    from typing_extensions import NotRequired  # noqa: F401

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
        _TD_DIR = frozenset(dir(TensorDictBase))
    return _TD_DIR


def _is_not_required(tp: Any) -> bool:
    """Check whether a type annotation is NotRequired[T]."""
    origin = getattr(tp, "__origin__", None)
    if origin is NotRequired:
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

    On Python 3.14+ (PEP 749), annotations are lazily evaluated and may
    not appear in ``cls.__dict__`` until accessed through the descriptor.
    We fall back to ``cls.__annotations__`` which on 3.10+ returns only
    the class's own annotations (triggering lazy evaluation on 3.14).
    """
    own_raw = cls.__dict__.get("__annotations__", None)
    if not own_raw:
        own_raw = getattr(cls, "__annotations__", None)
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
    subclasses (skipping TensorDictBase whose annotations use syntax
    that may not resolve at runtime).

    Returns (expected, required, optional) as frozensets of field names.
    """
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


def _make_init(cls: type) -> Callable:
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
        self._source = TensorDict(
            source=source,
            batch_size=batch_size,
            device=device,
            names=names,
            non_blocking=non_blocking,
            lock=lock,
        )
        if getattr(cls, "_frozen", False):
            self._source.lock_()

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


# ---------------------------------------------------------------------------
# Delegation helpers
# ---------------------------------------------------------------------------


def _make_delegate(method_name: str) -> Callable:
    """Create a method that delegates to ``self._source.<method_name>``."""

    def method(self, *args, **kwargs):
        return getattr(self._source, method_name)(*args, **kwargs)

    method.__name__ = method_name
    method.__qualname__ = f"TypedTensorDict.{method_name}"
    return method


def _make_delegate_wrap(method_name: str) -> Callable:
    """Delegate to ``self._source`` and wrap the TensorDictBase result."""

    def method(self, *args, **kwargs):
        result = getattr(self._source, method_name)(*args, **kwargs)
        if result is None:
            return None
        return type(self)._wrap_td(result)

    method.__name__ = method_name
    method.__qualname__ = f"TypedTensorDict.{method_name}"
    return method


def _make_delegate_inplace(method_name: str) -> Callable:
    """Delegate to ``self._source`` in-place and return ``self``."""

    def method(self, *args, **kwargs):
        getattr(self._source, method_name)(*args, **kwargs)
        return self

    method.__name__ = method_name
    method.__qualname__ = f"TypedTensorDict.{method_name}"
    return method


def _make_delegate_tuple_wrap(method_name: str) -> Callable:
    """Delegate to ``self._source`` and wrap each element of the returned tuple."""

    def method(self, *args, **kwargs):
        results = getattr(self._source, method_name)(*args, **kwargs)
        return tuple(type(self)._wrap_td(r) for r in results)

    method.__name__ = method_name
    method.__qualname__ = f"TypedTensorDict.{method_name}"
    return method


@dataclass_transform()
class _TypedTensorDictMeta(type(TensorDictBase)):
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
                if attr in td_dir:
                    raise AttributeError(
                        f"Field '{attr}' shadows a TensorDictBase attribute. "
                        f"Use TypedTensorDict['shadow'] to allow this."
                    )

        cls.__expected_keys__ = expected
        cls.__required_keys__ = required
        cls.__optional_keys__ = optional

        # Generate properties for fields that clash with TensorDictBase
        # attributes so they override the parent's version.
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


class TypedTensorDict(TensorDictBase, metaclass=_TypedTensorDictMeta):
    """A TensorDictBase subclass with typed field declarations and backend composition.

    TypedTensorDict combines TensorDict's tensor operations with TypedDict-style
    field declarations.  Subclasses declare fields as class annotations and get:

    - Typed attribute access (``state.eta``)
    - Typed construction (``PredictorState(eta=..., X=..., beta=..., batch_size=...)``)
    - Inheritance (``class ObservedState(PredictorState): ...``)
    - ``NotRequired`` fields for optional pipeline branches
    - Full TensorDict interop (``.to()``, ``.clone()``, slicing, ``**state`` spreading)
    - Backend composition via ``from_tensordict`` (H5, Redis, lazy stacks, …)

    Internally, a ``TypedTensorDict`` delegates all storage to an internal
    ``_source`` attribute (a ``TensorDictBase`` instance).  Direct construction
    creates a ``TensorDict`` as ``_source``; ``from_tensordict`` stores the
    given backend directly, enabling zero-copy typed access to any backend.

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
        >>>
        >>> # Wrap an existing TensorDict backend
        >>> td = TensorDict({"eta": torch.randn(5, 3), "X": torch.randn(5, 4)}, batch_size=[5])
        >>> state = PredictorState.from_tensordict(td)
        >>> state.eta.shape
        torch.Size([5, 3])
    """

    _source: TensorDictBase

    @property
    def _is_shared(self):
        return self._source._is_shared

    @_is_shared.setter
    def _is_shared(self, value):
        self._source._is_shared = value

    @property
    def _is_memmap(self):
        return self._source._is_memmap

    @_is_memmap.setter
    def _is_memmap(self, value):
        self._source._is_memmap = value

    # ------------------------------------------------------------------
    # Internal helper: wrap a TensorDictBase without key validation
    # ------------------------------------------------------------------
    @classmethod
    def _wrap_td(cls, td: TensorDictBase) -> TypedTensorDict:
        obj = cls.__new__(cls)
        obj._source = td
        return obj

    # ------------------------------------------------------------------
    # Public constructor: wrap with key validation
    # ------------------------------------------------------------------
    @classmethod
    def from_tensordict(
        cls, td: TensorDictBase, *, check: bool = True
    ) -> TypedTensorDict:
        """Wrap an existing TensorDictBase backend as a TypedTensorDict.

        The ``td`` is stored directly (no copy); mutations through the
        TypedTensorDict are reflected in the original backend and vice versa.

        Args:
            td: Any ``TensorDictBase`` instance (``TensorDict``,
                ``PersistentTensorDict``, ``LazyStackedTensorDict``, etc.).

        Keyword Args:
            check (bool): If ``True`` (default), validate that all required
                fields are present in ``td``.  Set to ``False`` to wrap an
                empty or partially-filled backend (e.g. a pre-allocated
                ``TensorDictStore``); missing fields will raise at access time
                rather than wrap time.

        Raises:
            TypeError: If ``check=True`` and required fields are missing.

        Returns:
            A new ``TypedTensorDict`` instance backed by ``td``.
        """
        if check:
            missing = cls.__required_keys__ - set(td.keys())
            if missing:
                missing_str = ", ".join(sorted(missing))
                raise TypeError(
                    f"{cls.__name__}.from_tensordict() missing required field(s): {missing_str}"
                )
        obj = cls.__new__(cls)
        obj._source = td
        return obj

    # ------------------------------------------------------------------
    # Attribute access for declared fields
    # ------------------------------------------------------------------
    def __getattr__(self, name: str) -> Any:
        expected = type(self).__expected_keys__
        if expected and name in expected:
            try:
                return self._get_str(name, NO_DEFAULT)
            except KeyError:
                raise AttributeError(
                    f"'{type(self).__name__}' has field '{name}' declared but not set"
                ) from None
        # Delegate internal TensorDictBase attributes to the backing source
        # (e.g. _last_op, _last_op_queue, _cache, …).
        try:
            source = self.__dict__["_source"]
        except KeyError:
            pass
        else:
            try:
                return getattr(source, name)
            except AttributeError:
                pass
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute {name!r}"
        )

    def __setattr__(self, name: str, value: Any) -> None:
        expected = type(self).__dict__.get("__expected_keys__", None)
        if expected is not None and name in expected:
            self[name] = value
            return
        object.__setattr__(self, name, value)

    # ------------------------------------------------------------------
    # Properties (delegated to _source)
    # ------------------------------------------------------------------
    @property
    def batch_size(self) -> torch.Size:
        return self._source.batch_size

    @batch_size.setter
    def batch_size(self, new_size: torch.Size) -> None:
        self._source.batch_size = new_size

    @property
    def device(self) -> torch.device | None:
        return self._source.device

    @device.setter
    def device(self, value) -> None:
        self._source.device = value

    @property
    def names(self):
        return self._source.names

    @names.setter
    def names(self, value):
        self._source.names = value

    def _has_names(self) -> bool:
        return self._source._has_names()

    def _set_names(self, names: Sequence[str] | None) -> None:
        self._source._set_names(names)

    def _erase_names(self):
        self._source._erase_names()

    def _rename_subtds(self, value):
        self._source._rename_subtds(value)

    def _change_batch_size(self, new_size: torch.Size) -> None:
        self._source._change_batch_size(new_size)

    @property
    def is_locked(self) -> bool:
        return self._source.is_locked

    @is_locked.setter
    def is_locked(self, value: bool) -> None:
        self._source.is_locked = value

    @_as_context_manager("is_locked")
    def lock_(self, *args, **kwargs):
        self._source.lock_(*args, **kwargs)
        return self

    @_as_context_manager("is_locked")
    def unlock_(self, *args, **kwargs):
        self._source.unlock_(*args, **kwargs)
        self._erase_cache()
        return self

    def _erase_cache(self):
        self._cache = None

    # ------------------------------------------------------------------
    # _select / _exclude: special handling for inplace
    # ------------------------------------------------------------------
    def _select(
        self,
        *keys,
        inplace: bool = False,
        strict: bool = True,
        set_shared: bool = True,
    ):
        if inplace:
            self._source._select(
                *keys, inplace=True, strict=strict, set_shared=set_shared
            )
            return self
        result = self._source._select(
            *keys, inplace=False, strict=strict, set_shared=set_shared
        )
        return type(self)._wrap_td(result)

    def _exclude(self, *keys, inplace: bool = False, set_shared: bool = True):
        if inplace:
            self._source._exclude(*keys, inplace=True, set_shared=set_shared)
            return self
        result = self._source._exclude(*keys, inplace=False, set_shared=set_shared)
        return type(self)._wrap_td(result)

    def _memmap_(self, *args, inplace=True, **kwargs):
        result = self._source._memmap_(*args, inplace=inplace, **kwargs)
        if inplace:
            return self
        if result is None:
            return None
        return type(self)._wrap_td(result)

    def where(self, condition, other, *, out=None, **kwargs):
        unwrapped_out = out._source if isinstance(out, TypedTensorDict) else out
        result = self._source.where(condition, other, out=unwrapped_out, **kwargs)
        if result is None:
            return None
        if out is not None:
            return out
        return type(self)._wrap_td(result)

    # ------------------------------------------------------------------
    # _apply_nest: unwrap *others, delegate, wrap result
    # ------------------------------------------------------------------
    def _apply_nest(
        self,
        fn,
        *others,
        batch_size=None,
        device=NO_DEFAULT,
        names=NO_DEFAULT,
        inplace: bool = False,
        checked: bool = False,
        call_on_nested: bool = False,
        default=NO_DEFAULT,
        named: bool = False,
        nested_keys: bool = False,
        prefix: tuple = (),
        filter_empty: bool | None = None,
        is_leaf=None,
        out=None,
        **constructor_kwargs,
    ):
        unwrapped_others = tuple(
            o._source if isinstance(o, TypedTensorDict) else o for o in others
        )
        unwrapped_out = out._source if isinstance(out, TypedTensorDict) else out
        result = self._source._apply_nest(
            fn,
            *unwrapped_others,
            batch_size=batch_size,
            device=device,
            names=names,
            inplace=inplace,
            checked=checked,
            call_on_nested=call_on_nested,
            default=default,
            named=named,
            nested_keys=nested_keys,
            prefix=prefix,
            filter_empty=filter_empty,
            is_leaf=is_leaf,
            out=unwrapped_out,
            **constructor_kwargs,
        )
        if result is None:
            return None
        if inplace or out is not None:
            return self if inplace else out
        return type(self)._wrap_td(result)

    def _multithread_apply_flat(
        self,
        fn,
        *others,
        call_on_nested: bool = False,
        default=NO_DEFAULT,
        named: bool = False,
        nested_keys: bool = False,
        prefix: tuple = (),
        is_leaf=None,
        executor=None,
        futures=None,
        **kwargs,
    ):
        unwrapped_others = tuple(
            o._source if isinstance(o, TypedTensorDict) else o for o in others
        )
        return self._source._multithread_apply_flat(
            fn,
            *unwrapped_others,
            call_on_nested=call_on_nested,
            default=default,
            named=named,
            nested_keys=nested_keys,
            prefix=prefix,
            is_leaf=is_leaf,
            executor=executor,
            futures=futures,
            **kwargs,
        )

    def _multithread_rebuild(
        self,
        *,
        batch_size=None,
        device=NO_DEFAULT,
        names=NO_DEFAULT,
        inplace: bool = False,
        checked: bool = False,
        out=None,
        filter_empty: bool | None = None,
        executor=None,
        futures=None,
        **kwargs,
    ):
        unwrapped_out = out._source if isinstance(out, TypedTensorDict) else out
        result = self._source._multithread_rebuild(
            batch_size=batch_size,
            device=device,
            names=names,
            inplace=inplace,
            checked=checked,
            out=unwrapped_out,
            filter_empty=filter_empty,
            executor=executor,
            futures=futures,
            **kwargs,
        )
        if result is None:
            return None
        if inplace or out is not None:
            return self if inplace else out
        return type(self)._wrap_td(result)

    # ------------------------------------------------------------------
    # from_dict / from_dict_instance: classmethods and instance methods
    # ------------------------------------------------------------------
    @classmethod
    def from_dict(
        cls,
        input_dict,
        *,
        auto_batch_size=None,
        batch_size=None,
        device=None,
        batch_dims=None,
        names=None,
    ):
        td = TensorDict.from_dict(
            input_dict,
            auto_batch_size=auto_batch_size,
            batch_size=batch_size,
            device=device,
            batch_dims=batch_dims,
            names=names,
        )
        return cls._wrap_td(td)

    def from_dict_instance(
        self,
        input_dict,
        *,
        auto_batch_size=None,
        batch_size=None,
        device=None,
        batch_dims=None,
        names=None,
    ):
        result = self._source.from_dict_instance(
            input_dict,
            auto_batch_size=auto_batch_size,
            batch_size=batch_size,
            device=device,
            batch_dims=batch_dims,
            names=names,
        )
        return type(self)._wrap_td(result)

    # ------------------------------------------------------------------
    # _load_memmap classmethod
    # ------------------------------------------------------------------
    @classmethod
    def _load_memmap(cls, prefix, metadata, device=None, out=None, *, robust_key):
        td = TensorDict._load_memmap(
            prefix,
            metadata,
            device=device,
            out=out,
            robust_key=robust_key,
        )
        return cls._wrap_td(td)

    # ------------------------------------------------------------------
    # _to_module
    # ------------------------------------------------------------------
    def _to_module(self, module, *, inplace=None, return_swap=True, **kwargs):
        return self._source._to_module(
            module,
            inplace=inplace,
            return_swap=return_swap,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Vmap cached methods
    # ------------------------------------------------------------------
    @cache  # noqa: B019
    def _add_batch_dim(self, *, in_dim: int, vmap_level: int):
        result = self._source._add_batch_dim(in_dim=in_dim, vmap_level=vmap_level)
        return type(self)._wrap_td(result)

    @cache  # noqa: B019
    def _remove_batch_dim(self, vmap_level: int, batch_size: int, out_dim: int):
        result = self._source._remove_batch_dim(vmap_level, batch_size, out_dim)
        return type(self)._wrap_td(result)

    @cache  # noqa: B019
    def _maybe_remove_batch_dim(
        self, funcname: str, vmap_level: int, batch_size: int, out_dim: int
    ):
        result = self._source._maybe_remove_batch_dim(
            funcname,
            vmap_level,
            batch_size,
            out_dim,
        )
        return type(self)._wrap_td(result)

    # ------------------------------------------------------------------
    # del_ and rename_key_: in-place, return self
    # ------------------------------------------------------------------
    def del_(self, key):
        self._source.del_(key)
        return self

    def rename_key_(self, old_key, new_key, safe=False):
        self._source.rename_key_(old_key, new_key, safe=safe)
        return self

    # ------------------------------------------------------------------
    # _new_unsafe: needed for torch.stack and other operations
    # ------------------------------------------------------------------
    @classmethod
    def _new_unsafe(
        cls,
        source=None,
        batch_size=None,
        device=None,
        names=None,
        non_blocking=None,
        lock=False,
        nested=True,
    ):
        td = TensorDict._new_unsafe(
            source=source,
            batch_size=batch_size,
            device=device,
            names=names,
            non_blocking=non_blocking,
            lock=lock,
            nested=nested,
        )
        return cls._wrap_td(td)

    # ------------------------------------------------------------------
    # _convert_to_tensordict
    # ------------------------------------------------------------------
    def _convert_to_tensordict(self, dict_value, non_blocking=None):
        return TensorDict(
            dict_value,
            batch_size=self.batch_size,
            device=self.device,
            names=(
                self._source._maybe_names()
                if hasattr(self._source, "_maybe_names")
                else None
            ),
            lock=self.is_locked,
            non_blocking=non_blocking,
        )


# ======================================================================
# Install delegated methods via helper factories
# ======================================================================

# Direct delegation: return value from _source as-is
_DIRECT_DELEGATES = [
    "_get_str",
    "_get_tuple",
    "_set_str",
    "_set_tuple",
    "_set_at_str",
    "_set_at_tuple",
    "_stack_onto_",
    "keys",
    "entry_class",
    "_check_is_shared",
    "_check_device",
    "is_contiguous",
    "popitem",
    "_cast_reduction",
    "__setitem__",
    "__ne__",
    "__xor__",
    "__or__",
    "__eq__",
    "__ge__",
    "__gt__",
    "__le__",
    "__lt__",
    "all",
    "any",
    "make_memmap",
    "make_memmap_from_storage",
    "make_memmap_from_tensor",
]

# Wrap result: delegate to _source, wrap TensorDictBase result
_WRAP_DELEGATES = [
    "_clone",
    "expand",
    "reshape",
    "_unsqueeze",
    "_squeeze",
    "_view",
    "_transpose",
    "_permute",
    "_repeat",
    "repeat_interleave",
    "contiguous",
    "masked_fill",
    "masked_select",
    "_index_tensordict",
]

# In-place: delegate to _source, return self
_INPLACE_DELEGATES = [
    "share_memory_",
    "detach_",
    "masked_fill_",
]

# Tuple wrap: delegate to _source, wrap each element
_TUPLE_WRAP_DELEGATES = [
    "_unbind",
    "chunk",
    "split",
]

for _name in _DIRECT_DELEGATES:
    setattr(TypedTensorDict, _name, _make_delegate(_name))

for _name in _WRAP_DELEGATES:
    setattr(TypedTensorDict, _name, _make_delegate_wrap(_name))

for _name in _INPLACE_DELEGATES:
    setattr(TypedTensorDict, _name, _make_delegate_inplace(_name))

for _name in _TUPLE_WRAP_DELEGATES:
    setattr(TypedTensorDict, _name, _make_delegate_tuple_wrap(_name))
