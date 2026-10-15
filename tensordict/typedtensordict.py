# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import sys
import warnings
from collections.abc import Callable, Sequence
from typing import Any, ClassVar

import torch
from tensordict._lazy import LazyStackedTensorDict
from tensordict._pytree import _register_td_node
from tensordict._td import _StateFromSource, TensorDict
from tensordict.base import (
    _is_tensor_collection,
    _PACKAGE_DIR,
    _register_tensor_class,
    NO_DEFAULT,
    TensorDictBase,
)
from tensordict.utils import (
    _as_context_manager,
    _cache_while_locked,
    _is_non_tensor,
    _is_tensorclass,
    _unravel_key_to_tuple,
)
from torch.compiler import is_compiling

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

__all__ = ["TypedTensorDict"]

# Annotation names that are class-level metadata, not user fields.
_META_FIELDS = frozenset(
    {
        "__expected_keys__",
        "__required_keys__",
        "__optional_keys__",
        "__field_defaults__",
        "_shadow",
        "_frozen",
        "_autocast",
        "_nocast",
        "_tensor_only",
    }
)

_OPTIONS = ("shadow", "frozen", "autocast", "nocast", "tensor_only")
# TensorClass options that TypedTensorDict accepts but ignores.
_DEPRECATED_OPTIONS = ("autocast", "nocast", "tensor_only")

_TD_DIR: frozenset[str] | None = None


def _get_td_dir() -> frozenset[str]:
    global _TD_DIR
    if _TD_DIR is None:
        _TD_DIR = frozenset(dir(TensorDictBase))
    return _TD_DIR


def _warn_deprecated_option(option: str) -> None:
    """Warn that ``option`` has no effect, at the caller of the metaclass method."""
    warnings.warn(
        f"The TypedTensorDict option {option!r} has no effect and is deprecated. "
        "It will be removed in TensorDict 0.17. TensorClass supports this option.",
        DeprecationWarning,
        stacklevel=3,
    )


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


def _collect_fields(
    cls: type,
) -> tuple[frozenset[str], frozenset[str], frozenset[str], dict[str, Any]]:
    """Collect field annotations from the class, separating required and optional.

    Walks the MRO but only collects annotations from TypedTensorDict
    subclasses (skipping TensorDictBase whose annotations use syntax
    that may not resolve at runtime).

    Returns (expected, required, optional, defaults). The first three values
    are frozensets of field names and defaults maps optional fields to their
    class-level default.
    """
    merged: dict[str, Any] = {}
    defaults: dict[str, Any] = {}
    for base in reversed(cls.__mro__):
        if base is TypedTensorDict or not issubclass(base, TypedTensorDict):
            continue
        own_hints = _resolve_own_hints(base)
        merged.update(own_hints)
        defaults.update(base.__dict__.get("__field_defaults__", {}))
        for field_name in own_hints:
            default = base.__dict__.get(field_name, NO_DEFAULT)
            if default is not NO_DEFAULT and not isinstance(default, property):
                defaults[field_name] = default

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
        if _is_not_required(field_type) or field_name in defaults:
            optional.add(field_name)
        else:
            required.add(field_name)

    defaults = {key: value for key, value in defaults.items() if key in expected}
    return frozenset(expected), frozenset(required), frozenset(optional), defaults


def _warn_missing_required(
    cls: type, td: TensorDictBase, call: str, opt_out: str
) -> None:
    """Warns if ``td``, the backend of a new ``cls`` built by ``call``, lacks a required field.

    ``call`` names the call, with ``{}`` for the class name.
    """
    # the backing dict: a keys() view costs about 2 us more per call.
    # from_dict passes other tensor collections through unchanged.
    keys = td._tensordict if type(td) is TensorDict else td.keys()
    missing = cls.__required_keys__.difference(keys)
    # Dynamo cannot trace warnings.warn: a compiled call stays silent.
    if not missing or is_compiling():
        return
    # point to the first frame outside tensordict: the user's call
    frame, stacklevel = sys._getframe(1), 2
    while frame is not None and frame.f_code.co_filename.startswith(_PACKAGE_DIR):
        frame, stacklevel = frame.f_back, stacklevel + 1
    warnings.warn(
        f"{call.format(cls.__name__)} returns an instance of {cls.__name__} that "
        f"misses the required field(s) {', '.join(sorted(missing))}. Starting "
        "with TensorDict 0.17, the default will change to raising the TypeError "
        f"that the constructor raises. Pass these fields, or {opt_out} to keep "
        "the instance without them.",
        FutureWarning,
        stacklevel=stacklevel,
    )


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
        _source: dict | None = None,
        *,
        batch_size: Sequence[int] | torch.Size | int | None = None,
        device: torch.device | str | int | None = None,
        names: Sequence[str] | None = None,
        non_blocking: bool | None = None,
        lock: bool = False,
        **kwargs: Any,
    ):
        if _source is not None:
            self._source = TensorDict(
                source=_source,
                batch_size=batch_size,
                device=device,
                names=names,
                non_blocking=non_blocking,
                lock=lock,
            )
            _warn_missing_required(
                cls,
                self._source,
                "{}(dict)",
                "use from_dict(..., check=False)",
            )
            return
        missing = required_keys - kwargs.keys()
        if missing:
            missing_str = ", ".join(sorted(missing))
            raise TypeError(
                f"{type(self).__name__}() missing required field(s): {missing_str}"
            )
        extra = kwargs.keys() - expected_keys
        if extra:
            extra_str = ", ".join(sorted(extra))
            raise TypeError(
                f"{type(self).__name__}() got unexpected field(s): {extra_str}"
            )
        source = {k: v for k, v in kwargs.items() if k in expected_keys}
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


def _install_field_property(
    cls: type, field_name: str, default: Any = NO_DEFAULT
) -> None:
    """Install a property that routes attribute access to the TensorDict entry."""

    def _getter(self, _key=field_name, _default=default):
        try:
            return self._get_str(_key, NO_DEFAULT)
        except KeyError:
            if _default is not NO_DEFAULT:
                return _default
            raise AttributeError(
                f"'{type(self).__name__}' has field '{_key}' declared but not set"
            ) from None

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
        if result is self._source:
            # The source returned itself (inplace=True, or a no-op such as
            # reshape to the same shape): return self, as TensorDict does.
            return self
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


def _reads_copies(td: TensorDictBase) -> bool:
    """Whether ``td``, or a tensordict that it holds or stacks, reads copies of its tensors."""
    if _is_tensorclass(type(td)):
        td = td._tensordict
    if td._reads_return_copies:
        return True
    if isinstance(td, LazyStackedTensorDict):
        return any(_reads_copies(member) for member in td.tensordicts)
    return any(
        _reads_copies(value)
        for value in td.values()
        if _is_tensor_collection(type(value)) and not _is_non_tensor(type(value))
    )


def _has_exclusive_keys(td: LazyStackedTensorDict) -> bool:
    """Whether the lazy stack ``td``, or a lazy stack that it stacks, has keys that only some of its members hold."""
    # the keys of a member that is a lazy stack are those all its members hold
    return td._has_exclusive_keys or any(
        _has_exclusive_keys(member)
        for member in td.tensordicts
        if isinstance(member, LazyStackedTensorDict)
    )


def _select_shared(value: Any, keys: list) -> Any:
    """The entries under ``keys`` of a tensor-collection operand, typed or not, for a method that runs on those keys of a lazy stack; other operands as they are."""
    if isinstance(value, TypedTensorDict):
        value = value._source
    if _is_tensor_collection(type(value)) and not _is_non_tensor(type(value)):
        return value.select(*keys)
    return value


def _make_delegate_tuple_wrap(method_name: str) -> Callable:
    """Delegate to ``self._source`` and wrap each element of the returned tuple."""

    def method(self, *args, **kwargs):
        results = getattr(self._source, method_name)(*args, **kwargs)
        return tuple(type(self)._wrap_td(r) for r in results)

    method.__name__ = method_name
    method.__qualname__ = f"TypedTensorDict.{method_name}"
    return method


@dataclass_transform(kw_only_default=True)
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
            elif opt in _DEPRECATED_OPTIONS and not name.startswith("TypedTensorDict_"):
                # __getitem__ warns for the options of the classes it creates
                _warn_deprecated_option(opt)
            setattr(cls, f"_{opt}", bool(val))

        if name == "TypedTensorDict":
            cls.__expected_keys__ = frozenset()
            cls.__required_keys__ = frozenset()
            cls.__optional_keys__ = frozenset()
            cls.__field_defaults__ = {}
            return cls

        # Intermediate option classes (created by __getitem__) have no
        # user-declared annotations to process.
        if name.startswith("TypedTensorDict_"):
            cls.__expected_keys__ = frozenset()
            cls.__required_keys__ = frozenset()
            cls.__optional_keys__ = frozenset()
            cls.__field_defaults__ = {}
            return cls

        expected, required, optional, defaults = _collect_fields(cls)

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
        cls.__field_defaults__ = defaults

        # Properties avoid the generic __getattr__ path for declared fields.
        # They also override TensorDictBase attributes when shadowing is enabled.
        for attr in expected:
            _install_field_property(cls, attr, defaults.get(attr, NO_DEFAULT))

        if "__init__" not in namespace:
            cls.__init__ = _make_init(cls)

        # Register with pytree so Dynamo can flatten/unflatten instances
        # during torch.compile tracing.  Without this, Dynamo falls back to
        # generic Python tracing for TensorDict subclass operations and hits
        # graph breaks (e.g. _has_mps -> torch.backends.mps.is_available()).
        _register_tensor_class(cls)
        try:
            _register_td_node(cls)
        except ValueError:
            pass

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
            if opt in _DEPRECATED_OPTIONS:
                _warn_deprecated_option(opt)
        suffix = "_".join(item)
        return _TypedTensorDictMeta(
            f"TypedTensorDict_{suffix}",
            (cls,),
            {},
            **{opt: True for opt in item},
        )


class TypedTensorDict(_StateFromSource, TensorDictBase, metaclass=_TypedTensorDictMeta):
    """A TensorDictBase subclass with typed field declarations and backend composition.

    .. warning::
        ``TypedTensorDict`` is experimental: its API can change in a minor
        release without a deprecation cycle.

    TypedTensorDict combines TensorDict's tensor operations with TypedDict-style
    field declarations. Subclasses declare fields as class annotations and get:

    - Typed attribute access (``state.eta``)
    - Typed construction (``PredictorState(eta=..., X=..., beta=..., batch_size=...)``)
    - Inheritance (``class ObservedState(PredictorState): ...``)
    - Default-valued fields for optional pipeline branches
    - Full TensorDict interop (``.to()``, ``.clone()``, slicing, ``**state`` spreading)
    - Backend composition via ``from_tensordict`` (H5, Redis, lazy stacks, and others)

    Use it instead of a :class:`~tensordict.TensorClass` when you need a
    :class:`~tensordict.TensorDictBase` subclass: ``state["key"]`` access,
    ``**state`` unpacking, or typed access to an existing backend. Use a
    :class:`~tensordict.TensorClass` instead when fields hold non-tensor
    values that must come back as plain Python objects.
    :ref:`choosing-a-container` compares the containers, and
    :ref:`typed-tensordict-guide` is the user guide.

    Internally, a ``TypedTensorDict`` delegates all storage to an internal
    ``_source`` attribute (a ``TensorDictBase`` instance). Direct construction
    creates a ``TensorDict`` as ``_source``. ``from_tensordict`` stores the
    given backend directly, so the typed access to that backend makes no copy.

    Examples:
        >>> import torch
        >>> from tensordict import TensorDict, TypedTensorDict
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
    _wrapped_attr = "_source"

    # Override: the flag lives on the source.
    @_StateFromSource._is_shared.setter
    def _is_shared(self, value):
        self._source._is_shared = value

    # Override: the flag lives on the source.
    @_StateFromSource._is_memmap.setter
    def _is_memmap(self, value):
        self._source._is_memmap = value

    @property
    def _reads_return_copies(self):
        # a lazy stack builds each entry that _get_str returns by stacking
        return self._source._reads_return_copies or isinstance(
            self._source, LazyStackedTensorDict
        )

    # Override: the members of a lazy source keep their batch size, so an
    # in-place shape op that changes the batch size raises before it writes,
    # also through a tensordict that holds self.
    @property
    def _inplace_batch_size_change(self):
        source = self._source
        return source._inplace_batch_size_change and not source._lazy

    # Override: a lazy source writes a gather with an index of the batch shape
    # into its members; the lazy stacks refuse it.
    @property
    def _gather_inplace_warns(self):
        return isinstance(self._source, LazyStackedTensorDict)

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
        expected = type(self).__expected_keys__
        if name in expected:
            self[name] = value
            return
        object.__setattr__(self, name, value)

    # ------------------------------------------------------------------
    # Properties (delegated to _source)
    # ------------------------------------------------------------------
    # Override: the batch size of the source.
    @property
    def batch_size(self) -> torch.Size:
        return self._source.batch_size

    @batch_size.setter
    def batch_size(self, new_size: torch.Size) -> None:
        self._source.batch_size = new_size

    # Override: the names of the source.
    @property
    def names(self):
        return self._source.names

    @names.setter
    def names(self, value):
        self._source.names = value

    def _set_names(self, names: Sequence[str] | None) -> None:
        self._source._set_names(names)

    def _erase_names(self):
        self._source._erase_names()

    def _rename_subtds(self, value):
        self._source._rename_subtds(value)

    def _change_batch_size(self, new_size: torch.Size) -> None:
        self._source._change_batch_size(new_size)

    # Override: locks the source; the base method builds the wrong lock graph.
    @_as_context_manager("is_locked")
    def lock_(self, *args, **kwargs):
        self._source.lock_(*args, **kwargs)
        return self

    # Override: unlocks the source; the base method builds the wrong lock graph.
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
    # Override: selects in _source and wraps the result.
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

    # Override: excludes in _source and wraps the result.
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

    # Override: the source's where, with a typed ``out`` unwrapped and the
    # result wrapped.
    def _where(self, condition, other, *, out, pad, update_batch_size):
        unwrapped_out = out._source if isinstance(out, TypedTensorDict) else out
        result = self._source.where(
            condition,
            other,
            out=unwrapped_out,
            pad=pad,
            update_batch_size=update_batch_size,
        )
        if out is not None:
            return out
        return type(self)._wrap_td(result)

    # Override: the source's in-place methods write into the members of a lazy
    # stack, which writing back the leaves read from self cannot do when they
    # are nested, also where the members read copies, such as h5 files (#2773).
    # Other sources take the inherited path. On a stack with exclusive keys,
    # also in a stack that it stacks (#2759), which the method leaves
    # unchanged, and where only the entries of the members read copies, it
    # changes a copy of the shared leaves, each read as one stacked tensor,
    # which the source writes into its members (#2739). Members that differ
    # in shape change their shared leaves through a selection of them.
    def _inplace_on_copies(self, method, *args, **kwargs):
        source = self._source
        if not isinstance(source, LazyStackedTensorDict):
            return super()._inplace_on_copies(method, *args, **kwargs)
        # the public name of a hook such as _zero_
        name = method.__name__.lstrip("_")
        if not _has_exclusive_keys(source) and (
            source._reads_return_copies or not _reads_copies(source)
        ):
            result = getattr(source, name)(*args, **kwargs)
            if result is not source:
                # broadcasting self returns a new tensordict and writes nothing
                return result
            return self
        try:
            items = list(source.items(True, True))
        except RuntimeError:
            # members that differ in shape cannot be read as one tensor: the
            # method runs on the shared leaves of each member, which a
            # selection holds (#2739), unless they read copies
            if _reads_copies(source):
                raise
            keys = list(source.keys(True, True))
            args = [_select_shared(a, keys) for a in args]
            kwargs = {k: _select_shared(v, keys) for k, v in kwargs.items()}
            shared = source.select(*keys)
            result = getattr(shared, name)(*args, **kwargs)
            # broadcasting self returns a new tensordict and writes nothing
            return self if result is shared else result
        leaves = TensorDict(batch_size=self.batch_size, device=self.device)
        for key, value in items:
            if isinstance(value, torch.Tensor):
                leaves._set_tuple(
                    _unravel_key_to_tuple(key),
                    value,
                    inplace=False,
                    validated=True,
                    non_blocking=False,
                )
        # a typed operand counts as its source, as for the dense copy
        args = [a._source if isinstance(a, TypedTensorDict) else a for a in args]
        kwargs = {
            k: v._source if isinstance(v, TypedTensorDict) else v
            for k, v in kwargs.items()
        }
        result = method(leaves, *args, **kwargs)
        if result is not leaves:
            # broadcasting self returns a new tensordict and writes nothing
            return result
        keys, values = leaves._items_list(True, True)
        self._write_back(keys, values, name)
        return self

    # Override: a lazy source indexes its members, so the result holds views of
    # them, as indexing the stack does; the base method would stack copies of
    # the flat leaves (#2759). Other sources index as before.
    def _index_tensordict(self, index, new_batch_size=None, names=None):
        source = self._source
        if (
            isinstance(source, LazyStackedTensorDict)
            and new_batch_size is None
            and names is None
        ):
            result = source[index]
        else:
            result = source._index_tensordict(index, new_batch_size, names)
        return type(self)._wrap_td(result)

    # Override: a lazy source splits the input over its members, which write
    # it in place, also when they are locked or nested, are files or lazy
    # stacks themselves (#2771, #2773); members that hold files update a
    # stacked copy instead. The keys that only some members hold are left
    # unchanged, as in the other in-place methods (#2739).
    def _update_(self, input_dict_or_td, *, clone, non_blocking, keys_to_update):
        source = self._source
        if not isinstance(source, LazyStackedTensorDict):
            return super()._update_(
                input_dict_or_td,
                clone=clone,
                non_blocking=non_blocking,
                keys_to_update=keys_to_update,
            )
        if isinstance(input_dict_or_td, TypedTensorDict):
            input_dict_or_td = input_dict_or_td._source
        if keys_to_update is None and _has_exclusive_keys(source):
            keys_to_update = [
                _unravel_key_to_tuple(key) for key in source.keys(True, True)
            ]
        if not source._reads_return_copies and _reads_copies(source):
            # the update_ of a member leaves the files that it holds
            # unchanged: update a stacked copy, written back as for add_
            return self._inplace_on_copies(
                TensorDict._update_,
                input_dict_or_td,
                clone=clone,
                non_blocking=non_blocking,
                keys_to_update=keys_to_update,
            )
        source._update_(
            input_dict_or_td,
            clone=clone,
            non_blocking=non_blocking,
            keys_to_update=keys_to_update,
        )
        return self

    # Override: a lazy source sets the data of each member from its part of
    # value, as the leaves that self lists are those of the members (#2772).
    # The value is split into a stack of its parts, which the base checks
    # before it sets anything.
    def _data_setter(self, value):
        source = self._source
        if (
            not isinstance(source, LazyStackedTensorDict)
            or value.batch_size != self.batch_size
        ):
            # the base raises for another batch size
            return super()._data_setter(value)
        if _reads_copies(source):
            raise RuntimeError(
                "Cannot assign .data: the members of the lazy stack read copies "
                "of their tensors, such as from h5 files, which .data cannot set."
            )
        pieces = value.unbind(source.stack_dim)
        super()._data_setter(LazyStackedTensorDict(*pieces, stack_dim=source.stack_dim))

    # Override: the traversals walk the members of a lazy source, as on the
    # source, so that the binary ops find the leaves that they list (#2772).
    def _members_as_lists(self):
        return self._source._members_as_lists()

    # Override: the source detaches, with its checks.
    def _detach_(self):
        self._source.detach_()

    # Override: the source fills, with its checks.
    def _masked_fill_(self, mask, value):
        self._source.masked_fill_(mask, value)
        return self

    # Override: the source moves to shared memory, with its checks.
    def _share_memory_(self):
        self._source.share_memory_()

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
        check: bool = True,
    ):
        """Builds an instance of the class from a dictionary of its fields.

        Keys of ``input_dict`` that the class does not declare are kept, as
        :meth:`from_tensordict` and :meth:`~tensordict.TensorDictBase.set`
        keep them.

        Args:
            input_dict (dict): the values of the fields. Nested dictionaries
                become :class:`~tensordict.TensorDict` instances.

        Keyword Args:
            auto_batch_size (bool, optional): whether to infer the batch size.
            batch_size (torch.Size, optional): the batch size.
            device (torch.device, optional): the device.
            batch_dims (int, optional): the number of batch dimensions to infer.
            names (list of str, optional): the names of the batch dimensions.
            check (bool): If ``True`` (default), warn with a
                :class:`FutureWarning` when ``input_dict`` misses a required
                field. Starting with TensorDict 0.17, the default will change
                to raising the ``TypeError`` that the constructor raises. Set
                it to ``False`` to build an instance without these fields;
                reading one then raises ``AttributeError``.

        Returns:
            A new instance of the class.
        """
        td = TensorDict.from_dict(
            input_dict,
            auto_batch_size=auto_batch_size,
            batch_size=batch_size,
            device=device,
            batch_dims=batch_dims,
            names=names,
        )
        if check:
            _warn_missing_required(cls, td, "{}.from_dict()", "pass check=False")
        return cls._wrap_td(td)

    # Override: builds from the source and wraps the result.
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
        """Builds an instance of the same class from a dictionary, keeping the types of the existing entries.

        Unlike :meth:`from_dict`, it does not check the declared fields: the
        result can miss required fields and hold undeclared keys. tensordict
        calls it with the entries that a nested write changes, as in
        ``outer[0:2] = {"inner": {"eta": value}}`` where ``outer["inner"]``
        is a ``TypedTensorDict``.

        The arguments are those of
        :meth:`~tensordict.TensorDictBase.from_dict_instance`.
        """
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
    def _load_memmap(
        cls,
        prefix,
        metadata,
        device=None,
        out=None,
        *,
        robust_key,
        allow_pickle: bool | None = None,
        mode: str | None = None,
    ):
        td = TensorDict._load_memmap(
            prefix,
            metadata,
            device=device,
            out=out,
            robust_key=robust_key,
            allow_pickle=allow_pickle,
            mode=mode,
        )
        return cls._wrap_td(td)

    # ------------------------------------------------------------------
    # _to_module
    # ------------------------------------------------------------------
    def _to_module(
        self,
        module,
        *,
        inplace=None,
        return_swap=True,
        preserve_module_state: bool = True,
        **kwargs,
    ):
        return self._source._to_module(
            module,
            inplace=inplace,
            return_swap=return_swap,
            preserve_module_state=preserve_module_state,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Vmap cached methods
    # ------------------------------------------------------------------
    @_cache_while_locked  # noqa: B019
    def _add_batch_dim(self, *, in_dim: int, vmap_level: int):
        result = self._source._add_batch_dim(in_dim=in_dim, vmap_level=vmap_level)
        return type(self)._wrap_td(result)

    @_cache_while_locked  # noqa: B019
    def _remove_batch_dim(self, vmap_level: int, batch_size: int, out_dim: int):
        result = self._source._remove_batch_dim(vmap_level, batch_size, out_dim)
        return type(self)._wrap_td(result)

    @_cache_while_locked  # noqa: B019
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
    # Override: perf, one call to the source (the base front and a _del hook
    # take 60-80% longer); the source checks its lock and the key.
    def del_(self, key):
        self._source.del_(key)
        return self

    # Override: the source checks its lock and the keys, and renames: a lazy
    # stack renames in each member, where the base would write a stacked copy.
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
        **kwargs,
    ):
        td = TensorDict._new_unsafe(
            source=source,
            batch_size=batch_size,
            device=device,
            names=names,
            non_blocking=non_blocking,
            lock=lock,
            nested=nested,
            **kwargs,
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
# The public methods in these lists override the base: ``is_contiguous`` (the
# base walks the leaves, several times slower), ``all`` and ``any`` (the base
# is 10-15% slower with a dim), ``reshape`` and ``repeat_interleave`` (with
# inplace=True, the base rewrites the leaves through self, 50% slower),
# ``popitem`` (one call to the source),
# ``__setitem__`` (a key write through the base is 15-25% slower: the base
# front calls _set_tuple, then the source's), ``make_memmap*`` (a memory-mapped
# source), ``contiguous`` (one call to the source), ``masked_fill`` (the
# source's copy: the base would clone a file), ``masked_select`` (the typed
# result).
_DIRECT_DELEGATES = [
    "_get_str",
    "_get_tuple",
    "_stack_onto_",
    "keys",
    "entry_class",
    "_check_is_shared",
    "_check_device",
    "is_contiguous",
    "popitem",
    "_cast_reduction",
    "__setitem__",
    "_compare",
    "all",
    "any",
    "make_memmap",
    "make_memmap_from_storage",
    "make_memmap_from_tensor",
]

# Wrap result: delegate to _source, wrap TensorDictBase result
_WRAP_DELEGATES = [
    "_clone",
    "_expand",
    "_reshape",
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
]

# In-place: delegate to _source, return self
_INPLACE_DELEGATES = [
    "_set_str",
    "_set_tuple",
    "_set_at_str",
    "_set_at_tuple",
]

# Tuple wrap: delegate to _source, wrap each element
_TUPLE_WRAP_DELEGATES = [
    "_unbind",
    "_chunk",
    "_split",
]

for _name in _DIRECT_DELEGATES:
    setattr(TypedTensorDict, _name, _make_delegate(_name))

for _name in _WRAP_DELEGATES:
    setattr(TypedTensorDict, _name, _make_delegate_wrap(_name))

for _name in _INPLACE_DELEGATES:
    setattr(TypedTensorDict, _name, _make_delegate_inplace(_name))

for _name in _TUPLE_WRAP_DELEGATES:
    setattr(TypedTensorDict, _name, _make_delegate_tuple_wrap(_name))
