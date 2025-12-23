# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from tensordict._lazy import LazyStackedTensorDict  # noqa: F401
from tensordict._td import TensorDict  # noqa: F401
from tensordict.base import (  # noqa: F401
    is_tensor_collection,
    NO_DEFAULT,
    TensorDictBase,
)
from tensordict.functional import (  # noqa: F401
    dense_stack_tds,
    make_tensordict,
    merge_tensordicts,
    pad,
    pad_sequence,
)
from tensordict.memmap import MemoryMappedTensor  # noqa: F401
from tensordict.utils import (  # noqa: F401
    assert_allclose_td,
    cache,
    convert_ellipsis_to_idx,
    erase_cache,
    expand_as_right,
    expand_right,
    implement_for,
    infer_size_impl,
    int_generator,
    is_nested_key,
    is_seq_of_nested_key,
    is_tensorclass,
    lock_blocked,
    NestedKey,
)

# TorchBind interop helpers (flat, single-device)
try:
    import torch  # local import to avoid issues if torch unavailable at parse time
    import weakref

    _clsns = getattr(torch, "classes", None)
    _tdns = getattr(_clsns, "tensordict", None) if _clsns is not None else None
    _TB_TensorDict = (
        getattr(_tdns, "TensorDict", None) if _tdns is not None else None
    )
    if _TB_TensorDict is not None:

        class _TorchBindContextManager:
            """Context manager wrapper for to_torchbind() that handles conversion back."""
            
            def __init__(self, td, tb):
                self._td_ref = weakref.ref(td)
                self._tb = tb
            
            def __enter__(self):
                return self._tb
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                if exc_type is not None:
                    return False
                # Convert TorchBind back to TensorDict and update original
                td = self._td_ref()
                if td is not None:
                    from tensordict._td import TensorDict as _TD_class
                    td_from_tb = _TD_class.from_torchbind(self._tb)
                    td.update(td_from_tb, inplace=False)
                return False

        def to_torchbind(td):  # type: ignore[no-redef]
            """Convert TensorDict to TorchBind format.
            
            Can be used as a context manager to automatically convert back:
                with td.to_torchbind() as tb:
                    # use tb (TorchBind object)
                # td is automatically updated from tb
            """
            if td.device is None:
                raise RuntimeError(
                    "TensorDict.to_torchbind requires a non-None device; call td = td.to(device) first"
                )
            keys = list(td.keys())
            values = [td.get(k) for k in keys]
            bs = list(td.batch_size)
            tb = _TB_TensorDict.from_pairs(keys, values, bs, td.device)
            # Return a context manager that wraps the TorchBind object
            return _TorchBindContextManager(td, tb)

        def _from_torchbind(cls, obj):  # type: ignore[no-redef]
            keys = list(obj.keys())
            d = {k: obj.get(k) for k in keys}
            return cls(d, batch_size=tuple(obj.batch_size()), device=obj.device())

        # Expose on class for convenience
        from tensordict._td import TensorDict as _TD  # local import to avoid cycles

        _TD.to_torchbind = to_torchbind  # type: ignore[attr-defined]
        _TD.from_torchbind = classmethod(_from_torchbind)  # type: ignore[attr-defined]
except Exception:
    # If the extension is not built/available, leave helpers undefined
    pass
