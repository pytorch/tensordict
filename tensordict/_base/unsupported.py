# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""The operations that some backends of :class:`~tensordict.TensorDictBase` cannot support.

Each mixin implements a group of abstract hooks of ``TensorDictBase`` with a
body that raises. A class lists the mixins before ``TensorDictBase`` in its
bases, so that the hooks stay abstract on ``TensorDictBase`` and a new backend
that forgets one still fails at instantiation.
"""

from __future__ import annotations


def _unsupported(self, op: str, hint: str = "Call `to_tensordict()` first.") -> str:
    return f"Cannot call `{op}` on a {type(self).__name__}. {hint}"


def _no_vmap(self) -> str:
    return (
        f"{type(self).__name__} cannot be used with vmap. Call `to_tensordict()` first."
    )


class _NoShapeOps:
    """Shape hooks for a backend that cannot give views of its entries."""

    def _view(self, *args, **kwargs):
        raise RuntimeError(
            _unsupported(self, "view", "Call `reshape` or `to_tensordict()` first.")
        )

    def _transpose(self, dim0, dim1):
        raise RuntimeError(_unsupported(self, "transpose"))

    def _permute(self, *args, **kwargs):
        raise RuntimeError(_unsupported(self, "permute"))

    def _squeeze(self, dim=None):
        raise RuntimeError(_unsupported(self, "squeeze"))

    def _unsqueeze(self, dim):
        raise RuntimeError(_unsupported(self, "unsqueeze"))


class _NoVmap:
    """Vmap hooks for a backend whose entries cannot be batched tensors."""

    def _add_batch_dim(self, *, in_dim, vmap_level):
        raise RuntimeError(_no_vmap(self))

    # A function that returns such a tensordict reaches these hooks without
    # _add_batch_dim.
    def _remove_batch_dim(self, vmap_level, batch_size, out_dim):
        raise RuntimeError(_no_vmap(self))

    def _maybe_remove_batch_dim(self, funcname, vmap_level, batch_size, out_dim):
        raise RuntimeError(_no_vmap(self))


class _NoSetNonTensor:
    """``set_non_tensor`` for a backend that cannot write a non-tensor entry this way."""

    def _set_non_tensor(self, key, value):
        raise NotImplementedError(
            f"set_non_tensor is not compatible with the tensordict type {type(self).__name__}."
        )


class _NoSelectExclude:
    """``select``, ``exclude`` and ``rename_keys`` for a backend whose reads copy the stored data.

    A shallow copy of such a backend shares its file or server, so renaming the
    keys of a copy would rename them in the storage.
    """

    def _select(self, *keys, inplace=False, strict=True, set_shared=True):
        raise NotImplementedError(
            f"Cannot call select on a {type(self).__name__}. "
            "Call to_tensordict() first."
        )

    def _exclude(self, *keys, inplace=False, set_shared=True):
        raise NotImplementedError(
            f"Cannot call exclude on a {type(self).__name__}. "
            "Call to_tensordict() first."
        )

    def _rename_keys(self, pairs, *, strict, prune_root):
        """Raises ``NotImplementedError``: see the class docstring."""
        raise NotImplementedError(
            f"Cannot call rename_keys on a {type(self).__name__}. "
            "Call to_tensordict() first."
        )


class _StoreUnsupported(_NoShapeOps, _NoVmap, _NoSetNonTensor, _NoSelectExclude):
    """The unsupported operations of the stores in :mod:`tensordict.store`."""

    def _memmap_(self, *, inplace, **kwargs):
        if inplace:
            raise RuntimeError(
                f"Cannot build a memmap TensorDict in-place from a {type(self).__name__}. "
                "Use `td.memmap()` instead."
            )
        return self.to_tensordict()._memmap_(inplace=False, **kwargs)

    def _stack_onto_(self, list_item, dim):
        raise RuntimeError(_unsupported(self, "_stack_onto_"))
