# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Cross-class compatibility tests for TensorClass and TypedTensorDict.

Tests every (wrapper x backend x operation) cell to build a compatibility
matrix between typed container wrappers and TensorDictBase backends.
"""

from __future__ import annotations

import tempfile

import pytest
import torch
from tensordict import TensorClass, TensorDict, TypedTensorDict
from tensordict._lazy import LazyStackedTensorDict
from tensordict.persistent import _has_h5 as _has_h5py
from tensordict.persistent import PersistentTensorDict
from tensordict.store._store import _has_redis, TensorDictStore
from torch import Tensor

BATCH = 4
FEAT_A = 3
FEAT_B = 5

# ---------------------------------------------------------------------------
# Typed containers
# ---------------------------------------------------------------------------


class MyTC(TensorClass):
    a: Tensor
    b: Tensor


class MyTTD(TypedTensorDict):
    a: Tensor
    b: Tensor


# ---------------------------------------------------------------------------
# Backend fixtures
# ---------------------------------------------------------------------------


def _make_base_td(device=None):
    return TensorDict(
        a=torch.randn(BATCH, FEAT_A),
        b=torch.randn(BATCH, FEAT_B),
        batch_size=[BATCH],
        device=device,
    )


def _make_tensordict():
    return _make_base_td()


def _make_lazy_stacked():
    tds = [
        TensorDict(
            a=torch.randn(FEAT_A),
            b=torch.randn(FEAT_B),
            batch_size=[],
        )
        for _ in range(BATCH)
    ]
    return LazyStackedTensorDict(*tds, stack_dim=0)


def _make_h5(tmp_path):
    td = _make_base_td()
    return PersistentTensorDict.from_dict(td, filename=str(tmp_path / "test.h5"))


def _make_memmap(tmp_path):
    td = _make_base_td()
    return td.memmap_(prefix=str(tmp_path / "memmap"))


def _make_typed_td():
    return MyTTD(
        a=torch.randn(BATCH, FEAT_A),
        b=torch.randn(BATCH, FEAT_B),
        batch_size=[BATCH],
    )


def _make_redis():
    td = _make_base_td()
    return TensorDictStore.from_tensordict(td)


# ---------------------------------------------------------------------------
# Parametrize helpers
# ---------------------------------------------------------------------------

BACKENDS_NO_INFRA = ["tensordict", "lazy_stacked", "typed_td"]
BACKENDS_H5 = ["h5"] if _has_h5py else []
BACKENDS_REDIS = ["redis"] if _has_redis else []
BACKENDS_MEMMAP = ["memmap"]

ALL_BACKENDS = BACKENDS_NO_INFRA + BACKENDS_MEMMAP + BACKENDS_H5 + BACKENDS_REDIS


def _get_backend(backend_name, tmp_path):
    if backend_name == "tensordict":
        return _make_tensordict()
    elif backend_name == "lazy_stacked":
        return _make_lazy_stacked()
    elif backend_name == "h5":
        return _make_h5(tmp_path)
    elif backend_name == "memmap":
        return _make_memmap(tmp_path)
    elif backend_name == "typed_td":
        return _make_typed_td()
    elif backend_name == "redis":
        return _make_redis()
    raise ValueError(f"Unknown backend: {backend_name}")


# ===================================================================
# TensorClass wrapping various backends
# ===================================================================


class TestTensorClassCompat:
    """Test TensorClass.from_tensordict(backend) for each backend."""

    @pytest.fixture(params=ALL_BACKENDS)
    def backend_td(self, request, tmp_path):
        return request.param, _get_backend(request.param, tmp_path)

    def test_construction(self, backend_td):
        name, td = backend_td
        tc = MyTC.from_tensordict(td)
        assert tc is not None
        assert tc.batch_size[0] == BATCH

    def test_attr_read(self, backend_td):
        name, td = backend_td
        tc = MyTC.from_tensordict(td)
        a = tc.a
        assert a.shape[-1] == FEAT_A
        b = tc.b
        assert b.shape[-1] == FEAT_B

    def test_attr_write(self, backend_td):
        name, td = backend_td
        if name == "memmap":
            pytest.skip("memmap TDs are locked; use set_() for in-place writes")
        tc = MyTC.from_tensordict(td)
        new_a = torch.ones(BATCH, FEAT_A)
        tc.a = new_a
        assert (tc.a == 1).all()

    def test_attr_write_inplace_memmap(self, tmp_path):
        td = _make_memmap(tmp_path)
        tc = MyTC.from_tensordict(td)
        tc.set_("a", torch.ones(BATCH, FEAT_A))
        assert (tc.a == 1).all()

    def test_index_single(self, backend_td):
        name, td = backend_td
        tc = MyTC.from_tensordict(td)
        item = tc[0]
        assert item.a.shape[-1] == FEAT_A

    def test_index_slice(self, backend_td):
        name, td = backend_td
        tc = MyTC.from_tensordict(td)
        sliced = tc[0:2]
        assert sliced.batch_size[0] == 2

    def test_to_tensordict(self, backend_td):
        name, td = backend_td
        tc = MyTC.from_tensordict(td)
        out = tc.to_tensordict()
        assert isinstance(out, TensorDict)
        assert set(out.keys()) == {"a", "b"}

    def test_clone(self, backend_td):
        name, td = backend_td
        tc = MyTC.from_tensordict(td)
        cloned = tc.clone()
        assert cloned.a.shape == tc.a.shape

    def test_stack(self, backend_td):
        name, td = backend_td
        tc = MyTC.from_tensordict(td)
        stacked = torch.stack([tc, tc], dim=0)
        assert stacked.batch_size[0] == 2

    def test_iteration(self, backend_td):
        name, td = backend_td
        tc = MyTC.from_tensordict(td)
        items = list(tc)
        assert len(items) == BATCH

    def test_update(self, backend_td):
        name, td = backend_td
        if name == "memmap":
            pytest.skip("memmap TDs are locked; use update_() for in-place writes")
        tc = MyTC.from_tensordict(td)
        tc.update({"a": torch.ones(BATCH, FEAT_A)})
        assert (tc.a == 1).all()

    def test_update_inplace_memmap(self, tmp_path):
        td = _make_memmap(tmp_path)
        tc = MyTC.from_tensordict(td)
        tc.update_({"a": torch.ones(BATCH, FEAT_A)})
        assert (tc.a == 1).all()


# ===================================================================
# TypedTensorDict interop with backends
# ===================================================================


class TestTypedTensorDictCompat:
    """Test TypedTensorDict interop patterns with various backends."""

    # --- Construction / conversion from backends ---

    def test_from_tensordict_data(self):
        td = _make_base_td()
        ttd = MyTTD(a=td["a"], b=td["b"], batch_size=td.batch_size)
        assert isinstance(ttd, MyTTD)
        assert isinstance(ttd, TensorDict)
        assert ttd.a.shape == (BATCH, FEAT_A)

    @pytest.mark.skipif(not _has_h5py, reason="h5py not available")
    def test_from_h5_data(self, tmp_path):
        h5 = _make_h5(tmp_path)
        materialized = h5.to_tensordict()
        ttd = MyTTD(a=materialized["a"], b=materialized["b"], batch_size=materialized.batch_size)
        assert isinstance(ttd, MyTTD)
        assert ttd.a.shape == (BATCH, FEAT_A)

    @pytest.mark.skipif(not _has_redis, reason="redis not available")
    def test_from_redis_data(self):
        store = _make_redis()
        materialized = store.to_tensordict()
        ttd = MyTTD(a=materialized["a"], b=materialized["b"], batch_size=materialized.batch_size)
        assert isinstance(ttd, MyTTD)

    def test_from_lazy_stack_data(self):
        ls = _make_lazy_stacked()
        materialized = ls.to_tensordict()
        ttd = MyTTD(a=materialized["a"], b=materialized["b"], batch_size=materialized.batch_size)
        assert isinstance(ttd, MyTTD)

    # --- Operations on TypedTensorDict ---

    def test_attr_read(self):
        ttd = _make_typed_td()
        assert ttd.a.shape == (BATCH, FEAT_A)
        assert ttd.b.shape == (BATCH, FEAT_B)

    def test_attr_write(self):
        ttd = _make_typed_td()
        ttd.a = torch.ones(BATCH, FEAT_A)
        assert (ttd.a == 1).all()

    def test_index_single(self):
        ttd = _make_typed_td()
        item = ttd[0]
        assert item.a.shape == (FEAT_A,)

    def test_index_slice(self):
        ttd = _make_typed_td()
        sliced = ttd[0:2]
        assert sliced.batch_size == torch.Size([2])

    def test_clone(self):
        ttd = _make_typed_td()
        cloned = ttd.clone()
        assert cloned.a.shape == ttd.a.shape
        assert set(cloned.keys()) == {"a", "b"}

    def test_to_tensordict(self):
        ttd = _make_typed_td()
        out = ttd.to_tensordict()
        assert isinstance(out, TensorDict)
        assert set(out.keys()) == {"a", "b"}

    def test_update(self):
        ttd = _make_typed_td()
        ttd.update({"a": torch.ones(BATCH, FEAT_A)})
        assert (ttd.a == 1).all()

    def test_iteration(self):
        ttd = _make_typed_td()
        items = list(ttd)
        assert len(items) == BATCH

    # --- Stacking ---

    def test_dense_stack(self):
        ttd1 = _make_typed_td()
        ttd2 = _make_typed_td()
        stacked = torch.stack([ttd1, ttd2], dim=0)
        assert stacked.batch_size[0] == 2
        assert stacked.batch_size[1] == BATCH

    def test_lazy_stack(self):
        ttd1 = _make_typed_td()
        ttd2 = _make_typed_td()
        ls = LazyStackedTensorDict(ttd1, ttd2, stack_dim=0)
        assert ls.batch_size[0] == 2
        assert ls[0].a.shape == (BATCH, FEAT_A)

    def test_lazy_stack_index_preserves_type(self):
        ttd1 = _make_typed_td()
        ttd2 = _make_typed_td()
        ls = LazyStackedTensorDict(ttd1, ttd2, stack_dim=0)
        item = ls[0]
        assert isinstance(item, MyTTD)

    # --- TypedTensorDict in/out of persistent backends ---

    @pytest.mark.skipif(not _has_h5py, reason="h5py not available")
    def test_to_h5_and_back(self, tmp_path):
        ttd = _make_typed_td()
        h5 = PersistentTensorDict.from_dict(ttd, filename=str(tmp_path / "ttd.h5"))
        assert set(h5.keys()) == {"a", "b"}
        materialized = h5.to_tensordict()
        assert materialized["a"].shape == (BATCH, FEAT_A)

    def test_to_memmap_and_back(self, tmp_path):
        ttd = _make_typed_td()
        mmap = ttd.memmap_(prefix=str(tmp_path / "ttd_mmap"))
        assert mmap.a.shape == (BATCH, FEAT_A)

    @pytest.mark.skipif(not _has_redis, reason="redis not available")
    def test_to_redis_and_back(self):
        ttd = _make_typed_td()
        store = TensorDictStore.from_tensordict(ttd)
        assert store["a"].shape == (BATCH, FEAT_A)
        back = store.to_tensordict()
        assert back["a"].shape == (BATCH, FEAT_A)


# ===================================================================
# TensorClass wrapping TypedTensorDict (expected overlap)
# ===================================================================


class TestTensorClassTypedTDOverlap:
    """Test the overlap between TensorClass and TypedTensorDict.

    The user expects this to be problematic since both enforce schemas.
    """

    def test_tc_from_typed_td(self):
        ttd = _make_typed_td()
        tc = MyTC.from_tensordict(ttd)
        assert tc.a.shape == (BATCH, FEAT_A)

    def test_tc_from_typed_td_attr_write(self):
        ttd = _make_typed_td()
        tc = MyTC.from_tensordict(ttd)
        tc.a = torch.ones(BATCH, FEAT_A)
        assert (tc.a == 1).all()

    def test_tc_from_typed_td_index(self):
        ttd = _make_typed_td()
        tc = MyTC.from_tensordict(ttd)
        item = tc[0]
        assert item.a.shape == (FEAT_A,)

    def test_tc_from_typed_td_clone(self):
        ttd = _make_typed_td()
        tc = MyTC.from_tensordict(ttd)
        cloned = tc.clone()
        assert cloned.a.shape == tc.a.shape

    def test_tc_from_typed_td_stack(self):
        ttd1 = _make_typed_td()
        ttd2 = _make_typed_td()
        tc1 = MyTC.from_tensordict(ttd1)
        tc2 = MyTC.from_tensordict(ttd2)
        stacked = torch.stack([tc1, tc2], dim=0)
        assert stacked.batch_size[0] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
