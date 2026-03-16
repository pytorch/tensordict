# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Cross-class compatibility tests for TensorClass and TypedTensorDict.

Tests every (wrapper x backend x operation) cell to build a compatibility
matrix between typed container wrappers and TensorDictBase backends.
"""

from __future__ import annotations

import pytest
import torch
from tensordict import TensorClass, TensorDict, TypedTensorDict
from tensordict._lazy import LazyStackedTensorDict
from tensordict.base import TensorDictBase
from tensordict.persistent import _has_h5 as _has_h5py, PersistentTensorDict
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
        assert isinstance(ttd, TensorDictBase)
        assert ttd.a.shape == (BATCH, FEAT_A)

    @pytest.mark.skipif(not _has_h5py, reason="h5py not available")
    def test_from_h5_data(self, tmp_path):
        h5 = _make_h5(tmp_path)
        materialized = h5.to_tensordict()
        ttd = MyTTD(
            a=materialized["a"], b=materialized["b"], batch_size=materialized.batch_size
        )
        assert isinstance(ttd, MyTTD)
        assert ttd.a.shape == (BATCH, FEAT_A)

    @pytest.mark.skipif(not _has_redis, reason="redis not available")
    def test_from_redis_data(self):
        store = _make_redis()
        materialized = store.to_tensordict()
        ttd = MyTTD(
            a=materialized["a"], b=materialized["b"], batch_size=materialized.batch_size
        )
        assert isinstance(ttd, MyTTD)

    def test_from_lazy_stack_data(self):
        ls = _make_lazy_stacked()
        materialized = ls.to_tensordict()
        ttd = MyTTD(
            a=materialized["a"], b=materialized["b"], batch_size=materialized.batch_size
        )
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


# ===================================================================
# TypedTensorDict.from_tensordict() wrapping various backends
# ===================================================================


# Backends that TypedTensorDict can wrap via from_tensordict
TTD_WRAP_BACKENDS_NO_INFRA = ["tensordict", "lazy_stacked"]
TTD_WRAP_BACKENDS_H5 = ["h5"] if _has_h5py else []
TTD_WRAP_BACKENDS_REDIS = ["redis"] if _has_redis else []
TTD_WRAP_BACKENDS_MEMMAP = ["memmap"]
TTD_WRAP_ALL_BACKENDS = (
    TTD_WRAP_BACKENDS_NO_INFRA
    + TTD_WRAP_BACKENDS_MEMMAP
    + TTD_WRAP_BACKENDS_H5
    + TTD_WRAP_BACKENDS_REDIS
)


class TestTypedTensorDictWrapping:
    """Test MyTTD.from_tensordict(backend) for each backend."""

    @pytest.fixture(params=TTD_WRAP_ALL_BACKENDS)
    def backend_td(self, request, tmp_path):
        return request.param, _get_backend(request.param, tmp_path)

    def test_construction(self, backend_td):
        name, td = backend_td
        ttd = MyTTD.from_tensordict(td)
        assert isinstance(ttd, MyTTD)
        assert isinstance(ttd, TensorDictBase)
        assert ttd.batch_size[0] == BATCH

    def test_attr_read(self, backend_td):
        name, td = backend_td
        ttd = MyTTD.from_tensordict(td)
        assert ttd.a.shape[-1] == FEAT_A
        assert ttd.b.shape[-1] == FEAT_B

    def test_attr_write(self, backend_td):
        name, td = backend_td
        if name == "memmap":
            pytest.skip("memmap TDs are locked; use set_() for in-place writes")
        ttd = MyTTD.from_tensordict(td)
        ttd.a = torch.ones_like(ttd.a)
        assert (ttd.a == 1).all()

    def test_attr_write_inplace_memmap(self, tmp_path):
        td = _make_memmap(tmp_path)
        ttd = MyTTD.from_tensordict(td)
        ttd.set_("a", torch.ones_like(ttd.a))
        assert (ttd.a == 1).all()

    def test_index(self, backend_td):
        name, td = backend_td
        ttd = MyTTD.from_tensordict(td)
        item = ttd[0]
        assert isinstance(item, MyTTD)
        assert item.a.shape[-1] == FEAT_A

    def test_slice(self, backend_td):
        name, td = backend_td
        ttd = MyTTD.from_tensordict(td)
        sliced = ttd[0:2]
        assert isinstance(sliced, MyTTD)
        assert sliced.batch_size[0] == 2

    def test_clone(self, backend_td):
        name, td = backend_td
        ttd = MyTTD.from_tensordict(td)
        cloned = ttd.clone()
        assert isinstance(cloned, MyTTD)
        assert cloned.a.shape == ttd.a.shape

    def test_update(self, backend_td):
        name, td = backend_td
        if name == "memmap":
            pytest.skip("memmap TDs are locked; use update_() for in-place writes")
        ttd = MyTTD.from_tensordict(td)
        ttd.update({"a": torch.ones(BATCH, FEAT_A)})
        assert (ttd.a == 1).all()

    def test_update_inplace_memmap(self, tmp_path):
        td = _make_memmap(tmp_path)
        ttd = MyTTD.from_tensordict(td)
        ttd.update_({"a": torch.ones(BATCH, FEAT_A)})
        assert (ttd.a == 1).all()

    def test_stack(self, backend_td):
        name, td = backend_td
        ttd1 = MyTTD.from_tensordict(td)
        td2 = (
            _get_backend(name, None)
            if name not in ("h5", "memmap")
            else _make_tensordict()
        )
        ttd2 = MyTTD.from_tensordict(td2)
        stacked = torch.stack([ttd1, ttd2])
        assert isinstance(stacked, MyTTD)
        assert stacked.batch_size[0] == 2

    def test_live_link(self, backend_td):
        """Mutations through TypedTensorDict reflect in the original backend."""
        name, td = backend_td
        if name == "memmap":
            pytest.skip("memmap TDs are locked")
        ttd = MyTTD.from_tensordict(td)
        ttd.a = torch.ones_like(ttd.a)
        assert (td["a"] == 1).all()

    def test_iterate(self, backend_td):
        name, td = backend_td
        ttd = MyTTD.from_tensordict(td)
        items = list(ttd)
        assert len(items) == BATCH
        assert isinstance(items[0], MyTTD)


# ===================================================================
# TensorDictStore.empty + TypedTensorDict pre-allocation workflow
# ===================================================================


@pytest.mark.skipif(not _has_redis, reason="redis not available")
class TestTensorDictStoreFromSchema:
    """Test TensorDictStore.from_schema() pre-allocation."""

    def test_from_schema_creates_keys(self):
        store = TensorDictStore.from_schema(
            {"a": ([FEAT_A], torch.float32), "b": ([FEAT_B], torch.float32)},
            batch_size=[BATCH],
        )
        try:
            assert set(store.keys()) == {"a", "b"}
            assert store["a"].shape == torch.Size([BATCH, FEAT_A])
            assert store["b"].shape == torch.Size([BATCH, FEAT_B])
            assert (store["a"] == 0).all()
        finally:
            store._run_sync(store._client.flushdb())

    def test_from_schema_write_and_read(self):
        store = TensorDictStore.from_schema(
            {"a": ([FEAT_A], torch.float32), "b": ([FEAT_B], torch.float32)},
            batch_size=[BATCH],
        )
        try:
            store[0] = TensorDict(
                a=torch.ones(FEAT_A),
                b=torch.ones(FEAT_B),
                batch_size=[],
            )
            assert (store[0]["a"] == 1).all()
            assert (store[1]["a"] == 0).all()
        finally:
            store._run_sync(store._client.flushdb())

    def test_from_schema_with_typed_td(self):
        store = TensorDictStore.from_schema(
            {"a": ([FEAT_A], torch.float32), "b": ([FEAT_B], torch.float32)},
            batch_size=[BATCH],
        )
        try:
            ttd = MyTTD.from_tensordict(store)
            assert isinstance(ttd, MyTTD)
            assert ttd.a.shape == torch.Size([BATCH, FEAT_A])

            ttd[0] = TensorDict(
                a=torch.ones(FEAT_A),
                b=torch.ones(FEAT_B),
                batch_size=[],
            )
            assert (ttd[0].a == 1).all()
        finally:
            store._run_sync(store._client.flushdb())

    def test_from_schema_scalar_shape(self):
        store = TensorDictStore.from_schema(
            {"reward": ([], torch.float32)},
            batch_size=[BATCH],
        )
        try:
            assert store["reward"].shape == torch.Size([BATCH])
        finally:
            store._run_sync(store._client.flushdb())


@pytest.mark.skipif(not _has_redis, reason="redis not available")
class TestTypedTDPreallocationWorkflow:
    """End-to-end test of the pre-allocation + iterative fill pattern."""

    def test_preallocate_and_fill(self):
        store = TensorDictStore.from_schema(
            {"a": ([FEAT_A], torch.float32), "b": ([FEAT_B], torch.float32)},
            batch_size=[BATCH],
        )
        try:
            ttd = MyTTD.from_tensordict(store)
            for i in range(BATCH):
                ttd[i] = TensorDict(
                    a=torch.full([FEAT_A], float(i)),
                    b=torch.full([FEAT_B], float(i)),
                    batch_size=[],
                )
            for i in range(BATCH):
                assert (ttd[i].a == float(i)).all()
                assert (ttd[i].b == float(i)).all()
        finally:
            store._run_sync(store._client.flushdb())

    def test_deferred_validation_then_fill(self):
        """Wrap empty store with check=False, then fill."""
        store = TensorDictStore(batch_size=[BATCH])
        try:
            ttd = MyTTD.from_tensordict(store, check=False)
            assert isinstance(ttd, MyTTD)
            assert len(list(ttd.keys())) == 0

            ttd[0] = TensorDict(
                a=torch.ones(FEAT_A),
                b=torch.ones(FEAT_B),
                batch_size=[],
            )
            assert (ttd[0].a == 1).all()
            assert set(ttd.keys()) == {"a", "b"}
        finally:
            store._run_sync(store._client.flushdb())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
