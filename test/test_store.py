# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import importlib
import pickle

import pytest
import torch
from tensordict import lazy_stack, TensorDict
from tensordict.base import TensorDictBase
from tensordict.store import LazyStackedTensorDictStore, TensorDictStore

_has_redis = importlib.util.find_spec("redis", None) is not None

# Ports used by each backend in CI.
_BACKEND_PORTS = {"redis": 6379, "dragonfly": 6380}


def _server_available(host: str, port: int) -> bool:
    """Check if a Redis-protocol server is reachable at *host*:*port*."""
    if not _has_redis:
        return False
    import redis

    try:
        r = redis.Redis(host=host, port=port, db=0, socket_connect_timeout=2)
        r.ping()
        r.close()
        return True
    except (redis.ConnectionError, redis.exceptions.ConnectionError, OSError):
        return False


@pytest.fixture(params=["redis", "dragonfly"])
def backend(request):
    """Yield ``(backend_name, port)`` for each available backend."""
    name = request.param
    port = _BACKEND_PORTS[name]
    if not _has_redis:
        pytest.skip("redis package not installed")
    if not _server_available("localhost", port):
        pytest.skip(f"{name} server not reachable on localhost:{port}")
    return name, port


@pytest.fixture
def store_kwargs(backend):
    """Common keyword arguments for constructing a store in tests."""
    name, port = backend
    return {"backend": name, "port": port, "db": 15}


class TestTensorDictStore:
    """Tests for TensorDictStore requiring a running store server."""

    @pytest.fixture(autouse=True)
    def store_td(self, store_kwargs):
        """Create a fresh TensorDictStore and clean up after the test."""
        td = TensorDictStore(batch_size=[10], **store_kwargs)
        yield td
        td.clear_redis()
        td.close()

    def test_basic_set_get(self, store_td):
        """Store and retrieve a tensor."""
        tensor = torch.randn(10, 3)
        store_td["obs"] = tensor
        result = store_td["obs"]
        assert isinstance(result, torch.Tensor)
        assert result.shape == (10, 3)
        assert torch.allclose(result, tensor)

    def test_multiple_keys(self, store_td):
        """Store and retrieve multiple tensors."""
        obs = torch.randn(10, 84)
        action = torch.randn(10, 4)
        reward = torch.randn(10, 1)
        store_td["obs"] = obs
        store_td["action"] = action
        store_td["reward"] = reward

        assert torch.allclose(store_td["obs"], obs)
        assert torch.allclose(store_td["action"], action)
        assert torch.allclose(store_td["reward"], reward)

    def test_dtypes(self, store_td):
        """Store and retrieve tensors of different dtypes."""
        for dtype in [
            torch.float32,
            torch.float64,
            torch.int32,
            torch.int64,
            torch.bool,
        ]:
            key = f"tensor_{dtype}".replace(".", "_")
            tensor = torch.zeros(10, 5, dtype=dtype)
            store_td[key] = tensor
            result = store_td[key]
            assert result.dtype == dtype, f"Failed for dtype {dtype}"
            assert result.shape == (10, 5)

    def test_nested_keys(self, store_td):
        """Store and retrieve tensors with nested keys."""
        tensor = torch.randn(10, 4)
        store_td["nested", "obs"] = tensor

        # Access the nested value
        result = store_td["nested", "obs"]
        assert torch.allclose(result, tensor)

        # Access the nested tensordict
        nested = store_td["nested"]

        assert isinstance(nested, TensorDictStore)
        result2 = nested["obs"]
        assert torch.allclose(result2, tensor)

    def test_nested_from_tensordict(self, store_td):
        """Store a nested TensorDict."""
        inner = TensorDict({"a": torch.randn(10, 3), "b": torch.randn(10, 2)}, [10])
        store_td["inner"] = inner

        assert torch.allclose(store_td["inner", "a"], inner["a"])
        assert torch.allclose(store_td["inner", "b"], inner["b"])

    def test_batch_size(self, store_td):
        """Verify batch_size is correctly stored and accessible."""
        assert store_td.batch_size == torch.Size([10])

    def test_from_dict(self, store_kwargs):
        """Construct from a dict."""

        source = {"obs": torch.randn(5, 3), "reward": torch.randn(5, 1)}
        td = TensorDictStore.from_dict(source, batch_size=[5], **store_kwargs)
        try:
            assert td.batch_size == torch.Size([5])
            assert torch.allclose(td["obs"], source["obs"])
            assert torch.allclose(td["reward"], source["reward"])
        finally:
            td.clear_redis()
            td.close()

    def test_from_dict_tensordict_input(self, store_kwargs):
        """Construct via from_dict with a TensorDict as input."""

        source = TensorDict(
            {"obs": torch.randn(5, 3), "reward": torch.randn(5, 1)}, [5]
        )
        td = TensorDictStore.from_dict(source, **store_kwargs)
        try:
            assert td.batch_size == torch.Size([5])
            assert torch.allclose(td["obs"], source["obs"])
            assert torch.allclose(td["reward"], source["reward"])
        finally:
            td.clear_redis()
            td.close()

    def test_to_local(self, store_td):
        """Materialize to a local TensorDict."""
        obs = torch.randn(10, 4)
        action = torch.randn(10, 2)
        store_td["obs"] = obs
        store_td["action"] = action

        local = store_td.to_local()
        assert isinstance(local, TensorDict)
        assert local.batch_size == torch.Size([10])
        assert torch.allclose(local["obs"], obs)
        assert torch.allclose(local["action"], action)

    def test_to_tensordict(self, store_td):
        """to_tensordict should be equivalent to to_local."""
        tensor = torch.randn(10, 3)
        store_td["x"] = tensor

        local = store_td.to_tensordict()
        assert isinstance(local, TensorDict)
        assert torch.allclose(local["x"], tensor)

    def test_contiguous(self, store_td):
        """contiguous should materialize."""
        tensor = torch.randn(10, 3)
        store_td["x"] = tensor
        local = store_td.contiguous()
        assert isinstance(local, TensorDict)
        assert torch.allclose(local["x"], tensor)

    def test_keys_view(self, store_td):
        """Test keys iteration."""
        store_td["a"] = torch.randn(10, 2)
        store_td["b"] = torch.randn(10, 3)
        store_td["nested", "c"] = torch.randn(10, 4)

        # Top-level keys (non-nested)
        keys = set(store_td.keys())
        assert "a" in keys
        assert "b" in keys
        assert "nested" in keys

    def test_keys_include_nested(self, store_td):
        """Test keys with include_nested=True."""
        store_td["a"] = torch.randn(10, 2)
        store_td["nested", "b"] = torch.randn(10, 3)

        nested_keys = set(store_td.keys(include_nested=True))
        assert "a" in nested_keys
        # The nested key should appear as a tuple
        assert ("nested", "b") in nested_keys or "nested" in nested_keys

    def test_keys_leaves_only(self, store_td):
        """Test keys with leaves_only=True."""
        store_td["a"] = torch.randn(10, 2)
        store_td["nested", "b"] = torch.randn(10, 3)

        leaf_keys = list(store_td.keys(leaves_only=True))
        assert "a" in leaf_keys
        # "nested" should not appear since it is not a leaf
        assert "nested" not in leaf_keys

    def test_contains(self, store_td):
        """Test __contains__ via 'in' operator."""
        store_td["obs"] = torch.randn(10, 3)
        store_td["nested", "val"] = torch.randn(10, 2)

        assert "obs" in store_td.keys()
        assert "nested" in store_td.keys()
        assert ("nested", "val") in store_td.keys(include_nested=True)
        assert "nonexistent" not in store_td.keys()

    def test_del(self, store_td):
        """Test key deletion."""
        store_td["obs"] = torch.randn(10, 3)
        store_td["action"] = torch.randn(10, 2)
        assert "obs" in store_td.keys()

        store_td.del_("obs")
        assert "obs" not in store_td.keys()
        assert "action" in store_td.keys()

    def test_del_nested(self, store_td):
        """Test deletion of nested keys."""
        store_td["nested", "a"] = torch.randn(10, 2)
        store_td["nested", "b"] = torch.randn(10, 3)
        assert "nested" in store_td.keys()

        store_td.del_("nested")
        assert "nested" not in store_td.keys()

    def test_locking(self, store_td):
        """Test lock/unlock."""
        store_td["obs"] = torch.randn(10, 3)
        store_td.lock_()
        assert store_td.is_locked

        # Should raise when trying to set a new key
        with pytest.raises(RuntimeError):
            store_td["new_key"] = torch.randn(10, 2)

        store_td.unlock_()
        assert not store_td.is_locked
        store_td["new_key"] = torch.randn(10, 2)

    def test_set_at_(self, store_td):
        """Test setting a value at a specific index."""
        tensor = torch.zeros(10, 3)
        store_td["obs"] = tensor
        new_vals = torch.ones(3)
        store_td.set_at_("obs", new_vals, 0)
        result = store_td["obs"]
        assert torch.allclose(result[0], new_vals)
        assert torch.allclose(result[1:], torch.zeros(9, 3))

    def test_update(self, store_td):
        """Test batch update from a dict."""
        source = TensorDict(
            {"obs": torch.randn(10, 3), "action": torch.randn(10, 2)}, [10]
        )
        store_td.update(source)
        assert torch.allclose(store_td["obs"], source["obs"])
        assert torch.allclose(store_td["action"], source["action"])

    def test_clone_recurse(self, store_td):
        """Test deep clone."""

        store_td["obs"] = torch.randn(10, 3)
        cloned = store_td.clone()
        try:
            assert isinstance(cloned, TensorDictStore)
            assert cloned._td_id != store_td._td_id
            assert torch.allclose(cloned["obs"], store_td["obs"])
        finally:
            cloned.clear_redis()
            cloned.close()

    def test_clone_no_recurse(self, store_td):
        """Test shallow clone (same Redis data)."""

        store_td["obs"] = torch.randn(10, 3)
        shallow = store_td.clone(False)
        assert isinstance(shallow, TensorDictStore)
        assert shallow._td_id == store_td._td_id
        assert torch.allclose(shallow["obs"], store_td["obs"])

    def test_pickling(self, store_td):
        """Test pickle round-trip."""
        store_td["obs"] = torch.randn(10, 3)
        data = pickle.dumps(store_td)
        restored = pickle.loads(data)
        try:
            assert torch.allclose(restored["obs"], store_td["obs"])
            assert restored._td_id == store_td._td_id
        finally:
            restored.close()

    def test_entry_class(self, store_td):
        """Test entry_class returns correct types."""

        store_td["obs"] = torch.randn(10, 3)
        store_td["nested", "a"] = torch.randn(10, 2)
        assert store_td.entry_class("obs") is torch.Tensor
        assert store_td.entry_class("nested") is TensorDictStore

    def test_device(self, store_kwargs):
        """Test device attribute."""

        td = TensorDictStore(batch_size=[5], device="cpu", **store_kwargs)
        try:
            td["x"] = torch.randn(5, 3)
            assert td.device == torch.device("cpu")
            result = td["x"]
            assert result.device == torch.device("cpu")
        finally:
            td.clear_redis()
            td.close()

    def test_repr(self, store_td):
        """Test string representation."""
        store_td["obs"] = torch.randn(10, 3)
        s = repr(store_td)
        assert "TensorDictStore" in s
        assert "obs" in s

    def test_empty(self, store_td):
        """Test empty()."""
        store_td["obs"] = torch.randn(10, 3)
        empty = store_td.empty()
        assert isinstance(empty, TensorDict)
        assert empty.batch_size == torch.Size([10])
        assert len(list(empty.keys())) == 0

    def test_fill_(self, store_td):
        """Test fill_."""
        store_td["obs"] = torch.randn(10, 3)
        store_td.fill_("obs", 42.0)
        result = store_td["obs"]
        assert torch.allclose(result, torch.full((10, 3), 42.0))

    def test_is_contiguous(self, store_td):
        """Redis TDs are not contiguous."""
        assert not store_td.is_contiguous()

    def test_shape_ops_raise(self, store_td):
        """Shape ops raise on TensorDictStore but work after to_local()."""
        store_td["obs"] = torch.randn(10, 3)

        with pytest.raises(RuntimeError):
            store_td.view(2, 5)
        with pytest.raises(RuntimeError):
            store_td.permute(0)
        with pytest.raises(RuntimeError):
            store_td.unsqueeze(0)
        with pytest.raises(RuntimeError):
            store_td.squeeze(0)

        # Escape hatch: materialize first, then shape ops work
        local = store_td.to_local()
        assert local.view(2, 5).shape == torch.Size([2, 5])
        assert local.unsqueeze(0).shape == torch.Size([1, 10])
        assert local.squeeze(0).shape == torch.Size([10])

    def test_share_memory_raises(self, store_td):
        """share_memory_ should raise."""
        with pytest.raises(NotImplementedError):
            store_td.share_memory_()

    def test_reconnect_by_id(self, store_kwargs):
        """Connect to an existing TensorDictStore by ID."""

        td1 = TensorDictStore(batch_size=[5], **store_kwargs)
        try:
            td1["obs"] = torch.randn(5, 3)
            saved_obs = td1["obs"].clone()
            td_id = td1._td_id

            # Create a new instance pointing to the same data
            td2 = TensorDictStore(batch_size=[5], td_id=td_id, **store_kwargs)
            try:
                assert torch.allclose(td2["obs"], saved_obs)
            finally:
                td2.close()
        finally:
            td1.clear_redis()
            td1.close()

    def test_popitem(self, store_td):
        """Test popitem."""
        store_td["a"] = torch.randn(10, 2)
        store_td["b"] = torch.randn(10, 3)
        key, val = store_td.popitem()
        assert key in ("a", "b")
        assert isinstance(val, torch.Tensor)
        # Only one key should remain
        remaining = list(store_td.keys())
        assert len(remaining) == 1

    def test_rename_key(self, store_td):
        """Test rename_key_."""
        tensor = torch.randn(10, 3)
        store_td["old_name"] = tensor
        store_td.rename_key_("old_name", "new_name")
        assert "old_name" not in store_td.keys()
        assert "new_name" in store_td.keys()
        assert torch.allclose(store_td["new_name"], tensor)

    def test_from_tensordict(self, store_kwargs):
        """Test from_tensordict classmethod."""

        source = TensorDict(
            {"obs": torch.randn(5, 3), "action": torch.randn(5, 2)}, [5]
        )
        td = TensorDictStore.from_tensordict(source, **store_kwargs)
        try:
            assert td.batch_size == torch.Size([5])
            assert torch.allclose(td["obs"], source["obs"])
            assert torch.allclose(td["action"], source["action"])
        finally:
            td.clear_redis()
            td.close()

    def test_from_tensordict_preserves_device(self, store_kwargs):
        """from_tensordict should preserve the source device by default."""

        source = TensorDict({"x": torch.randn(3)}, [3], device="cpu")
        td = TensorDictStore.from_tensordict(source, **store_kwargs)
        try:
            assert td.device == torch.device("cpu")
        finally:
            td.clear_redis()
            td.close()

    def test_from_store(self, store_kwargs):
        """Test from_store: reconnect to existing data by td_id."""

        # Writer creates data
        writer = TensorDictStore(batch_size=[5], **store_kwargs)
        try:
            obs = torch.randn(5, 3)
            writer["obs"] = obs
            td_id = writer._td_id

            # Reader reconnects from a different handle
            reader = TensorDictStore.from_store(td_id=td_id, **store_kwargs)
            try:
                assert reader.batch_size == torch.Size([5])
                assert torch.allclose(reader["obs"], obs)
            finally:
                reader.close()
        finally:
            writer.clear_redis()
            writer.close()

    def test_from_store_not_found(self, store_kwargs):
        """from_store should raise KeyError for unknown td_id."""

        with pytest.raises(KeyError, match="No TensorDictStore"):
            TensorDictStore.from_store(td_id="nonexistent-uuid", **store_kwargs)

    def test_from_store_with_device_override(self, store_kwargs):
        """from_store should allow overriding the device."""

        writer = TensorDictStore(batch_size=[3], device="cpu", **store_kwargs)
        try:
            writer["x"] = torch.randn(3)
            td_id = writer._td_id

            reader = TensorDictStore.from_store(
                td_id=td_id, device="cpu", **store_kwargs
            )
            try:
                assert reader.device == torch.device("cpu")
                assert reader["x"].device == torch.device("cpu")
            finally:
                reader.close()
        finally:
            writer.clear_redis()
            writer.close()

    # ---- Byte-range indexed read tests ----

    def test_indexed_read_int(self, store_td):
        """td[i] should return the correct slice via GETRANGE."""
        obs = torch.randn(10, 3)
        store_td["obs"] = obs
        sub = store_td[5]
        result = sub["obs"]
        assert result.shape == torch.Size([3])
        assert torch.allclose(result, obs[5])

    def test_indexed_read_slice(self, store_td):
        """td[2:5] should return the correct slice via GETRANGE."""
        obs = torch.randn(10, 4)
        store_td["obs"] = obs
        sub = store_td[2:5]
        result = sub["obs"]
        assert result.shape == torch.Size([3, 4])
        assert torch.allclose(result, obs[2:5])

    def test_indexed_read_fancy(self, store_td):
        """td[tensor_idx] should return correct rows via GETRANGE."""
        obs = torch.randn(10, 3)
        store_td["obs"] = obs
        idx = torch.tensor([1, 4, 7])
        sub = store_td[idx]
        result = sub["obs"]
        assert result.shape == torch.Size([3, 3])
        assert torch.allclose(result, obs[idx])

    def test_indexed_read_bool_mask(self, store_td):
        """td[bool_mask] should return correct rows via GETRANGE."""
        obs = torch.randn(10, 3)
        store_td["obs"] = obs
        mask = torch.zeros(10, dtype=torch.bool)
        mask[0] = True
        mask[3] = True
        mask[9] = True
        sub = store_td[mask]
        result = sub["obs"]
        assert result.shape == torch.Size([3, 3])
        assert torch.allclose(result, obs[mask])

    def test_indexed_read_multiple_keys(self, store_td):
        """Indexed read should work across multiple leaf keys."""
        obs = torch.randn(10, 4)
        action = torch.randn(10, 2)
        store_td["obs"] = obs
        store_td["action"] = action
        sub = store_td[3]
        assert torch.allclose(sub["obs"], obs[3])
        assert torch.allclose(sub["action"], action[3])

    # ---- Byte-range indexed write tests ----

    def test_indexed_write_int(self, store_td):
        """td[5] = subtd should only modify row 5 via SETRANGE."""
        store_td["obs"] = torch.zeros(10, 3)
        store_td["action"] = torch.zeros(10, 2)

        new_obs = torch.ones(3)
        new_action = torch.ones(2) * 2.0
        store_td[5] = TensorDict({"obs": new_obs, "action": new_action}, [])

        full_obs = store_td["obs"]
        full_action = store_td["action"]
        # Row 5 should be updated
        assert torch.allclose(full_obs[5], new_obs)
        assert torch.allclose(full_action[5], new_action)
        # Other rows should be untouched
        assert torch.allclose(full_obs[:5], torch.zeros(5, 3))
        assert torch.allclose(full_obs[6:], torch.zeros(4, 3))

    def test_indexed_write_slice(self, store_td):
        """td[2:5] = subtd should modify rows 2-4 via SETRANGE."""
        store_td["obs"] = torch.zeros(10, 3)

        new_vals = torch.ones(3, 3) * 7.0
        store_td[2:5] = TensorDict({"obs": new_vals}, [3])

        full = store_td["obs"]
        assert torch.allclose(full[2:5], new_vals)
        assert torch.allclose(full[:2], torch.zeros(2, 3))
        assert torch.allclose(full[5:], torch.zeros(5, 3))

    def test_indexed_write_list(self, store_td):
        """td[[0, 3, 7]] = subtd should modify selected rows."""
        store_td["obs"] = torch.zeros(10, 3)
        idx = [0, 3, 7]
        new_vals = torch.ones(3, 3) * 5.0
        store_td[idx] = TensorDict({"obs": new_vals}, [3])

        full = store_td["obs"]
        for i, pos in enumerate(idx):
            assert torch.allclose(full[pos], new_vals[i])
        # Unselected rows should be zero
        for pos in [1, 2, 4, 5, 6, 8, 9]:
            assert torch.allclose(full[pos], torch.zeros(3))

    def test_indexed_write_tensor(self, store_td):
        """td[tensor_idx] = subtd should modify selected rows."""
        store_td["obs"] = torch.zeros(10, 4)
        idx = torch.tensor([2, 5, 8])
        new_vals = torch.ones(3, 4) * 3.0
        store_td[idx] = TensorDict({"obs": new_vals}, [3])

        full = store_td["obs"]
        assert torch.allclose(full[idx], new_vals)
        # Check untouched
        untouched = torch.tensor([0, 1, 3, 4, 6, 7, 9])
        assert torch.allclose(full[untouched], torch.zeros(7, 4))

    def test_indexed_write_bool_mask(self, store_td):
        """td[mask] = subtd should modify masked rows."""
        store_td["obs"] = torch.zeros(10, 3)
        mask = torch.zeros(10, dtype=torch.bool)
        mask[1] = True
        mask[4] = True
        mask[9] = True
        new_vals = torch.ones(3, 3) * 9.0
        store_td[mask] = TensorDict({"obs": new_vals}, [3])

        full = store_td["obs"]
        assert torch.allclose(full[mask], new_vals)
        assert torch.allclose(full[~mask], torch.zeros(7, 3))

    def test_indexed_write_ellipsis(self, store_td):
        """td[...] = subtd should overwrite all rows via SETRANGE."""
        store_td["obs"] = torch.zeros(10, 3)
        new_vals = torch.ones(10, 3) * 4.0
        store_td[...] = TensorDict({"obs": new_vals}, [10])
        assert torch.allclose(store_td["obs"], new_vals)

    def test_indexed_read_ellipsis(self, store_td):
        """td[...] should return all rows."""
        obs = torch.randn(10, 3)
        store_td["obs"] = obs
        sub = store_td[...]
        assert torch.allclose(sub["obs"], obs)

    def test_indexed_read_step_slice(self, store_td):
        """td[::2] should return every other row via covering range."""
        obs = torch.randn(10, 3)
        store_td["obs"] = obs
        sub = store_td[::2]
        assert sub["obs"].shape == torch.Size([5, 3])
        assert torch.allclose(sub["obs"], obs[::2])

    def test_indexed_read_step3(self, store_td):
        """td[1::3] should fetch covering range and stride locally."""
        obs = torch.randn(10, 3)
        store_td["obs"] = obs
        sub = store_td[1::3]
        assert sub["obs"].shape == torch.Size([3, 3])
        assert torch.allclose(sub["obs"], obs[1::3])

    def test_indexed_write_step_slice(self, store_td):
        """td[::2] = subtd should use partial covering-range RMW."""
        store_td["obs"] = torch.zeros(10, 3)
        new_vals = torch.ones(5, 3) * 9.0
        store_td[::2] = TensorDict({"obs": new_vals}, [5])

        full = store_td["obs"]
        assert torch.allclose(full[::2], new_vals)
        # Odd rows should remain zero
        assert torch.allclose(full[1::2], torch.zeros(5, 3))

    def test_indexed_write_step3(self, store_td):
        """td[1::3] = subtd should use partial covering-range RMW."""
        store_td["obs"] = torch.arange(30, dtype=torch.float).reshape(10, 3)
        original = store_td["obs"].clone()
        new_vals = torch.ones(3, 3) * -1.0
        store_td[1::3] = TensorDict({"obs": new_vals}, [3])

        full = store_td["obs"]
        assert torch.allclose(full[1::3], new_vals)
        # Unmodified rows should stay the same
        for i in range(10):
            if i not in (1, 4, 7):
                assert torch.allclose(full[i], original[i])

    # ---- set_at_ via byte-range ----

    def test_set_at_byte_range(self, store_td):
        """set_at_ should use SETRANGE for a single key."""
        store_td["obs"] = torch.zeros(10, 3)
        new_val = torch.ones(3) * 42.0
        store_td.set_at_("obs", new_val, 3)

        full = store_td["obs"]
        assert torch.allclose(full[3], new_val)
        assert torch.allclose(full[:3], torch.zeros(3, 3))
        assert torch.allclose(full[4:], torch.zeros(6, 3))

    # ---- Metadata caching tests ----

    def test_cache_metadata_default(self, store_td):
        """Metadata cache should be populated after writes by default."""
        store_td["obs"] = torch.randn(10, 3)
        assert store_td._meta_cache is not None
        # The cache should contain the key's metadata
        key_path = store_td._full_key_path("obs")
        assert key_path in store_td._meta_cache
        shape, dtype = store_td._meta_cache[key_path]
        assert shape == [10, 3]
        assert dtype == torch.float32

    def test_cache_metadata_disabled(self, store_kwargs):
        """cache_metadata=False should disable the local cache."""
        td = TensorDictStore(batch_size=[5], cache_metadata=False, **store_kwargs)
        try:
            assert td._meta_cache is None
            td["x"] = torch.randn(5, 3)
            assert td._meta_cache is None
        finally:
            td.clear_redis()
            td.close()

    def test_cache_evicted_on_delete(self, store_td):
        """Deleting a key should evict it from the metadata cache."""
        store_td["obs"] = torch.randn(10, 3)
        key_path = store_td._full_key_path("obs")
        assert key_path in store_td._meta_cache
        store_td.del_("obs")
        assert key_path not in store_td._meta_cache


class TestLazyStackedTensorDictStore:
    """Tests for LazyStackedTensorDictStore requiring a running store server."""

    @pytest.fixture(autouse=True)
    def store_stack(self, store_kwargs):
        """Create a LazyStackedTensorDictStore from a homogeneous lazy stack."""
        tds = [
            TensorDict({"a": torch.randn(4, 3), "b": torch.randn(4)}, batch_size=[4])
            for _ in range(5)
        ]
        lazy_td = lazy_stack(tds)
        store_td = LazyStackedTensorDictStore.from_lazy_stack(lazy_td, **store_kwargs)
        yield store_td, tds, lazy_td
        store_td.clear_redis()
        store_td.close()

    # ---- Construction ----

    def test_from_lazy_stack_batch_size(self, store_stack):
        store_td, tds, lazy_td = store_stack
        assert store_td.batch_size == torch.Size([5, 4])
        assert store_td._count == 5
        assert store_td._stack_dim == 0
        assert store_td._inner_batch_size == torch.Size([4])

    def test_from_lazy_stack_keys(self, store_stack):
        store_td, tds, lazy_td = store_stack
        assert set(store_td.keys()) == {"a", "b"}

    def test_repr(self, store_stack):
        store_td, _, _ = store_stack
        r = repr(store_td)
        assert "LazyStackedTensorDictStore" in r
        assert "count=5" in r

    # ---- to_store() convenience ----

    def test_to_store_from_lazy_stack(self, store_kwargs):
        tds = [TensorDict({"x": torch.randn(3, 2)}, batch_size=[3]) for _ in range(4)]
        lazy_td = lazy_stack(tds)
        store_td = lazy_td.to_store(**store_kwargs)
        try:
            assert isinstance(store_td, LazyStackedTensorDictStore)
            assert store_td.batch_size == torch.Size([4, 3])
            result = store_td["x"]
            assert result.shape == torch.Size([4, 3, 2])
        finally:
            store_td.clear_redis()
            store_td.close()

    # ---- Read: td[int] ----

    def test_getitem_int(self, store_stack):
        store_td, tds, lazy_td = store_stack
        elem = store_td[0]
        assert isinstance(elem, TensorDictBase)
        assert elem.batch_size == torch.Size([4])
        assert torch.allclose(elem["a"], tds[0]["a"])
        assert torch.allclose(elem["b"], tds[0]["b"])

    def test_getitem_int_last(self, store_stack):
        store_td, tds, lazy_td = store_stack
        elem = store_td[4]
        assert torch.allclose(elem["a"], tds[4]["a"])

    def test_getitem_int_negative(self, store_stack):
        store_td, tds, lazy_td = store_stack
        elem = store_td[-1]
        assert torch.allclose(elem["a"], tds[4]["a"])

    # ---- Read: td[key] ----

    def test_getitem_key(self, store_stack):
        store_td, tds, lazy_td = store_stack
        full_a = store_td["a"]
        assert full_a.shape == torch.Size([5, 4, 3])
        for i in range(5):
            assert torch.allclose(full_a[i], tds[i]["a"])

    def test_getitem_key_1d(self, store_stack):
        store_td, tds, lazy_td = store_stack
        full_b = store_td["b"]
        assert full_b.shape == torch.Size([5, 4])
        for i in range(5):
            assert torch.allclose(full_b[i], tds[i]["b"])

    # ---- Read: td[int][key] ----

    def test_getitem_int_then_key(self, store_stack):
        store_td, tds, lazy_td = store_stack
        val = store_td[0]["a"]
        assert val.shape == torch.Size([4, 3])
        assert torch.allclose(val, tds[0]["a"])

    # ---- Read: td[slice] ----

    def test_getitem_slice(self, store_stack):
        store_td, tds, lazy_td = store_stack
        sub = store_td[1:3]
        assert isinstance(sub, TensorDictBase)
        assert sub.batch_size == torch.Size([2, 4])
        assert torch.allclose(sub["a"][0], tds[1]["a"])
        assert torch.allclose(sub["a"][1], tds[2]["a"])

    def test_getitem_slice_step(self, store_stack):
        store_td, tds, lazy_td = store_stack
        sub = store_td[::2]
        assert sub.batch_size == torch.Size([3, 4])
        assert torch.allclose(sub["a"][0], tds[0]["a"])
        assert torch.allclose(sub["a"][1], tds[2]["a"])
        assert torch.allclose(sub["a"][2], tds[4]["a"])

    # ---- Read: td[tensor_index] ----

    def test_getitem_tensor_index(self, store_stack):
        store_td, tds, lazy_td = store_stack
        idx = torch.tensor([0, 3, 4])
        sub = store_td[idx]
        assert isinstance(sub, TensorDictBase)
        assert sub.batch_size == torch.Size([3, 4])
        assert torch.allclose(sub["a"][0], tds[0]["a"])
        assert torch.allclose(sub["a"][1], tds[3]["a"])
        assert torch.allclose(sub["a"][2], tds[4]["a"])

    # ---- Write: td[int] = subtd ----

    def test_setitem_int(self, store_stack):
        store_td, tds, lazy_td = store_stack
        new_a = torch.ones(4, 3)
        new_b = torch.ones(4)
        store_td[0] = TensorDict({"a": new_a, "b": new_b}, batch_size=[4])
        elem = store_td[0]
        assert torch.allclose(elem["a"], new_a)
        assert torch.allclose(elem["b"], new_b)
        # Other elements unchanged
        assert torch.allclose(store_td[1]["a"], tds[1]["a"])

    # ---- Write: td[slice] = subtd ----

    def test_setitem_slice(self, store_stack):
        store_td, tds, lazy_td = store_stack
        new_val = TensorDict(
            {"a": torch.zeros(2, 4, 3), "b": torch.zeros(2, 4)},
            batch_size=[2, 4],
        )
        store_td[1:3] = new_val
        assert torch.allclose(store_td[1]["a"], torch.zeros(4, 3))
        assert torch.allclose(store_td[2]["a"], torch.zeros(4, 3))
        # Unchanged
        assert torch.allclose(store_td[0]["a"], tds[0]["a"])

    # ---- Write: td[tensor_index] = subtd ----

    def test_setitem_tensor_index(self, store_stack):
        store_td, tds, lazy_td = store_stack
        idx = torch.tensor([0, 4])
        new_val = TensorDict(
            {"a": torch.ones(2, 4, 3), "b": torch.ones(2, 4)},
            batch_size=[2, 4],
        )
        store_td[idx] = new_val
        assert torch.allclose(store_td[0]["a"], torch.ones(4, 3))
        assert torch.allclose(store_td[4]["a"], torch.ones(4, 3))
        # Middle element unchanged
        assert torch.allclose(store_td[2]["a"], tds[2]["a"])

    # ---- to_tensordict / to_local ----

    def test_to_tensordict(self, store_stack):
        store_td, tds, lazy_td = store_stack
        local = store_td.to_tensordict()
        assert isinstance(local, TensorDict)
        assert local.batch_size == torch.Size([5, 4])
        for i in range(5):
            assert torch.allclose(local["a"][i], tds[i]["a"])
            assert torch.allclose(local["b"][i], tds[i]["b"])

    def test_to_local(self, store_stack):
        store_td, tds, lazy_td = store_stack
        local = store_td.to_local()
        assert isinstance(local, TensorDict)
        assert local.batch_size == torch.Size([5, 4])

    # ---- td[idx].to_tensordict() pattern ----

    def test_indexed_to_tensordict(self, store_stack):
        store_td, tds, lazy_td = store_stack
        local = store_td[1:3].to_tensordict()
        assert local.batch_size == torch.Size([2, 4])
        assert torch.allclose(local["a"][0], tds[1]["a"])

    # ---- Heterogeneous shapes ----

    def test_heterogeneous_shapes(self, store_kwargs):
        """Test lazy stack with different feature dims per element."""
        td0 = TensorDict({"a": torch.randn(3, 4)}, batch_size=[3])
        td1 = TensorDict({"a": torch.randn(3, 8)}, batch_size=[3])
        lazy_td = lazy_stack([td0, td1])
        store_td = LazyStackedTensorDictStore.from_lazy_stack(lazy_td, **store_kwargs)
        try:
            assert store_td.batch_size == torch.Size([2, 3])
            # Read element 0
            elem0 = store_td[0]
            assert torch.allclose(elem0["a"], td0["a"])
            # Read element 1
            elem1 = store_td[1]
            assert torch.allclose(elem1["a"], td1["a"])
        finally:
            store_td.clear_redis()
            store_td.close()

    # ---- Pickling ----

    def test_pickle_roundtrip(self, store_stack):
        store_td, tds, lazy_td = store_stack
        data = pickle.dumps(store_td)
        restored = pickle.loads(data)
        try:
            assert restored.batch_size == store_td.batch_size
            assert restored._count == store_td._count
            assert torch.allclose(restored[0]["a"], tds[0]["a"])
        finally:
            restored.close()

    # ---- from_store reconnect ----

    def test_from_store_reconnect(self, store_stack, store_kwargs):
        store_td, tds, lazy_td = store_stack
        td_id = store_td._td_id
        restored = LazyStackedTensorDictStore.from_store(td_id=td_id, **store_kwargs)
        try:
            assert restored.batch_size == store_td.batch_size
            assert restored._count == 5
            assert torch.allclose(restored[0]["a"], tds[0]["a"])
        finally:
            restored.close()

    # ---- Nested keys ----

    def test_nested_keys(self, store_kwargs):
        """Test lazy stack with nested TensorDicts."""
        tds = [
            TensorDict(
                {
                    "obs": torch.randn(3),
                    "nested": TensorDict({"x": torch.randn(3, 2)}, [3]),
                },
                batch_size=[3],
            )
            for _ in range(4)
        ]
        lazy_td = lazy_stack(tds)
        store_td = LazyStackedTensorDictStore.from_lazy_stack(lazy_td, **store_kwargs)
        try:
            assert store_td.batch_size == torch.Size([4, 3])
            assert "obs" in store_td.keys()
            # Read element
            elem = store_td[0]
            assert torch.allclose(elem["obs"], tds[0]["obs"])
            assert torch.allclose(elem[("nested", "x")], tds[0]["nested", "x"])
        finally:
            store_td.clear_redis()
            store_td.close()

    # ---- Write-through view tests ----

    def test_view_set_propagates(self, store_stack):
        """rltd[0].set('a', val) should propagate to Redis."""
        store_td, tds, lazy_td = store_stack
        new_a = torch.ones(4, 3) * 42.0
        view = store_td[0]
        view.set("a", new_a, inplace=True)
        # Re-read: should see the change
        reread = store_td[0]["a"]
        assert torch.allclose(reread, new_a)
        # Other elements unaffected
        assert torch.allclose(store_td[1]["a"], tds[1]["a"])

    def test_view_setitem_key_propagates(self, store_stack):
        """rltd[0]['a'] = val should propagate to Redis."""
        store_td, tds, lazy_td = store_stack
        new_a = torch.ones(4, 3) * 99.0
        view = store_td[0]
        view["a"] = new_a
        reread = store_td[0]["a"]
        assert torch.allclose(reread, new_a)

    def test_view_shape_change_raises(self, store_stack):
        """Changing element shape through the view should raise."""
        store_td, tds, lazy_td = store_stack
        view = store_td[0]
        with pytest.raises((ValueError, RuntimeError)):
            view.set("a", torch.randn(10, 10), inplace=True)

    def test_view_to_tensordict(self, store_stack):
        """view.to_tensordict() should return a regular TensorDict."""
        store_td, tds, lazy_td = store_stack
        view = store_td[0]
        local = view.to_tensordict()
        assert isinstance(local, TensorDict)
        assert local.batch_size == torch.Size([4])
        assert torch.allclose(local["a"], tds[0]["a"])

    def test_view_nested_set(self, store_kwargs):
        """Write-through on nested keys."""
        tds = [
            TensorDict(
                {
                    "obs": torch.randn(3),
                    "nested": TensorDict({"x": torch.randn(3, 2)}, [3]),
                },
                batch_size=[3],
            )
            for _ in range(4)
        ]
        lazy_td = lazy_stack(tds)
        store_td = LazyStackedTensorDictStore.from_lazy_stack(lazy_td, **store_kwargs)
        try:
            view = store_td[0]
            new_x = torch.ones(3, 2) * 7.0
            view.set(("nested", "x"), new_x, inplace=True)
            reread = store_td[0][("nested", "x")]
            assert torch.allclose(reread, new_x)
        finally:
            store_td.clear_redis()
            store_td.close()


class TestBackendParam:
    """Tests for backend parameter."""

    def test_backend_default(self, store_kwargs):
        td = TensorDictStore(batch_size=[5], **store_kwargs)
        try:
            assert "backend=" in repr(td)
        finally:
            td.clear_redis()
            td.close()

    def test_backend_from_tensordict(self, store_kwargs):
        local = TensorDict({"a": torch.randn(5)}, [5])
        td = TensorDictStore.from_tensordict(local, **store_kwargs)
        try:
            assert td._backend == store_kwargs["backend"]
        finally:
            td.clear_redis()
            td.close()

    def test_backend_to_store(self, store_kwargs):
        local = TensorDict({"a": torch.randn(5)}, [5])
        td = local.to_store(**store_kwargs)
        try:
            assert td._backend == store_kwargs["backend"]
        finally:
            td.clear_redis()
            td.close()

    def test_pickle_preserves_backend(self, store_kwargs):
        td = TensorDictStore(batch_size=[5], **store_kwargs)
        td["x"] = torch.randn(5, 3)
        try:
            data = pickle.dumps(td)
            restored = pickle.loads(data)
            assert restored._backend == store_kwargs["backend"]
            assert torch.allclose(restored["x"], td["x"])
            restored.close()
        finally:
            td.clear_redis()
            td.close()

    def test_lazy_stack_backend(self, store_kwargs):
        tds = [TensorDict({"a": torch.randn(4)}, batch_size=[4]) for _ in range(3)]
        ltd = lazy_stack(tds)
        rltd = LazyStackedTensorDictStore.from_lazy_stack(ltd, **store_kwargs)
        try:
            assert rltd._backend == store_kwargs["backend"]
            assert "backend=" in repr(rltd)
        finally:
            rltd.clear_redis()
            rltd.close()


from tensordict import tensorclass


@tensorclass
class _MyData:
    obs: torch.Tensor
    reward: torch.Tensor


@tensorclass
class _MyDataWithNonTensor:
    obs: torch.Tensor
    label: str


class TestTensorClassStore:
    """Tests for TensorClass round-trip through TensorDictStore."""

    def test_from_tensordict_stores_class_path(self, store_kwargs):
        tc = _MyData(obs=torch.randn(5, 3), reward=torch.randn(5), batch_size=[5])
        store = TensorDictStore.from_tensordict(tc, **store_kwargs)
        try:
            assert store._tensorclass_cls is not None
            assert "_MyData" in store._tensorclass_cls
            assert torch.allclose(store["obs"], tc.obs)
            assert torch.allclose(store["reward"], tc.reward)
        finally:
            store.clear_redis()
            store.close()

    def test_to_store_returns_tensorclass(self, store_kwargs):
        """tc.to_store() returns a TensorClass wrapping a TensorDictStore."""
        tc = _MyData(obs=torch.randn(5, 3), reward=torch.randn(5), batch_size=[5])
        result = tc.to_store(**store_kwargs)
        try:
            assert type(result).__name__ == "_MyData"
            assert isinstance(result._tensordict, TensorDictStore)
            assert torch.allclose(result.obs, tc.obs)
            assert torch.allclose(result.reward, tc.reward)
        finally:
            result._tensordict.clear_redis()
            result._tensordict.close()

    def test_from_store_auto_import(self, store_kwargs):
        tc = _MyData(obs=torch.randn(5, 3), reward=torch.randn(5), batch_size=[5])
        store = TensorDictStore.from_tensordict(tc, **store_kwargs)
        td_id = store._td_id
        try:
            restored = TensorDictStore.from_store(td_id=td_id, **store_kwargs)
            assert type(restored).__name__ == "_MyData"
            assert isinstance(restored._tensordict, TensorDictStore)
            assert torch.allclose(restored.obs, tc.obs)
            assert torch.allclose(restored.reward, tc.reward)
        finally:
            store.clear_redis()
            store.close()

    def test_from_store_explicit_cls(self, store_kwargs):
        tc = _MyData(obs=torch.randn(5, 3), reward=torch.randn(5), batch_size=[5])
        store = TensorDictStore.from_tensordict(tc, **store_kwargs)
        td_id = store._td_id
        try:
            restored = TensorDictStore.from_store(
                td_id=td_id, tensorclass_cls=_MyData, **store_kwargs
            )
            assert type(restored).__name__ == "_MyData"
            assert torch.allclose(restored.obs, tc.obs)
        finally:
            store.clear_redis()
            store.close()

    def test_from_store_explicit_cls_string(self, store_kwargs):
        tc = _MyData(obs=torch.randn(5, 3), reward=torch.randn(5), batch_size=[5])
        store = TensorDictStore.from_tensordict(tc, **store_kwargs)
        td_id = store._td_id
        cls_path = f"{_MyData.__module__}.{_MyData.__qualname__}"
        try:
            restored = TensorDictStore.from_store(
                td_id=td_id, tensorclass_cls=cls_path, **store_kwargs
            )
            assert type(restored).__name__ == "_MyData"
            assert torch.allclose(restored.obs, tc.obs)
        finally:
            store.clear_redis()
            store.close()

    def test_from_store_no_tensorclass_returns_store(self, store_kwargs):
        td = TensorDict({"x": torch.randn(5, 3)}, batch_size=[5])
        store = TensorDictStore.from_tensordict(td, **store_kwargs)
        td_id = store._td_id
        try:
            restored = TensorDictStore.from_store(td_id=td_id, **store_kwargs)
            assert type(restored) is TensorDictStore
        finally:
            store.clear_redis()
            store.close()

    def test_tensorclass_attribute_access_through_store(self, store_kwargs):
        tc = _MyData(obs=torch.randn(5, 3), reward=torch.randn(5), batch_size=[5])
        store = TensorDictStore.from_tensordict(tc, **store_kwargs)
        td_id = store._td_id
        try:
            restored = TensorDictStore.from_store(td_id=td_id, **store_kwargs)
            # Attribute access should work and fetch from Redis
            assert restored.obs.shape == torch.Size([5, 3])
            assert restored.reward.shape == torch.Size([5])
            # Write through attribute
            new_obs = torch.zeros(5, 3)
            restored.obs = new_obs
            assert torch.allclose(restored.obs, new_obs)
        finally:
            store.clear_redis()
            store.close()

    def test_tensorclass_non_tensor_fields(self, store_kwargs):
        tc = _MyDataWithNonTensor(obs=torch.randn(5, 3), label="hello", batch_size=[5])
        store = TensorDictStore.from_tensordict(tc, **store_kwargs)
        td_id = store._td_id
        try:
            restored = TensorDictStore.from_store(td_id=td_id, **store_kwargs)
            assert type(restored).__name__ == "_MyDataWithNonTensor"
            assert torch.allclose(restored.obs, tc.obs)
            assert restored.label == "hello"
        finally:
            store.clear_redis()
            store.close()

    def test_pickle_tensorclass_store(self, store_kwargs):
        tc = _MyData(obs=torch.randn(5, 3), reward=torch.randn(5), batch_size=[5])
        store = TensorDictStore.from_tensordict(tc, **store_kwargs)
        try:
            assert store._tensorclass_cls is not None
            data = pickle.dumps(store)
            restored = pickle.loads(data)
            assert restored._tensorclass_cls == store._tensorclass_cls
            assert torch.allclose(restored["obs"], store["obs"])
            restored.close()
        finally:
            store.clear_redis()
            store.close()


class TestNonTensorIndexing:
    """Tests for per-element non-tensor indexing (ISSUE #1) and writing
    to empty stores at an index (ISSUE #2)."""

    def test_write_to_empty_store_at_index(self, store_kwargs):
        """ISSUE #2: ``store[i] = td`` must work even when the store has no
        keys yet (first indexed write)."""
        store = TensorDictStore(batch_size=[5], **store_kwargs)
        try:
            store[0] = TensorDict({"obs": torch.randn(4)}, [])
            assert store[0]["obs"].shape == torch.Size([4])
            # Other elements should be zero-initialised
            assert (store[1]["obs"] == 0).all()
        finally:
            store.clear_redis()
            store.close()

    def test_non_tensor_per_element_read(self, store_kwargs):
        """ISSUE #1: ``store[i]["label"]`` must return the i-th element,
        not the whole blob."""
        td = TensorDict({"obs": torch.zeros(5, 4), "label": "placeholder"}, [5])
        store = TensorDictStore.from_tensordict(td, **store_kwargs)
        try:
            # Before any per-element write, all elements should be the same
            assert store[0]["label"] == "placeholder"
            assert store[3]["label"] == "placeholder"
        finally:
            store.clear_redis()
            store.close()

    def test_non_tensor_per_element_write(self, store_kwargs):
        """ISSUE #1: ``store[i] = td`` must update the non-tensor field
        at index *i* only."""
        td = TensorDict({"obs": torch.zeros(5, 4), "label": "placeholder"}, [5])
        store = TensorDictStore.from_tensordict(td, **store_kwargs)
        try:
            store[3] = TensorDict({"obs": torch.ones(4), "label": "cat"}, [])
            assert store[3]["label"] == "cat"
            assert store[0]["label"] == "placeholder"
            assert store[4]["label"] == "placeholder"
        finally:
            store.clear_redis()
            store.close()

    def test_non_tensor_slice_write(self, store_kwargs):
        """Per-element write with a slice index."""
        td = TensorDict({"obs": torch.zeros(5, 4), "label": "a"}, [5])
        store = TensorDictStore.from_tensordict(td, **store_kwargs)
        try:
            store[1:3] = TensorDict({"obs": torch.ones(2, 4), "label": "b"}, [2])
            assert store[0]["label"] == "a"
            assert store[1]["label"] == "b"
            assert store[2]["label"] == "b"
            assert store[3]["label"] == "a"
        finally:
            store.clear_redis()
            store.close()

    def test_non_tensor_to_tensordict_after_write(self, store_kwargs):
        """``to_tensordict()`` must reflect per-element non-tensor writes."""
        td = TensorDict({"val": torch.randn(3, 2), "tag": "x"}, [3])
        store = TensorDictStore.from_tensordict(td, **store_kwargs)
        try:
            store[1] = TensorDict({"val": torch.zeros(2), "tag": "y"}, [])
            local = store.to_tensordict()
            tags = [local[i]["tag"] for i in range(3)]
            assert tags == ["x", "y", "x"]
        finally:
            store.clear_redis()
            store.close()

    def test_write_non_tensor_to_empty_store_at_index(self, store_kwargs):
        """ISSUE #2 + non-tensor: writing non-tensor to a fresh store at
        an index must work."""
        store = TensorDictStore(batch_size=[4], **store_kwargs)
        try:
            store[0] = TensorDict({"obs": torch.randn(3), "label": "hello"}, [])
            assert store[0]["obs"].shape == torch.Size([3])
            assert store[0]["label"] == "hello"
            assert store[1]["label"] is None  # uninitialised slot
        finally:
            store.clear_redis()
            store.close()

    def test_setitem_with_nested_tensorclass(self, store_kwargs):
        """Nested TensorClass values must not be misclassified as non-tensor leaves.

        TensorClass does not inherit from TensorDictBase, so an isinstance check
        against TensorDictBase would wrongly route it to non-tensor serialisation.
        """

        @tensorclass
        class History:
            obs: torch.Tensor
            label: str

        @tensorclass
        class Transition:
            obs: torch.Tensor
            history: History

        history = History(obs=torch.randn(3), label="step0", batch_size=[])
        t = Transition(obs=torch.randn(4), history=history, batch_size=[])

        td = TensorDict({"obs": torch.zeros(5, 4)}, [5])
        td["history"] = TensorDict({"obs": torch.zeros(5, 3)}, [5])
        td["history"].set_non_tensor("label", "init")

        store = TensorDictStore.from_tensordict(td, **store_kwargs)
        try:
            store[0] = t.to_tensordict()
            recovered = store[0]
            assert torch.allclose(recovered["obs"], t.obs)
            assert torch.allclose(recovered["history", "obs"], history.obs)
            assert recovered.get_non_tensor(("history", "label")) == "step0"
        finally:
            store.clear_redis()
            store.close()


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
