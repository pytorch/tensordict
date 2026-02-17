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
from tensordict.store import (
    LazyStackedTensorDictStore,
    LazyStackedTensorDictStore as RedisLazyStackedTensorDict,
    TensorDictStore,
    TensorDictStore as RedisTensorDict,
)

_has_redis = importlib.util.find_spec("redis", None) is not None


def _redis_available():
    """Check if a Redis server is reachable on localhost:6379."""
    if not _has_redis:
        return False
    import redis

    try:
        r = redis.Redis(host="localhost", port=6379, db=0, socket_connect_timeout=2)
        r.ping()
        r.close()
        return True
    except (redis.ConnectionError, redis.exceptions.ConnectionError, OSError):
        return False


_skip_no_redis_pkg = pytest.mark.skipif(
    not _has_redis, reason="redis package not installed."
)
_skip_no_redis_server = pytest.mark.skipif(
    not _redis_available(), reason="Redis server not reachable on localhost:6379."
)

skip_redis = pytest.mark.usefixtures()


@_skip_no_redis_pkg
@_skip_no_redis_server
class TestRedisTensorDict:
    """Tests for RedisTensorDict requiring a running Redis server."""

    @pytest.fixture(autouse=True)
    def redis_td(self):
        """Create a fresh RedisTensorDict and clean up after the test."""

        td = RedisTensorDict(batch_size=[10], db=15)
        yield td
        td.clear_redis()
        td.close()

    def test_basic_set_get(self, redis_td):
        """Store and retrieve a tensor."""
        tensor = torch.randn(10, 3)
        redis_td["obs"] = tensor
        result = redis_td["obs"]
        assert isinstance(result, torch.Tensor)
        assert result.shape == (10, 3)
        assert torch.allclose(result, tensor)

    def test_multiple_keys(self, redis_td):
        """Store and retrieve multiple tensors."""
        obs = torch.randn(10, 84)
        action = torch.randn(10, 4)
        reward = torch.randn(10, 1)
        redis_td["obs"] = obs
        redis_td["action"] = action
        redis_td["reward"] = reward

        assert torch.allclose(redis_td["obs"], obs)
        assert torch.allclose(redis_td["action"], action)
        assert torch.allclose(redis_td["reward"], reward)

    def test_dtypes(self, redis_td):
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
            redis_td[key] = tensor
            result = redis_td[key]
            assert result.dtype == dtype, f"Failed for dtype {dtype}"
            assert result.shape == (10, 5)

    def test_nested_keys(self, redis_td):
        """Store and retrieve tensors with nested keys."""
        tensor = torch.randn(10, 4)
        redis_td["nested", "obs"] = tensor

        # Access the nested value
        result = redis_td["nested", "obs"]
        assert torch.allclose(result, tensor)

        # Access the nested tensordict
        nested = redis_td["nested"]

        assert isinstance(nested, RedisTensorDict)
        result2 = nested["obs"]
        assert torch.allclose(result2, tensor)

    def test_nested_from_tensordict(self, redis_td):
        """Store a nested TensorDict."""
        inner = TensorDict({"a": torch.randn(10, 3), "b": torch.randn(10, 2)}, [10])
        redis_td["inner"] = inner

        assert torch.allclose(redis_td["inner", "a"], inner["a"])
        assert torch.allclose(redis_td["inner", "b"], inner["b"])

    def test_batch_size(self, redis_td):
        """Verify batch_size is correctly stored and accessible."""
        assert redis_td.batch_size == torch.Size([10])

    def test_from_dict(self):
        """Construct from a dict."""

        source = {"obs": torch.randn(5, 3), "reward": torch.randn(5, 1)}
        td = RedisTensorDict.from_dict(source, batch_size=[5], db=15)
        try:
            assert td.batch_size == torch.Size([5])
            assert torch.allclose(td["obs"], source["obs"])
            assert torch.allclose(td["reward"], source["reward"])
        finally:
            td.clear_redis()
            td.close()

    def test_from_dict_tensordict_input(self):
        """Construct via from_dict with a TensorDict as input."""

        source = TensorDict(
            {"obs": torch.randn(5, 3), "reward": torch.randn(5, 1)}, [5]
        )
        td = RedisTensorDict.from_dict(source, db=15)
        try:
            assert td.batch_size == torch.Size([5])
            assert torch.allclose(td["obs"], source["obs"])
            assert torch.allclose(td["reward"], source["reward"])
        finally:
            td.clear_redis()
            td.close()

    def test_to_local(self, redis_td):
        """Materialize to a local TensorDict."""
        obs = torch.randn(10, 4)
        action = torch.randn(10, 2)
        redis_td["obs"] = obs
        redis_td["action"] = action

        local = redis_td.to_local()
        assert isinstance(local, TensorDict)
        assert local.batch_size == torch.Size([10])
        assert torch.allclose(local["obs"], obs)
        assert torch.allclose(local["action"], action)

    def test_to_tensordict(self, redis_td):
        """to_tensordict should be equivalent to to_local."""
        tensor = torch.randn(10, 3)
        redis_td["x"] = tensor

        local = redis_td.to_tensordict()
        assert isinstance(local, TensorDict)
        assert torch.allclose(local["x"], tensor)

    def test_contiguous(self, redis_td):
        """contiguous should materialize."""
        tensor = torch.randn(10, 3)
        redis_td["x"] = tensor
        local = redis_td.contiguous()
        assert isinstance(local, TensorDict)
        assert torch.allclose(local["x"], tensor)

    def test_keys_view(self, redis_td):
        """Test keys iteration."""
        redis_td["a"] = torch.randn(10, 2)
        redis_td["b"] = torch.randn(10, 3)
        redis_td["nested", "c"] = torch.randn(10, 4)

        # Top-level keys (non-nested)
        keys = set(redis_td.keys())
        assert "a" in keys
        assert "b" in keys
        assert "nested" in keys

    def test_keys_include_nested(self, redis_td):
        """Test keys with include_nested=True."""
        redis_td["a"] = torch.randn(10, 2)
        redis_td["nested", "b"] = torch.randn(10, 3)

        nested_keys = set(redis_td.keys(include_nested=True))
        assert "a" in nested_keys
        # The nested key should appear as a tuple
        assert ("nested", "b") in nested_keys or "nested" in nested_keys

    def test_keys_leaves_only(self, redis_td):
        """Test keys with leaves_only=True."""
        redis_td["a"] = torch.randn(10, 2)
        redis_td["nested", "b"] = torch.randn(10, 3)

        leaf_keys = list(redis_td.keys(leaves_only=True))
        assert "a" in leaf_keys
        # "nested" should not appear since it is not a leaf
        assert "nested" not in leaf_keys

    def test_contains(self, redis_td):
        """Test __contains__ via 'in' operator."""
        redis_td["obs"] = torch.randn(10, 3)
        redis_td["nested", "val"] = torch.randn(10, 2)

        assert "obs" in redis_td.keys()
        assert "nested" in redis_td.keys()
        assert ("nested", "val") in redis_td.keys(include_nested=True)
        assert "nonexistent" not in redis_td.keys()

    def test_del(self, redis_td):
        """Test key deletion."""
        redis_td["obs"] = torch.randn(10, 3)
        redis_td["action"] = torch.randn(10, 2)
        assert "obs" in redis_td.keys()

        redis_td.del_("obs")
        assert "obs" not in redis_td.keys()
        assert "action" in redis_td.keys()

    def test_del_nested(self, redis_td):
        """Test deletion of nested keys."""
        redis_td["nested", "a"] = torch.randn(10, 2)
        redis_td["nested", "b"] = torch.randn(10, 3)
        assert "nested" in redis_td.keys()

        redis_td.del_("nested")
        assert "nested" not in redis_td.keys()

    def test_locking(self, redis_td):
        """Test lock/unlock."""
        redis_td["obs"] = torch.randn(10, 3)
        redis_td.lock_()
        assert redis_td.is_locked

        # Should raise when trying to set a new key
        with pytest.raises(RuntimeError):
            redis_td["new_key"] = torch.randn(10, 2)

        redis_td.unlock_()
        assert not redis_td.is_locked
        redis_td["new_key"] = torch.randn(10, 2)

    def test_set_at_(self, redis_td):
        """Test setting a value at a specific index."""
        tensor = torch.zeros(10, 3)
        redis_td["obs"] = tensor
        new_vals = torch.ones(3)
        redis_td.set_at_("obs", new_vals, 0)
        result = redis_td["obs"]
        assert torch.allclose(result[0], new_vals)
        assert torch.allclose(result[1:], torch.zeros(9, 3))

    def test_update(self, redis_td):
        """Test batch update from a dict."""
        source = TensorDict(
            {"obs": torch.randn(10, 3), "action": torch.randn(10, 2)}, [10]
        )
        redis_td.update(source)
        assert torch.allclose(redis_td["obs"], source["obs"])
        assert torch.allclose(redis_td["action"], source["action"])

    def test_clone_recurse(self, redis_td):
        """Test deep clone."""

        redis_td["obs"] = torch.randn(10, 3)
        cloned = redis_td.clone()
        try:
            assert isinstance(cloned, RedisTensorDict)
            assert cloned._td_id != redis_td._td_id
            assert torch.allclose(cloned["obs"], redis_td["obs"])
        finally:
            cloned.clear_redis()
            cloned.close()

    def test_clone_no_recurse(self, redis_td):
        """Test shallow clone (same Redis data)."""

        redis_td["obs"] = torch.randn(10, 3)
        shallow = redis_td.clone(False)
        assert isinstance(shallow, RedisTensorDict)
        assert shallow._td_id == redis_td._td_id
        assert torch.allclose(shallow["obs"], redis_td["obs"])

    def test_pickling(self, redis_td):
        """Test pickle round-trip."""
        redis_td["obs"] = torch.randn(10, 3)
        data = pickle.dumps(redis_td)
        restored = pickle.loads(data)
        try:
            assert torch.allclose(restored["obs"], redis_td["obs"])
            assert restored._td_id == redis_td._td_id
        finally:
            restored.close()

    def test_entry_class(self, redis_td):
        """Test entry_class returns correct types."""

        redis_td["obs"] = torch.randn(10, 3)
        redis_td["nested", "a"] = torch.randn(10, 2)
        assert redis_td.entry_class("obs") is torch.Tensor
        assert redis_td.entry_class("nested") is RedisTensorDict

    def test_device(self):
        """Test device attribute."""

        td = RedisTensorDict(batch_size=[5], device="cpu", db=15)
        try:
            td["x"] = torch.randn(5, 3)
            assert td.device == torch.device("cpu")
            result = td["x"]
            assert result.device == torch.device("cpu")
        finally:
            td.clear_redis()
            td.close()

    def test_repr(self, redis_td):
        """Test string representation."""
        redis_td["obs"] = torch.randn(10, 3)
        s = repr(redis_td)
        assert "TensorDictStore" in s
        assert "obs" in s

    def test_empty(self, redis_td):
        """Test empty()."""
        redis_td["obs"] = torch.randn(10, 3)
        empty = redis_td.empty()
        assert isinstance(empty, TensorDict)
        assert empty.batch_size == torch.Size([10])
        assert len(list(empty.keys())) == 0

    def test_fill_(self, redis_td):
        """Test fill_."""
        redis_td["obs"] = torch.randn(10, 3)
        redis_td.fill_("obs", 42.0)
        result = redis_td["obs"]
        assert torch.allclose(result, torch.full((10, 3), 42.0))

    def test_is_contiguous(self, redis_td):
        """Redis TDs are not contiguous."""
        assert not redis_td.is_contiguous()

    def test_shape_ops_raise(self, redis_td):
        """Shape ops raise on RedisTensorDict but work after to_local()."""
        redis_td["obs"] = torch.randn(10, 3)

        with pytest.raises(RuntimeError):
            redis_td.view(2, 5)
        with pytest.raises(RuntimeError):
            redis_td.permute(0)
        with pytest.raises(RuntimeError):
            redis_td.unsqueeze(0)
        with pytest.raises(RuntimeError):
            redis_td.squeeze(0)

        # Escape hatch: materialize first, then shape ops work
        local = redis_td.to_local()
        assert local.view(2, 5).shape == torch.Size([2, 5])
        assert local.unsqueeze(0).shape == torch.Size([1, 10])
        assert local.squeeze(0).shape == torch.Size([10])

    def test_share_memory_raises(self, redis_td):
        """share_memory_ should raise."""
        with pytest.raises(NotImplementedError):
            redis_td.share_memory_()

    def test_reconnect_by_id(self):
        """Connect to an existing RedisTensorDict by ID."""

        td1 = RedisTensorDict(batch_size=[5], db=15)
        try:
            td1["obs"] = torch.randn(5, 3)
            saved_obs = td1["obs"].clone()
            td_id = td1._td_id

            # Create a new instance pointing to the same data
            td2 = RedisTensorDict(batch_size=[5], db=15, td_id=td_id)
            try:
                assert torch.allclose(td2["obs"], saved_obs)
            finally:
                td2.close()
        finally:
            td1.clear_redis()
            td1.close()

    def test_popitem(self, redis_td):
        """Test popitem."""
        redis_td["a"] = torch.randn(10, 2)
        redis_td["b"] = torch.randn(10, 3)
        key, val = redis_td.popitem()
        assert key in ("a", "b")
        assert isinstance(val, torch.Tensor)
        # Only one key should remain
        remaining = list(redis_td.keys())
        assert len(remaining) == 1

    def test_rename_key(self, redis_td):
        """Test rename_key_."""
        tensor = torch.randn(10, 3)
        redis_td["old_name"] = tensor
        redis_td.rename_key_("old_name", "new_name")
        assert "old_name" not in redis_td.keys()
        assert "new_name" in redis_td.keys()
        assert torch.allclose(redis_td["new_name"], tensor)

    def test_from_tensordict(self):
        """Test from_tensordict classmethod."""

        source = TensorDict(
            {"obs": torch.randn(5, 3), "action": torch.randn(5, 2)}, [5]
        )
        td = RedisTensorDict.from_tensordict(source, db=15)
        try:
            assert td.batch_size == torch.Size([5])
            assert torch.allclose(td["obs"], source["obs"])
            assert torch.allclose(td["action"], source["action"])
        finally:
            td.clear_redis()
            td.close()

    def test_from_tensordict_preserves_device(self):
        """from_tensordict should preserve the source device by default."""

        source = TensorDict({"x": torch.randn(3)}, [3], device="cpu")
        td = RedisTensorDict.from_tensordict(source, db=15)
        try:
            assert td.device == torch.device("cpu")
        finally:
            td.clear_redis()
            td.close()

    def test_from_redis(self):
        """Test from_redis: reconnect to existing data by td_id."""

        # Writer creates data
        writer = RedisTensorDict(batch_size=[5], db=15)
        try:
            obs = torch.randn(5, 3)
            writer["obs"] = obs
            td_id = writer._td_id

            # Reader reconnects from a different handle
            reader = RedisTensorDict.from_redis(td_id=td_id, db=15)
            try:
                assert reader.batch_size == torch.Size([5])
                assert torch.allclose(reader["obs"], obs)
            finally:
                reader.close()
        finally:
            writer.clear_redis()
            writer.close()

    def test_from_redis_not_found(self):
        """from_redis should raise KeyError for unknown td_id."""

        with pytest.raises(KeyError, match="No TensorDictStore"):
            RedisTensorDict.from_redis(td_id="nonexistent-uuid", db=15)

    def test_from_redis_with_device_override(self):
        """from_redis should allow overriding the device."""

        writer = RedisTensorDict(batch_size=[3], device="cpu", db=15)
        try:
            writer["x"] = torch.randn(3)
            td_id = writer._td_id

            reader = RedisTensorDict.from_redis(td_id=td_id, db=15, device="cpu")
            try:
                assert reader.device == torch.device("cpu")
                assert reader["x"].device == torch.device("cpu")
            finally:
                reader.close()
        finally:
            writer.clear_redis()
            writer.close()

    # ---- Byte-range indexed read tests ----

    def test_indexed_read_int(self, redis_td):
        """td[i] should return the correct slice via GETRANGE."""
        obs = torch.randn(10, 3)
        redis_td["obs"] = obs
        sub = redis_td[5]
        result = sub["obs"]
        assert result.shape == torch.Size([3])
        assert torch.allclose(result, obs[5])

    def test_indexed_read_slice(self, redis_td):
        """td[2:5] should return the correct slice via GETRANGE."""
        obs = torch.randn(10, 4)
        redis_td["obs"] = obs
        sub = redis_td[2:5]
        result = sub["obs"]
        assert result.shape == torch.Size([3, 4])
        assert torch.allclose(result, obs[2:5])

    def test_indexed_read_fancy(self, redis_td):
        """td[tensor_idx] should return correct rows via GETRANGE."""
        obs = torch.randn(10, 3)
        redis_td["obs"] = obs
        idx = torch.tensor([1, 4, 7])
        sub = redis_td[idx]
        result = sub["obs"]
        assert result.shape == torch.Size([3, 3])
        assert torch.allclose(result, obs[idx])

    def test_indexed_read_bool_mask(self, redis_td):
        """td[bool_mask] should return correct rows via GETRANGE."""
        obs = torch.randn(10, 3)
        redis_td["obs"] = obs
        mask = torch.zeros(10, dtype=torch.bool)
        mask[0] = True
        mask[3] = True
        mask[9] = True
        sub = redis_td[mask]
        result = sub["obs"]
        assert result.shape == torch.Size([3, 3])
        assert torch.allclose(result, obs[mask])

    def test_indexed_read_multiple_keys(self, redis_td):
        """Indexed read should work across multiple leaf keys."""
        obs = torch.randn(10, 4)
        action = torch.randn(10, 2)
        redis_td["obs"] = obs
        redis_td["action"] = action
        sub = redis_td[3]
        assert torch.allclose(sub["obs"], obs[3])
        assert torch.allclose(sub["action"], action[3])

    # ---- Byte-range indexed write tests ----

    def test_indexed_write_int(self, redis_td):
        """td[5] = subtd should only modify row 5 via SETRANGE."""
        redis_td["obs"] = torch.zeros(10, 3)
        redis_td["action"] = torch.zeros(10, 2)

        new_obs = torch.ones(3)
        new_action = torch.ones(2) * 2.0
        redis_td[5] = TensorDict({"obs": new_obs, "action": new_action}, [])

        full_obs = redis_td["obs"]
        full_action = redis_td["action"]
        # Row 5 should be updated
        assert torch.allclose(full_obs[5], new_obs)
        assert torch.allclose(full_action[5], new_action)
        # Other rows should be untouched
        assert torch.allclose(full_obs[:5], torch.zeros(5, 3))
        assert torch.allclose(full_obs[6:], torch.zeros(4, 3))

    def test_indexed_write_slice(self, redis_td):
        """td[2:5] = subtd should modify rows 2-4 via SETRANGE."""
        redis_td["obs"] = torch.zeros(10, 3)

        new_vals = torch.ones(3, 3) * 7.0
        redis_td[2:5] = TensorDict({"obs": new_vals}, [3])

        full = redis_td["obs"]
        assert torch.allclose(full[2:5], new_vals)
        assert torch.allclose(full[:2], torch.zeros(2, 3))
        assert torch.allclose(full[5:], torch.zeros(5, 3))

    def test_indexed_write_list(self, redis_td):
        """td[[0, 3, 7]] = subtd should modify selected rows."""
        redis_td["obs"] = torch.zeros(10, 3)
        idx = [0, 3, 7]
        new_vals = torch.ones(3, 3) * 5.0
        redis_td[idx] = TensorDict({"obs": new_vals}, [3])

        full = redis_td["obs"]
        for i, pos in enumerate(idx):
            assert torch.allclose(full[pos], new_vals[i])
        # Unselected rows should be zero
        for pos in [1, 2, 4, 5, 6, 8, 9]:
            assert torch.allclose(full[pos], torch.zeros(3))

    def test_indexed_write_tensor(self, redis_td):
        """td[tensor_idx] = subtd should modify selected rows."""
        redis_td["obs"] = torch.zeros(10, 4)
        idx = torch.tensor([2, 5, 8])
        new_vals = torch.ones(3, 4) * 3.0
        redis_td[idx] = TensorDict({"obs": new_vals}, [3])

        full = redis_td["obs"]
        assert torch.allclose(full[idx], new_vals)
        # Check untouched
        untouched = torch.tensor([0, 1, 3, 4, 6, 7, 9])
        assert torch.allclose(full[untouched], torch.zeros(7, 4))

    def test_indexed_write_bool_mask(self, redis_td):
        """td[mask] = subtd should modify masked rows."""
        redis_td["obs"] = torch.zeros(10, 3)
        mask = torch.zeros(10, dtype=torch.bool)
        mask[1] = True
        mask[4] = True
        mask[9] = True
        new_vals = torch.ones(3, 3) * 9.0
        redis_td[mask] = TensorDict({"obs": new_vals}, [3])

        full = redis_td["obs"]
        assert torch.allclose(full[mask], new_vals)
        assert torch.allclose(full[~mask], torch.zeros(7, 3))

    def test_indexed_write_ellipsis(self, redis_td):
        """td[...] = subtd should overwrite all rows via SETRANGE."""
        redis_td["obs"] = torch.zeros(10, 3)
        new_vals = torch.ones(10, 3) * 4.0
        redis_td[...] = TensorDict({"obs": new_vals}, [10])
        assert torch.allclose(redis_td["obs"], new_vals)

    def test_indexed_read_ellipsis(self, redis_td):
        """td[...] should return all rows."""
        obs = torch.randn(10, 3)
        redis_td["obs"] = obs
        sub = redis_td[...]
        assert torch.allclose(sub["obs"], obs)

    def test_indexed_read_step_slice(self, redis_td):
        """td[::2] should return every other row via covering range."""
        obs = torch.randn(10, 3)
        redis_td["obs"] = obs
        sub = redis_td[::2]
        assert sub["obs"].shape == torch.Size([5, 3])
        assert torch.allclose(sub["obs"], obs[::2])

    def test_indexed_read_step3(self, redis_td):
        """td[1::3] should fetch covering range and stride locally."""
        obs = torch.randn(10, 3)
        redis_td["obs"] = obs
        sub = redis_td[1::3]
        assert sub["obs"].shape == torch.Size([3, 3])
        assert torch.allclose(sub["obs"], obs[1::3])

    def test_indexed_write_step_slice(self, redis_td):
        """td[::2] = subtd should use partial covering-range RMW."""
        redis_td["obs"] = torch.zeros(10, 3)
        new_vals = torch.ones(5, 3) * 9.0
        redis_td[::2] = TensorDict({"obs": new_vals}, [5])

        full = redis_td["obs"]
        assert torch.allclose(full[::2], new_vals)
        # Odd rows should remain zero
        assert torch.allclose(full[1::2], torch.zeros(5, 3))

    def test_indexed_write_step3(self, redis_td):
        """td[1::3] = subtd should use partial covering-range RMW."""
        redis_td["obs"] = torch.arange(30, dtype=torch.float).reshape(10, 3)
        original = redis_td["obs"].clone()
        new_vals = torch.ones(3, 3) * -1.0
        redis_td[1::3] = TensorDict({"obs": new_vals}, [3])

        full = redis_td["obs"]
        assert torch.allclose(full[1::3], new_vals)
        # Unmodified rows should stay the same
        for i in range(10):
            if i not in (1, 4, 7):
                assert torch.allclose(full[i], original[i])

    # ---- set_at_ via byte-range ----

    def test_set_at_byte_range(self, redis_td):
        """set_at_ should use SETRANGE for a single key."""
        redis_td["obs"] = torch.zeros(10, 3)
        new_val = torch.ones(3) * 42.0
        redis_td.set_at_("obs", new_val, 3)

        full = redis_td["obs"]
        assert torch.allclose(full[3], new_val)
        assert torch.allclose(full[:3], torch.zeros(3, 3))
        assert torch.allclose(full[4:], torch.zeros(6, 3))

    # ---- Metadata caching tests ----

    def test_cache_metadata_default(self, redis_td):
        """Metadata cache should be populated after writes by default."""
        redis_td["obs"] = torch.randn(10, 3)
        assert redis_td._meta_cache is not None
        # The cache should contain the key's metadata
        key_path = redis_td._full_key_path("obs")
        assert key_path in redis_td._meta_cache
        shape, dtype = redis_td._meta_cache[key_path]
        assert shape == [10, 3]
        assert dtype == torch.float32

    def test_cache_metadata_disabled(self):
        """cache_metadata=False should disable the local cache."""
        td = RedisTensorDict(batch_size=[5], db=15, cache_metadata=False)
        try:
            assert td._meta_cache is None
            td["x"] = torch.randn(5, 3)
            assert td._meta_cache is None
        finally:
            td.clear_redis()
            td.close()

    def test_cache_evicted_on_delete(self, redis_td):
        """Deleting a key should evict it from the metadata cache."""
        redis_td["obs"] = torch.randn(10, 3)
        key_path = redis_td._full_key_path("obs")
        assert key_path in redis_td._meta_cache
        redis_td.del_("obs")
        assert key_path not in redis_td._meta_cache


@_skip_no_redis_pkg
@_skip_no_redis_server
class TestRedisLazyStackedTensorDict:
    """Tests for RedisLazyStackedTensorDict requiring a running Redis server."""

    @pytest.fixture(autouse=True)
    def redis_stack(self):
        """Create a RedisLazyStackedTensorDict from a homogeneous lazy stack."""
        tds = [
            TensorDict({"a": torch.randn(4, 3), "b": torch.randn(4)}, batch_size=[4])
            for _ in range(5)
        ]
        lazy_td = lazy_stack(tds)
        redis_td = RedisLazyStackedTensorDict.from_lazy_stack(lazy_td, db=15)
        yield redis_td, tds, lazy_td
        redis_td.clear_redis()
        redis_td.close()

    # ---- Construction ----

    def test_from_lazy_stack_batch_size(self, redis_stack):
        redis_td, tds, lazy_td = redis_stack
        assert redis_td.batch_size == torch.Size([5, 4])
        assert redis_td._count == 5
        assert redis_td._stack_dim == 0
        assert redis_td._inner_batch_size == torch.Size([4])

    def test_from_lazy_stack_keys(self, redis_stack):
        redis_td, tds, lazy_td = redis_stack
        assert set(redis_td.keys()) == {"a", "b"}

    def test_repr(self, redis_stack):
        redis_td, _, _ = redis_stack
        r = repr(redis_td)
        assert "LazyStackedTensorDictStore" in r
        assert "count=5" in r

    # ---- to_redis() convenience ----

    def test_to_redis_from_lazy_stack(self):
        tds = [TensorDict({"x": torch.randn(3, 2)}, batch_size=[3]) for _ in range(4)]
        lazy_td = lazy_stack(tds)
        redis_td = lazy_td.to_redis(db=15)
        try:
            assert isinstance(redis_td, LazyStackedTensorDictStore)
            assert redis_td.batch_size == torch.Size([4, 3])
            result = redis_td["x"]
            assert result.shape == torch.Size([4, 3, 2])
        finally:
            redis_td.clear_redis()
            redis_td.close()

    # ---- Read: td[int] ----

    def test_getitem_int(self, redis_stack):
        redis_td, tds, lazy_td = redis_stack
        elem = redis_td[0]
        assert isinstance(elem, TensorDictBase)
        assert elem.batch_size == torch.Size([4])
        assert torch.allclose(elem["a"], tds[0]["a"])
        assert torch.allclose(elem["b"], tds[0]["b"])

    def test_getitem_int_last(self, redis_stack):
        redis_td, tds, lazy_td = redis_stack
        elem = redis_td[4]
        assert torch.allclose(elem["a"], tds[4]["a"])

    def test_getitem_int_negative(self, redis_stack):
        redis_td, tds, lazy_td = redis_stack
        elem = redis_td[-1]
        assert torch.allclose(elem["a"], tds[4]["a"])

    # ---- Read: td[key] ----

    def test_getitem_key(self, redis_stack):
        redis_td, tds, lazy_td = redis_stack
        full_a = redis_td["a"]
        assert full_a.shape == torch.Size([5, 4, 3])
        for i in range(5):
            assert torch.allclose(full_a[i], tds[i]["a"])

    def test_getitem_key_1d(self, redis_stack):
        redis_td, tds, lazy_td = redis_stack
        full_b = redis_td["b"]
        assert full_b.shape == torch.Size([5, 4])
        for i in range(5):
            assert torch.allclose(full_b[i], tds[i]["b"])

    # ---- Read: td[int][key] ----

    def test_getitem_int_then_key(self, redis_stack):
        redis_td, tds, lazy_td = redis_stack
        val = redis_td[0]["a"]
        assert val.shape == torch.Size([4, 3])
        assert torch.allclose(val, tds[0]["a"])

    # ---- Read: td[slice] ----

    def test_getitem_slice(self, redis_stack):
        redis_td, tds, lazy_td = redis_stack
        sub = redis_td[1:3]
        assert isinstance(sub, TensorDictBase)
        assert sub.batch_size == torch.Size([2, 4])
        assert torch.allclose(sub["a"][0], tds[1]["a"])
        assert torch.allclose(sub["a"][1], tds[2]["a"])

    def test_getitem_slice_step(self, redis_stack):
        redis_td, tds, lazy_td = redis_stack
        sub = redis_td[::2]
        assert sub.batch_size == torch.Size([3, 4])
        assert torch.allclose(sub["a"][0], tds[0]["a"])
        assert torch.allclose(sub["a"][1], tds[2]["a"])
        assert torch.allclose(sub["a"][2], tds[4]["a"])

    # ---- Read: td[tensor_index] ----

    def test_getitem_tensor_index(self, redis_stack):
        redis_td, tds, lazy_td = redis_stack
        idx = torch.tensor([0, 3, 4])
        sub = redis_td[idx]
        assert isinstance(sub, TensorDictBase)
        assert sub.batch_size == torch.Size([3, 4])
        assert torch.allclose(sub["a"][0], tds[0]["a"])
        assert torch.allclose(sub["a"][1], tds[3]["a"])
        assert torch.allclose(sub["a"][2], tds[4]["a"])

    # ---- Write: td[int] = subtd ----

    def test_setitem_int(self, redis_stack):
        redis_td, tds, lazy_td = redis_stack
        new_a = torch.ones(4, 3)
        new_b = torch.ones(4)
        redis_td[0] = TensorDict({"a": new_a, "b": new_b}, batch_size=[4])
        elem = redis_td[0]
        assert torch.allclose(elem["a"], new_a)
        assert torch.allclose(elem["b"], new_b)
        # Other elements unchanged
        assert torch.allclose(redis_td[1]["a"], tds[1]["a"])

    # ---- Write: td[slice] = subtd ----

    def test_setitem_slice(self, redis_stack):
        redis_td, tds, lazy_td = redis_stack
        new_val = TensorDict(
            {"a": torch.zeros(2, 4, 3), "b": torch.zeros(2, 4)},
            batch_size=[2, 4],
        )
        redis_td[1:3] = new_val
        assert torch.allclose(redis_td[1]["a"], torch.zeros(4, 3))
        assert torch.allclose(redis_td[2]["a"], torch.zeros(4, 3))
        # Unchanged
        assert torch.allclose(redis_td[0]["a"], tds[0]["a"])

    # ---- Write: td[tensor_index] = subtd ----

    def test_setitem_tensor_index(self, redis_stack):
        redis_td, tds, lazy_td = redis_stack
        idx = torch.tensor([0, 4])
        new_val = TensorDict(
            {"a": torch.ones(2, 4, 3), "b": torch.ones(2, 4)},
            batch_size=[2, 4],
        )
        redis_td[idx] = new_val
        assert torch.allclose(redis_td[0]["a"], torch.ones(4, 3))
        assert torch.allclose(redis_td[4]["a"], torch.ones(4, 3))
        # Middle element unchanged
        assert torch.allclose(redis_td[2]["a"], tds[2]["a"])

    # ---- to_tensordict / to_local ----

    def test_to_tensordict(self, redis_stack):
        redis_td, tds, lazy_td = redis_stack
        local = redis_td.to_tensordict()
        assert isinstance(local, TensorDict)
        assert local.batch_size == torch.Size([5, 4])
        for i in range(5):
            assert torch.allclose(local["a"][i], tds[i]["a"])
            assert torch.allclose(local["b"][i], tds[i]["b"])

    def test_to_local(self, redis_stack):
        redis_td, tds, lazy_td = redis_stack
        local = redis_td.to_local()
        assert isinstance(local, TensorDict)
        assert local.batch_size == torch.Size([5, 4])

    # ---- td[idx].to_tensordict() pattern ----

    def test_indexed_to_tensordict(self, redis_stack):
        redis_td, tds, lazy_td = redis_stack
        local = redis_td[1:3].to_tensordict()
        assert local.batch_size == torch.Size([2, 4])
        assert torch.allclose(local["a"][0], tds[1]["a"])

    # ---- Heterogeneous shapes ----

    def test_heterogeneous_shapes(self):
        """Test lazy stack with different feature dims per element."""
        td0 = TensorDict({"a": torch.randn(3, 4)}, batch_size=[3])
        td1 = TensorDict({"a": torch.randn(3, 8)}, batch_size=[3])
        lazy_td = lazy_stack([td0, td1])
        redis_td = RedisLazyStackedTensorDict.from_lazy_stack(lazy_td, db=15)
        try:
            assert redis_td.batch_size == torch.Size([2, 3])
            # Read element 0
            elem0 = redis_td[0]
            assert torch.allclose(elem0["a"], td0["a"])
            # Read element 1
            elem1 = redis_td[1]
            assert torch.allclose(elem1["a"], td1["a"])
        finally:
            redis_td.clear_redis()
            redis_td.close()

    # ---- Pickling ----

    def test_pickle_roundtrip(self, redis_stack):
        redis_td, tds, lazy_td = redis_stack
        data = pickle.dumps(redis_td)
        restored = pickle.loads(data)
        try:
            assert restored.batch_size == redis_td.batch_size
            assert restored._count == redis_td._count
            assert torch.allclose(restored[0]["a"], tds[0]["a"])
        finally:
            restored.close()

    # ---- from_redis reconnect ----

    def test_from_redis_reconnect(self, redis_stack):
        redis_td, tds, lazy_td = redis_stack
        td_id = redis_td._td_id
        restored = RedisLazyStackedTensorDict.from_redis(td_id=td_id, db=15)
        try:
            assert restored.batch_size == redis_td.batch_size
            assert restored._count == 5
            assert torch.allclose(restored[0]["a"], tds[0]["a"])
        finally:
            restored.close()

    # ---- Nested keys ----

    def test_nested_keys(self):
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
        redis_td = RedisLazyStackedTensorDict.from_lazy_stack(lazy_td, db=15)
        try:
            assert redis_td.batch_size == torch.Size([4, 3])
            assert "obs" in redis_td.keys()
            # Read element
            elem = redis_td[0]
            assert torch.allclose(elem["obs"], tds[0]["obs"])
            assert torch.allclose(elem[("nested", "x")], tds[0]["nested", "x"])
        finally:
            redis_td.clear_redis()
            redis_td.close()

    # ---- Write-through view tests ----

    def test_view_set_propagates(self, redis_stack):
        """rltd[0].set('a', val) should propagate to Redis."""
        redis_td, tds, lazy_td = redis_stack
        new_a = torch.ones(4, 3) * 42.0
        view = redis_td[0]
        view.set("a", new_a, inplace=True)
        # Re-read: should see the change
        reread = redis_td[0]["a"]
        assert torch.allclose(reread, new_a)
        # Other elements unaffected
        assert torch.allclose(redis_td[1]["a"], tds[1]["a"])

    def test_view_setitem_key_propagates(self, redis_stack):
        """rltd[0]['a'] = val should propagate to Redis."""
        redis_td, tds, lazy_td = redis_stack
        new_a = torch.ones(4, 3) * 99.0
        view = redis_td[0]
        view["a"] = new_a
        reread = redis_td[0]["a"]
        assert torch.allclose(reread, new_a)

    def test_view_shape_change_raises(self, redis_stack):
        """Changing element shape through the view should raise."""
        redis_td, tds, lazy_td = redis_stack
        view = redis_td[0]
        with pytest.raises((ValueError, RuntimeError)):
            view.set("a", torch.randn(10, 10), inplace=True)

    def test_view_to_tensordict(self, redis_stack):
        """view.to_tensordict() should return a regular TensorDict."""
        redis_td, tds, lazy_td = redis_stack
        view = redis_td[0]
        local = view.to_tensordict()
        assert isinstance(local, TensorDict)
        assert local.batch_size == torch.Size([4])
        assert torch.allclose(local["a"], tds[0]["a"])

    def test_view_nested_set(self):
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
        redis_td = RedisLazyStackedTensorDict.from_lazy_stack(lazy_td, db=15)
        try:
            view = redis_td[0]
            new_x = torch.ones(3, 2) * 7.0
            view.set(("nested", "x"), new_x, inplace=True)
            reread = redis_td[0][("nested", "x")]
            assert torch.allclose(reread, new_x)
        finally:
            redis_td.clear_redis()
            redis_td.close()


@_skip_no_redis_pkg
@_skip_no_redis_server
class TestBackendAndCompat:
    """Tests for backend parameter and backward-compat aliases."""

    def test_backend_default(self):
        td = TensorDictStore(batch_size=[5], db=15)
        try:
            assert td._backend == "redis"
            assert "backend='redis'" in repr(td)
        finally:
            td.clear_redis()
            td.close()

    def test_backend_dragonfly(self):
        # Dragonfly uses the same wire protocol, so this connects to Redis
        # but records the backend name for documentation purposes.
        td = TensorDictStore(backend="dragonfly", batch_size=[5], db=15)
        try:
            assert td._backend == "dragonfly"
            assert "backend='dragonfly'" in repr(td)
            td["x"] = torch.randn(5, 3)
            assert torch.allclose(td["x"], td["x"])
        finally:
            td.clear_redis()
            td.close()

    def test_backend_from_tensordict(self):
        local = TensorDict({"a": torch.randn(5)}, [5])
        td = TensorDictStore.from_tensordict(local, backend="dragonfly", db=15)
        try:
            assert td._backend == "dragonfly"
        finally:
            td.clear_redis()
            td.close()

    def test_backend_to_store(self):
        local = TensorDict({"a": torch.randn(5)}, [5])
        td = local.to_store(backend="dragonfly", db=15)
        try:
            assert td._backend == "dragonfly"
        finally:
            td.clear_redis()
            td.close()

    def test_to_redis_alias(self):
        local = TensorDict({"a": torch.randn(5)}, [5])
        td = local.to_redis(db=15)
        try:
            assert td._backend == "redis"
            assert isinstance(td, TensorDictStore)
        finally:
            td.clear_redis()
            td.close()

    def test_backward_compat_import_redis(self):
        """Old import path ``from tensordict.redis import RedisTensorDict``."""
        from tensordict.redis import RedisTensorDict as OldName

        assert OldName is TensorDictStore

    def test_backward_compat_import_lazy(self):
        """Old ``from tensordict.redis import RedisLazyStackedTensorDict``."""
        from tensordict.redis import RedisLazyStackedTensorDict as OldLazy

        assert OldLazy is LazyStackedTensorDictStore

    def test_from_redis_alias(self):
        """from_redis should still work as an alias of from_store."""
        td = TensorDictStore(batch_size=[5], db=15)
        td["x"] = torch.randn(5, 3)
        td_id = td._td_id
        try:
            restored = TensorDictStore.from_redis(td_id=td_id, db=15)
            assert torch.allclose(restored["x"], td["x"])
            restored.close()
        finally:
            td.clear_redis()
            td.close()

    def test_pickle_preserves_backend(self):
        td = TensorDictStore(backend="dragonfly", batch_size=[5], db=15)
        td["x"] = torch.randn(5, 3)
        try:
            data = pickle.dumps(td)
            restored = pickle.loads(data)
            assert restored._backend == "dragonfly"
            assert torch.allclose(restored["x"], td["x"])
            restored.close()
        finally:
            td.clear_redis()
            td.close()

    def test_lazy_stack_backend(self):
        tds = [TensorDict({"a": torch.randn(4)}, batch_size=[4]) for _ in range(3)]
        ltd = lazy_stack(tds)
        rltd = LazyStackedTensorDictStore.from_lazy_stack(
            ltd, backend="dragonfly", db=15
        )
        try:
            assert rltd._backend == "dragonfly"
            assert "backend='dragonfly'" in repr(rltd)
        finally:
            rltd.clear_redis()
            rltd.close()


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
