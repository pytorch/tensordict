# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import pickle

import pytest
import torch
from tensordict import TensorDict

try:
    import redis

    _has_redis = True
except ImportError:
    _has_redis = False


def _redis_available():
    """Check if a Redis server is reachable on localhost:6379."""
    if not _has_redis:
        return False
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
        from tensordict.redis import RedisTensorDict

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
        for dtype in [torch.float32, torch.float64, torch.int32, torch.int64, torch.bool]:
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
        from tensordict.redis import RedisTensorDict

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
        from tensordict.redis import RedisTensorDict

        source = {"obs": torch.randn(5, 3), "reward": torch.randn(5, 1)}
        td = RedisTensorDict.from_dict(source, batch_size=[5], db=15)
        try:
            assert td.batch_size == torch.Size([5])
            assert torch.allclose(td["obs"], source["obs"])
            assert torch.allclose(td["reward"], source["reward"])
        finally:
            td.clear_redis()
            td.close()

    def test_from_tensordict(self):
        """Construct from a TensorDict."""
        from tensordict.redis import RedisTensorDict

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
        from tensordict.redis import RedisTensorDict

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
        from tensordict.redis import RedisTensorDict

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
        from tensordict.redis import RedisTensorDict

        redis_td["obs"] = torch.randn(10, 3)
        redis_td["nested", "a"] = torch.randn(10, 2)
        assert redis_td.entry_class("obs") is torch.Tensor
        assert redis_td.entry_class("nested") is RedisTensorDict

    def test_device(self):
        """Test device attribute."""
        from tensordict.redis import RedisTensorDict

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
        assert "RedisTensorDict" in s
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
        """Shape ops should raise RuntimeError."""
        redis_td["obs"] = torch.randn(10, 3)
        with pytest.raises(RuntimeError):
            redis_td.view(2, 5)
        with pytest.raises(RuntimeError):
            redis_td.permute(0)
        with pytest.raises(RuntimeError):
            redis_td.unsqueeze(0)
        with pytest.raises(RuntimeError):
            redis_td.squeeze(0)

    def test_share_memory_raises(self, redis_td):
        """share_memory_ should raise."""
        with pytest.raises(NotImplementedError):
            redis_td.share_memory_()

    def test_reconnect_by_id(self):
        """Connect to an existing RedisTensorDict by ID."""
        from tensordict.redis import RedisTensorDict

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


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
