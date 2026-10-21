# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import functools
import gc
import importlib
import inspect
import io
import pickle
import re
import ssl
import uuid
import warnings
from unittest import mock

import numpy as np
import pytest
import torch
from tensordict import (
    assert_allclose_td,
    is_tensor_collection,
    lazy_stack,
    LazyStackedTensorDict,
    NonTensorData,
    NonTensorStack,
    TensorClass,
    TensorDict,
)
from tensordict.base import TensorDictBase
from tensordict.store import LazyStackedTensorDictStore, TensorDictStore
from tensordict.store._connection import _client_settings, _connect_kwargs
from tensordict.store._lazy import _StoreStackElementView

_has_redis = importlib.util.find_spec("redis", None) is not None

# Ports used by each backend in CI.
_BACKEND_PORTS = {"redis": 6379, "dragonfly": 6380}


# Probe each server once: redis-py retries a refused connection for about
# 5 s, and the backend fixture runs for every store test.
@functools.cache
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


def _delete_namespace(prefix, store_kwargs):
    """Delete every server key under ``prefix``, also of a store that failed to build."""
    import redis

    client = redis.Redis(port=store_kwargs["port"], db=store_kwargs["db"])
    try:
        for key in client.scan_iter(match=f"{prefix}:*"):
            client.delete(key)
    finally:
        client.close()


def _stored_batch_size(store, store_kwargs):
    """Read the batch size of ``store`` from the server, with a new handle."""
    reader = TensorDictStore.from_store(td_id=store._td_id, **store_kwargs)
    try:
        return reader.batch_size
    finally:
        reader.close()


def _count_round_trips(store, monkeypatch):
    """Patch ``type(store)._run_sync`` to count the round trips to the server."""
    calls = []
    run_sync = type(store)._run_sync

    def counting_run_sync(self, coro):
        calls.append(None)
        return run_sync(self, coro)

    monkeypatch.setattr(type(store), "_run_sync", counting_run_sync)
    return calls


@pytest.mark.parametrize(
    "cls,what",
    [
        (TensorDictStore, "TensorDictStore"),
        (LazyStackedTensorDictStore, "LazyStackedTensorDictStore"),
        (_StoreStackElementView, r"LazyStackedTensorDictStore\[i\]"),
    ],
)
def test_to_local_is_deprecated(cls, what, monkeypatch):
    # no server needed: to_local only forwards to to_tensordict
    local = TensorDict({"a": torch.zeros(2)}, [2])
    monkeypatch.setattr(cls, "to_tensordict", lambda self: local)
    store = cls.__new__(cls)
    with pytest.warns(
        DeprecationWarning,
        match=rf"^{what}\.to_local\(\) is deprecated and will be removed in "
        r"TensorDict 0\.17\. Use to_tensordict\(\) instead\.$",
    ) as record:
        assert store.to_local() is local
    assert record[0].filename == __file__


_MISSING_KEY_WARNING = (
    r"of a missing key .* does nothing today\. Starting with TensorDict 0\.17, "
    r"the default will change to raising a KeyError"
)


def _rename_source():
    """A tensordict for the rename_key_ tests of the stores."""
    return TensorDict(
        a=torch.arange(3.0),
        b={"c": torch.arange(3)},
        other={"d": torch.zeros(3)},
        nested={"d": {"y": torch.ones(3), "d": {"y": torch.full((3,), 2.0)}}},
        s={"t": torch.zeros(3)},
        batch_size=[3],
    )


# (old_key, new_key) with a destination that is not free
_RENAME_OCCUPIED = [
    ("b", "a"),
    ("b", "other"),
    (("nested", "d"), ("other", "d")),
    ("a", "b"),
    (("b", "c"), "nested"),
    # nested.d.y moves to nested.y before nested.d.d.y moves to nested.d.y
    (("nested", "d"), "nested"),
]

# (old_key, new_key, safe, match): renames that raise and change nothing
_RENAME_INVALID = [
    ("b", ("a", "q"), False, "below a leaf"),
    ("a", ("b", "c", "q"), False, "below a leaf"),
    ("b", "other", True, "already present"),
    ("a", "b", True, "already present"),
]

# (old_key, new_key): a new key inside the old one, where the entry moves
_RENAME_INSIDE = [
    ("s", ("s", "s")),
    ("s", ("s", "x", "y")),
    ("a", ("a", "b")),
    # s.t moves to s.t.u.t: the leaf above the destination moves too
    ("s", ("s", "t", "u")),
    # nested.d.d.y moves to nested.d.d.d.y before nested.d.y moves to
    # nested.d.d.y
    (("nested", "d"), ("nested", "d", "d")),
]


def _assert_store_matches(store, expected):
    assert set(store.keys(True, True)) == set(expected.keys(True, True))
    # the store reads, element by element too (it keeps no empty
    # sub-tensordict, so only the leaves are compared)
    store[0:1]
    elements = [store[i].to_tensordict() for i in range(len(expected))]
    for key in expected.keys(True, True):
        torch.testing.assert_close(store[key], expected[key])
        for i, element in enumerate(elements):
            torch.testing.assert_close(element[key], expected[i][key])


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

    def test_batch_size_setter(self, store_td, store_kwargs):
        """Setting batch_size checks the entries and persists the new size."""
        store_td["obs"] = torch.zeros(10, 3)
        store_td["nested", "x"] = torch.zeros(10, 3, 2)
        store_td["text"] = "hello"

        store_td.batch_size = [10, 3]
        assert store_td.batch_size == torch.Size([10, 3])
        assert _stored_batch_size(store_td, store_kwargs) == torch.Size([10, 3])
        with pytest.raises(RuntimeError, match="incompatible with the batch-size"):
            store_td.batch_size = [10, 2]
        assert store_td.batch_size == torch.Size([10, 3])
        assert _stored_batch_size(store_td, store_kwargs) == torch.Size([10, 3])
        # the batch size of a nested view is not stored
        store_td["nested"].batch_size = [10, 3, 2]
        assert store_td["nested"].batch_size == torch.Size([10, 3, 2])
        assert store_td.batch_size == torch.Size([10, 3])
        assert _stored_batch_size(store_td, store_kwargs) == torch.Size([10, 3])

    def test_batch_size_setter_copy(self, store_td, store_kwargs):
        """A shallow copy of the store, such as the one a tensordict holds when
        the store goes in with another batch size, sets its batch size in
        memory only."""
        store_td["obs"] = torch.zeros(10, 3, 2)
        td = TensorDict({"s": store_td}, batch_size=[10, 3])
        assert td["s"].batch_size == torch.Size([10, 3])
        assert store_td.batch_size == torch.Size([10])
        assert _stored_batch_size(store_td, store_kwargs) == torch.Size([10])
        # an unpickled copy is still a copy
        copy = pickle.loads(pickle.dumps(store_td.copy()))
        try:
            copy.batch_size = [10, 3, 2]
            assert copy.batch_size == torch.Size([10, 3, 2])
            assert _stored_batch_size(store_td, store_kwargs) == torch.Size([10])
        finally:
            copy.close()

    def test_batch_size_setter_non_tensor(self, store_td):
        """A non-tensor entry written per element holds one value per element
        of the first batch dim, so it fixes that dim."""
        store_td[0] = TensorDict(label="a", batch_size=[])
        with pytest.raises(RuntimeError, match="label"):
            store_td.batch_size = [5]
        assert store_td.batch_size == torch.Size([10])
        store_td.batch_size = []
        assert store_td.batch_size == torch.Size([])

    def test_batch_size_list_no_read(self, store_td, monkeypatch):
        """Setting an unchanged batch size as a list reads no entry."""
        store_td["obs"] = torch.zeros(10, 3)
        store_td["nested", "x"] = torch.zeros(10, 2)
        calls = []
        aget_tensor = TensorDictStore._aget_tensor

        async def counting_aget_tensor(self, *args, **kwargs):
            calls.append(None)
            return await aget_tensor(self, *args, **kwargs)

        monkeypatch.setattr(TensorDictStore, "_aget_tensor", counting_aget_tensor)
        store_td.batch_size = [10]
        store_td.batch_size = (10,)
        assert store_td.batch_size == torch.Size([10])
        assert not calls
        store_td["obs"]
        assert calls

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

    @pytest.mark.parametrize("auto_batch_size", [False, True])
    def test_from_dict_auto_batch_size(self, store_kwargs, auto_batch_size):
        """from_dict infers the batch size of a dict input when auto_batch_size=True,
        as TensorDict.from_dict does."""

        source = {"obs": torch.randn(5, 3), "nested": {"x": torch.randn(5, 3, 2)}}
        td = TensorDictStore.from_dict(
            source, auto_batch_size=auto_batch_size, **store_kwargs
        )
        try:
            expected = TensorDict.from_dict(source, auto_batch_size=auto_batch_size)
            assert expected.batch_size == (
                torch.Size([5, 3]) if auto_batch_size else torch.Size([])
            )
            assert td.batch_size == expected.batch_size
            assert td["nested"].batch_size == expected["nested"].batch_size
            assert torch.allclose(td["obs"], source["obs"])
            assert torch.allclose(td["nested", "x"], source["nested"]["x"])
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

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"batch_dims": 1},
            {"batch_size": [5, 3], "names": ["x", "y"]},
        ],
    )
    def test_from_dict_batch_dims_names(self, store_kwargs, kwargs):
        source = {"obs": torch.randn(5, 3)}
        expected = TensorDict.from_dict(source, **kwargs)
        prefix = f"td-test-{uuid.uuid4().hex}"
        try:
            td = TensorDictStore.from_dict(
                source, prefix=prefix, **kwargs, **store_kwargs
            )
            assert td.batch_size == expected.batch_size
            assert td.names == expected.names
            assert torch.equal(td["obs"], source["obs"])
            td.close()
        finally:
            _delete_namespace(prefix, store_kwargs)

    def test_from_schema(self, store_kwargs):
        prefix = f"td-test-{uuid.uuid4().hex}"
        try:
            td = TensorDictStore.from_schema(
                {"a": ([2], torch.int64), "r": ([], torch.float32)},
                batch_size=[4],
                prefix=prefix,
                **store_kwargs,
            )
            assert td.batch_size == (4,)
            assert list(td.keys()) == ["a", "r"]
            assert torch.equal(td["a"], torch.zeros(4, 2, dtype=torch.int64))
            assert torch.equal(td["r"], torch.zeros(4))
            td.close()
        finally:
            _delete_namespace(prefix, store_kwargs)

    def test_from_schema_nested(self, store_kwargs):
        prefix = f"td-test-{uuid.uuid4().hex}"
        try:
            td = TensorDictStore.from_schema(
                {("n", "b"): ([2], torch.int64), "a": ([], torch.float32)},
                batch_size=[4],
                prefix=prefix,
                **store_kwargs,
            )
            expected = TensorDict.from_schema(
                {("n", "b"): ([2], torch.int64), "a": ([], torch.float32)},
                batch_size=[4],
            )
            assert td.batch_size == (4,)
            assert (td.to_tensordict() == expected).all()
            td.close()
        finally:
            _delete_namespace(prefix, store_kwargs)

    def test_to_local(self, store_td):
        """Materialize to a local TensorDict."""
        obs = torch.randn(10, 4)
        action = torch.randn(10, 2)
        store_td["obs"] = obs
        store_td["action"] = action

        with pytest.warns(DeprecationWarning, match="to_local"):
            local = store_td.to_local()
        assert isinstance(local, TensorDict)
        assert local.batch_size == torch.Size([10])
        assert torch.allclose(local["obs"], obs)
        assert torch.allclose(local["action"], action)

    def test_to_tensordict(self, store_td):
        """to_tensordict materializes to a local TensorDict."""
        tensor = torch.randn(10, 3)
        store_td["x"] = tensor

        local = store_td.to_tensordict()
        assert isinstance(local, TensorDict)
        assert torch.allclose(local["x"], tensor)

    def test_zero_numel_read(self, store_td, store_kwargs):
        """A tensor with zero elements reads back with its shape and dtype."""
        store_td["a"] = torch.zeros(10, 0, dtype=torch.int32)
        for result in (store_td["a"], store_td.to_tensordict()["a"]):
            assert result.shape == (10, 0)
            assert result.dtype == torch.int32
        td = TensorDictStore(batch_size=[0], **store_kwargs)
        try:
            td["c"] = torch.zeros(0)
            assert td["c"].shape == (0,)
            assert td.to_tensordict()["c"].shape == (0,)
        finally:
            td.clear_redis()
            td.close()

    def test_contiguous(self, store_td):
        """contiguous should materialize."""
        tensor = torch.randn(10, 3)
        store_td["x"] = tensor
        local = store_td.contiguous()
        assert isinstance(local, TensorDict)
        assert torch.allclose(local["x"], tensor)

    @pytest.mark.parametrize("canonical", [False, True])
    def test_contiguous_one_round_trip(self, store_td, monkeypatch, canonical):
        """contiguous() reads the store as to_tensordict() does."""
        store_td["x"] = torch.randn(10, 3)
        store_td["nested", "y"] = torch.randn(10, 2)
        calls = _count_round_trips(store_td, monkeypatch)
        expected = store_td.to_tensordict()
        round_trips = len(calls)
        calls.clear()
        out = store_td.contiguous(canonical=canonical)
        assert len(calls) == round_trips
        assert type(out) is TensorDict
        assert_allclose_td(out, expected)
        with pytest.raises(
            NotImplementedError,
            match=r"contiguous\(inplace=True\) is not supported on TensorDictStore",
        ):
            store_td.contiguous(inplace=True)

    def test_pin_memory_raises(self, store_td):
        store_td["x"] = torch.randn(10, 3)
        with pytest.raises(
            RuntimeError,
            match=r"^Cannot pin memory of a TensorDictStore\. Call to_tensordict\(\) "
            r"before making this call\.$",
        ):
            store_td.pin_memory()

    def test_chunk_zero_size_dim(self, store_kwargs):
        # torch.chunk gives `chunks` empty pieces along a zero-size dim
        td = TensorDictStore(batch_size=[0, 3], **store_kwargs)
        try:
            for dim in (0, -2):
                pieces = td.chunk(2, dim)
                assert [p.batch_size for p in pieces] == [torch.Size([0, 3])] * 2
        finally:
            td.clear_redis()
            td.close()

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
        """keys(include_nested=True) yields the sub-tensordict keys, as TensorDict does."""
        store_td["a"] = torch.randn(10, 2)
        store_td["n", "b"] = torch.randn(10, 3)
        store_td["n", "m", "c"] = torch.randn(10)

        keys = ["a", ("n", "b"), ("n", "m", "c"), ("n", "m"), "n"]
        assert list(store_td.keys(include_nested=True)) == keys
        assert len(store_td.keys(include_nested=True)) == 5
        assert list(store_td.to_tensordict().keys(include_nested=True)) == keys
        # the view of a sub-tensordict
        assert list(store_td["n"].keys(include_nested=True)) == [
            "b",
            ("m", "c"),
            "m",
        ]
        # leaves_only is unchanged
        assert list(store_td.keys(True, True)) == ["a", ("n", "b"), ("n", "m", "c")]
        # update() checks that the keys of the source cover the destination
        dest = TensorDict(
            {
                "a": torch.zeros(3, 2),
                "n": {"b": torch.zeros(3, 3), "m": {"c": torch.zeros(3)}},
            },
            batch_size=[3],
        )
        dest.update(store_td, update_batch_size=True)
        assert dest.batch_size == (10,)
        assert (dest["n", "m", "c"] == store_td["n", "m", "c"]).all()

    def test_keys_leaves_only(self, store_td):
        """Test keys with leaves_only=True."""
        store_td["a"] = torch.randn(10, 2)
        store_td["nested", "b"] = torch.randn(10, 3)

        leaf_keys = list(store_td.keys(leaves_only=True))
        assert "a" in leaf_keys
        # "nested" should not appear since it is not a leaf
        assert "nested" not in leaf_keys

    def test_keys_values_items_order(self, store_td):
        """keys(), values() and items() follow the sorted order of the key paths."""
        td = TensorDict(
            {
                "x": torch.zeros(10),
                "R": torch.ones(10),
                "n": {"b": torch.full((10,), 2.0), "a": torch.full((10,), 3.0)},
                "n-x": torch.full((10,), 4.0),
            },
            batch_size=[10],
        )
        for key in td.keys(True, True):
            store_td[key] = td[key]
        # "n-x" sorts before "n.a"
        keys = ["R", "n-x", ("n", "a"), ("n", "b"), "x"]
        assert list(store_td.keys(True, True)) == keys
        assert [key for key, _ in store_td.items(True, True)] == keys
        for key, value in zip(keys, store_td.values(True, True), strict=True):
            assert (value == td[key]).all()
        assert list(store_td.keys()) == ["R", "n-x", "n", "x"]
        assert list(store_td["n"].keys()) == ["a", "b"]
        assert list(store_td.to_tensordict().keys(True, True)) == keys

    def test_contains(self, store_td):
        """Test __contains__ via 'in' operator."""
        store_td["obs"] = torch.randn(10, 3)
        store_td["nested", "val"] = torch.randn(10, 2)

        assert "obs" in store_td.keys()
        assert "nested" in store_td.keys()
        assert ("nested", "val") in store_td.keys(include_nested=True)
        assert "nonexistent" not in store_td.keys()

    def test_contains_nested_key_without_include_nested(self, store_td):
        # The store finds a nested key in a view without nesting, with a
        # FutureWarning: it returns False in 0.17, as TensorDict does (#2060).
        store_td["nested", "val"] = torch.randn(10, 2)
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            assert "nested" in store_td.keys()
            assert ("nested",) in store_td.keys()
            assert ("nested", "val") in store_td.keys(include_nested=True)
            assert ("nested", "x") not in store_td.keys()
        for keys in (store_td.keys(), store_td.keys(leaves_only=True)):
            with pytest.warns(FutureWarning, match="returns False"):
                assert ("nested", "val") in keys

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

    @pytest.mark.parametrize(
        "method,key",
        [
            ("del_", "obs"),
            ("del_", "nested"),
            ("del_", ("nested", "a")),
            ("pop", "obs"),
            ("pop", "nested"),
            ("popitem", None),
            ("rename_key_", "obs"),
            ("rename_key_", "nested"),
            ("rename_key_", ("nested", "a")),
        ],
    )
    def test_existing_key_does_not_warn(self, store_td, method, key):
        # del_ and rename_key_ also act on every key under a sub-tensordict
        store_td["obs"] = torch.randn(10, 3)
        store_td["nested", "a"] = torch.randn(10, 2)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            if method == "popitem":
                store_td.popitem()
            elif method == "rename_key_":
                store_td.rename_key_(key, "renamed")
                assert "renamed" in store_td.keys()
            else:
                getattr(store_td, method)(key)
                assert key not in store_td.keys(True)

    @pytest.mark.parametrize("method", ["del_", "rename_key_"])
    @pytest.mark.parametrize("key", ["missing", ("nested", "missing")])
    def test_missing_key_warns(self, store_td, method, key):
        # https://github.com/pytorch/tensordict/issues/2062
        store_td["nested", "a"] = torch.randn(10, 2)
        args = (key,) if method == "del_" else (key, "renamed")
        with pytest.warns(
            FutureWarning, match=rf"^TensorDictStore\.{method} {_MISSING_KEY_WARNING}"
        ) as record:
            assert getattr(store_td, method)(*args) is store_td
        # count only FutureWarnings: under xdist on free-threaded Python, the
        # record once held 12 other warnings from code outside the call
        warned = [w for w in record if issubclass(w.category, FutureWarning)]
        assert len(warned) == 1, [str(w.message) for w in record]
        assert warned[0].filename == __file__
        assert set(store_td.keys(True, True)) == {("nested", "a")}

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

    def test_update_self_writes_copy(self, store_td):
        """update({key: store}) writes a copy of the data, where a TensorDict raises (#2741)."""
        store_td["obs"] = torch.ones(10, 3)
        store_td.update({"me": store_td})
        torch.testing.assert_close(store_td["me", "obs"], torch.ones(10, 3))
        assert "me" not in store_td["me"].keys()

    def test_replace_does_not_write_into_arg(self, store_td):
        """replace(store, **kwargs) leaves the store passed to it unchanged (#2984)."""
        store_td["obs"] = torch.zeros(10, 3)
        td = TensorDict(obs=torch.ones(10, 3), batch_size=[10])
        out = td.replace(store_td, extra=torch.ones(10))
        assert set(store_td.keys()) == {"obs"}
        assert type(out) is TensorDict
        assert set(out.keys()) == {"obs", "extra"}
        torch.testing.assert_close(out["obs"], torch.zeros(10, 3))
        torch.testing.assert_close(out["extra"], torch.ones(10))

    @pytest.mark.parametrize("stack_dim", [0, 1])
    def test_update__copy__lazy_source(self, store_td, stack_dim):
        """copy_ and update_ write a lazy stack into the store (#2593).

        Before, they changed copies of the stored entries and wrote nothing.
        """
        store_td.batch_size = [2, 2]

        def source(value, heterogeneous=False):
            return lazy_stack(
                [
                    TensorDict(
                        a=torch.full((2,), value + i),
                        n=TensorDict(b=torch.full((2, 3), value - i), batch_size=[2]),
                        **(
                            {"h": torch.full((2, 1 + i), value)}
                            if heterogeneous
                            else {}
                        ),
                        batch_size=[2],
                    )
                    for i in range(2)
                ],
                stack_dim,
            )

        expected = source(0.0).to_tensordict()
        store_td.update(expected)
        for td in (store_td, expected):
            assert td.copy_(source(1.0)) is td
            assert td.update_(source(2.0), keys_to_update=[("n", "b")]) is td
            # a lazy stack nested in the input
            td.update_(TensorDict(n=source(3.0).get("n"), batch_size=[2, 2]))
            td["n"].copy_(source(4.0).get("n"))
            # keys_to_update leaves out a key whose elements differ in shape
            td.update_(source(5.0, heterogeneous=True), keys_to_update=["a"])
        assert_allclose_td(store_td.to_tensordict(), expected)

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

    def test_pickle_nested_and_locked(self, store_td):
        """A pickled nested view keeps its prefix, and a pickled locked store is locked."""
        store_td["a"] = torch.arange(10.0)
        store_td["n", "b"] = torch.ones(10, 2)
        copies = []
        try:
            nested = pickle.loads(pickle.dumps(store_td["n"]))
            copies.append(nested)
            assert nested._prefix == "n"
            assert list(nested.keys()) == ["b"]
            assert nested.batch_size == torch.Size([10])
            assert torch.equal(nested["b"], torch.ones(10, 2))
            assert not nested.is_locked

            store_td.lock_()
            root = pickle.loads(pickle.dumps(store_td))
            copies.append(root)
            assert root.is_locked
            assert sorted(root.keys()) == ["a", "n"]
            assert root.batch_size == torch.Size([10])
            locked_nested = pickle.loads(pickle.dumps(store_td["n"]))
            copies.append(locked_nested)
            assert locked_nested.is_locked
            assert locked_nested._prefix == "n"
            assert torch.equal(locked_nested["b"], torch.ones(10, 2))
        finally:
            store_td.unlock_()
            for copy in copies:
                copy.close()

    def test_nested_clear_redis(self, store_td, store_kwargs):
        """clear_redis() on a nested view deletes the entries under the view only."""
        store_td["a"] = torch.arange(10.0)
        store_td["n", "b"] = torch.ones(10, 2)
        store_td["n", "m", "c"] = torch.ones(10)
        store_td["nn"] = torch.zeros(10)  # starts with "n", but is not under it
        store_td["n"].clear_redis()
        assert sorted(store_td.keys(True, True)) == ["a", "nn"]
        server_keys = _server_keys(store_td, store_kwargs)
        assert not [key for key in server_keys if ":n." in key]
        assert store_td._batch_size_key in server_keys
        reader = TensorDictStore.from_store(td_id=store_td._td_id, **store_kwargs)
        try:
            assert sorted(reader.keys(True, True)) == ["a", "nn"]
            assert torch.equal(reader["a"], torch.arange(10.0))
        finally:
            reader.close()

    def test_get_at_stale_key_cache(self, store_td, store_kwargs):
        """get_at of a key that another handle deleted, with a stale key cache."""
        store_td["a"] = torch.arange(10.0)
        store_td["n", "b"] = torch.ones(10, 2)
        reader = TensorDictStore.from_store(td_id=store_td._td_id, **store_kwargs)
        try:
            assert sorted(reader.keys()) == ["a", "n"]
            # the metadata of the reader is not cached yet
            assert torch.equal(reader.get_at(("n", "b"), 1), torch.ones(2))
            store_td.del_("a")
            with pytest.raises(KeyError, match="key a not found in TensorDictStore"):
                reader.get_at("a", 0)
        finally:
            reader.close()

    def test_entry_class(self, store_td):
        """Test entry_class returns correct types."""

        store_td["obs"] = torch.randn(10, 3)
        store_td["nested", "a"] = torch.randn(10, 2)
        assert store_td.entry_class("obs") is torch.Tensor
        assert store_td.entry_class("nested") is TensorDictStore

    def test_store_device_setter_moves_reads(self, store_td):
        """The device of a store is the device of what it reads."""
        store_td["x"] = torch.randn(10, 3)
        store_td["nested", "y"] = torch.randn(10, 2)
        store_td.device = "meta"
        assert store_td.device == torch.device("meta")
        assert store_td["x"].device == torch.device("meta")
        assert store_td["nested", "y"].device == torch.device("meta")
        store_td.device = None
        assert store_td.device is None
        assert store_td["x"].device == torch.device("cpu")

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

    def test_to_tensordict_and_repr_do_not_read_leaves(self, store_td, monkeypatch):
        source = TensorDict(
            obs=torch.arange(10.0),
            nested=TensorDict(x=torch.arange(20).view(10, 2), batch_size=[10]),
            batch_size=[10],
        )
        store_td.update(source)

        def no_read(self, *args, **kwargs):
            raise AssertionError("a leaf was read")

        monkeypatch.setattr(TensorDictStore, "_get_str", no_read)
        monkeypatch.setattr(TensorDictStore, "_aget_tensor", no_read)
        assert repr(store_td) == (
            "TensorDictStore(\n"
            "    keys=['nested', 'obs'],\n"
            "    batch_size=torch.Size([10]),\n"
            "    device=None,\n"
            f"    backend={store_td._backend!r},\n"
            f"    td_id={store_td._td_id!r})"
        )
        local = store_td.to_tensordict()
        monkeypatch.undo()
        assert type(local) is TensorDict
        assert local.batch_size == (10,)
        assert (local == source).all()

    @pytest.mark.parametrize(
        "with_string",
        [
            False,
            True,
        ],
    )
    def test_to_dtype_store(self, store_td, with_string):
        source = TensorDict(
            obs=torch.arange(10.0),
            nested=TensorDict(x=torch.arange(20).view(10, 2), batch_size=[10]),
            batch_size=[10],
        )
        if with_string:
            source["s"] = "hello"
        store_td.update(source)
        for args in ((torch.float64,), ("cpu",)):
            out = store_td.to(*args)
            expected = source.to(*args)
            assert type(out) is TensorDict
            assert out.device == expected.device
            assert out["nested", "x"].dtype == expected["nested", "x"].dtype
            assert (out == expected).all()
            if with_string:
                assert out["s"] == "hello"

    def test_to_context_manager_same_device(self, store_kwargs):
        # to() on the device of the store returns the store, so the block
        # writes to it and nothing is written back on exit
        store = TensorDictStore(batch_size=[10], device="cpu", **store_kwargs)
        try:
            store["obs"] = torch.zeros(10)
            store["nested", "x"] = torch.zeros(10, 2)
            calls = []
            run_sync = store._run_sync

            def counting_run_sync(coro):
                calls.append(None)
                return run_sync(coro)

            with store.to("cpu") as same:
                assert same is store
                same["obs"] = torch.ones(10)
                store._run_sync = counting_run_sync
            del store._run_sync
            assert calls == []
            assert (store["obs"] == 1).all()
        finally:
            store.clear_redis()
            store.close()

    def test_empty(self, store_td):
        """Test empty()."""
        store_td["obs"] = torch.randn(10, 3)
        empty = store_td.empty()
        assert isinstance(empty, TensorDict)
        assert empty.batch_size == torch.Size([10])
        assert len(list(empty.keys())) == 0

    def test_store_empty_recurse_reads_no_leaf(self, store_td, monkeypatch):
        store_td["a"] = torch.zeros(10, 2)
        store_td["s"] = NonTensorData("text", batch_size=[10])
        store_td["n"] = TensorDict(
            x=torch.ones(10),
            m=TensorDict(y=torch.ones(10, 4), batch_size=[10, 4]),
            batch_size=[10],
        )
        reads = []
        get_tensor = TensorDictStore._aget_tensor

        async def counting_get_tensor(self, key_path):
            reads.append(key_path)
            return await get_tensor(self, key_path)

        monkeypatch.setattr(TensorDictStore, "_aget_tensor", counting_get_tensor)
        for td in (store_td, store_td["n"]):
            empty = td.empty(recurse=True)
            assert type(empty) is TensorDict
            assert empty.batch_size == torch.Size([10])
        empty = store_td.empty(recurse=True)
        # the sub-tensordicts keep their batch size; non-tensor data is left out
        assert set(empty.keys(True)) == {"n", ("n", "m")}
        assert empty["n", "m"].batch_size == torch.Size([10, 4])
        assert store_td.empty(recurse=True, batch_size=[])["n", "m"].batch_size == (
            torch.Size([])
        )
        assert reads == []

    def test_store_setitem_scalar_mask_list(self, store_td):
        dense = TensorDict(
            a=torch.arange(30.0).view(10, 3),
            n=TensorDict(c=torch.arange(10), batch_size=[10]),
            batch_size=[10],
        )
        store_td.update(dense)
        mask = torch.arange(10) % 3 == 0
        for index, value in (
            (mask, 0),
            ([1, 2], -1),
            (slice(4, 6), {"a": torch.full((2, 3), 7.0)}),
            ((Ellipsis, 7), 9),
            ([8, 9], dense[:2] * 0 + 4),
            (Ellipsis, 3),
        ):
            store_td[index] = value
            dense[index] = value
            assert (store_td.to_tensordict() == dense).all(), index
        with pytest.raises(TypeError, match="not a tensor"):
            store_td[[0, 1]] = torch.zeros(2)

    def test_store_setitem_ellipsis_0d(self, store_kwargs):
        # with an empty batch size, the Ellipsis covers no dim: a 0-d entry
        # takes the scalar too
        store = TensorDictStore.from_tensordict(
            TensorDict(a=torch.zeros(()), b=torch.zeros(3)), **store_kwargs
        )
        try:
            store[...] = 3
            assert (store["a"] == 3).all() and (store["b"] == 3).all()
            store[...] = TensorDict(a=torch.ones(()), b=torch.ones(3))
            assert (store["a"] == 1).all() and (store["b"] == 1).all()
        finally:
            store.clear_redis()
            store.close()

    def test_store_setitem_self_nesting(self, store_td):
        store_td["a"] = torch.zeros(10)
        with pytest.raises(
            ValueError, match=r"Cannot set a tensordict inside itself \(key 'me'\)"
        ):
            store_td["me"] = store_td
        assert set(store_td.keys()) == {"a"}

    def test_fill_(self, store_td):
        """Test fill_."""
        store_td["obs"] = torch.randn(10, 3)
        store_td.fill_("obs", 42.0)
        result = store_td["obs"]
        assert torch.allclose(result, torch.full((10, 3), 42.0))

    def test_fill_nested(self, store_td):
        """fill_ on a nested key fills each of its leaves on the server."""
        store_td["nested"] = TensorDict(
            a=torch.zeros(10, 3),
            m=TensorDict(b=torch.zeros(10), batch_size=[10]),
            batch_size=[10],
        )
        assert store_td.fill_("nested", 2.0) is store_td
        assert (store_td["nested", "a"] == 2).all()
        assert (store_td["nested", "m", "b"] == 2).all()

    @pytest.mark.parametrize("key", ["missing", ("missing", "x")])
    def test_fill_missing_key(self, store_td, key):
        """fill_ of a missing key raises a KeyError, as on a TensorDict."""
        store_td["obs"] = torch.zeros(10, 3)
        with pytest.raises(KeyError):
            store_td.fill_(key, 1.0)

    def test_detach_no_read(self, store_td, monkeypatch):
        """detach_ returns the store without calling the server."""
        store_td["obs"] = torch.zeros(10, 3)

        def no_round_trip(self, coro):
            coro.close()
            raise AssertionError("the server was called")

        with monkeypatch.context() as patch:
            patch.setattr(TensorDictStore, "_run_sync", no_round_trip)
            assert store_td.detach_() is store_td

    def test_inplace_methods_write(self, store_td):
        """The in-place methods, update_ and copy_ write to the server (#2242)."""
        source = TensorDict(
            a=torch.randn(10, 3),
            nested=TensorDict(b=torch.randn(10, 2), batch_size=[10]),
            batch_size=[10],
        )
        store_td.update(source)
        expected = source.clone()
        other = source.apply(lambda x: torch.rand_like(x) + 1)
        for td in (store_td, expected):
            assert td.abs_() is td
            assert td.add_(other, alpha=2.0) is td
            td *= 2
            assert td.clamp_max_(other) is td
            assert td.update_(other.select("a") + 1) is td
            assert td.update_(other * 3, keys_to_update=[("nested", "b")]) is td
        assert_allclose_td(store_td.to_tensordict(), expected)
        store_td.copy_(other)
        assert_allclose_td(store_td.to_tensordict(), other)

    def test_inplace_methods_non_tensor(self, store_td):
        """The in-place methods skip a non-tensor entry, as on a TensorDict."""
        source = TensorDict(
            a=torch.randn(10, 3),
            nested=TensorDict(b=torch.randn(10, 2), batch_size=[10]),
            batch_size=[10],
        )
        store_td.update(source.clone().set_non_tensor("s", "hello"))
        expected = source.clone()
        other = source.apply(lambda x: torch.rand_like(x) + 1)
        for td in (store_td, expected):
            assert td.add_(1.0) is td
            assert td.abs_() is td
            assert td.lerp_(other, 0.5) is td
            assert td.mul_(other) is td
        assert_allclose_td(store_td.to_tensordict().exclude("s"), expected)
        assert store_td.zero_() is store_td
        assert (store_td["a"] == 0).all()
        assert (store_td["nested", "b"] == 0).all()
        assert store_td["s"] == "hello"

    def test_update_from_store_round_trips(self, store_td, store_kwargs):
        """update_ from a store reads it as often as a TensorDict destination does."""
        td = TensorDict(
            {
                **{f"k{i}": torch.ones(10, 3) for i in range(5)},
                "nested": TensorDict(b=torch.ones(10, 2), batch_size=[10]),
            },
            batch_size=[10],
        )
        store_td.update(td * 0)
        source = TensorDictStore.from_tensordict(td, **store_kwargs)
        try:
            calls = []
            run_sync = source._run_sync

            def counting_run_sync(coro):
                calls.append(None)
                return run_sync(coro)

            source._run_sync = counting_run_sync
            for keys_to_update in (None, ["k0"]):
                calls.clear()
                (td * 0).update_(source, keys_to_update=keys_to_update)
                expected = len(calls)
                calls.clear()
                store_td.update_(source, keys_to_update=keys_to_update)
                assert len(calls) == expected
            assert (store_td["k0"] == 1).all()
        finally:
            del source._run_sync
            source.clear_redis()
            source.close()

    def test_masked_fill_(self, store_td):
        """masked_fill_ fills the masked batch elements, as on a TensorDict."""
        source = TensorDict(
            {
                "a": torch.zeros(10, 10),
                "b": torch.zeros(10, 2),
                "c": torch.zeros(10),
                "nested": TensorDict({"d": torch.zeros(10, 4)}, [10]),
            },
            [10],
        )
        store_td.update(source)
        mask = torch.zeros(10, dtype=torch.bool)
        mask[[1, 4]] = True
        assert store_td.masked_fill_(mask, 1.0) is store_td
        expected = source.masked_fill_(mask, 1.0)
        for key in expected.keys(include_nested=True, leaves_only=True):
            torch.testing.assert_close(store_td[key], expected[key])

    def test_masked_fill_non_tensor(self, store_td):
        """masked_fill_ leaves a non-tensor entry unchanged, as on a TensorDict."""
        store_td["a"] = torch.zeros(10, 3)
        store_td["text"] = "hello"
        mask = torch.zeros(10, dtype=torch.bool)
        mask[[1, 4]] = True
        assert store_td.masked_fill_(mask, 1.0) is store_td
        torch.testing.assert_close(
            store_td["a"], torch.zeros(10, 3).masked_fill_(mask[:, None], 1.0)
        )
        assert store_td["text"] == "hello"

    def test_masked_select_inplace(self, store_td):
        """masked_select(inplace=True) masks the entries of the store itself."""
        source = TensorDict(
            {
                "a": torch.arange(10.0),
                "nested": TensorDict({"d": torch.ones(10, 4)}, [10]),
            },
            [10],
        )
        store_td.update(source)
        mask = torch.zeros(10, dtype=torch.bool)
        mask[[1, 4, 7]] = True
        assert store_td.masked_select(mask, inplace=True) is store_td
        assert store_td.batch_size == torch.Size([3])
        expected = source.masked_select(mask)
        for key in expected.keys(include_nested=True, leaves_only=True):
            torch.testing.assert_close(store_td[key], expected[key])

    def test_masked_select_copy(self, store_td, monkeypatch):
        """masked_select returns a dense copy, read in one to_tensordict call."""
        source = TensorDict(
            {
                "a": torch.arange(10.0),
                "nested": TensorDict({"d": torch.ones(10, 4)}, [10]),
            },
            [10],
        )
        store_td.update(source)
        mask = torch.zeros(10, dtype=torch.bool)
        mask[[1, 4, 7]] = True
        calls = []
        to_tensordict = TensorDictStore.to_tensordict

        def counting_to_tensordict(self, *args, **kwargs):
            calls.append(self)
            return to_tensordict(self, *args, **kwargs)

        monkeypatch.setattr(TensorDictStore, "to_tensordict", counting_to_tensordict)
        result = store_td.masked_select(mask)
        assert len(calls) == 1 and calls[0] is store_td
        assert type(result) is TensorDict
        assert (result == source.masked_select(mask)).all()
        assert store_td.batch_size == (10,)
        assert (store_td.to_tensordict() == source).all()

    def test_is_contiguous(self, store_td):
        """Redis TDs are not contiguous."""
        assert not store_td.is_contiguous()

    def test_shape_ops_raise(self, store_td):
        """Shape ops raise on TensorDictStore but work after to_tensordict()."""
        store_td["obs"] = torch.randn(10, 3)

        with pytest.raises(RuntimeError):
            store_td.view(2, 5)
        with pytest.raises(RuntimeError):
            store_td.permute(0)
        with pytest.raises(RuntimeError):
            store_td.unsqueeze(0)
        with pytest.raises(RuntimeError):
            store_td.squeeze()
        # squeezing a dim of size != 1 is a no-op, as on TensorDict
        assert store_td.squeeze(0) is store_td

        # Escape hatch: materialize first, then shape ops work
        local = store_td.to_tensordict()
        assert local.view(2, 5).shape == torch.Size([2, 5])
        assert local.unsqueeze(0).shape == torch.Size([1, 10])
        assert local.squeeze(0).shape == torch.Size([10])

    def test_share_memory_raises(self, store_td):
        """share_memory_ should raise."""
        with pytest.raises(NotImplementedError):
            store_td.share_memory_()

    def test_vmap_raises(self, store_td):
        """vmap raises, also on a function that returns the store."""
        store_td["obs"] = torch.randn(10, 3)
        with pytest.raises(RuntimeError, match="cannot be used with vmap"):
            torch.vmap(lambda x: x)(store_td)
        # this used to return None
        with pytest.raises(RuntimeError, match="cannot be used with vmap"):
            torch.vmap(lambda x: store_td)(torch.zeros(2))

    def test_memmap(self, store_td, tmp_path):
        """memmap and save write a memory-mapped copy; only memmap_ raises."""
        source = TensorDict(
            {"obs": torch.randn(10, 3), "nested": {"a": torch.randn(10, 2)}}, [10]
        )
        store_td.update(source)
        for out in (
            store_td.memmap(tmp_path / "memmap"),
            store_td.save(tmp_path / "save"),
            store_td.memmap(),
        ):
            assert type(out) is TensorDict
            assert out.is_memmap() and out.is_locked
            assert (out == source).all()
        assert (TensorDict.load_memmap(tmp_path / "save") == source).all()
        with pytest.raises(RuntimeError, match="in-place"):
            store_td.memmap_(tmp_path / "memmap_")

    def test_close_releases_event_loop(self, store_kwargs):
        """close() closes the event loop, which releases the sockets that it holds."""
        td = TensorDictStore(batch_size=[5], **store_kwargs)
        # a closed store must not appear in a failed assertion: its repr would
        # wait on the loop forever
        loop = td._loop
        td.clear_redis()
        td.close()
        assert loop.is_closed()
        td.close()  # closing again is a no-op

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

    def test_pop_nested(self, store_td):
        """pop and popitem return the data of a nested entry that they delete."""
        a, c, d = torch.randn(10, 2), torch.randn(10, 3), torch.randn(10)
        store_td["nested", "a"] = a
        store_td["nested", "b", "c"] = c
        store_td["other", "d"] = d

        value = store_td.pop(("nested", "b"))
        assert isinstance(value, TensorDict)
        assert torch.equal(value["c"], c)
        assert ("nested", "b") not in store_td.keys(include_nested=True)

        value = store_td.pop("other")
        assert isinstance(value, TensorDict)
        assert torch.equal(value["d"], d)
        assert "other" not in store_td.keys()

        key, value = store_td.popitem()
        assert key == "nested"
        assert isinstance(value, TensorDict)
        assert torch.equal(value["a"], a)
        assert list(store_td.keys()) == []

    def test_rename_key(self, store_td):
        """Test rename_key_."""
        tensor = torch.randn(10, 3)
        store_td["old_name"] = tensor
        store_td.rename_key_("old_name", "new_name")
        assert "old_name" not in store_td.keys()
        assert "new_name" in store_td.keys()
        assert torch.allclose(store_td["new_name"], tensor)

    def test_rename_key_locked_warns(self, store_td):
        # a locked store renamed its keys without a check: it still does, with
        # a warning, until 0.17
        store_td["obs"] = torch.randn(10, 3)
        store_td.lock_()
        with pytest.warns(
            FutureWarning,
            match=r"^TensorDictStore\.rename_key_ renames keys of a locked "
            r"tensordict today\. Starting with TensorDict 0\.17",
        ) as record:
            assert store_td.rename_key_("obs", "renamed") is store_td
        # count only FutureWarnings: under xdist on free-threaded Python, the
        # record once held 12 other warnings from code outside the call
        warned = [w for w in record if issubclass(w.category, FutureWarning)]
        assert len(warned) == 1, [str(w.message) for w in record]
        assert warned[0].filename == __file__
        assert set(store_td.keys()) == {"renamed"}

    def test_rename_key_bad_new_key(self, store_td):
        # the store moved "obs" to the empty key ""
        store_td["obs"] = torch.randn(10, 3)
        with pytest.raises(TypeError, match="Expected new_name to be a string"):
            store_td.rename_key_("obs", 1)
        assert set(store_td.keys()) == {"obs"}

    def test_rename_key_safe_nested_destination(self, store_td):
        # safe=True checks the sub-tensordicts too: the store renamed "obs" to
        # "nested", which then was both a leaf and a sub-tensordict
        store_td["obs"] = torch.randn(10, 3)
        store_td["nested", "a"] = torch.randn(10, 2)
        with pytest.raises(KeyError, match="key nested already present"):
            store_td.rename_key_("obs", "nested", safe=True)
        assert set(store_td.keys(True, True)) == {"obs", ("nested", "a")}

    @pytest.mark.parametrize("old_key,new_key", _RENAME_OCCUPIED)
    def test_rename_key_occupied_destination(self, store_kwargs, old_key, new_key):
        # the entry replaces what is at the destination, as in TensorDict: the
        # store kept the old leaves there, or a leaf that was also a
        # sub-tensordict, and could not be read
        expected = _rename_source().rename_key_(old_key, new_key)
        store_td = TensorDictStore.from_tensordict(_rename_source(), **store_kwargs)
        try:
            store_td.rename_key_(old_key, new_key)
            _assert_store_matches(store_td, expected)
        finally:
            store_td.clear_redis()
            store_td.close()

    @pytest.mark.parametrize("old_key,new_key,safe,match", _RENAME_INVALID)
    def test_rename_key_invalid_destination(
        self, store_kwargs, old_key, new_key, safe, match
    ):
        # the store moved the entry below a leaf
        store_td = TensorDictStore.from_tensordict(_rename_source(), **store_kwargs)
        try:
            with pytest.raises(KeyError, match=match):
                store_td.rename_key_(old_key, new_key, safe=safe)
            _assert_store_matches(store_td, _rename_source())
        finally:
            store_td.clear_redis()
            store_td.close()

    @pytest.mark.parametrize("safe", [False, True])
    @pytest.mark.parametrize("old_key,new_key", _RENAME_INSIDE)
    def test_rename_key_inside_old_key(self, store_kwargs, old_key, new_key, safe):
        # the entry moves, as pop and set do on a TensorDict, and no data is
        # lost
        expected = _rename_source()
        expected.set(new_key, expected.pop(old_key))
        store_td = TensorDictStore.from_tensordict(_rename_source(), **store_kwargs)
        try:
            store_td.rename_key_(old_key, new_key, safe=safe)
            _assert_store_matches(store_td, expected)
        finally:
            store_td.clear_redis()
            store_td.close()

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

    def test_names_setter_keeps_nested_trailing_name(self, store_kwargs):
        """Setting or clearing the parent names keeps the child's own names."""
        source = TensorDict(
            {"agents": TensorDict({"obs": torch.zeros(2, 3, 4)}, [2, 3])}, [2]
        )
        td = TensorDictStore.from_tensordict(source, **store_kwargs)
        try:
            td["agents"].names = [None, "agent"]
            td.names = ["batch"]
            assert td["agents"].names == ["batch", "agent"]
            td.names = None
            assert td["agents"].names == [None, "agent"]
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

    def test_from_store_cache_metadata(self, store_kwargs):
        """from_store passes cache_metadata to the store, not to the Redis client."""
        writer = TensorDictStore(batch_size=[3], **store_kwargs)
        try:
            writer["x"] = torch.arange(3.0)
            reader = TensorDictStore.from_store(
                td_id=writer._td_id, cache_metadata=False, **store_kwargs
            )
            try:
                assert reader._meta_cache is None
                assert torch.equal(reader["x"], torch.arange(3.0))
            finally:
                reader.close()
        finally:
            writer.clear_redis()
            writer.close()

    def test_from_store_not_found(self, store_kwargs):
        """from_store should raise KeyError for unknown td_id."""

        with pytest.raises(KeyError, match="No TensorDictStore"):
            TensorDictStore.from_store(td_id="nonexistent-uuid", **store_kwargs)

    def test_from_store_closes_its_client_when_the_read_fails(self, store_kwargs):
        """from_store closes the client it reads the metadata with, also on an error."""
        import redis

        writer = TensorDictStore(batch_size=[2], **store_kwargs)
        try:
            client = redis.Redis(port=store_kwargs["port"], db=store_kwargs["db"])
            try:
                # GET on a set fails with WRONGTYPE once the read is connected
                key = writer._redis_key("__device__")
                client.delete(key)
                client.sadd(key, "not a string")
            finally:
                client.close()
            gc.collect()  # garbage of earlier tests is finalized outside the check
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ResourceWarning)
                with pytest.raises(redis.ResponseError, match="WRONGTYPE"):
                    TensorDictStore.from_store(td_id=writer._td_id, **store_kwargs)
                gc.collect()
            assert not [w for w in caught if issubclass(w.category, ResourceWarning)]
        finally:
            writer.clear_redis()
            writer.close()

    def test_get_at_missing_key(self, store_td):
        store_td["a"] = torch.arange(10.0)
        store_td["n", "b"] = torch.ones(10, 2)
        assert store_td.get_at("zz", 0, None) is None
        assert torch.equal(store_td.get_at("n", 1)["b"], torch.ones(2))

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

    def test_indexed_read_bool_list(self, store_td):
        """td[list_of_bools] is a boolean mask, as for a tensor."""
        obs = torch.randn(10, 3)
        store_td["obs"] = obs
        mask = [i in (0, 3, 9) for i in range(10)]
        sub = store_td[mask]
        assert sub.batch_size == torch.Size([3])
        assert torch.allclose(sub["obs"], obs[torch.tensor(mask)])

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

    def test_nd_mask(self, store_kwargs):
        """Reads and writes through a 2-D mask match the same ops on a tensordict."""
        td = TensorDict({"x": torch.arange(60.0).view(3, 4, 5)}, [3, 4])
        store = TensorDictStore.from_tensordict(td, **store_kwargs)
        try:
            mask = torch.tensor([[True, False, True, False]] * 3)
            torch.testing.assert_close(store[mask]["x"], td[mask]["x"])
            torch.testing.assert_close(store.get_at("x", mask), td["x"][mask])
            value = -torch.ones(6, 5)
            store.set_at_("x", value, mask)
            td.set_at_("x", value, mask)
            store[~mask] = TensorDict({"x": torch.zeros(6, 5)}, [6])
            td[~mask] = TensorDict({"x": torch.zeros(6, 5)}, [6])
            torch.testing.assert_close(store["x"], td["x"])
        finally:
            store.clear_redis()
            store.close()

    @pytest.mark.filterwarnings("ignore:indexing with dtype torch.uint8")
    def test_uint8_mask(self, store_kwargs):
        """A uint8 tensor is a mask (deprecated in torch), not a list of rows."""
        td = TensorDict({"x": torch.arange(5.0)}, [5])
        store = TensorDictStore.from_tensordict(td, **store_kwargs)
        try:
            mask = torch.tensor([1, 0, 1, 0, 2], dtype=torch.uint8)
            torch.testing.assert_close(store[mask]["x"], td[mask]["x"])
            torch.testing.assert_close(store.get_at("x", mask), td["x"][mask])
            store[mask] = TensorDict({"x": -torch.ones(3)}, [3])
            td[mask] = TensorDict({"x": -torch.ones(3)}, [3])
            torch.testing.assert_close(store["x"], td["x"])
        finally:
            store.clear_redis()
            store.close()

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

    @pytest.mark.parametrize(
        "index", [(..., 1), (0, ..., 1), (1, ...)], ids=["front", "middle", "end"]
    )
    def test_indexed_write_ellipsis_feature_dims(self, store_kwargs, index):
        """An Ellipsis in an index written to the store stands for the batch dims, as on a TensorDict (#2599)."""
        td = TensorDict(
            a=torch.arange(6.0).view(2, 3),
            b=torch.arange(24.0).view(2, 3, 4),
            n=TensorDict(c=torch.arange(30).view(2, 3, 5), batch_size=[2, 3]),
            batch_size=[2, 3],
        )
        store_td = TensorDictStore.from_tensordict(td.clone(), **store_kwargs)
        try:
            value = td[index].apply(lambda x: -x)
            for dest in (store_td, td):
                dest[index] = 7.0
                dest[index] = value.exclude("a")
            assert (store_td.to_tensordict() == td).all()
            assert (store_td[index] == td[index]).all()
        finally:
            store_td.clear_redis()
            store_td.close()

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

    def test_get_at_set_at_bool_list(self, store_td):
        """get_at and set_at_ read a list of bools as a mask, as store[mask] does."""
        obs = torch.randn(10, 3)
        store_td["obs"] = obs
        mask = [i in (0, 3, 9) for i in range(10)]
        torch.testing.assert_close(store_td.get_at("obs", mask), obs[mask])
        store_td.set_at_("obs", torch.ones(3, 3), mask)
        obs[mask] = 1
        torch.testing.assert_close(store_td["obs"], obs)
        # one row per list element is not the selected shape
        with pytest.raises(RuntimeError):
            store_td.set_at_("obs", torch.zeros(10, 3), mask)
        torch.testing.assert_close(store_td["obs"], obs)

    @pytest.mark.parametrize(
        "index", [1, [0, 2], torch.tensor([1, 3]), slice(2, 5), [True, False] * 5]
    )
    def test_set_at_python_scalar(self, store_td, index):
        """set_at_ writes a Python scalar in the dtype of the entry, as torch does."""
        store_td["count"] = torch.zeros(10, 2, dtype=torch.long)
        store_td["obs"] = torch.zeros(10, 3)
        count, obs = torch.zeros(10, 2, dtype=torch.long), torch.zeros(10, 3)
        store_td.set_at_("count", -3.5, index)
        store_td.set_at_("obs", 2, index)
        count[index] = -3.5
        obs[index] = 2
        torch.testing.assert_close(store_td["count"], count)
        torch.testing.assert_close(store_td["obs"], obs)

    def test_set_at_byte_range(self, store_td):
        """set_at_ should use SETRANGE for a single key."""
        store_td["obs"] = torch.zeros(10, 3)
        new_val = torch.ones(3) * 42.0
        store_td.set_at_("obs", new_val, 3)

        full = store_td["obs"]
        assert torch.allclose(full[3], new_val)
        assert torch.allclose(full[:3], torch.zeros(3, 3))
        assert torch.allclose(full[4:], torch.zeros(6, 3))

    @pytest.mark.parametrize(
        "index",
        [3, slice(2, 5), [0, 9], torch.tensor([1, 3]), torch.arange(10) < 4, (..., 3)],
    )
    def test_set_at_update_at_nested(self, store_td, index):
        """set_at_ and update_at_ write each leaf of a tensordict value at the
        index, as on a TensorDict."""
        td = TensorDict(
            a=torch.zeros(10),
            n=TensorDict(
                b=torch.zeros(10),
                m=TensorDict(c=torch.zeros(10, 2, dtype=torch.long), batch_size=[10]),
                batch_size=[10],
            ),
            batch_size=[10],
        )
        td["n"].set_non_tensor("s", "x")
        store_td.update(td)
        value = td[index] + 1
        value["n"].set_non_tensor("s", "y")
        nested = value["n"] + 1
        nested.set_non_tensor("s", "z")
        for target in (store_td, td):
            target.update_at_(value, index)
            target.set_at_("n", nested, index)
            target.set_at_(("n", "m"), nested["m"] + 1, index)

        def check():
            for key in td.keys(True, True):
                torch.testing.assert_close(store_td[key], td[key])
            assert store_td.get(("n", "s")).tolist() == td.get(("n", "s")).tolist()

        check()
        # a leaf that the store lacks raises before anything is written
        missing = (nested + 2).set("z", nested["b"])
        missing.set_non_tensor("s", "w")
        with pytest.raises(KeyError, match="not found"):
            store_td.set_at_("n", missing, index)
        check()
        assert "z" not in store_td["n"].keys()

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


def _indexed_store(store_kwargs, stacked, source):
    if stacked:
        return LazyStackedTensorDictStore.from_lazy_stack(
            lazy_stack(source.unbind(0)), **store_kwargs
        )
    return TensorDictStore.from_tensordict(source, **store_kwargs)


# One case per store write path. The assignment rules themselves are checked
# against PyTorch in test_store_utils_split.py.
_MASK = torch.zeros(10, dtype=torch.bool)
_MASK[[0, 2]] = True


@pytest.mark.parametrize("stacked", [False, True])
@pytest.mark.parametrize(
    "idx,value_shape",
    [
        (-1, (3,)),  # single row
        (slice(0, 2), ()),  # contiguous slice, broadcast scalar
        (slice(0, 4, 2), (3,)),  # stepped slice, broadcast row
        (range(4, 0, -2), (2, 3)),  # range, reversed
        ([0, -1], (2, 3)),  # list with a negative index
        (torch.tensor([[0, -1], [2, 3]]), (3,)),  # multidimensional tensor
        (_MASK, ()),  # boolean mask
    ],
)
def test_store_indexed_write_matches_torch(store_kwargs, stacked, idx, value_shape):
    source = TensorDict(
        x=torch.arange(30, dtype=torch.float32).reshape(10, 3), batch_size=[10]
    )
    store = _indexed_store(store_kwargs, stacked, source)
    try:
        # float64 is converted for slices and integers, and rejected for list
        # and tensor indices, as in PyTorch.
        value = torch.full(value_shape, 7.0, dtype=torch.float64)
        expected = source["x"].clone()
        try:
            expected[idx] = value
        except RuntimeError:
            with pytest.raises(RuntimeError):
                store.set_at_("x", value, idx)
            expected[idx] = value.float()
            store.set_at_("x", value.float(), idx)
        else:
            store.set_at_("x", value, idx)
        torch.testing.assert_close(store["x"], expected)
        torch.testing.assert_close(store.get_at("x", idx), expected[idx])
    finally:
        store.clear_redis()
        store.close()


@pytest.mark.parametrize("stacked", [False, True])
@pytest.mark.parametrize("idx", [10, [-11, 0], range(9, 11), _MASK[:9]])
def test_store_index_bounds(store_kwargs, stacked, idx):
    source = TensorDict(
        x=torch.arange(30, dtype=torch.float32).reshape(10, 3), batch_size=[10]
    )
    store = _indexed_store(store_kwargs, stacked, source)
    try:
        with pytest.raises(IndexError):
            store.get_at("x", idx)
        with pytest.raises(IndexError):
            store[idx]
        value = torch.ones(3)
        with pytest.raises(IndexError):
            store.set_at_("x", value, idx)
        with pytest.raises(IndexError):
            store[idx] = TensorDict(x=value, batch_size=[])
        torch.testing.assert_close(store["x"], source["x"])
    finally:
        store.clear_redis()
        store.close()


@pytest.mark.parametrize("stacked", [False, True])
@pytest.mark.parametrize(
    "idx,value_shape",
    [
        ((0, slice(None)), (3,)),  # tuple index: full read-modify-write
        (slice(5, 5), (3,)),  # empty selection: nothing is written
    ],
)
def test_store_indexed_write_fallbacks(store_kwargs, stacked, idx, value_shape):
    source = TensorDict(
        x=torch.arange(30, dtype=torch.float32).reshape(10, 3), batch_size=[10]
    )
    store = _indexed_store(store_kwargs, stacked, source)
    try:
        value = torch.full(value_shape, 7, dtype=torch.int64)
        expected = source["x"].clone()
        expected[idx] = value
        store.set_at_("x", value, idx)
        torch.testing.assert_close(store["x"], expected)
    finally:
        store.clear_redis()
        store.close()


@pytest.mark.parametrize("stacked", [False, True])
@pytest.mark.parametrize(
    "idx", [0, slice(0, 2), slice(0, 4, 2), [0, 2], (0, slice(None))]
)
def test_store_rejected_write_keeps_data(store_kwargs, stacked, idx):
    source = TensorDict(
        x=torch.arange(30, dtype=torch.float32).reshape(10, 3), batch_size=[10]
    )
    store = _indexed_store(store_kwargs, stacked, source)
    try:
        with pytest.raises(RuntimeError):
            source["x"].clone()[idx] = torch.ones(4)
        with pytest.raises(RuntimeError):
            store.set_at_("x", torch.ones(4), idx)
        torch.testing.assert_close(store["x"], source["x"])
    finally:
        store.clear_redis()
        store.close()


@pytest.mark.parametrize("idx", [10, torch.tensor([-11, 0])])
def test_first_indexed_write_bounds(store_kwargs, idx):
    store = TensorDictStore(batch_size=[10], **store_kwargs)
    try:
        value = torch.ones(3) if isinstance(idx, int) else torch.ones(2, 3)
        with pytest.raises(IndexError):
            store.set_at_("x", value, idx)
        assert not list(store.keys())
    finally:
        store.clear_redis()
        store.close()


@pytest.mark.parametrize("heterogeneous", [False, True])
@pytest.mark.parametrize("method", ["element", "key"])
def test_stack_element_write_value(store_kwargs, heterogeneous, method):
    tds = [
        TensorDict(x=torch.zeros(2 + i if heterogeneous else 2), batch_size=[])
        for i in range(3)
    ]
    store = LazyStackedTensorDictStore.from_lazy_stack(lazy_stack(tds), **store_kwargs)
    try:
        # int64 is converted to the stored float32; a scalar is broadcast.
        for value in [torch.full((2,), 7), torch.tensor(8)]:
            if method == "element":
                store[-3] = TensorDict(x=value, batch_size=[])
            else:
                store[-3].set_("x", value)
            torch.testing.assert_close(store[0]["x"], value.float().expand(2))
        for i in [1, 2]:
            torch.testing.assert_close(store[i]["x"], tds[i]["x"])
    finally:
        store.clear_redis()
        store.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA")
@pytest.mark.parametrize("stacked", [False, True])
@pytest.mark.parametrize(
    "idx,value",
    [
        (slice(0, 4, 2), lambda: torch.full((3,), 7.0, device="cuda")),
        (
            [[0, 2], [1, 3]],
            lambda: torch.full((2, 2, 3), 7.0, device="cuda"),
        ),
        ("mask", lambda: torch.tensor(7.0)),
        ("mask", lambda: torch.tensor(7.0, device="cuda")),
    ],
)
def test_store_indexed_write_cuda(store_kwargs, stacked, idx, value):
    source = TensorDict(
        x=torch.arange(15, dtype=torch.float32, device="cuda").reshape(5, 3),
        batch_size=[5],
        device="cuda",
    )
    if idx == "mask":
        idx = torch.tensor([True, False, True, False, False], device="cuda")
    elif isinstance(idx, list):
        idx = torch.tensor(idx, device="cuda")
    value = value()
    store = _indexed_store(store_kwargs, stacked, source)
    try:
        expected = source["x"].clone()
        expected[idx] = value
        store.set_at_("x", value, idx)
        torch.testing.assert_close(store["x"], expected)
        torch.testing.assert_close(store.get_at("x", idx), expected[idx])
    finally:
        store.clear_redis()
        store.close()


# ---- Connection, pickling and cleanup, on both stores ----

_SMALL_A = torch.arange(6.0).view(2, 3)


def _small_store(store_kwargs, stacked, **kwargs):
    """A store with batch size [2], an entry "a" and a nested entry ("n", "b")."""
    source = TensorDict(
        a=_SMALL_A, n=TensorDict(b=torch.ones(2, 2), batch_size=[2]), batch_size=[2]
    )
    if stacked:
        return LazyStackedTensorDictStore.from_lazy_stack(
            lazy_stack(source.unbind(0)), **store_kwargs, **kwargs
        )
    return TensorDictStore.from_tensordict(source, **store_kwargs, **kwargs)


def _server_keys(store, store_kwargs):
    """The keys of ``store`` on the server, read with another client."""
    import redis

    client = redis.Redis(port=store_kwargs["port"], db=store_kwargs["db"])
    try:
        pattern = f"{store._namespace}:{{{store._td_id}}}:*"
        return sorted(key.decode() for key in client.scan_iter(match=pattern))
    finally:
        client.close()


@pytest.mark.parametrize("stacked", [False, True])
def test_clear_redis_deletes_server_keys(store_kwargs, stacked):
    store = _small_store(store_kwargs, stacked)
    try:
        assert torch.equal(store["a"], _SMALL_A)
        assert _server_keys(store, store_kwargs)
        store.clear_redis()
        assert _server_keys(store, store_kwargs) == []
        assert store._meta_cache == {}
        assert getattr(store, "_keys_cache", [None]) == [None]
    finally:
        store.close()


@pytest.mark.parametrize("stacked", [False, True])
def test_clear_redis_then_write(store_kwargs, stacked):
    """A cleared store writes again, but from_store cannot find it: its metadata is gone."""
    store = _small_store(store_kwargs, stacked)
    cls = type(store)
    try:
        store.clear_redis()
        store["c"] = torch.full((2, 3), 2.0)
        assert list(store.keys()) == ["c"]
        assert torch.equal(store["c"], torch.full((2, 3), 2.0))
        with pytest.raises(KeyError, match=f"No {cls.__name__} with td_id"):
            cls.from_store(td_id=store._td_id, **store_kwargs)
    finally:
        store.clear_redis()
        store.close()


@pytest.mark.parametrize("stacked", [False, True])
def test_reader_sees_new_keys(store_kwargs, stacked):
    """A TensorDictStore caches the key registry; a LazyStackedTensorDictStore reads it each time."""
    writer = _small_store(store_kwargs, stacked)
    reader = type(writer).from_store(td_id=writer._td_id, **store_kwargs)
    try:
        assert sorted(reader.keys()) == ["a", "n"]
        writer["c"] = torch.zeros(2, 3)
        assert sorted(reader.keys()) == (["a", "c", "n"] if stacked else ["a", "n"])
    finally:
        reader.close()
        writer.clear_redis()
        writer.close()


@pytest.mark.parametrize("cache_metadata", [True, False])
@pytest.mark.parametrize("stacked", [False, True])
def test_del_key_removes_server_keys(store_kwargs, stacked, cache_metadata):
    if stacked:
        # "h" has elements of different shapes, so it also has an offset table
        source = lazy_stack(
            [
                TensorDict(a=torch.zeros(3), h=torch.zeros(2)),
                TensorDict(a=torch.ones(3), h=torch.zeros(4)),
            ]
        )
        store = LazyStackedTensorDictStore.from_lazy_stack(
            source, cache_metadata=cache_metadata, **store_kwargs
        )
        entries = {"a": ["d:a", "m:a"], "h": ["d:h", "d:h:idx", "m:h"]}
    else:
        store = _small_store(store_kwargs, stacked, cache_metadata=cache_metadata)
        entries = {"a": ["d:a", "m:a"], ("n", "b"): ["d:n.b", "m:n.b"]}
    try:
        for key, suffixes in entries.items():
            entry_keys = {store._redis_key(suffix) for suffix in suffixes}
            assert entry_keys <= set(_server_keys(store, store_kwargs))
            store.del_(key)
            assert not entry_keys & set(_server_keys(store, store_kwargs))
            assert key not in store.keys(True)
        assert store._run_sync(store._aget_all_keys()) == set()
        assert store._meta_cache == ({} if cache_metadata else None)
    finally:
        store.clear_redis()
        store.close()


@pytest.mark.parametrize("stacked", [False, True])
def test_close_twice_and_nested_close(store_kwargs, stacked):
    store = _small_store(store_kwargs, stacked)
    try:
        if stacked:
            # an element reads through the connection of the stack, and has no close
            assert not hasattr(store[0], "close")
        else:
            nested = store["n"]
            nested.close()  # a nested view does not own the connection
            nested.__del__()
            assert torch.equal(nested["b"], torch.ones(2, 2))
        assert torch.equal(store["a"], _SMALL_A)
        store.clear_redis()
    finally:
        store.close()
    assert store._loop.is_closed()
    store.close()  # closing again is a no-op


@pytest.mark.parametrize("stacked", [False, True])
def test_close_swallows_errors(store_kwargs, stacked, monkeypatch):
    """close() stops the loop when closing the client fails, and __del__ never raises."""
    store = _small_store(store_kwargs, stacked)
    store.clear_redis()
    aclose = store._client.aclose

    async def failing_aclose():
        # close the connection first: the stopped loop cannot close it later
        await aclose()
        raise ConnectionError("connection lost")

    def failing_close():
        raise RuntimeError("close failed")

    monkeypatch.setattr(store._client, "aclose", failing_aclose)
    monkeypatch.setattr(store, "close", failing_close)
    store.__del__()
    monkeypatch.delattr(store, "close")
    store.close()
    assert store._loop.is_closed()
    # the connection is not left for the garbage collector to finalize
    monkeypatch.undo()
    gc.collect()  # garbage of earlier tests is finalized outside the check
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ResourceWarning)
        del store
        gc.collect()
    assert not [w for w in caught if issubclass(w.category, ResourceWarning)]


@pytest.mark.parametrize("stacked", [False, True])
def test_setstate_unix_socket_path(store_kwargs, stacked, tmp_path):
    """An unpickled store connects through its unix socket, not through host and port."""
    store = _small_store(store_kwargs, stacked)
    try:
        state = store.__getstate__()
    finally:
        store.clear_redis()
        store.close()
    state["_unix_socket_path"] = str(tmp_path / "redis.sock")
    copy = type(store).__new__(type(store))
    copy.__setstate__(state)
    try:
        kwargs = copy._client.connection_pool.connection_kwargs
        assert kwargs["path"] == str(tmp_path / "redis.sock")
        assert "port" not in kwargs
    finally:
        copy.close()


@pytest.mark.parametrize("cls", [TensorDictStore, LazyStackedTensorDictStore])
def test_from_store_unix_socket_path(cls, tmp_path):
    """from_store reads the metadata through the unix socket, not through host and port."""
    if not _has_redis:
        pytest.skip("redis package not installed")
    import redis
    from redis.asyncio.retry import Retry
    from redis.backoff import NoBackoff

    with pytest.raises(redis.exceptions.ConnectionError, match="missing.sock"):
        cls.from_store(
            td_id="nonexistent-uuid",
            unix_socket_path=str(tmp_path / "missing.sock"),
            retry=Retry(NoBackoff(), 0),
        )


@pytest.mark.parametrize("copy", [None, "pickle", "clone", "from_store"])
@pytest.mark.parametrize("stacked", [False, True])
def test_client_kwarg(store_kwargs, stacked, copy):
    """A store built with client= uses it, and its copies connect to the same server."""
    import redis.asyncio as aioredis

    def client():
        return aioredis.Redis(port=store_kwargs["port"], db=store_kwargs["db"])

    # no port and no db: the store must take them from the client
    store = _small_store({"backend": store_kwargs["backend"]}, stacked, client=client())
    try:
        assert torch.equal(store["a"], _SMALL_A)
        assert store._redis_key("m:a") in _server_keys(store, store_kwargs)
        if copy is None:
            return
        if copy == "pickle":
            other = pickle.loads(pickle.dumps(store))
        elif copy == "clone":
            other = store.clone()
        else:
            other = type(store).from_store(td_id=store._td_id, client=client())
        try:
            assert torch.equal(other["a"], _SMALL_A)
            assert other._redis_key("m:a") in _server_keys(other, store_kwargs)
        finally:
            if copy == "clone":
                other.clear_redis()
            other.close()
    finally:
        store.clear_redis()
        store.close()


class _ClientWithoutPool:
    """A client that does not expose its server address."""

    def __init__(self, client):
        self._client = client

    def __getattr__(self, name):
        if name == "connection_pool":
            raise AttributeError(name)
        return getattr(self._client, name)


@pytest.mark.parametrize("stacked", [False, True])
def test_client_without_settings(store_kwargs, stacked):
    """A store whose client hides its server works, but cannot be copied."""
    import redis.asyncio as aioredis

    client = aioredis.Redis(port=store_kwargs["port"], db=store_kwargs["db"])
    store = _small_store(
        {"backend": store_kwargs["backend"]},
        stacked,
        client=_ClientWithoutPool(client),
    )
    try:
        assert torch.equal(store["a"], _SMALL_A)
        with pytest.raises(RuntimeError, match="Cannot connect a copy"):
            pickle.dumps(store)
        with pytest.raises(RuntimeError, match="Cannot connect a copy"):
            store.clone()
        if stacked:
            # a shallow copy of a lazy stack opens its own connection too
            with pytest.raises(RuntimeError, match="Cannot connect a copy"):
                store.copy()
    finally:
        store.clear_redis()
        store.close()


def test_client_settings():
    """_client_settings reads the server and the credentials of a client."""
    if not _has_redis:
        pytest.skip("redis package not installed")
    import redis.asyncio as aioredis

    # building a client does not connect, so no server is needed
    client = aioredis.Redis(port=6380, db=3, username="u", password="p")
    assert _client_settings(client) == (
        "localhost",
        6380,
        3,
        None,
        {"username": "u", "password": "p"},
    )
    client = aioredis.Redis(unix_socket_path="/tmp/redis.sock", db=2)
    host, port, db, path, credentials = _client_settings(client)
    assert (db, path, credentials) == (2, "/tmp/redis.sock", {})
    assert _client_settings(None) is None


@pytest.mark.parametrize("transport", ["tcp", "unix", "custom", "opaque"])
def test_client_settings_non_tls(transport):
    """Address and credential reconstruction remains available for non-TLS clients."""
    if not _has_redis:
        pytest.skip("redis package not installed")
    import redis.asyncio as aioredis

    class CustomConnection(aioredis.Connection):
        pass

    if transport == "opaque":
        assert _client_settings(_ClientWithoutPool(aioredis.Redis())) is None
        return
    options = {"db": 2, "username": "u", "password": "p"}
    if transport == "unix":
        options["unix_socket_path"] = "/tmp/redis.sock"
        client = aioredis.Redis(**options)
        expected_class = aioredis.UnixDomainSocketConnection
    elif transport == "custom":
        client = aioredis.Redis(
            connection_pool=aioredis.ConnectionPool(
                connection_class=CustomConnection, **options
            )
        )
        expected_class = aioredis.Connection
    else:
        client = aioredis.Redis(host="store.example", port=6381, **options)
        expected_class = aioredis.Connection
    reconstructed = aioredis.Redis(**_connect_kwargs(*_client_settings(client)))
    pool = reconstructed.connection_pool
    assert pool.connection_class is expected_class
    assert _client_settings(reconstructed) == _client_settings(client)


def test_client_unrepresentable_tls_settings():
    """Unrepresentable TLS options are reported, not silently discarded."""
    if not _has_redis:
        pytest.skip("redis package not installed")
    import redis.asyncio as aioredis

    class CustomSSLConnection(aioredis.SSLConnection):
        def __init__(self, ssl_custom_option=True, **kwargs):
            self.ssl_custom_option = ssl_custom_option
            super().__init__(**kwargs)

    client = aioredis.Redis(
        connection_pool=aioredis.ConnectionPool(
            connection_class=CustomSSLConnection, ssl_custom_option=True
        )
    )
    with pytest.raises(
        RuntimeError, match="Cannot reproduce Redis TLS settings: ssl_custom_option"
    ):
        _client_settings(client)


@pytest.mark.parametrize("cls", [TensorDictStore, LazyStackedTensorDictStore])
@pytest.mark.parametrize("origin", ["constructor", "url", "subclass"])
@pytest.mark.parametrize("configured", [False, True])
def test_client_tls_settings(cls, origin, configured, monkeypatch):
    """Copies and metadata readers retain the supplied client's TLS settings."""
    if not _has_redis:
        pytest.skip("redis package not installed")
    import redis.asyncio as aioredis

    options = {}
    if configured:
        options = {
            "ssl_keyfile": "client.key",
            "ssl_certfile": "client.crt",
            "ssl_cert_reqs": "required",
            "ssl_ca_certs": "ca.crt",
            "ssl_ca_data": "certificate data",
            "ssl_ca_path": "certificates",
            "ssl_check_hostname": False,
            "ssl_min_version": ssl.TLSVersion.TLSv1_2,
            "ssl_max_version": ssl.TLSVersion.TLSv1_3,
            "ssl_ciphers": "ECDHE+AESGCM",
            "ssl_password": "key-password",
            "ssl_include_verify_flags": [ssl.VERIFY_X509_STRICT],
            "ssl_exclude_verify_flags": [ssl.VERIFY_X509_TRUSTED_FIRST],
        }
        # Exercise the options supplied by the installed redis-py, including
        # older releases that predate protocol bounds and verification flags.
        parameters = inspect.signature(aioredis.Redis).parameters
        options = {key: value for key, value in options.items() if key in parameters}
    address = {
        "host": "store.example",
        "port": 6381,
        "db": 3,
        "username": "u",
        "password": "p",
    }
    if origin == "constructor":
        client = aioredis.Redis(ssl=True, **address, **options)
    elif origin == "url":
        client = aioredis.Redis.from_url("rediss://u:p@store.example:6381/3", **options)
    else:

        class CustomSSLConnection(aioredis.SSLConnection):
            pass

        client = aioredis.Redis(
            connection_pool=aioredis.ConnectionPool(
                connection_class=CustomSSLConnection, **address, **options
            )
        )
    expected = {
        name: options.get(name, parameter.default)
        for name, parameter in inspect.signature(
            aioredis.SSLConnection
        ).parameters.items()
        if name.startswith("ssl_")
    }
    clients = []
    stacked = cls is LazyStackedTensorDictStore
    metadata = (
        [b"lazy_stack", b"2", b"0", b"[]", b""] if stacked else [b"[]", b"", None]
    )

    def pipeline(self, *args, **kwargs):
        clients.append(self)
        return mock.Mock(execute=mock.AsyncMock(return_value=metadata))

    monkeypatch.setattr(aioredis.Redis, "pipeline", pipeline)
    monkeypatch.setattr(cls, "to_tensordict", lambda self: {})
    store = cls(
        client=client,
        **({"count": 2, "inner_batch_size": []} if stacked else {}),
        # As with host and credentials, the supplied client takes precedence.
        ssl=False,
        ssl_cert_reqs="none",
        ssl_check_hostname=not expected["ssl_check_hostname"],
        username="ignored",
        password="ignored",
    )
    copies = []
    try:
        copies.append(pickle.loads(pickle.dumps(store)))
        copies.append(store.clone())
        copies.append(store.clone(False))
        copies.append(cls.from_store(td_id=store._td_id, client=client, ssl=False))
        # Includes the temporary client used by _read_stored_metadata.
        for reconstructed in [*(copy._client for copy in copies), *clients]:
            pool = reconstructed.connection_pool
            assert issubclass(pool.connection_class, aioredis.SSLConnection)
            for name, value in {**address, **expected}.items():
                if reconstructed is client and name not in pool.connection_kwargs:
                    continue  # URL/pool clients can leave SSL defaults implicit.
                assert pool.connection_kwargs[name] == value
    finally:
        for copy in copies:
            copy.close()
        store.close()


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

    def test_keys_values_items_order(self, store_kwargs):
        """keys(), values() and items() follow the sorted order of the key paths."""
        td = TensorDict(
            {
                "x": torch.zeros(3, 2),
                "R": torch.ones(3, 2),
                "n": {"b": torch.full((3, 2), 2.0), "a": torch.full((3, 2), 3.0)},
                "n-x": torch.full((3, 2), 4.0),
            },
            batch_size=[3, 2],
        )
        store_td = LazyStackedTensorDictStore.from_lazy_stack(
            lazy_stack(list(td.unbind(0))), **store_kwargs
        )
        try:
            # "n-x" sorts before "n.a"
            keys = ["R", "n-x", ("n", "a"), ("n", "b"), "x"]
            # the store and the view of one of its elements
            for store, expected in ((store_td, td), (store_td[0], td[0])):
                assert list(store.keys(True, True)) == keys
                assert [key for key, _ in store.items(True, True)] == keys
                for key, value in zip(keys, store.values(True, True), strict=True):
                    assert (value == expected[key]).all()
                assert list(store.keys()) == ["R", "n-x", "n", "x"]
                assert list(store["n"].keys()) == ["a", "b"]
                assert list(store.to_tensordict().keys(True, True)) == keys
        finally:
            store_td.clear_redis()
            store_td.close()

    def test_keys_include_nested(self, store_kwargs):
        """keys(include_nested=True) yields the sub-tensordict keys, as TensorDict does."""
        td = TensorDict(
            {
                "a": torch.zeros(3, 2),
                "n": {"b": torch.ones(3, 2), "m": {"c": torch.ones(3, 2)}},
            },
            batch_size=[3, 2],
        )
        store_td = LazyStackedTensorDictStore.from_lazy_stack(
            lazy_stack(list(td.unbind(0))), **store_kwargs
        )
        try:
            keys = ["a", ("n", "b"), ("n", "m", "c"), ("n", "m"), "n"]
            # the store and the view of one of its elements
            for store in (store_td, store_td[0]):
                assert list(store.keys(include_nested=True)) == keys
                assert len(store.keys(include_nested=True)) == 5
                assert list(store.to_tensordict().keys(include_nested=True)) == keys
                # leaves_only is unchanged
                assert list(store.keys(True, True)) == [
                    "a",
                    ("n", "b"),
                    ("n", "m", "c"),
                ]
        finally:
            store_td.clear_redis()
            store_td.close()

    def test_contains_nested_key_without_include_nested(self, store_stack):
        # As on a lazy stack, a nested key in a view without nesting is found,
        # with a FutureWarning: it returns False in 0.17 (#2060).
        store_td, _, _ = store_stack
        store_td["nested", "val"] = torch.zeros(5, 4)
        # the store and the view of one of its elements
        for store in (store_td, store_td[0]):
            with warnings.catch_warnings():
                warnings.simplefilter("error", FutureWarning)
                assert "nested" in store.keys()
                assert ("nested", "val") in store.keys(include_nested=True)
                assert ("nested", "x") not in store.keys()
            with pytest.warns(FutureWarning, match="returns False"):
                assert ("nested", "val") in store.keys()

    def test_repr(self, store_stack):
        store_td, _, _ = store_stack
        r = repr(store_td)
        assert "LazyStackedTensorDictStore" in r
        assert "count=5" in r

    def test_repr_and_to_tensordict_do_not_read_leaves(self, store_stack, monkeypatch):
        store_td, tds, _ = store_stack
        element = store_td[1]

        def no_read(self, *args, **kwargs):
            raise AssertionError("a leaf was read")

        monkeypatch.setattr(LazyStackedTensorDictStore, "_get_str", no_read)
        monkeypatch.setattr(_StoreStackElementView, "_get_str", no_read)
        assert repr(store_td) == (
            "LazyStackedTensorDictStore(\n"
            "    keys=['a', 'b'],\n"
            "    batch_size=torch.Size([5, 4]),\n"
            "    count=5,\n"
            "    stack_dim=0,\n"
            "    device=None,\n"
            f"    backend={store_td._backend!r},\n"
            f"    td_id={store_td._td_id!r})"
        )
        assert repr(element) == (
            "_StoreStackElementView(\n"
            f"    parent_td_id={store_td._td_id!r},\n"
            "    element_idx=1,\n"
            "    keys=['a', 'b'],\n"
            "    batch_size=torch.Size([4]),\n"
            "    device=None)"
        )
        local = element.to_tensordict()
        monkeypatch.undo()
        assert type(local) is TensorDict
        assert local.batch_size == (4,)
        assert (local == tds[1]).all()

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

    def test_close_releases_event_loop(self, store_kwargs):
        """close() closes the event loop, which releases the sockets that it holds."""
        store = LazyStackedTensorDictStore.from_lazy_stack(
            lazy_stack([TensorDict({"a": torch.zeros(2)}, [2])]), **store_kwargs
        )
        loop = store._loop
        store.clear_redis()
        store.close()
        assert loop.is_closed()
        store.close()  # closing again is a no-op

    # ---- Read: td[list_of_bools] ----

    def test_getitem_bool_list(self, store_stack):
        store_td, tds, lazy_td = store_stack
        mask = [True, False, False, True, True]
        sub = store_td[mask]
        assert sub.batch_size == torch.Size([3, 4])
        torch.testing.assert_close(sub["a"], lazy_td[torch.tensor(mask)]["a"])

    def test_get_at_set_at_bool_list(self, store_stack):
        store_td, tds, lazy_td = store_stack
        mask = [True, False, False, True, True]
        expected = lazy_td["a"]
        torch.testing.assert_close(store_td.get_at("a", mask), expected[mask])
        store_td.set_at_("a", torch.zeros(3, 4, 3), mask)
        expected[mask] = 0
        torch.testing.assert_close(store_td["a"], expected)

    @pytest.mark.parametrize("index", [1, [0, 2], (slice(None), [1, 3])])
    def test_set_at_python_scalar(self, store_stack, index):
        store_td, tds, lazy_td = store_stack
        expected = lazy_td["a"]
        store_td.set_at_("a", -2, index)
        expected[index] = -2
        torch.testing.assert_close(store_td["a"], expected)

    @pytest.mark.parametrize(
        "index",
        [3, slice(1, 4), [0, 4], (slice(None), 1), (..., 1), torch.arange(5) > 2],
    )
    def test_set_at_update_at_nested(self, store_kwargs, index):
        """set_at_ and update_at_ write each leaf of a tensordict value at the
        index, as on a lazy stack."""
        td = TensorDict(
            a=torch.zeros(5, 4),
            n=TensorDict(
                b=torch.zeros(5, 4),
                m=TensorDict(
                    c=torch.zeros(5, 4, 2, dtype=torch.long), batch_size=[5, 4]
                ),
                batch_size=[5, 4],
            ),
            batch_size=[5, 4],
        )
        lazy_td = lazy_stack(list(td.unbind(0)))
        store_td = LazyStackedTensorDictStore.from_lazy_stack(lazy_td, **store_kwargs)
        try:
            value = td[index] + 1
            for target in (store_td, lazy_td):
                target.update_at_(value, index)
                target.set_at_("n", value["n"] + 1, index)
                target.set_at_(("n", "m"), value["n", "m"] + 2, index)

            def check():
                for key in lazy_td.keys(True, True):
                    torch.testing.assert_close(store_td[key], lazy_td[key])

            check()
            # a leaf that the store lacks raises before anything is written
            missing = (value["n"] + 3).set("z", value["n", "b"])
            with pytest.raises(KeyError):
                store_td.set_at_("n", missing, index)
            check()
            assert "z" not in store_td["n"].keys()
        finally:
            store_td.clear_redis()
            store_td.close()

    @pytest.mark.parametrize("index", [1, [0, 2], (slice(None), [1, 3])])
    def test_setitem_scalar(self, store_stack, index):
        store_td, tds, lazy_td = store_stack
        store_td[index] = -2
        lazy_td[index] = -2
        torch.testing.assert_close(store_td["a"], lazy_td["a"])
        torch.testing.assert_close(store_td["b"], lazy_td["b"])
        # an element of the stack takes a scalar too
        element = store_td[3]
        element[1] = 5
        lazy_td[3, 1] = 5
        torch.testing.assert_close(store_td["a"], lazy_td["a"])
        with pytest.raises(TypeError, match="not a tensor"):
            store_td[index] = torch.ones(3)

    def test_nd_mask(self, store_stack):
        """A 2-D mask over the stack dim and an inner dim, as on the lazy stack."""
        store_td, tds, lazy_td = store_stack
        mask = torch.tensor([[True, False, True, False]] * 5)
        expected = lazy_td["a"]
        torch.testing.assert_close(store_td[mask]["a"], expected[mask])
        torch.testing.assert_close(store_td.get_at("a", mask), expected[mask])
        store_td.set_at_("a", torch.zeros(10, 3), mask)
        expected[mask] = 0
        torch.testing.assert_close(store_td["a"], expected)

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

    @pytest.mark.parametrize(
        "idx",
        [
            (0, slice(None)),
            (slice(1, 3), 0),
            (1, 2),
            (torch.tensor([0, 2]), 1),
            (slice(1, 3), ...),
            (..., 1),
            None,
        ],
    )
    def test_getitem_multidim_index(self, store_stack, idx):
        # These indices are not handled by the byte-range reads.
        store_td, tds, lazy_td = store_stack
        torch.testing.assert_close(store_td.get_at("a", idx), lazy_td.get_at("a", idx))
        sub = store_td[idx]
        expected = lazy_td[idx]
        assert sub.batch_size == expected.batch_size
        torch.testing.assert_close(sub["a"], expected["a"])
        torch.testing.assert_close(sub["b"], expected["b"])

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

    def test_lstore_setitem_dict(self, store_stack):
        # a dict value at an element, at a slice and on an element view
        store_td, tds, lazy_td = store_stack
        for target, index, value in (
            (store_td, 1, {"a": torch.ones(4, 3), "b": torch.ones(4)}),
            (store_td, slice(2, 4), {"a": torch.zeros(4, 3), "b": torch.zeros(4)}),
            (store_td[4], slice(0, 2), {"a": torch.full((3,), 5.0)}),
        ):
            target[index] = value
        lazy_td[1] = TensorDict(a=torch.ones(4, 3), b=torch.ones(4), batch_size=[4])
        lazy_td[2:4] = 0
        lazy_td[4, 0:2] = TensorDict(a=torch.full((2, 3), 5.0), batch_size=[2])
        assert (store_td.to_tensordict() == lazy_td.to_tensordict()).all()

    @pytest.mark.parametrize("as_dict", [False, True])
    def test_lstore_setitem_partial_element(self, store_stack, as_dict):
        store_td, tds, lazy_td = store_stack
        value = TensorDict(a=torch.ones(4, 3), batch_size=[4])
        store_td[0] = value.to_dict() if as_dict else value
        lazy_td[0] = value
        assert (store_td.to_tensordict() == lazy_td.to_tensordict()).all()
        # a key that the store lacks raises before anything is written, as at
        # a slice
        for value in (
            TensorDict(zz=torch.ones(4), batch_size=[4]),
            TensorDict(
                a=torch.ones(4, 3), b=torch.ones(4), zz=torch.ones(4), batch_size=[4]
            ),
        ):
            with pytest.raises(
                KeyError, match="key zz not found in LazyStackedTensorDictStore"
            ):
                store_td[1] = value.to_dict() if as_dict else value
            with pytest.raises(
                KeyError, match="key zz not found in LazyStackedTensorDictStore"
            ):
                store_td[1:2] = value.unsqueeze(0)
        assert set(store_td.keys()) == {"a", "b"}
        assert (store_td.to_tensordict() == lazy_td.to_tensordict()).all()

    def test_lstore_setitem_self_nesting(self, store_stack):
        store_td, tds, lazy_td = store_stack
        for td in (store_td, store_td[0]):
            with pytest.raises(
                ValueError,
                match=r"Cannot set a tensordict inside itself \(key 'me'\)",
            ):
                td["me"] = td
        assert set(store_td.keys()) == {"a", "b"}

    def test_lstore_empty_skeleton(self, store_kwargs, monkeypatch):
        # empty() builds a TensorDict from the key registry, without reading data
        tds = [
            TensorDict(
                a=torch.zeros(3),
                n=TensorDict(x=torch.zeros(3, 2), batch_size=[3, 2]),
                batch_size=[3],
            )
            for _ in range(2)
        ]
        store_td = LazyStackedTensorDictStore.from_lazy_stack(
            lazy_stack(tds), **store_kwargs
        )
        try:
            for td in (store_td, store_td[1]):
                calls = _count_round_trips(td, monkeypatch)
                empty = td.empty()
                assert type(empty) is TensorDict
                assert empty.batch_size == td.batch_size
                assert list(empty.keys()) == []
                empty = td.empty(device="meta")
                assert empty.device == torch.device("meta")
                assert calls == []
                empty = td.empty(
                    recurse=True, batch_size=td.batch_size[:1], names=["z"]
                )
                assert list(empty.keys(True)) == ["n"]
                assert empty.names == ["z"]
                # the sub-tensordicts take the batch size of the result
                assert empty["n"].batch_size == td.batch_size[:1]
                # at most one read of the key registry
                assert len(calls) <= 1
                monkeypatch.undo()
        finally:
            store_td.clear_redis()
            store_td.close()

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
        with pytest.warns(DeprecationWarning, match="to_local"):
            local = store_td.to_local()
        assert isinstance(local, TensorDict)
        assert local.batch_size == torch.Size([5, 4])

    def test_memmap(self, store_stack, tmp_path):
        """memmap and save write a memory-mapped copy; only memmap_ raises."""
        store_td, tds, lazy_td = store_stack
        source = lazy_td.to_tensordict()
        for out in (
            store_td.memmap(tmp_path / "memmap"),
            store_td.save(tmp_path / "save"),
            store_td.memmap(),
        ):
            assert type(out) is TensorDict
            assert out.is_memmap() and out.is_locked
            assert (out == source).all()
        assert (TensorDict.load_memmap(tmp_path / "save") == source).all()
        elem = store_td[1].memmap(tmp_path / "elem")
        assert elem.is_memmap()
        assert (elem == tds[1]).all()
        with pytest.raises(RuntimeError, match="in-place"):
            store_td.memmap_(tmp_path / "memmap_")
        with pytest.raises(RuntimeError, match="in-place"):
            store_td[1].memmap_(tmp_path / "elem_")

    def test_vmap_raises(self, store_stack):
        """vmap raises, also on a function that returns the store or an element."""
        store_td, tds, lazy_td = store_stack
        for td in (store_td, store_td[0]):
            with pytest.raises(RuntimeError, match="cannot be used with vmap"):
                torch.vmap(lambda x: x)(td)
            # this used to return None
            with pytest.raises(RuntimeError, match="cannot be used with vmap"):
                torch.vmap(lambda x, td=td: td)(torch.zeros(2))

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

    def test_heterogeneous_key_stack_level(self, store_kwargs):
        """A key whose elements differ in shape is not read or written as one tensor (#2030).

        As on a lazy stack, stack-level reads, set_ and fill_ raise, while
        zero_, the other in-place methods and the reads and writes of one
        element use the shape of each element. Before, they used the shape
        of element 0 for every element.
        """
        tds = [
            TensorDict(
                a=torch.arange(12.0 * (i + 1)).view(3, 4 * (i + 1)) - 1,
                b=torch.ones(3),
                n=TensorDict(
                    c=torch.zeros(3, 2), h=torch.ones(3, i + 1), batch_size=[3]
                ),
                batch_size=[3],
            )
            for i in range(2)
        ]
        lazy_td = lazy_stack([td.clone() for td in tds])
        store_td = LazyStackedTensorDictStore.from_lazy_stack(
            lazy_stack(tds), **store_kwargs
        )
        try:
            for read in (
                lambda: store_td["a"],
                store_td.to_tensordict,
                lambda: store_td.fill_("a", 1.0),
                lambda: store_td.set_("a", torch.ones(2, 3, 4)),
                lambda: store_td.update_(a=torch.ones(2, 3, 4)),
            ):
                with pytest.raises(RuntimeError, match="differ in shape"):
                    read()
            torch.testing.assert_close(store_td[1, 0]["a"], tds[1]["a"][0])
            torch.testing.assert_close(store_td.get_at("a", 1), tds[1]["a"])
            torch.testing.assert_close(store_td[1:2]["a"], lazy_td[1:2]["a"])
            assert store_td.get(("n", "zz"), None) is None
            assert store_td.get(("a", "x"), None) is None
            with pytest.raises(ValueError, match="not a tensordict"):
                store_td.get(("a", "x"))
            for td in (store_td, lazy_td):
                td[1, 0] = 7.0
                td[0:1] = 3.0
                td["b"] = torch.full((2, 3), 2.0)
                td.abs_()
                td.mul_(2)
                td.mul_(torch.tensor(2.0))
                td.update_(
                    {
                        "b": torch.zeros(2, 3),
                        "n": {"c": torch.ones(2, 3, 2)},
                        "extra": torch.ones(2, 3),
                    }
                )
                td.update_(
                    {"n": {"c": torch.full((2, 3, 2), 2.0)}},
                    keys_to_update=[("n", "c")],
                )
            for i in range(2):
                torch.testing.assert_close(store_td[i]["a"], lazy_td[i]["a"])
                torch.testing.assert_close(store_td[i]["b"], lazy_td[i]["b"])
                assert (store_td[i]["n"] == lazy_td[i]["n"]).all()
            assert store_td.zero_() is store_td
            for i in range(2):
                assert store_td[i]["a"].shape == tds[i]["a"].shape
                assert (store_td[i]["a"] == 0).all()
        finally:
            store_td.clear_redis()
            store_td.close()

    @staticmethod
    def _layout_store(store_kwargs, lazy_td, written, legacy):
        """A store of ``lazy_td`` written by ``from_lazy_stack`` or by ``set``.

        With ``legacy``, the ``layout`` field is removed, as in the metadata
        that versions before #2030 wrote; their data bytes are the same.
        """
        if written == "upload":
            store_td = LazyStackedTensorDictStore.from_lazy_stack(
                lazy_td.clone(), **store_kwargs
            )
        else:
            store_td = LazyStackedTensorDictStore(
                count=len(lazy_td.tensordicts),
                stack_dim=lazy_td.stack_dim,
                inner_batch_size=lazy_td.tensordicts[0].batch_size,
                **store_kwargs,
            )
            for key in lazy_td.keys(True, True):
                store_td.set(key, lazy_td.get(key).clone())
        if legacy:
            for kp in store_td._run_sync(store_td._aget_all_keys()):
                store_td._run_sync(
                    store_td._client.hdel(store_td._meta_key(kp), "layout")
                )
        store_td.close()
        return LazyStackedTensorDictStore.from_store(
            td_id=store_td._td_id, **store_kwargs
        )

    @staticmethod
    def _layout_lazy_td(stack_dim, count, inner):
        n = torch.Size(inner).numel()
        return lazy_stack(
            [
                TensorDict(
                    a=torch.arange(float(n)).view(inner) + 100 * i,
                    n=TensorDict(
                        b=torch.arange(2 * n).view(*inner, 2) + 100 * i,
                        batch_size=inner,
                    ),
                    batch_size=inner,
                )
                for i in range(count)
            ],
            stack_dim,
        )

    @pytest.mark.parametrize("legacy", [False, True])
    @pytest.mark.parametrize("written", ["upload", "set"])
    @pytest.mark.parametrize("stack_dim", [1, 2])
    def test_stack_dim_nonzero(self, store_kwargs, stack_dim, written, legacy):
        """A store stacked along a dim other than 0 reads and writes as the lazy stack (#2030).

        from_lazy_stack stores an entry stack dim first and set in the batch
        layout. Before, the indexed reads and writes took every entry to be
        in the batch layout, and the element views every entry to be stack
        dim first. Each entry is now read in its own layout, with or without
        the layout field.
        """
        lazy_td = self._layout_lazy_td(stack_dim, 2, [3, 4][:stack_dim])
        store_td = self._layout_store(store_kwargs, lazy_td, written, legacy)
        try:
            assert store_td.batch_size == lazy_td.batch_size
            for read in (
                lambda td: td["a"],
                lambda td: td["n", "b"],
                lambda td: td.to_tensordict()["a"],
                lambda td: td[0]["a"],
                lambda td: td[:, 1]["n", "b"],
                lambda td: td[1:, ::2]["a"],
                lambda td: td.get_at("a", (0, 1)),
            ):
                torch.testing.assert_close(read(store_td), read(lazy_td))
            view = _StoreStackElementView(store_td, 1)
            assert (view.to_tensordict() == lazy_td.tensordicts[1]).all()
            mask = torch.zeros(lazy_td.batch_size, dtype=torch.bool)
            mask[0, 1] = True
            for td, element in (
                (store_td, view),
                (lazy_td, lazy_td.tensordicts[1]),
            ):
                td.set_at_("a", -1.0, (1, 1))
                td[2, 0] = 5.0
                td.masked_fill_(mask, -2.0)
                element.fill_(("n", "b"), -3)
            reader = LazyStackedTensorDictStore.from_store(
                td_id=store_td._td_id, **store_kwargs
            )
            try:
                assert (reader.to_tensordict() == lazy_td).all()
            finally:
                reader.close()
        finally:
            store_td.clear_redis()
            store_td.close()

    @pytest.mark.parametrize("stack_dim", [1, 2])
    def test_stack_dim_nonzero_round_trips(self, store_kwargs, stack_dim, monkeypatch):
        """A write that gives the same bytes in both layouts takes as many round trips on either (#2030).

        A scalar written over the dims up to the stack dim, or a write at one
        index repeated on each of those dims, costs no more on an entry
        stored stack dim first than on one stored in the batch layout.
        """
        import redis.asyncio as aioredis

        calls = []
        execute = aioredis.client.Pipeline.execute
        execute_command = aioredis.Redis.execute_command

        async def counting_execute(pipe, *args, **kwargs):
            if pipe.command_stack:
                calls.append(None)
            return await execute(pipe, *args, **kwargs)

        async def counting_execute_command(client, *args, **kwargs):
            if not isinstance(client, aioredis.client.Pipeline):
                calls.append(None)
            return await execute_command(client, *args, **kwargs)

        monkeypatch.setattr(aioredis.client.Pipeline, "execute", counting_execute)
        monkeypatch.setattr(aioredis.Redis, "execute_command", counting_execute_command)
        lazy_td = self._layout_lazy_td(stack_dim, 2, [3, 4][:stack_dim])
        element = torch.tensor([6, 5])
        writes = (
            lambda td: td.__setitem__(slice(None), 7.0),
            lambda td: td.set_at_("a", 8.0, slice(None)),
            lambda td: td.set_at_("a", 9.0, (1, 1)),
            lambda td: td.set_at_(("n", "b"), element, (1,) * (stack_dim + 1)),
        )
        counts = {}
        for written in ("upload", "set"):
            store_td = self._layout_store(store_kwargs, lazy_td, written, False)
            try:
                store_td["a"]
                counts[written] = []
                for write in writes:
                    calls.clear()
                    write(store_td)
                    counts[written].append(len(calls))
                expected = lazy_td.clone()
                for write in writes:
                    write(expected)
                assert (store_td.to_tensordict() == expected).all()
            finally:
                store_td.clear_redis()
                store_td.close()
        assert counts["upload"] == counts["set"]

    @pytest.mark.parametrize(
        "index",
        [(..., 1), (0, ..., 1), (1, ...), (...,)],
        ids=["front", "middle", "end", "only"],
    )
    @pytest.mark.parametrize("written", ["upload", "set"])
    @pytest.mark.parametrize("stack_dim", [0, 1])
    def test_setitem_ellipsis(self, store_kwargs, stack_dim, written, index):
        """An Ellipsis in an index written to the store stands for the batch dims, as on the lazy stack (#2599).

        Before, the store applied the index to the full shape of each entry,
        so store[..., 1] = 7.0 wrote column 1 of the feature dim of ("n", "b")
        and a tensordict value raised.
        """
        lazy_td = self._layout_lazy_td(stack_dim, 2, [3, 4])
        store_td = self._layout_store(store_kwargs, lazy_td, written, False)
        try:
            value = lazy_td[index].to_tensordict().apply(lambda x: -x)
            for td, element in (
                (store_td, _StoreStackElementView(store_td, 1)),
                (lazy_td, lazy_td.tensordicts[1]),
            ):
                td[index] = 7.0
                element[index] = 8.0
                td[index] = value.exclude("a")
            assert (store_td.to_tensordict() == lazy_td).all()
            assert (store_td[index] == lazy_td[index]).all()
        finally:
            store_td.clear_redis()
            store_td.close()

    @pytest.mark.parametrize("written", ["upload", "set"])
    def test_setitem_ellipsis_0d_elements(self, store_kwargs, written):
        """An Ellipsis written to an element of empty batch size selects its 0-d entries whole (#2599)."""
        lazy_td = self._layout_lazy_td(0, 2, [])
        store_td = self._layout_store(store_kwargs, lazy_td, written, False)
        try:
            for td, element in (
                (store_td, _StoreStackElementView(store_td, 1)),
                (lazy_td, lazy_td.tensordicts[1]),
            ):
                td[(...,)] = 7.0
                element[(...,)] = 8.0
            assert (store_td.to_tensordict() == lazy_td).all()
        finally:
            store_td.clear_redis()
            store_td.close()

    @pytest.mark.parametrize("legacy", [False, True])
    @pytest.mark.parametrize("written", ["upload", "set"])
    def test_stack_dim_nonzero_square(self, store_kwargs, written, legacy):
        """The layout of an entry whose shape fits both layouts comes from the layout field (#2030).

        Without the field, as written before #2030, the indexed reads take
        the entry to be in the batch layout and the element views stack dim
        first, as they did then.
        """
        lazy_td = self._layout_lazy_td(1, 3, [3])
        store_td = self._layout_store(store_kwargs, lazy_td, written, legacy)
        try:
            # the data as written: stack dim first, or in the batch layout
            stored = lazy_td["a"].movedim(1, 0) if written == "upload" else lazy_td["a"]
            indexed = stored if legacy else lazy_td["a"]
            torch.testing.assert_close(store_td["a"], indexed)
            torch.testing.assert_close(store_td[:, 1]["a"], indexed[:, 1])
            torch.testing.assert_close(store_td.get_at("a", (0, 1)), indexed[0, 1])
            element = stored[1] if legacy else lazy_td.tensordicts[1]["a"]
            torch.testing.assert_close(
                _StoreStackElementView(store_td, 1)["a"], element
            )
        finally:
            store_td.clear_redis()
            store_td.close()

    @pytest.mark.parametrize(
        "index", [slice(0, 2), slice(None, None, -1), (slice(None), 1)]
    )
    def test_index_several_elements_heterogeneous(self, store_kwargs, index):
        """An index that picks several elements of a store whose elements differ in shape (#2030).

        As on a lazy stack, it gives the lazy stack of the elements, each
        with its own shapes. Before, it raised.
        """
        tds = [
            TensorDict(
                a=torch.arange(12.0 * (i + 1)).view(3, 4 * (i + 1)),
                b=torch.arange(3.0) + 10 * i,
                batch_size=[3],
            )
            for i in range(2)
        ]
        lazy_td = lazy_stack([td.clone() for td in tds])
        store_td = LazyStackedTensorDictStore.from_lazy_stack(
            lazy_stack(tds), **store_kwargs
        )
        try:
            result = store_td[index]
            expected = lazy_td[index]
            assert isinstance(result, LazyStackedTensorDict)
            assert result.batch_size == expected.batch_size
            for got, want in zip(result.tensordicts, expected.tensordicts, strict=True):
                assert (got == want).all()
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

    def test_pickle_locked_and_names(self, store_stack):
        store_td, tds, lazy_td = store_stack
        store_td.names = ["stack", "inner"]
        store_td.lock_()
        try:
            restored = pickle.loads(pickle.dumps(store_td))
            try:
                assert restored.is_locked
                assert restored.names == ["stack", "inner"]
                assert restored.batch_size == torch.Size([5, 4])
                assert torch.equal(restored[1]["b"], tds[1]["b"])
            finally:
                restored.close()
        finally:
            store_td.unlock_()

    def test_from_store_not_found(self, store_kwargs):
        with pytest.raises(
            KeyError,
            match=r"No LazyStackedTensorDictStore with td_id='nonexistent-uuid' found\.",
        ):
            LazyStackedTensorDictStore.from_store(
                td_id="nonexistent-uuid", **store_kwargs
            )

    def test_get_at_missing_key(self, store_kwargs):
        store = _small_store(store_kwargs, stacked=True)
        try:
            assert store.get_at("zz", 0, None) is None
            assert store.get_at(("n", "zz"), 0, None) is None
            assert torch.equal(store.get_at("n", 1)["b"], torch.ones(2))
            assert torch.equal(store.get_at(("n", "b"), 1), torch.ones(2))
            with pytest.raises(ValueError, match="'a' is a Tensor, not a tensordict"):
                store.get_at(("a", "x"), 0)
        finally:
            store.clear_redis()
            store.close()

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

    def test_from_store_cache_metadata(self, store_stack, store_kwargs):
        """from_store passes cache_metadata to the store, not to the Redis client."""
        store_td, tds, lazy_td = store_stack
        restored = LazyStackedTensorDictStore.from_store(
            td_id=store_td._td_id, cache_metadata=False, **store_kwargs
        )
        try:
            assert restored._meta_cache is None
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

    # ---- rename_key_ ----

    def test_rename_key(self, store_stack):
        store_td, tds, lazy_td = store_stack
        # the elements of "a" have one shape, so "a" has no offset table
        store_td.rename_key_("a", "c")
        assert set(store_td.keys()) == {"b", "c"}
        torch.testing.assert_close(store_td["c"], lazy_td["a"])
        # replaces "b", whose shape differs
        store_td.rename_key_("c", "b")
        assert set(store_td.keys()) == {"b"}
        torch.testing.assert_close(store_td["b"], lazy_td["a"])

    def test_rename_key_locked_warns(self, store_stack):
        # a locked store renamed its keys without a check: it still does, with
        # a warning, until 0.17
        store_td, tds, lazy_td = store_stack
        store_td.lock_()
        with pytest.warns(
            FutureWarning,
            match=r"^LazyStackedTensorDictStore\.rename_key_ renames keys of a "
            r"locked tensordict today\. Starting with TensorDict 0\.17",
        ) as record:
            assert store_td.rename_key_("a", "c") is store_td
        # count only FutureWarnings: under xdist on free-threaded Python, the
        # record once held 12 other warnings from code outside the call
        warned = [w for w in record if issubclass(w.category, FutureWarning)]
        assert len(warned) == 1, [str(w.message) for w in record]
        assert warned[0].filename == __file__
        assert set(store_td.keys()) == {"b", "c"}

    def test_rename_key_leaf_kinds(self, store_kwargs):
        tds = [
            TensorDict(
                {
                    "het": torch.randn(3, i + 1),
                    "empty": torch.randn(3, 0),
                    "nested": {"x": torch.randn(3, 2)},
                },
                batch_size=[3],
            )
            for i in range(4)
        ]
        store_td = LazyStackedTensorDictStore.from_lazy_stack(
            lazy_stack(tds), **store_kwargs
        )
        try:
            # "het" has an offset table, as its elements have different
            # shapes, "empty" has no data, and ("nested", "x") is a leaf of
            # one shape in a sub-tensordict
            store_td.rename_key_("het", "het2")
            store_td.rename_key_("empty", "empty2")
            store_td.rename_key_(("nested", "x"), ("nested", "y"))
            assert set(store_td.keys(True, True)) == {
                "het2",
                "empty2",
                ("nested", "y"),
            }
            for i, td in enumerate(tds):
                torch.testing.assert_close(store_td[i]["het2"], td["het"])
            assert store_td["empty2"].shape == (4, 3, 0)
            torch.testing.assert_close(
                store_td["nested", "y"], torch.stack([td["nested", "x"] for td in tds])
            )
        finally:
            store_td.clear_redis()
            store_td.close()

    def test_rename_key_sub_tensordict(self, store_kwargs):
        tds = [
            TensorDict(
                {
                    "nested": {
                        "het": torch.randn(3, i + 1),
                        "d": {"y": torch.randn(3)},
                    },
                    "x": torch.randn(3, 2),
                },
                batch_size=[3],
            )
            for i in range(4)
        ]
        store_td = LazyStackedTensorDictStore.from_lazy_stack(
            lazy_stack(tds), **store_kwargs
        )
        try:
            store_td.rename_key_("nested", ("n", "m"))
            keys = {("n", "m", "het"), ("n", "m", "d", "y"), "x"}
            assert set(store_td.keys(True, True)) == keys
            for i, td in enumerate(tds):
                torch.testing.assert_close(
                    store_td[i]["n", "m", "het"], td["nested", "het"]
                )
                torch.testing.assert_close(
                    store_td[i]["n", "m", "d", "y"], td["nested", "d", "y"]
                )
                torch.testing.assert_close(store_td[i]["x"], td["x"])
        finally:
            store_td.clear_redis()
            store_td.close()

    @pytest.mark.parametrize("old_key,new_key", _RENAME_OCCUPIED)
    def test_rename_key_occupied_destination(self, store_kwargs, old_key, new_key):
        # the entry replaces what is at the destination, as in TensorDict: the
        # store was left unchanged, or with a leaf that was also a
        # sub-tensordict, and could not be read
        expected = _rename_source().rename_key_(old_key, new_key)
        store_td = LazyStackedTensorDictStore.from_lazy_stack(
            lazy_stack(_rename_source().unbind(0)), **store_kwargs
        )
        try:
            store_td.rename_key_(old_key, new_key)
            _assert_store_matches(store_td, expected)
        finally:
            store_td.clear_redis()
            store_td.close()

    @pytest.mark.parametrize("old_key,new_key,safe,match", _RENAME_INVALID)
    def test_rename_key_invalid_destination(
        self, store_kwargs, old_key, new_key, safe, match
    ):
        # the store moved a leaf below a leaf, and was left unchanged for a
        # sub-tensordict
        store_td = LazyStackedTensorDictStore.from_lazy_stack(
            lazy_stack(_rename_source().unbind(0)), **store_kwargs
        )
        try:
            with pytest.raises(KeyError, match=match):
                store_td.rename_key_(old_key, new_key, safe=safe)
            _assert_store_matches(store_td, _rename_source())
        finally:
            store_td.clear_redis()
            store_td.close()

    @pytest.mark.parametrize("safe", [False, True])
    @pytest.mark.parametrize("old_key,new_key", _RENAME_INSIDE)
    def test_rename_key_inside_old_key(self, store_kwargs, old_key, new_key, safe):
        # the entry moves, as pop and set do on a TensorDict, and no data is
        # lost
        expected = _rename_source()
        expected.set(new_key, expected.pop(old_key))
        store_td = LazyStackedTensorDictStore.from_lazy_stack(
            lazy_stack(_rename_source().unbind(0)), **store_kwargs
        )
        try:
            store_td.rename_key_(old_key, new_key, safe=safe)
            _assert_store_matches(store_td, expected)
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

    def test_masked_select_inplace_raises(self, store_stack):
        """masked_select(inplace=True) raises on the stack and its element views."""
        store_td, tds, lazy_td = store_stack
        with pytest.raises(
            NotImplementedError,
            match="masked_select.inplace=True. is not supported on "
            "LazyStackedTensorDictStore",
        ):
            store_td.masked_select(torch.ones(5, 4, dtype=torch.bool), inplace=True)
        with pytest.raises(
            NotImplementedError,
            match="masked_select.inplace=True. is not supported on "
            "_StoreStackElementView",
        ):
            store_td[1].masked_select(torch.ones(4, dtype=torch.bool), inplace=True)
        assert store_td.batch_size == torch.Size([5, 4])
        # the store is still readable
        assert (store_td.to_tensordict() == lazy_td).all()

    @pytest.mark.parametrize("element", [False, True])
    def test_masked_select_copy(self, store_stack, element, monkeypatch):
        """masked_select returns a dense copy, read in one to_tensordict call."""
        store_td, tds, lazy_td = store_stack
        target, expected = (store_td[1], tds[1]) if element else (store_td, lazy_td)
        mask = torch.zeros(target.shape, dtype=torch.bool)
        mask[1] = True
        mask[-1] = True
        calls = []
        cls = type(target)
        to_tensordict = cls.to_tensordict

        def counting_to_tensordict(self, *args, **kwargs):
            calls.append(self)
            return to_tensordict(self, *args, **kwargs)

        monkeypatch.setattr(cls, "to_tensordict", counting_to_tensordict)
        result = target.masked_select(mask)
        assert len(calls) == 1 and calls[0] is target
        assert type(result) is TensorDict
        assert (result == expected.masked_select(mask)).all()
        monkeypatch.undo()
        assert (store_td.to_tensordict() == lazy_td).all()

    @pytest.mark.parametrize("key", ["missing", ("missing", "x")])
    def test_fill_missing_key(self, store_stack, key):
        """fill_ of a missing key raises a KeyError, on the stack and on an element view."""
        store_td, tds, lazy_td = store_stack
        for td in (store_td, store_td[1]):
            with pytest.raises(KeyError):
                td.fill_(key, 1.0)

    def test_detach_no_read(self, store_stack, monkeypatch):
        """detach_ returns the stack or the element view without calling the server."""
        store_td, tds, lazy_td = store_stack
        view = store_td[1]

        def no_round_trip(self, coro):
            coro.close()
            raise AssertionError("the server was called")

        with monkeypatch.context() as patch:
            for cls in (LazyStackedTensorDictStore, _StoreStackElementView):
                patch.setattr(cls, "_run_sync", no_round_trip)
            assert store_td.detach_() is store_td
            assert view.detach_() is view

    def test_view_fill_nested(self, store_kwargs):
        """fill_ on a nested key of an element view fills that element only."""
        lazy_td = lazy_stack(
            [
                TensorDict(
                    a=torch.zeros(3),
                    n=TensorDict(
                        b=torch.zeros(3, 2),
                        m=TensorDict(c=torch.zeros(3), batch_size=[3]),
                        batch_size=[3],
                    ),
                    batch_size=[3],
                )
                for _ in range(4)
            ]
        )
        store_td = LazyStackedTensorDictStore.from_lazy_stack(lazy_td, **store_kwargs)
        try:
            for td in (store_td, lazy_td):
                view = td[1]
                assert view.fill_("n", 5.0) is view
            assert_allclose_td(store_td.to_tensordict(), lazy_td.to_tensordict())
            assert (store_td["n", "m", "c"][1] == 5).all()
            assert (store_td["n", "b"][2] == 0).all()
        finally:
            store_td.clear_redis()
            store_td.close()

    def test_view_masked_fill_(self, store_stack):
        """rltd[1].masked_fill_ fills the masked rows of element 1 only."""
        store_td, tds, lazy_td = store_stack
        mask = torch.tensor([True, False, False, True])
        view = store_td[1]
        assert view.masked_fill_(mask, 1.0) is view
        expected = tds[1].clone().masked_fill_(mask, 1.0)
        for key in ("a", "b"):
            torch.testing.assert_close(store_td[1][key], expected[key])
            torch.testing.assert_close(store_td[0][key], tds[0][key])

    @pytest.mark.parametrize(
        "shapes",
        [
            {"a": (3,), "f": (3, 2), ("n", "g"): (3, 2, 2)},
            # the entry ends with the batch shape [4, 3]: torch's broadcasting
            # from the right would apply the mask to the last two dims
            {"s": (3, 4, 3)},
        ],
    )
    def test_masked_fill_(self, store_kwargs, shapes):
        """masked_fill_ with a batch-shaped mask fills the masked batch elements (#2224)."""
        lazy_td = lazy_stack(
            [
                TensorDict(
                    {key: torch.zeros(shape) for key, shape in shapes.items()},
                    batch_size=[3],
                )
                for _ in range(4)
            ]
        )
        store_td = LazyStackedTensorDictStore.from_lazy_stack(lazy_td, **store_kwargs)
        try:
            mask = torch.zeros(4, 3, dtype=torch.bool)
            mask[1, 2] = True
            mask[3, 0] = True
            assert store_td.masked_fill_(mask, 1.0) is store_td
            expected = lazy_td.clone().masked_fill_(mask, 1.0)
            assert_allclose_td(store_td.to_tensordict(), expected.to_tensordict())
        finally:
            store_td.clear_redis()
            store_td.close()

    @pytest.mark.parametrize("batch_size,mask_shape", [[(5, 4), (4,)], [(5, 5), (5,)]])
    def test_masked_fill_trailing_mask(self, store_kwargs, batch_size, mask_shape):
        """A mask that is not batch-shaped still broadcasts from the right, with a warning.

        TensorDict.masked_fill_ aligns a mask with the leading batch dims, but
        the store has always broadcast these masks from the right. This is
        kept until TensorDict 0.17.
        """
        lazy_td = lazy_stack(
            [
                TensorDict(b=torch.zeros(batch_size[1:]), batch_size=batch_size[1:])
                for _ in range(batch_size[0])
            ]
        )
        store_td = LazyStackedTensorDictStore.from_lazy_stack(lazy_td, **store_kwargs)
        try:
            mask = torch.zeros(mask_shape, dtype=torch.bool)
            mask[1] = True
            match = (
                r"^Calling LazyStackedTensorDictStore\.masked_fill_ with a mask of "
                r"shape .* broadcasts the mask from the right .* will be removed "
                r"in TensorDict 0\.17\."
            )
            with pytest.warns(FutureWarning, match=match) as record:
                assert store_td.masked_fill_(mask, 1.0) is store_td
            # The redis client can record a warning of its own first (seen on
            # free-threaded Python 3.14t), so check the warning under test.
            (warning,) = [w for w in record if re.search(match, str(w.message))]
            assert warning.filename == __file__
            torch.testing.assert_close(
                store_td["b"], torch.zeros(batch_size).masked_fill(mask, 1.0)
            )
        finally:
            store_td.clear_redis()
            store_td.close()

    def test_masked_fill_entry_shaped_mask(self, store_kwargs):
        """A mask of the shape of every entry fills it as TensorDict does, without a warning."""
        lazy_td = lazy_stack(
            [TensorDict(a=torch.zeros(3, 2), batch_size=[3]) for _ in range(4)]
        )
        store_td = LazyStackedTensorDictStore.from_lazy_stack(lazy_td, **store_kwargs)
        try:
            mask = torch.zeros(4, 3, 2, dtype=torch.bool)
            mask[1, 2, 0] = True
            expected = torch.zeros(4, 3, 2).masked_fill(mask, 1.0)
            with warnings.catch_warnings():
                warnings.simplefilter("error", FutureWarning)
                assert store_td.masked_fill_(mask, 1.0) is store_td
            torch.testing.assert_close(store_td["a"], expected)
            # a mask with the entry's dims that broadcasts from a dim of size 1
            # fills the same elements either way
            for mask in (
                torch.ones(4, 1, 2, dtype=torch.bool),
                torch.ones(1, 3, 2, dtype=torch.bool),
            ):
                mask[0, 0, 1] = False
                expected = store_td["a"].masked_fill(mask, 2.0)
                with warnings.catch_warnings():
                    warnings.simplefilter("error", FutureWarning)
                    assert store_td.masked_fill_(mask, 2.0) is store_td
                torch.testing.assert_close(store_td["a"], expected)
        finally:
            store_td.clear_redis()
            store_td.close()

    def test_masked_fill_warns_before_writing(self, store_kwargs):
        """The FutureWarning for a mask that is not batch-shaped comes before any write."""
        lazy_td = lazy_stack(
            [
                TensorDict(a=torch.zeros(3, 2), d=torch.zeros(3, 2, 5), batch_size=[3])
                for _ in range(4)
            ]
        )
        store_td = LazyStackedTensorDictStore.from_lazy_stack(lazy_td, **store_kwargs)
        try:
            mask = torch.ones(4, 3, 2, dtype=torch.bool)
            with warnings.catch_warnings():
                warnings.simplefilter("error", FutureWarning)
                with pytest.raises(FutureWarning, match="broadcasts the mask"):
                    store_td.masked_fill_(mask, 1.0)
            assert_allclose_td(store_td.to_tensordict(), lazy_td.to_tensordict())
        finally:
            store_td.clear_redis()
            store_td.close()

    def test_masked_fill_heterogeneous(self, store_kwargs):
        """masked_fill_ on a store whose elements differ in shape (#2030).

        As on a lazy stack, a batch-shaped mask fills each element with its
        slice of the mask, and a 0-d mask fills each element.
        """
        tds = [
            TensorDict(a=torch.zeros(3), h=torch.zeros(3, 2 + i), batch_size=[3])
            for i in range(2)
        ]
        lazy_td = lazy_stack([td.clone() for td in tds])
        store_td = LazyStackedTensorDictStore.from_lazy_stack(
            lazy_stack(tds), **store_kwargs
        )
        try:
            mask = torch.tensor([[True, False, True], [False, True, False]])
            assert store_td.masked_fill_(mask, 2.0) is store_td
            lazy_td.masked_fill_(mask, 2.0)
            for i in range(2):
                assert (store_td[i] == lazy_td[i]).all()
            assert store_td.masked_fill_(torch.tensor(True), 1.0) is store_td
            torch.testing.assert_close(store_td["a"], torch.ones(2, 3))
            for i in range(2):
                torch.testing.assert_close(store_td[i]["h"], torch.ones(3, 2 + i))
        finally:
            store_td.clear_redis()
            store_td.close()

    def test_fill_zero_nested(self, store_kwargs):
        """fill_ on a nested key and zero_ write nested entries, as on a lazy stack."""
        lazy_td = lazy_stack(
            [
                TensorDict(
                    a=torch.ones(3),
                    n=TensorDict(
                        b=torch.ones(3, 2),
                        m=TensorDict(c=torch.ones(3), batch_size=[3]),
                        batch_size=[3],
                    ),
                    batch_size=[3],
                )
                for _ in range(4)
            ]
        )
        store_td = LazyStackedTensorDictStore.from_lazy_stack(lazy_td, **store_kwargs)
        try:
            for td in (store_td, lazy_td):
                assert td.fill_("n", 5.0) is td
                view = td[1]
                assert view.fill_(("n", "m"), 7.0) is view
                td[2].zero_()
            assert_allclose_td(store_td.to_tensordict(), lazy_td.to_tensordict())
            assert store_td.zero_() is store_td
            assert_allclose_td(
                store_td.to_tensordict(), lazy_td.zero_().to_tensordict()
            )
        finally:
            store_td.clear_redis()
            store_td.close()

    def test_fill_zero_nested_heterogeneous(self, store_kwargs):
        """fill_ on the store raises for heterogeneous nested entries, zero_ zeroes them.

        Their elements differ in shape, so they cannot be read or written as
        one tensor (#2030). zero_ and the element views write each element
        with its own shape.
        """
        tds = [
            TensorDict(
                n=TensorDict(b=torch.ones(3), h=torch.ones(3, 2 + i), batch_size=[3]),
                batch_size=[3],
            )
            for i in range(2)
        ]
        store_td = LazyStackedTensorDictStore.from_lazy_stack(
            lazy_stack(tds), **store_kwargs
        )
        try:
            with pytest.raises(RuntimeError, match="differ in shape"):
                store_td.fill_("n", 5.0)
            store_td.fill_(("n", "b"), 5.0)
            torch.testing.assert_close(store_td["n", "b"], torch.full((2, 3), 5.0))
            store_td.zero_()
            torch.testing.assert_close(store_td["n", "b"], torch.zeros(2, 3))
            for i in range(2):
                torch.testing.assert_close(
                    store_td[i]["n", "h"], torch.zeros_like(tds[i]["n", "h"])
                )
            store_td[1].fill_("n", 5.0)
            torch.testing.assert_close(store_td[1]["n", "h"], torch.full((3, 3), 5.0))
            torch.testing.assert_close(
                store_td[0]["n", "h"], torch.zeros_like(tds[0]["n", "h"])
            )
        finally:
            store_td.clear_redis()
            store_td.close()

    def test_update_nested(self, store_kwargs):
        """update() with a sub-tensordict value writes nested entries, as on a lazy stack (#2154)."""
        lazy_td = self._nested_lazy_td()
        store_td = LazyStackedTensorDictStore.from_lazy_stack(
            self._nested_lazy_td(), **store_kwargs
        )
        try:
            for td in (store_td, lazy_td):
                td.update(
                    TensorDict(
                        n=TensorDict(b=torch.ones(5, 3, 2), batch_size=[5, 3]),
                        batch_size=[5, 3],
                    )
                )
                td.update(
                    {
                        ("n", "m"): TensorDict(
                            c=torch.full((5, 3), 2.0), batch_size=[5, 3]
                        )
                    }
                )
                td.update(
                    {
                        "n": {
                            "b": torch.full((5, 3, 2), 9.0),
                            "m": {"c": torch.full((5, 3), 3.0)},
                        }
                    },
                    keys_to_update=[("n", "m", "c")],
                )
                # a new leaf under an existing sub-tensordict
                td.update({"n": {"d": torch.full((5, 3), 4.0)}})
                td[1].update({"n": {"b": torch.full((3, 2), 5.0)}})
                # the context manager writes back through update()
                with td.flatten_keys() as flat:
                    flat.set_("n.m.c", torch.full((5, 3), 6.0))
            assert_allclose_td(store_td.to_tensordict(), lazy_td.to_tensordict())
            torch.testing.assert_close(store_td[1]["n", "b"], torch.full((3, 2), 5.0))
            torch.testing.assert_close(store_td["n", "d"], torch.full((5, 3), 4.0))
            # a locked store takes an in-place update of an existing nested leaf
            store_td.lock_()
            store_td.update({"n": {"b": torch.full((5, 3, 2), 7.0)}}, inplace=True)
            torch.testing.assert_close(store_td["n", "b"], torch.full((5, 3, 2), 7.0))
            with pytest.raises(RuntimeError, match="Cannot modify locked"):
                store_td.update({"n": {"e": torch.ones(5, 3)}}, inplace=True)
            assert ("n", "e") not in store_td.keys(True)
            store_td.unlock_()
        finally:
            store_td.clear_redis()
            store_td.close()

    def test_update_self_writes_copy(self, store_kwargs):
        """update({key: store}) writes a copy of the data, where a TensorDict raises (#2741)."""
        store_td = LazyStackedTensorDictStore.from_lazy_stack(
            self._nested_lazy_td(), **store_kwargs
        )
        try:
            store_td.update({"me": store_td})
            torch.testing.assert_close(store_td["me", "a"], store_td["a"])
            assert "me" not in store_td["me"].keys()
            element = store_td[1]
            element.update({"el": element})
            torch.testing.assert_close(element["el", "a"], element["a"])
        finally:
            store_td.clear_redis()
            store_td.close()

    def test_replace_does_not_write_into_arg(self, store_stack):
        """replace(store, **kwargs) leaves the store passed to it unchanged (#2984)."""
        store_td, _, lazy_td = store_stack
        td = TensorDict(a=torch.zeros(5, 4, 3), batch_size=[5, 4])
        out = td.replace(store_td, extra=torch.ones(5, 4))
        assert set(store_td.keys()) == {"a", "b"}
        assert type(out) is TensorDict
        assert set(out.keys()) == {"a", "b", "extra"}
        torch.testing.assert_close(out["a"], lazy_td["a"])
        torch.testing.assert_close(out["extra"], torch.ones(5, 4))

    def test_update_nested_heterogeneous(self, store_kwargs):
        """update() of a nested leaf leaves a sibling whose elements differ in shape intact."""
        tds = [
            TensorDict(
                n=TensorDict(b=torch.ones(3), h=torch.ones(3, 2 + i), batch_size=[3]),
                batch_size=[3],
            )
            for i in range(2)
        ]
        store_td = LazyStackedTensorDictStore.from_lazy_stack(
            lazy_stack(tds), **store_kwargs
        )
        try:
            store_td.update({"n": {"b": torch.full((2, 3), 5.0)}})
            torch.testing.assert_close(store_td["n", "b"], torch.full((2, 3), 5.0))
            for i in range(2):
                torch.testing.assert_close(store_td[i]["n", "h"], tds[i]["n", "h"])
            store_td[1].update({"n": {"h": torch.zeros(3, 3)}})
            torch.testing.assert_close(store_td[1]["n", "h"], torch.zeros(3, 3))
            torch.testing.assert_close(store_td[0]["n", "h"], tds[0]["n", "h"])
            # a locked store takes an in-place update of an existing leaf only
            store_td.lock_()
            store_td.update({"n": {"b": torch.full((2, 3), 7.0)}}, inplace=True)
            torch.testing.assert_close(store_td["n", "b"], torch.full((2, 3), 7.0))
            with pytest.raises(RuntimeError, match="Cannot modify locked"):
                store_td.update({"n": {"e": torch.ones(2, 3)}}, inplace=True)
            assert ("n", "e") not in store_td.keys(True)
            store_td.unlock_()
        finally:
            store_td.clear_redis()
            store_td.close()

    def test_lstore_update__heterogeneous_missing_key(self, store_kwargs):
        """update_ of a store whose elements differ in shape, with keys that the store lacks."""
        tds = [
            TensorDict(a=torch.zeros(3), h=torch.zeros(3, 2 + i), batch_size=[3])
            for i in range(2)
        ]
        store_td = LazyStackedTensorDictStore.from_lazy_stack(
            lazy_stack(tds), **store_kwargs
        )
        try:
            # a key that the store lacks raises, as on every other class,
            # unless another key matches and keys_to_update is not given
            with pytest.raises(KeyError, match="zz"):
                store_td.update_({"zz": torch.ones(2, 3)})
            assert "zz" not in store_td.keys()
            store_td.update_({"a": torch.ones(2, 3), "zz": torch.ones(2, 3)})
            torch.testing.assert_close(store_td["a"], torch.ones(2, 3))
            store_td.update_(
                {"a": torch.full((2, 3), 2.0), "zz": torch.ones(2, 3)},
                keys_to_update=["a"],
            )
            torch.testing.assert_close(store_td["a"], torch.full((2, 3), 2.0))
            store_td.update_(
                TensorDict(a=torch.zeros(2, 3), batch_size=[2, 3]),
                a=torch.full((2, 3), 3.0),
            )
            torch.testing.assert_close(store_td["a"], torch.full((2, 3), 3.0))
            with pytest.raises(KeyError, match="zz"):
                store_td.update_({"zz": torch.ones(2, 3)}, keys_to_update=["zz"])
            assert "zz" not in store_td.keys()
        finally:
            store_td.clear_redis()
            store_td.close()

    def test_inplace_methods_write(self, store_kwargs):
        """The in-place methods and update_ write to the server, as on a lazy stack (#2242).

        A key whose elements differ in shape is written element by element
        (#2030).
        """
        tds = [
            TensorDict(
                a=torch.randn(3, 2),
                h=torch.randn(3, 1 + i),
                n=TensorDict(b=torch.randn(3), batch_size=[3]),
                batch_size=[3],
            )
            for i in range(2)
        ]
        lazy_td = lazy_stack([td.clone() for td in tds])
        store_td = LazyStackedTensorDictStore.from_lazy_stack(
            lazy_stack(tds), **store_kwargs
        )
        try:
            for td in (store_td, lazy_td):
                assert td.abs_() is td
                td *= 2
                view = td[1]
                assert view.add_(1.0) is view
            other = TensorDict(a=lazy_td["a"] + 1, batch_size=[2, 3])
            assert store_td.update_(other) is store_td
            lazy_td.update_(other)
            for i in range(2):
                torch.testing.assert_close(store_td[i]["a"], lazy_td[i]["a"])
                torch.testing.assert_close(store_td[i]["n", "b"], lazy_td[i]["n", "b"])
                torch.testing.assert_close(store_td[i]["h"], lazy_td[i]["h"])
            assert store_td.zero_() is store_td
            for i in range(2):
                assert (store_td[i]["h"] == 0).all()
            assert (store_td["a"] == 0).all()
        finally:
            store_td.clear_redis()
            store_td.close()

    @pytest.mark.parametrize("heterogeneous", [False, True])
    @pytest.mark.parametrize("stack_dim", [0, 1])
    def test_update__copy__lazy_source(self, store_kwargs, stack_dim, heterogeneous):
        """copy_ and update_ write a lazy stack into the store, as into a lazy stack (#2593).

        Before, they changed copies of the stored entries and wrote nothing,
        or raised for a key whose elements differ in shape.
        """

        def source(value):
            return lazy_stack(
                [
                    TensorDict(
                        a=torch.full((3,), value + i),
                        n=TensorDict(b=torch.full((3, 2), value - i), batch_size=[3]),
                        **(
                            {"h": torch.full((3, 1 + i), value)}
                            if heterogeneous
                            else {}
                        ),
                        batch_size=[3],
                    )
                    for i in range(2)
                ],
                stack_dim,
            )

        def element_source(value):
            # a lazy stack for the element at 1, along its batch dim
            return lazy_stack(
                [
                    TensorDict(
                        a=torch.tensor(value + j),
                        n=TensorDict(b=torch.full((2,), value)),
                        **({"h": torch.full((2,), value - j)} if heterogeneous else {}),
                    )
                    for j in range(3)
                ]
            )

        lazy_td = source(0.0)
        store_td = LazyStackedTensorDictStore.from_lazy_stack(
            source(0.0), **store_kwargs
        )
        try:
            for td, element in (
                (store_td, _StoreStackElementView(store_td, 1)),
                (lazy_td, lazy_td.tensordicts[1]),
            ):
                assert td.copy_(source(1.0)) is td
                assert td.update_(source(2.0), keys_to_update=[("n", "b")]) is td
                td.update_(source(3.0), keys_to_update=["a", "h"])
                element.copy_(element_source(4.0))
            for i in range(2):
                assert_allclose_td(
                    _StoreStackElementView(store_td, i).to_tensordict(),
                    lazy_td.tensordicts[i],
                )
        finally:
            store_td.clear_redis()
            store_td.close()

    def test_locked_update_inplace(self, store_kwargs):
        """A locked store and element view write their keys with inplace=True, as a lazy stack does (#2403)."""
        tds = [
            TensorDict(a=torch.zeros(3), n=TensorDict(b=torch.zeros(3)))
            for _ in range(2)
        ]
        lazy_td = lazy_stack([td.clone() for td in tds])
        store_td = LazyStackedTensorDictStore.from_lazy_stack(
            lazy_stack(tds), **store_kwargs
        )
        try:
            for td in (store_td, lazy_td, store_td[1], lazy_td[1]):
                td.lock_()
                td.update({"a": td["a"] + 1}, inplace=True)
                td.set(("n", "b"), td["n", "b"] + 2, inplace=True)
                td.set(
                    "n",
                    TensorDict(b=td["n", "b"] + 3, batch_size=td.shape),
                    inplace=True,
                )
                # a new key still raises, also as a leaf of a tensordict value
                with pytest.raises(RuntimeError, match="Cannot modify locked"):
                    td.update({"c": td["a"]}, inplace=True)
                with pytest.raises(RuntimeError, match="Cannot modify locked"):
                    td.set(("n", "c"), td["a"], inplace=True)
                with pytest.raises(RuntimeError, match="Cannot modify locked"):
                    td["n"] = TensorDict(c=td["a"], batch_size=td.shape)
                with pytest.raises(RuntimeError, match="Cannot modify locked"):
                    td["n"] = td["a"]
                with pytest.raises(RuntimeError, match="Cannot modify locked"):
                    td["n", "m"] = TensorDict(c=td["a"], batch_size=td.shape)
                td.unlock_()
            assert_allclose_td(store_td.to_tensordict(), lazy_td.to_tensordict())
        finally:
            store_td.clear_redis()
            store_td.close()

    def test_view_of_locked_store(self, store_kwargs):
        """The view of an element of a locked store raises for a new key and writes the keys the store has (#2404)."""
        store_td = LazyStackedTensorDictStore.from_lazy_stack(
            lazy_stack(
                [
                    TensorDict(a=torch.zeros(3), n=TensorDict(b=torch.zeros(3)))
                    for _ in range(2)
                ]
            ),
            **store_kwargs,
        )
        try:
            store_td.lock_()
            view = store_td[1]
            # as documented, the view itself and the results of its
            # operations are not locked
            assert not view.is_locked
            assert not (view + 1).is_locked
            for key in ("x", ("n", "x")):
                with pytest.raises(RuntimeError, match="Cannot modify locked"):
                    view.set(key, torch.ones(3))
            with pytest.raises(RuntimeError, match="Cannot modify locked"):
                view.update({"x": torch.ones(3)}, inplace=True)
            with pytest.raises(RuntimeError, match="Cannot modify locked"):
                view["n"] = TensorDict(c=torch.ones(3))
            with pytest.raises(RuntimeError, match="Cannot modify locked"):
                view.set("n", TensorDict(b=torch.ones(3), c=torch.ones(3)))
            with pytest.raises(RuntimeError, match="Cannot modify locked"):
                view["n"] = torch.ones(3)
            assert list(store_td.keys(True, True)) == ["a", ("n", "b")]
            # the keys the store has are written, with or without inplace
            view.set("a", torch.ones(3))
            view.set(("n", "b"), torch.ones(3))
            view.update({"a": view["a"] + 1})
            view.update({"a": view["a"] + 1}, inplace=True)
            torch.testing.assert_close(store_td["a"][1], torch.full((3,), 3.0))
            torch.testing.assert_close(store_td["n", "b"][1], torch.ones(3))
            torch.testing.assert_close(store_td["a"][0], torch.zeros(3))
            # the view follows the lock of the store
            store_td.unlock_()
            view.set("x", torch.ones(3))
            torch.testing.assert_close(store_td["x"][1], torch.ones(3))
            # a locked view still needs inplace for the keys the store has
            view.lock_()
            with pytest.raises(RuntimeError, match="Cannot modify locked"):
                view.set("a", torch.ones(3))
            view.set("a", torch.zeros(3), inplace=True)
            torch.testing.assert_close(store_td["a"][1], torch.zeros(3))
        finally:
            store_td.clear_redis()
            store_td.close()

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

    def test_view_set_at_nested(self, store_kwargs):
        """set_at_ and update_at_ on an element view write nested entries (#2154)."""
        lazy_td = self._nested_lazy_td()
        store_td = LazyStackedTensorDictStore.from_lazy_stack(lazy_td, **store_kwargs)
        try:
            for td in (store_td, lazy_td):
                td[1].set_at_(("n", "b"), torch.tensor([7.0, 8.0]), 0)
                td[2].set_at_(("n", "m", "c"), 4.0, slice(1, None))
                td[0].set_at_(
                    "n",
                    TensorDict(b=torch.full((2,), 5.0), m=TensorDict(c=6.0)),
                    2,
                )
                td[3].set_at_(("n", "m"), TensorDict(c=torch.tensor(9.0)), -1)
                td[4].update_at_(
                    TensorDict(a=torch.tensor(1.0), n=TensorDict(b=torch.ones(2))), 1
                )
            assert_allclose_td(store_td.to_tensordict(), lazy_td.to_tensordict())

            # a key path under a leaf does not exist
            keys = set(store_td.keys(True, True))
            with pytest.raises(KeyError):
                store_td[0].set_at_(("a", "b"), torch.tensor(1.0), 0)
            assert set(store_td.keys(True, True)) == keys

            # a leaf the store lacks is created, zero-filled for the other
            # elements
            store_td[-1].set_at_(
                "n", TensorDict(b=torch.zeros(2), d=torch.tensor(3.0)), 0
            )
            expected = torch.zeros(5, 3)
            expected[-1, 0] = 3.0
            torch.testing.assert_close(store_td["n", "d"], expected)
        finally:
            store_td.clear_redis()
            store_td.close()

    def test_view_set_at_nested_heterogeneous(self, store_kwargs):
        """Element-view set_at_ writes nested entries whose elements differ in shape."""
        tds = [
            TensorDict(
                n=TensorDict(
                    b=torch.zeros(2), h=torch.zeros(k, dtype=torch.long), batch_size=[]
                ),
                batch_size=[],
            )
            for k in (2, 3, 4)
        ]
        lazy_td = lazy_stack([td.clone() for td in tds])
        store_td = LazyStackedTensorDictStore.from_lazy_stack(
            lazy_stack(tds), **store_kwargs
        )
        try:
            for td in (store_td, lazy_td):
                td[1].set_at_(("n", "h"), torch.tensor(7), 2)
                td[2].set_at_("n", TensorDict(h=torch.full((4,), 3)), ())
                td[0].set_at_(("n", "b"), torch.tensor(1.5), 0)
            for i in range(3):
                assert store_td[i]["n", "h"].dtype == torch.long
                assert_allclose_td(store_td[i].to_tensordict(), lazy_td[i])
        finally:
            store_td.clear_redis()
            store_td.close()

    def test_element_view_index_and_chunk(self, store_stack):
        """An element view reads one element for indexing and chunk, with the values of its dense copy."""
        store_td, tds, lazy_td = store_stack
        view = store_td[1]
        assert isinstance(view, _StoreStackElementView)
        dense = view.to_tensordict()
        for index in (0, slice(1, 3), torch.tensor([3, 0]), (Ellipsis,)):
            out = view[index]
            assert (out == dense[index]).all()
            assert out.batch_size == dense[index].batch_size
        pieces = view.chunk(3)
        ref = dense.chunk(3)
        assert [p.batch_size for p in pieces] == [r.batch_size for r in ref]
        for piece, r in zip(pieces, ref):
            assert (piece == r).all()

    def test_view_set_new_key_last_element(self, store_kwargs):
        """A new key set on any element view, the last one too, reads back exactly."""
        lazy_td = self._nested_lazy_td()
        store_td = LazyStackedTensorDictStore.from_lazy_stack(lazy_td, **store_kwargs)
        try:
            value = torch.tensor([1.0, 2.0, 3.0])
            for i in range(5):
                store_td[i].set(f"z{i}", value)
                store_td[i].set(("n", f"z{i}"), value)
            for i in range(5):
                for key in (f"z{i}", ("n", f"z{i}")):
                    expected = torch.zeros(5, 3)
                    expected[i] = value
                    torch.testing.assert_close(store_td[key], expected)
        finally:
            store_td.clear_redis()
            store_td.close()

    # ---- empty ----

    def test_empty_recurse(self, store_kwargs):
        """empty(recurse=True) keeps the sub-tensordicts, as on a lazy stack."""
        tds = [
            TensorDict(
                {
                    "a": torch.randn(3),
                    "b": {"c": torch.randn(3), "d": {"e": torch.randn(3, 2)}},
                },
                batch_size=[3],
            )
            for _ in range(4)
        ]
        store_td = LazyStackedTensorDictStore.from_lazy_stack(
            lazy_stack(tds), **store_kwargs
        )
        try:
            # the store and the view of one element
            for td in (store_td, store_td[1]):
                empty = td.empty(recurse=True)
                assert isinstance(empty, TensorDict)
                assert set(empty.keys(True)) == {"b", ("b", "d")}
                assert empty.batch_size == td.batch_size
                assert empty["b", "d"].batch_size == td.batch_size
                assert len(td.empty().keys()) == 0
        finally:
            store_td.clear_redis()
            store_td.close()

    # ---- del_, entry_class, contiguous, popitem, chunk, (un)flatten_keys ----

    @staticmethod
    def _nested_lazy_td():
        return lazy_stack(
            [
                TensorDict(
                    a=torch.arange(3.0) + i,
                    n=TensorDict(
                        b=torch.full((3, 2), float(i)),
                        m=TensorDict(c=torch.full((3,), -float(i)), batch_size=[3]),
                        batch_size=[3],
                    ),
                    batch_size=[3],
                )
                for i in range(5)
            ]
        )

    def test_del_(self, store_kwargs):
        """del_ removes a leaf, or every leaf under a sub-tensordict, as on a lazy stack."""
        lazy_td = self._nested_lazy_td()
        store_td = LazyStackedTensorDictStore.from_lazy_stack(lazy_td, **store_kwargs)
        try:
            for key in (("n", "b"), "n"):
                for td in (store_td, lazy_td):
                    assert td.del_(key) is td
                assert set(store_td.keys(True, True)) == set(lazy_td.keys(True, True))
                assert_allclose_td(store_td.to_tensordict(), lazy_td.to_tensordict())
            assert list(store_td.keys()) == ["a"]
        finally:
            store_td.clear_redis()
            store_td.close()

    @pytest.mark.parametrize(
        "method,key",
        [
            ("del_", "a"),
            ("del_", "n"),
            ("del_", ("n", "m")),
            ("pop", "a"),
            ("pop", "n"),
            ("popitem", None),
            ("rename_key_", "a"),
            ("rename_key_", "n"),
            ("rename_key_", ("n", "m")),
        ],
    )
    def test_existing_key_does_not_warn(self, store_kwargs, method, key):
        # del_ and rename_key_ also act on every key under a sub-tensordict
        store_td = LazyStackedTensorDictStore.from_lazy_stack(
            self._nested_lazy_td(), **store_kwargs
        )
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                if method == "popitem":
                    store_td.popitem()
                elif method == "rename_key_":
                    store_td.rename_key_(key, "renamed")
                    assert "renamed" in store_td.keys()
                else:
                    getattr(store_td, method)(key)
                    assert key not in store_td.keys(True)
        finally:
            store_td.clear_redis()
            store_td.close()

    @pytest.mark.parametrize("method", ["del_", "rename_key_"])
    @pytest.mark.parametrize("key", ["missing", ("n", "missing")])
    def test_missing_key_warns(self, store_kwargs, method, key):
        # https://github.com/pytorch/tensordict/issues/2062
        lazy_td = self._nested_lazy_td()
        store_td = LazyStackedTensorDictStore.from_lazy_stack(lazy_td, **store_kwargs)
        args = (key,) if method == "del_" else (key, "renamed")
        try:
            with pytest.warns(
                FutureWarning,
                match=rf"^LazyStackedTensorDictStore\.{method} {_MISSING_KEY_WARNING}",
            ) as record:
                assert getattr(store_td, method)(*args) is store_td
            # count only FutureWarnings: under xdist on free-threaded Python, the
            # record once held 12 other warnings from code outside the call
            warned = [w for w in record if issubclass(w.category, FutureWarning)]
            assert len(warned) == 1, [str(w.message) for w in record]
            assert warned[0].filename == __file__
            assert set(store_td.keys(True, True)) == set(lazy_td.keys(True, True))
        finally:
            store_td.clear_redis()
            store_td.close()

    @staticmethod
    def _key_edit_stores(store_kwargs):
        """A TensorDictStore, a LazyStackedTensorDictStore and an element view of it."""
        td = TensorDict(
            a=torch.arange(6.0).view(2, 3),
            n=TensorDict(x=torch.ones(2, 3), batch_size=[2, 3]),
            batch_size=[2, 3],
        )
        tds = TensorDictStore.from_tensordict(td, **store_kwargs)
        lstds = LazyStackedTensorDictStore.from_lazy_stack(
            lazy_stack(list(td.unbind(0))), **store_kwargs
        )
        return td, {"tds": tds, "lstds": lstds, "view": lstds[0]}, (tds, lstds)

    @pytest.mark.parametrize("kind", ["tds", "lstds", "view"])
    def test_get_at_missing_node_and_through_tensor(self, store_kwargs, kind):
        td, stores, owned = self._key_edit_stores(store_kwargs)
        try:
            store = stores[kind]
            assert store.get_at("zz", 0, None) is None
            assert store.get_at("zz", 0) is None
            node = store.get_at("n", 0)
            assert type(node) is TensorDict
            assert (node["x"] == 1).all()
            with pytest.raises(ValueError, match="'a' is a Tensor, not a tensordict"):
                store.get_at(("a", "x"), 0)
        finally:
            for store in owned:
                store.clear_redis()
                store.close()

    @pytest.mark.parametrize("op", ["flatten_keys", "unflatten_keys"])
    @pytest.mark.parametrize("kind", ["tds", "lstds", "view"])
    def test_flatten_inplace_raises_before_read(self, store_kwargs, kind, op):
        td, stores, owned = self._key_edit_stores(store_kwargs)
        try:
            store = stores[kind]
            keys = list(store.keys(True, True))
            if kind == "view":
                # the view ignores inplace and returns a copy, until 0.17
                with pytest.warns(DeprecationWarning, match="inplace=True"):
                    out = getattr(store, op)(inplace=True)
                assert type(out) is TensorDict
            else:
                with mock.patch.object(
                    type(store), "to_tensordict", side_effect=AssertionError
                ):
                    with pytest.raises(
                        ValueError, match=f"Cannot call {op} in_place with a"
                    ):
                        getattr(store, op)(inplace=True)
            assert list(store.keys(True, True)) == keys
            assert_allclose_td(store.to_tensordict(), td[0] if kind == "view" else td)
            flat = store.flatten_keys()
            assert type(flat) is TensorDict
            assert set(flat.keys()) == {"a", "n.x"}

            def is_leaf(cls):
                return cls is TensorDict

            flat = store.flatten_keys(is_leaf=is_leaf)
            dense = td[0] if kind == "view" else td
            assert list(flat.keys()) == list(dense.flatten_keys(is_leaf=is_leaf).keys())
        finally:
            for store in owned:
                store.clear_redis()
                store.close()

    @pytest.mark.parametrize("kind", ["tds", "lstds", "view"])
    def test_select_exclude_unsupported_messages(self, store_kwargs, kind):
        td, stores, owned = self._key_edit_stores(store_kwargs)
        try:
            store = stores[kind]
            name = type(store).__name__
            for op in ("select", "exclude"):
                with pytest.raises(
                    NotImplementedError, match=f"Cannot call {op} on a {name}."
                ):
                    getattr(store, op)("a")
        finally:
            for store in owned:
                store.clear_redis()
                store.close()

    @pytest.mark.parametrize("kind", ["tds", "lstds"])
    @pytest.mark.parametrize("old,new", [("a", ""), ("a", ("",)), ("", "b")])
    def test_rename_key_empty_string_key(self, store_kwargs, kind, old, new):
        # "" is a valid key: only the empty tuple is not
        td = TensorDict({old: torch.zeros(2, 3)}, batch_size=[2, 3])
        if kind == "tds":
            store = TensorDictStore.from_tensordict(td, **store_kwargs)
        else:
            store = LazyStackedTensorDictStore.from_lazy_stack(
                lazy_stack(list(td.unbind(0))), **store_kwargs
            )
        try:
            assert store.rename_key_(old, new) is store
            new_key = new if isinstance(new, str) else new[0]
            assert list(store.keys()) == [new_key]
            assert (store.get(new_key) == 0).all()
        finally:
            store.clear_redis()
            store.close()

    @pytest.mark.parametrize("kind", ["tds", "lstds", "view"])
    def test_key_edits_empty_key(self, store_kwargs, kind):
        td, stores, owned = self._key_edit_stores(store_kwargs)
        try:
            store = stores[kind]
            if kind == "view":
                with pytest.raises(RuntimeError, match="Cannot delete keys"):
                    store.del_("a")
                with pytest.raises(RuntimeError, match="Cannot rename keys"):
                    store.rename_key_("a", "a")
                with pytest.raises(RuntimeError, match="Cannot delete keys"):
                    store.popitem()
            elif kind == "tds":
                with pytest.raises(IndexError):
                    store.del_(())
            else:
                with pytest.warns(FutureWarning, match="missing key"):
                    store.del_(())
            assert list(store.keys(True, True)) == ["a", ("n", "x")]
        finally:
            for store in owned:
                store.clear_redis()
                store.close()

    def test_non_tensor_items_store_round_trips(self, store_kwargs):
        td = TensorDict(
            a=torch.zeros(2),
            s=NonTensorData("x", batch_size=[2]),
            n=TensorDict(t=NonTensorData("y", batch_size=[2]), batch_size=[2]),
            batch_size=[2],
        )
        store = TensorDictStore.from_tensordict(td, **store_kwargs)
        try:
            assert [key for key, _ in td.non_tensor_items(True)] == ["s", ("n", "t")]
            # the store returns no non-tensor leaf
            assert list(store.non_tensor_items(True)) == []
            assert store.non_tensor_items() == ()
        finally:
            store.clear_redis()
            store.close()

    def test_entry_class(self, store_kwargs):
        lazy_td = self._nested_lazy_td()
        store_td = LazyStackedTensorDictStore.from_lazy_stack(lazy_td, **store_kwargs)
        try:
            for key in ("a", ("n", "b"), ("n", "m", "c")):
                assert store_td.entry_class(key) is torch.Tensor
                assert store_td.entry_class(key) is lazy_td.entry_class(key)
            with pytest.raises(KeyError, match="not found"):
                store_td.entry_class("missing")
        finally:
            store_td.clear_redis()
            store_td.close()

    def test_entry_class_sub_tensordict(self, store_kwargs):
        store_td = LazyStackedTensorDictStore.from_lazy_stack(
            self._nested_lazy_td(), **store_kwargs
        )
        try:
            for key in ("n", ("n", "m")):
                assert store_td.entry_class(key) is type(store_td.get(key))
                assert store_td[0].entry_class(key) is type(store_td[0].get(key))
        finally:
            store_td.clear_redis()
            store_td.close()

    @pytest.mark.parametrize("canonical", [False, True])
    def test_contiguous(self, store_stack, canonical):
        store_td, _, lazy_td = store_stack
        out = store_td.contiguous(canonical=canonical)
        assert type(out) is TensorDict
        assert_allclose_td(out, lazy_td.contiguous(canonical=canonical))

    @pytest.mark.parametrize("element", [False, True])
    @pytest.mark.parametrize("canonical", [False, True])
    def test_contiguous_one_round_trip(
        self, store_stack, monkeypatch, element, canonical
    ):
        """contiguous() reads the store as to_tensordict() does."""
        store_td, tds, _ = store_stack
        td = store_td[0] if element else store_td
        calls = _count_round_trips(store_td, monkeypatch)
        expected = td.to_tensordict()
        round_trips = len(calls)
        calls.clear()
        out = td.contiguous(canonical=canonical)
        assert len(calls) == round_trips
        assert type(out) is TensorDict
        assert_allclose_td(out, expected)
        assert td.is_contiguous() is False
        with pytest.raises(
            NotImplementedError,
            match=rf"contiguous\(inplace=True\) is not supported on {type(td).__name__}",
        ):
            td.contiguous(inplace=True)

    @pytest.mark.parametrize("element", [False, True])
    def test_pin_memory_raises(self, store_stack, element):
        store_td, _, _ = store_stack
        td = store_td[0] if element else store_td
        with pytest.raises(
            RuntimeError,
            match=rf"^Cannot pin memory of a {type(td).__name__}\. Call to_tensordict\(\) "
            r"before making this call\.$",
        ):
            td.pin_memory()

    def test_store_device_setter_moves_reads(self, store_stack):
        """The device of the store is the device of what it reads; the device
        of an element is a label for its leaf reads."""
        store_td, _, _ = store_stack
        store_td.device = "meta"
        assert store_td.device == torch.device("meta")
        assert store_td["a"].device == torch.device("meta")
        assert store_td[0]["a"].device == torch.device("meta")
        store_td.device = None
        element = store_td[0]
        with pytest.warns(
            FutureWarning,
            match=r"the leaves that it reads stay on the device of the store, while "
            r"its nested tensordicts, to_tensordict\(\) and contiguous\(\) use the "
            r"new device\. Starting with TensorDict 0\.17, the default will change "
            r"to raising",
        ):
            element.device = "meta"
        assert element.device == torch.device("meta")
        assert element["a"].device == torch.device("cpu")
        assert element.to_tensordict()["a"].device == torch.device("meta")
        assert element.contiguous()["a"].device == torch.device("meta")

    def test_batch_size_setter(self, store_stack, monkeypatch):
        """The new batch size is checked against the stored shapes, without
        reading the entries."""
        store_td, _, _ = store_stack

        def no_read(*args, **kwargs):
            raise AssertionError("the batch size setter read an entry")

        store_td.batch_size = [5]
        assert store_td.batch_size == torch.Size([5])
        assert store_td.to_tensordict().batch_size == torch.Size([5])
        element = store_td[0]
        for name in (
            "_abatch_get_at",
            "_abatch_get_element",
            "_abatch_get_element_keys",
        ):
            monkeypatch.setattr(LazyStackedTensorDictStore, name, no_read)
        store_td.batch_size = [5, 4]
        element.batch_size = []
        assert element.batch_size == torch.Size([])
        element.batch_size = [4]
        for td, size in ((store_td, [17]), (store_td, [5, 4, 3]), (element, [17])):
            with pytest.raises(RuntimeError, match="incompatible with the batch-size"):
                td.batch_size = size
        assert store_td.batch_size == torch.Size([5, 4])
        assert element.batch_size == torch.Size([4])

    def test_batch_size_setter_heterogeneous(self, store_kwargs):
        """A dim along which the elements differ fits no batch size."""
        lazy_td = lazy_stack(
            [
                TensorDict(a=torch.zeros(2, 3), batch_size=[2]),
                TensorDict(a=torch.zeros(2, 4), batch_size=[2]),
            ]
        )
        store_td = LazyStackedTensorDictStore.from_lazy_stack(lazy_td, **store_kwargs)
        try:
            store_td.batch_size = [2]
            assert store_td.batch_size == torch.Size([2])
            store_td.batch_size = [2, 2]
            with pytest.raises(RuntimeError, match="incompatible with the batch-size"):
                store_td.batch_size = [2, 2, 3]
            element = store_td[1]
            element.batch_size = [2, 4]
            assert element.batch_size == torch.Size([2, 4])
            with pytest.raises(RuntimeError, match="incompatible with the batch-size"):
                element.batch_size = [2, 3]
        finally:
            store_td.clear_redis()
            store_td.close()

    def test_lock_names(self, store_stack):
        store_td, _, _ = store_stack
        element = store_td[0]
        assert element.lock_() is element
        assert element.is_locked
        assert element.unlock_() is element
        assert not element.is_locked
        store_td.names = ["s", "x"]
        assert store_td.names == ["s", "x"]
        element.names = ["y"]
        assert element.names == ["y"]
        store_td.names = None
        assert store_td.names == [None, None]

    def test_popitem(self, store_stack):
        """popitem removes and returns the last key, as on a lazy stack."""
        store_td, _, lazy_td = store_stack
        for expected_key in ("b", "a"):
            key, value = store_td.popitem()
            assert key == expected_key
            torch.testing.assert_close(value, lazy_td[key])
            assert key not in store_td.keys()
        assert len(store_td.keys()) == 0
        with pytest.raises(KeyError, match="is empty"):
            store_td.popitem()

    @pytest.mark.parametrize("dim", [0, 1])
    def test_chunk(self, store_stack, dim):
        store_td, _, lazy_td = store_stack
        chunks = store_td.chunk(2, dim)
        expected = lazy_td.chunk(2, dim)
        assert len(chunks) == len(expected) == 2
        for chunk, expected_chunk in zip(chunks, expected):
            assert type(chunk) is TensorDict
            assert_allclose_td(chunk, expected_chunk.to_tensordict())

    @pytest.mark.parametrize("dim", [1, -1])
    def test_chunk_zero_size_dim(self, store_kwargs, dim):
        # torch.chunk gives `chunks` empty pieces along a zero-size dim
        lazy_td = lazy_stack(
            [TensorDict({"a": torch.zeros(0, 4)}, [0]) for _ in range(3)]
        )
        store_td = LazyStackedTensorDictStore.from_lazy_stack(lazy_td, **store_kwargs)
        try:
            for chunks in (2, 3):
                pieces = store_td.chunk(chunks, dim)
                assert len(pieces) == chunks
                for piece in pieces:
                    assert piece.batch_size == torch.Size([3, 0])
                    assert piece["a"].shape == torch.Size([3, 0, 4])
        finally:
            store_td.clear_redis()
            store_td.close()

    @pytest.mark.parametrize("separator", [".", "/"])
    def test_flatten_keys(self, store_kwargs, separator):
        lazy_td = self._nested_lazy_td()
        store_td = LazyStackedTensorDictStore.from_lazy_stack(lazy_td, **store_kwargs)
        try:
            out = store_td.flatten_keys(separator)
            assert type(out) is TensorDict
            assert_allclose_td(out, lazy_td.flatten_keys(separator).to_tensordict())
            with pytest.raises(ValueError, match="Cannot call flatten_keys in_place"):
                store_td.flatten_keys(separator, inplace=True)
        finally:
            store_td.clear_redis()
            store_td.close()

    def test_unflatten_keys(self, store_kwargs):
        # The store joins nested keys with ".", so the flat keys use "/"
        lazy_td = self._nested_lazy_td()
        store_td = LazyStackedTensorDictStore.from_lazy_stack(
            lazy_td.flatten_keys("/"), **store_kwargs
        )
        try:
            assert set(store_td.keys()) == {"a", "n/b", "n/m/c"}
            out = store_td.unflatten_keys("/")
            assert type(out) is TensorDict
            assert_allclose_td(out, lazy_td.to_tensordict())
            with pytest.raises(ValueError, match="Cannot call unflatten_keys in_place"):
                store_td.unflatten_keys("/", inplace=True)
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


class _SubclassData(TensorClass):
    obs: torch.Tensor
    reward: torch.Tensor


class _NotATensorClass:
    @classmethod
    def _from_tensordict(cls, *args, **kwargs):
        raise AssertionError("Non-TensorClass reconstruction must not be called")


class TestTensorClassStore:
    """Tests for TensorClass round-trip through TensorDictStore."""

    @pytest.fixture
    def stored_metadata(self):
        # Stub only I/O: from_store still resolves and wraps the real class.
        with mock.patch(
            "tensordict.store._store._read_stored_metadata",
            return_value=(b"[2]", None, None),
        ) as metadata:
            yield metadata

    @pytest.mark.parametrize("tc_type", [_MyData, _SubclassData])
    @pytest.mark.parametrize(
        "source", ["class", "path", "stored", "override_class", "override_path"]
    )
    def test_from_store_resolved_type(self, tc_type, source, stored_metadata):
        path = f"{tc_type.__module__}.{tc_type.__name__}"
        explicit = None
        raw_stored = None
        if source in ("class", "override_class"):
            explicit = tc_type
        elif source in ("path", "override_path"):
            explicit = path
        if source == "stored":
            raw_stored = path.encode()
        elif source.startswith("override"):
            raw_stored = b"missing_module.UnusedClass"
        stored_metadata.return_value = (b"[2]", None, raw_stored)
        local = TensorDict({"obs": torch.zeros(2), "reward": torch.ones(2)}, [2])
        constructor = mock.Mock(return_value=local)

        result = TensorDictStore.from_store.__func__(
            constructor, td_id="test", tensorclass_cls=explicit
        )

        assert type(result) is tc_type
        assert result._tensordict is local
        assert local._tensorclass_cls == path
        assert torch.equal(result.obs, torch.zeros(2))
        constructor.assert_called_once()

    @pytest.mark.parametrize(
        "source,value",
        [
            (source, value)
            for value in (int, len, _NotATensorClass)
            for source in ("class", "path", "stored")
            if source != "class" or isinstance(value, type)
        ],
    )
    def test_from_store_rejects_invalid_type(self, source, value, stored_metadata):
        path = f"{value.__module__}.{value.__name__}"
        explicit = value if source == "class" else path
        if source == "stored":
            explicit = None
            stored_metadata.return_value = (b"[2]", None, path.encode())
        constructor = mock.Mock(return_value=TensorDict({}, [2]))
        with mock.patch.object(_NotATensorClass, "_from_tensordict") as rebuild:
            with pytest.raises(TypeError, match="Expected a TensorClass type"):
                TensorDictStore.from_store.__func__(
                    constructor, td_id="test", tensorclass_cls=explicit
                )
            rebuild.assert_not_called()

    def test_from_store_rejects_imported_instance(self, stored_metadata, monkeypatch):
        instance = _MyData(obs=torch.zeros(2), reward=torch.ones(2), batch_size=[2])
        path = f"{__name__}._stored_instance"
        monkeypatch.setattr(path, instance, raising=False)
        stored_metadata.return_value = (b"[2]", None, path.encode())
        with pytest.raises(TypeError, match="Expected a TensorClass type"):
            TensorDictStore.from_store.__func__(
                mock.Mock(return_value=TensorDict({}, [2])), td_id="test"
            )

    def test_from_store_without_resolved_type(self, stored_metadata):
        local = TensorDict({}, [2])
        result = TensorDictStore.from_store.__func__(
            mock.Mock(return_value=local), td_id="test"
        )
        assert result is local

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
            restored._tensordict.close()
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

    def test_clear_redis_deletes_tensorclass_key(self, store_kwargs):
        import redis

        tc = _MyData(obs=torch.randn(5, 3), reward=torch.randn(5), batch_size=[5])
        store = TensorDictStore.from_tensordict(tc, **store_kwargs)
        td_id = store._td_id
        store.clear_redis()
        store.close()
        client = redis.Redis(port=store_kwargs["port"], db=store_kwargs["db"])
        try:
            assert client.keys(f"*{td_id}*") == []
        finally:
            client.close()


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

    @pytest.mark.parametrize(
        "index",
        [3, [0, 2], torch.tensor([1, 3]), slice(1, 4), [True, False] * 2 + [True]],
    )
    def test_setitem_scalar(self, store_kwargs, index):
        """A scalar is written to every tensor entry, in its dtype, as in a
        TensorDict; the non-tensor entries are left as they are."""
        td = TensorDict(
            {
                "count": torch.zeros(5, 2, dtype=torch.long),
                "obs": torch.zeros(5, 4),
                "nested": {"x": torch.zeros(5)},
                "label": "placeholder",
            },
            [5],
        )
        store = TensorDictStore.from_tensordict(td, **store_kwargs)
        try:
            store[index] = -3.5
            expected = td.exclude("label")
            expected[index] = -3.5
            for key in ("count", "obs", ("nested", "x")):
                torch.testing.assert_close(store[key], expected[key])
            assert store[0]["label"] == "placeholder"
            assert store[4]["label"] == "placeholder"
            with pytest.raises(TypeError, match="not a tensor"):
                store[index] = torch.ones(4)
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

    def test_non_tensor_to_tensordict_is_writable(self, store_kwargs):
        """A per-element entry is read as a NonTensorStack, so the copy that
        ``to_tensordict()`` returns takes indexed writes."""
        td = TensorDict({"val": torch.randn(3, 2), "tag": "x"}, [3])
        store = TensorDictStore.from_tensordict(td, **store_kwargs)
        try:
            store[1] = TensorDict({"val": torch.zeros(2), "tag": "y"}, [])
            local = store.to_tensordict()
            assert isinstance(local.get("tag"), NonTensorStack)
            local[[0, 1]] = TensorDict({"val": torch.ones(2, 2), "tag": "z"}, [2])
            assert local.get("tag").tolist() == ["z", "z", "x"]
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

    def test_setitem_with_nontensorstack(self, store_kwargs):
        """NonTensorStack leaves must not be silently dropped on write.

        NonTensorStack inherits from LazyStackedTensorDict (a TensorDictBase),
        so it passes the is_tensor_collection check and was silently skipped.
        """
        from tensordict import set_list_to_stack

        with set_list_to_stack(True):
            td = TensorDict(
                {
                    "reward": torch.tensor([1.0]),
                    "chat": TensorDict(
                        {
                            "prompt": TensorDict(
                                {
                                    "role": ["user"],
                                    "content": ["hello"],
                                },
                                batch_size=[1],
                            ),
                        },
                        batch_size=[],
                    ),
                },
                batch_size=[],
            )

        store_td = TensorDict({"reward": torch.zeros(5)}, batch_size=[5])
        store_td["chat"] = TensorDict(
            {"prompt": TensorDict({}, batch_size=[5, 1])}, batch_size=[5]
        )
        store_td["chat", "prompt"].set_non_tensor("role", "init")
        store_td["chat", "prompt"].set_non_tensor("content", "init")

        store = TensorDictStore.from_tensordict(store_td, **store_kwargs)
        try:
            store[0] = td
            result = store[0]
            assert torch.allclose(result["reward"], torch.tensor([1.0]))
            role = result.get_non_tensor(("chat", "prompt", "role"))
            content = result.get_non_tensor(("chat", "prompt", "content"))
            assert role is not None, "NonTensorStack 'role' was silently dropped"
            assert content is not None, "NonTensorStack 'content' was silently dropped"
        finally:
            store.clear_redis()
            store.close()

    def test_setitem_tensor_index_non_tensor(self, store_kwargs):
        """_aset_non_tensor_at must accept torch.Tensor indices.

        When extend() is called, TorchRL computes indices as a torch.Tensor
        (e.g. tensor([0, 1, 2, ...])) and passes them through __setitem__.
        """
        batch = 4
        td = TensorDict({"obs": torch.randn(batch, 3)}, batch_size=[batch])
        td.set_non_tensor("label", "init")

        store = TensorDictStore.from_tensordict(td, **store_kwargs)
        try:
            # Write with a tensor index
            new_labels = TensorDict({"obs": torch.ones(2, 3)}, batch_size=[2])
            new_labels.set_non_tensor("label", "updated")
            idx = torch.tensor([1, 3])
            store[idx] = new_labels

            # Verify the indexed elements were updated
            r1 = store[1]
            assert r1.get_non_tensor("label") == "updated"
            r3 = store[3]
            assert r3.get_non_tensor("label") == "updated"

            # Verify untouched elements are still "init"
            r0 = store[0]
            assert r0.get_non_tensor("label") == "init"
            r2 = store[2]
            assert r2.get_non_tensor("label") == "init"
        finally:
            store.clear_redis()
            store.close()

    @staticmethod
    def _values(value):
        """The Python values held by a non-tensor result."""
        if is_tensor_collection(value):
            return value.tolist() if value.batch_dims else value.data
        return value

    @pytest.mark.parametrize(
        "index",
        [
            (3,),
            slice(1, 4),
            slice(None, None, 2),
            [-1, 0],
            torch.tensor([0, 3]),
            np.array([0, 3]),
            range(1, 3),
            torch.tensor([True, False, False, True, False]),
        ],
    )
    def test_non_tensor_indexed_read(self, store_kwargs, index):
        """Indexed reads and get_at of a per-element non-tensor entry follow the
        index, as on a tensordict."""
        td = TensorDict({"obs": torch.zeros(5), "label": "a"}, [5])
        store = TensorDictStore.from_tensordict(td, **store_kwargs)
        try:
            # a single stored value is broadcast, as store.get returns it
            assert store.get_at("label", index) == "a"
            for target in (store, td):
                target[3] = TensorDict({"obs": torch.ones(()), "label": "d"}, [])
            sub, expected = store[index], td[index]
            assert sub.batch_size == expected.batch_size
            assert self._values(sub.get("label")) == self._values(expected.get("label"))
            assert self._values(store.get_at("label", index)) == self._values(
                td.get_at("label", index)
            )
        finally:
            store.clear_redis()
            store.close()

    def test_non_tensor_empty_index_read(self, store_kwargs):
        """An empty index selects no element of a per-element non-tensor entry."""
        td = TensorDict({"obs": torch.zeros(5), "label": "a"}, [5])
        store = TensorDictStore.from_tensordict(td, **store_kwargs)
        try:
            store[3] = TensorDict({"obs": torch.ones(()), "label": "d"}, [])
            sub = store[[]]
            assert sub.batch_size == torch.Size([0])
            assert sub.get("label").tolist() == []
            assert store.get_at("label", []).tolist() == []
        finally:
            store.clear_redis()
            store.close()

    @pytest.mark.parametrize(
        "index",
        [
            torch.tensor([True, False, False, True, False]),
            np.array([0, 3]),
            (slice(1, 3),),
            Ellipsis,
        ],
    )
    def test_non_tensor_indexed_write(self, store_kwargs, index):
        """Indexed writes change the selected elements of a non-tensor entry."""
        td = TensorDict({"obs": torch.zeros(5), "label": "a"}, [5])
        store = TensorDictStore.from_tensordict(td, **store_kwargs)
        try:
            for target in (store, td):
                target[3] = TensorDict({"obs": torch.ones(()), "label": "d"}, [])
            value_shape = td[index].batch_size
            value = TensorDict(
                {"obs": torch.ones(value_shape), "label": "z"}, value_shape
            )
            store[index] = value
            td[index] = value
            assert [store[i]["label"] for i in range(5)] == td.get("label").tolist()
            torch.testing.assert_close(store["obs"], td["obs"])
        finally:
            store.clear_redis()
            store.close()

    def test_non_tensor_set_at_mask(self, store_kwargs):
        """set_at_ on a non-tensor key changes the masked elements."""
        td = TensorDict({"obs": torch.zeros(5), "label": "a"}, [5])
        store = TensorDictStore.from_tensordict(td, **store_kwargs)
        try:
            mask = torch.tensor([True, False, False, True, False])
            store.set_at_("label", "z", mask)
            assert [store[i]["label"] for i in range(5)] == ["z", "a", "a", "z", "a"]
        finally:
            store.clear_redis()
            store.close()

    def test_non_tensor_unsupported_index_writes_nothing(self, store_kwargs):
        """An index that a non-tensor entry cannot take is rejected before
        anything is written."""
        td = TensorDict({"obs": torch.zeros(5, 2), "label": "a"}, [5, 2])
        store = TensorDictStore.from_tensordict(td, **store_kwargs)
        try:
            store[3] = TensorDict({"obs": torch.ones(2), "label": "d"}, [2])
            obs = store["obs"].clone()
            with pytest.raises(TypeError, match="Non-tensor indexed writes"):
                store[0, :] = TensorDict({"obs": torch.ones(2), "label": "z"}, [2])
            torch.testing.assert_close(store["obs"], obs)
        finally:
            store.clear_redis()
            store.close()

    @pytest.mark.parametrize("key", ["label", ("nested", "label")])
    @pytest.mark.parametrize("method", ["from_tensordict", "set"])
    def test_non_tensor_stack_write(self, store_kwargs, key, method):
        """A NonTensorStack written whole is stored per element, as indexed
        writes store it, so indexed reads and writes follow the index."""
        td = TensorDict({"obs": torch.arange(5.0)}, [5])
        td[key] = NonTensorStack(*[f"s{i}" for i in range(5)])
        if method == "from_tensordict":
            store = TensorDictStore.from_tensordict(td, **store_kwargs)
        else:
            store = TensorDictStore(batch_size=[5], **store_kwargs)
            store["obs"] = td["obs"]
            store[key] = td.get(key)
        try:
            assert store[1:3][key] == td[1:3][key] == ["s1", "s2"]
            assert store[1][key] == "s1"
            assert store.get_at(key, 1) == "s1"
            assert store[key] == td[key]
            assert isinstance(store.get(key), NonTensorStack)
            assert store.flip(0)[key] == td.flip(0)[key]
            assert store.roll(1, 0)[key] == td.roll(1, 0)[key]
            store.set_at_(key, "z", 2)
            assert store[key] == ["s0", "s1", "z", "s3", "s4"]
        finally:
            store.clear_redis()
            store.close()

    def test_non_tensor_blob_read(self, store_kwargs):
        """An entry stored as one blob holds one value for every element: a
        NonTensorData of a list, or a NonTensorStack written by an earlier
        version."""
        store = TensorDictStore(batch_size=[3], **store_kwargs)
        try:
            store["shared"] = NonTensorData(["x", "y", "z"], batch_size=[3])
            # earlier versions stored a whole NonTensorStack as one blob
            store._run_sync(store._aset_non_tensor("old", ["x", "y", "z"]))
            for key in ("shared", "old"):
                assert type(store.get(key)) is list
                assert store[key] == store[1][key] == ["x", "y", "z"]
                assert store.get_at(key, 1) == ["x", "y", "z"]
        finally:
            store.clear_redis()
            store.close()

    def test_non_tensor_pickle_read(self, store_kwargs):
        """A value that JSON cannot encode is stored with pickle and reads
        back as it was written."""
        store = TensorDictStore(batch_size=[3], **store_kwargs)
        try:
            store["label"] = NonTensorData({1, 2})
            assert store["label"] == {1, 2}
            assert store[1]["label"] == {1, 2}
            assert store.get_at("label", 1) == {1, 2}
            assert store.to_tensordict()["label"] == {1, 2}
        finally:
            store.clear_redis()
            store.close()


def test_lazy_store_pickle_path():
    # LazyStackedTensorDictStore moved from tensordict.store._store to
    # tensordict.store._lazy. Pickles made before the move name the old module,
    # and unpickling looks the classes up there.
    import tensordict.store._lazy as lazy_module

    unpickler = pickle.Unpickler(io.BytesIO())
    for name in ("LazyStackedTensorDictStore", "_StoreStackElementView"):
        cls = unpickler.find_class("tensordict.store._store", name)
        assert cls is getattr(lazy_module, name)


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
