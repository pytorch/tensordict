# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the UCXX transport layer (tensordict._ucxx).

Tests are split into two groups:

1. **Unit tests** that exercise protocol helpers and schema hashing without
   requiring ucxx to be installed.
2. **Integration tests** that require a working ucxx installation and exercise
   real UCXX endpoint communication on localhost.
"""

from __future__ import annotations

import asyncio
import json
import struct
import subprocess
import sys
import textwrap
import threading
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from tensordict import TensorDict

# Protocol helpers and classes are importable even without ucxx installed
from tensordict._ucxx import (
    _can_recv_into,
    _compute_total_bytes,
    _metadata_hash,
    _NEW_SCHEMA_FLAG,
    _SAME_SCHEMA_FLAG,
    recv_tensordict,
    send_tensordict,
    TensorDictPipe,
)

try:
    import ucxx  # noqa: F401

    _has_ucxx = True
except ImportError:
    _has_ucxx = False

requires_ucxx = pytest.mark.skipif(not _has_ucxx, reason="ucxx not installed")
requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)


# ---------------------------------------------------------------------------
# Unit tests — no ucxx required
# ---------------------------------------------------------------------------


class TestMetadataHash:
    def test_deterministic(self):
        meta = {
            "leaves": {"a": ("torch.float32", [2, 3], 0, 24, 0)},
            "cls": "TensorDict",
            "non_tensors": {},
            "cls_metadata": {"batch_size": [2], "device": None, "names": None},
        }
        h1 = _metadata_hash(meta)
        h2 = _metadata_hash(meta)
        assert h1 == h2

    def test_different_metadata_different_hash(self):
        meta1 = {
            "leaves": {"a": ("torch.float32", [2, 3], 0, 24, 0)},
            "cls": "TensorDict",
            "non_tensors": {},
            "cls_metadata": {"batch_size": [2], "device": None, "names": None},
        }
        meta2 = {
            "leaves": {"b": ("torch.float64", [4], 0, 32, 0)},
            "cls": "TensorDict",
            "non_tensors": {},
            "cls_metadata": {"batch_size": [4], "device": None, "names": None},
        }
        assert _metadata_hash(meta1) != _metadata_hash(meta2)

    def test_key_order_invariant(self):
        meta1 = {"a": 1, "b": 2}
        meta2 = {"b": 2, "a": 1}
        assert _metadata_hash(meta1) == _metadata_hash(meta2)


class TestComputeTotalBytes:
    def test_single_leaf(self):
        meta = {
            "leaves": {"x": ("torch.float32", [10], 0, 40, 0)},
            "cls": "TensorDict",
            "non_tensors": {},
            "cls_metadata": {},
        }
        assert _compute_total_bytes(meta) == 40

    def test_multiple_leaves(self):
        meta = {
            "leaves": {
                "a": ("torch.float32", [4], 0, 16, 0),
                "b": ("torch.float64", [4], 16, 48, 0),
            },
            "cls": "TensorDict",
            "non_tensors": {},
            "cls_metadata": {},
        }
        assert _compute_total_bytes(meta) == 48

    def test_six_field_leaf(self):
        meta = {"leaves": {"<NJT_OFFSETS>x": ("torch.int64", [3], 32, 56, 0, 2)}}
        assert _compute_total_bytes(meta) == 56

    def test_nested(self):
        meta = {
            "leaves": {"a": ("torch.float32", [4], 0, 16, 0)},
            "cls": "TensorDict",
            "non_tensors": {},
            "cls_metadata": {},
            "sub": {
                "leaves": {"b": ("torch.float64", [4], 16, 48, 0)},
                "cls": "TensorDict",
                "non_tensors": {},
                "cls_metadata": {},
            },
        }
        assert _compute_total_bytes(meta) == 48


class _FakeEndpoint:
    """A mock UCXX endpoint that records sends and replays them on recv."""

    def __init__(self):
        self._buffers: list[bytes] = []
        self._recv_idx = 0

    async def send(self, buf):
        if isinstance(buf, np.ndarray):
            self._buffers.append(bytes(buf))
        elif isinstance(buf, torch.Tensor):
            self._buffers.append(bytes(buf.cpu().numpy()))
        else:
            self._buffers.append(bytes(buf))

    async def recv(self, buf):
        data = self._buffers[self._recv_idx]
        assert buf.nbytes == len(data)
        self._recv_idx += 1
        if isinstance(buf, np.ndarray):
            buf[:] = np.frombuffer(data, dtype=buf.dtype)[: len(buf)]
        elif isinstance(buf, torch.Tensor):
            arr = np.frombuffer(data, dtype=np.uint8)
            buf.copy_(torch.from_numpy(arr.copy()))

    async def close(self):
        pass


class TestProtocolConsolidated:
    """Test the consolidated send/recv protocol using a fake endpoint."""

    def _make_pipe(self, endpoint):
        return TensorDictPipe(endpoint, consolidated=True)

    @pytest.fixture
    def td(self):
        return TensorDict(
            {"a": torch.randn(3, 4), "b": torch.randn(3, 2)},
            batch_size=[3],
        )

    def test_first_send_includes_metadata(self, td):
        ep = _FakeEndpoint()
        pipe = self._make_pipe(ep)
        asyncio.run(pipe.asend(td))

        # First buffer should be the flag byte
        assert ep._buffers[0] == _NEW_SCHEMA_FLAG

        # Second buffer should be metadata length (8 bytes)
        meta_len = struct.unpack("<Q", ep._buffers[1])[0]
        assert meta_len > 0

        # Third buffer should be parseable JSON metadata
        meta = json.loads(ep._buffers[2])
        assert "_total_bytes" in meta

    def test_second_send_same_schema_skips_metadata(self, td):
        ep = _FakeEndpoint()
        pipe = self._make_pipe(ep)
        asyncio.run(pipe.asend(td))
        first_count = len(ep._buffers)

        asyncio.run(pipe.asend(td))
        # Second send: flag (1) + storage (1) = 2 buffers
        second_count = len(ep._buffers) - first_count
        assert second_count == 2
        assert ep._buffers[first_count] == _SAME_SCHEMA_FLAG

    def test_schema_change_resends_metadata(self, td):
        ep = _FakeEndpoint()
        pipe = self._make_pipe(ep)
        asyncio.run(pipe.asend(td))
        first_count = len(ep._buffers)

        td2 = TensorDict({"x": torch.randn(5)}, batch_size=[5])
        asyncio.run(pipe.asend(td2))
        # Schema changed: flag (1) + meta_len (1) + meta (1) + storage (1) = 4
        second_count = len(ep._buffers) - first_count
        assert second_count == 4
        assert ep._buffers[first_count] == _NEW_SCHEMA_FLAG


class TestRoundtrip:
    """Test full send → recv roundtrip with fake endpoint."""

    @pytest.mark.parametrize("stateless", [False, True])
    @pytest.mark.parametrize("lengths", [False, True])
    @pytest.mark.parametrize("ragged_idx", [1, 2])
    @pytest.mark.parametrize("offsets", [[0, 3, 8], [0], [0, 0, 0]])
    def test_jagged_roundtrip(self, stateless, lengths, ragged_idx, offsets):
        offsets = torch.tensor(offsets)
        values = torch.arange(offsets[-1].item() * 2.0).reshape(-1, 2)
        if ragged_idx == 2:
            values = values.T.contiguous()
        jagged = torch.nested.nested_tensor_from_jagged(
            values,
            offsets=offsets,
            lengths=(offsets.diff() - 1).clamp_min(0) if lengths else None,
            jagged_dim=ragged_idx,
        )
        td = TensorDict({"nested": {"x": jagged}}, batch_size=[])
        ep = _FakeEndpoint()
        if stateless:
            asyncio.run(send_tensordict.__wrapped__(ep, td))
            result = asyncio.run(recv_tensordict.__wrapped__(ep))
        else:
            asyncio.run(TensorDictPipe(ep).asend(td))
            result = asyncio.run(TensorDictPipe(ep).arecv())
        assert len(result["nested", "x"].unbind()) == len(jagged.unbind())
        for actual, expected in zip(result["nested", "x"].unbind(), jagged.unbind()):
            torch.testing.assert_close(actual, expected)
        assert result["nested", "x"]._ragged_idx == ragged_idx

    @pytest.mark.parametrize("stateless", [False, True])
    @pytest.mark.parametrize("data", [{}, {"x": torch.empty(0)}, {"nested": {}}])
    def test_consolidated_empty_roundtrip(self, stateless, data):
        td = TensorDict(data, batch_size=[])
        ep = _FakeEndpoint()
        if stateless:
            asyncio.run(send_tensordict.__wrapped__(ep, td))
            result = asyncio.run(recv_tensordict.__wrapped__(ep))
        else:
            asyncio.run(TensorDictPipe(ep).asend(td))
            result = asyncio.run(TensorDictPipe(ep).arecv())
        torch.testing.assert_close(result.to_dict(), td.to_dict())

    @pytest.mark.parametrize(
        "mismatch", [None, "key", "dtype", "shape", "batch", "size"]
    )
    def test_stateless_destination_layout(self, mismatch):
        td = TensorDict({"x": torch.arange(4.0)}, batch_size=[4])
        target = TensorDict(
            {
                "y" if mismatch == "key" else "x": torch.zeros(
                    (2, 2) if mismatch == "shape" else 4,
                    dtype=torch.int32 if mismatch == "dtype" else torch.float32,
                )
            },
            batch_size=[] if mismatch in ("batch", "shape") else [4],
        ).consolidate(metadata=True)
        if mismatch == "size":
            target._consolidated["storage"] = torch.zeros(24, dtype=torch.uint8)
        before = target.clone()
        ep = _FakeEndpoint()
        asyncio.run(send_tensordict.__wrapped__(ep, td))
        result = asyncio.run(recv_tensordict.__wrapped__(ep, target))
        torch.testing.assert_close(result["x"], td["x"])
        assert result.batch_size == td.batch_size
        if mismatch is None:
            assert result is target
        else:
            assert result is not target
            torch.testing.assert_close(target.to_dict(), before.to_dict())

    @pytest.mark.parametrize("stateless", [False, True])
    @pytest.mark.parametrize("device", ["cpu", "cpu:0"])
    def test_equivalent_cpu_destination(self, stateless, device):
        td = TensorDict({"x": torch.arange(4.0)}, batch_size=[])
        target = torch.zeros_like(td).consolidate(metadata=True)
        ep = _FakeEndpoint()
        if stateless:
            asyncio.run(send_tensordict.__wrapped__(ep, td))
            result = asyncio.run(recv_tensordict.__wrapped__(ep, target, device=device))
        else:
            asyncio.run(TensorDictPipe(ep).asend(td))
            result = asyncio.run(TensorDictPipe(ep).arecv(target, device=device))
        assert result is target
        torch.testing.assert_close(target["x"], td["x"])

    @pytest.mark.parametrize("current_index", [0, 1])
    @pytest.mark.parametrize("storage_index", [0, 1])
    @pytest.mark.parametrize("device", ["cuda", "cuda:0", "cuda:1"])
    def test_cuda_destination_current_device(
        self, current_index, storage_index, device, monkeypatch
    ):
        monkeypatch.setattr(torch.cuda, "current_device", lambda: current_index)
        # Only the compatibility check needs a CUDA device; no GPU is allocated.
        target = SimpleNamespace(
            is_consolidated=lambda: True,
            _consolidated={
                "storage": SimpleNamespace(
                    device=torch.device("cuda", storage_index), numel=lambda: 16
                ),
                "metadata": {},
            },
        )
        expected_index = (
            current_index if device == "cuda" else torch.device(device).index
        )
        assert _can_recv_into(target, 16, torch.device(device), _metadata_hash({})) == (
            storage_index == expected_index
        )

    @pytest.mark.parametrize("device", [None, "cpu", "meta"])
    @pytest.mark.parametrize("mismatch", [False, True])
    def test_stateless_destination_device(self, device, mismatch, monkeypatch):
        td = TensorDict({"x": torch.arange(4.0)}, batch_size=[])
        target = TensorDict(
            {"y" if mismatch else "x": torch.zeros(4)}, batch_size=[]
        ).consolidate(metadata=True, device="meta")
        ep = _FakeEndpoint()
        asyncio.run(send_tensordict.__wrapped__(ep, td))
        # A meta target lets this CPU-only test observe device selection without
        # allocating GPU storage. Replace its receive view with a CPU buffer.
        views = []
        allocations = []
        torch_empty = torch.empty

        def allocate(*args, **kwargs):
            allocations.append(kwargs.get("device"))
            return torch_empty(*args, **kwargs)

        def receive_view(storage):
            views.append(storage)
            return np.empty(storage.numel(), dtype=np.uint8)

        monkeypatch.setattr(torch, "empty", allocate)
        monkeypatch.setattr("tensordict._ucxx._tensor_to_numpy", receive_view)
        result = asyncio.run(recv_tensordict.__wrapped__(ep, target, device=device))
        reuse = not mismatch and device != "cpu"
        expected_device = torch.device("cpu" if device == "cpu" else "meta")
        assert (result is target) == reuse
        assert views[0].device == expected_device
        assert allocations == ([] if reuse else [expected_device])
        assert set(result.keys()) == {"x"}
        assert set(target.keys()) == {"y" if mismatch else "x"}

    @pytest.mark.parametrize("flag", [b"\x02", b"\xff"])
    @pytest.mark.parametrize("handshake", [False, True])
    def test_unknown_flag_does_not_receive_payload(self, flag, handshake):
        ep = _FakeEndpoint()
        receiver = TensorDictPipe(ep)
        if handshake:
            asyncio.run(TensorDictPipe(ep).asend(TensorDict(x=torch.ones(2))))
            cached = asyncio.run(receiver.arecv())
            before = cached.clone()
        ep._buffers.extend([flag, bytes(8)])
        with pytest.raises(RuntimeError, match="Unknown.*flag"):
            asyncio.run(receiver.arecv())
        assert ep._recv_idx == len(ep._buffers) - 1
        if handshake:
            torch.testing.assert_close(cached.to_dict(), before.to_dict())

    @pytest.mark.parametrize(
        "header",
        [
            [],
            {},
            {"keys": "xy", "dtypes": ["torch.float32"] * 2, "shapes": [[1], [1]]},
            {"keys": ["x"], "dtypes": "torch.float32", "shapes": [[1]]},
            {"keys": ["x"], "dtypes": ["torch.float32"], "shapes": {"x": [1]}},
            {"keys": ["x", "y"], "dtypes": ["torch.float32"], "shapes": [[1], [1]]},
            {"keys": ["x", "y"], "dtypes": ["torch.float32"] * 2, "shapes": [[1]]},
            {"keys": ["x", "x"], "dtypes": ["torch.float32"] * 2, "shapes": [[1], [1]]},
            {
                "keys": ["a", ["a", "b"]],
                "dtypes": ["torch.float32"] * 2,
                "shapes": [[1], [1]],
            },
            {
                "keys": [["a", "b"], "a"],
                "dtypes": ["torch.float32"] * 2,
                "shapes": [[1], [1]],
            },
            {
                "keys": ["x", ["x"]],
                "dtypes": ["torch.float32"] * 2,
                "shapes": [[1], [1]],
            },
            {
                "keys": [["x", "y"], ["x", ["y"]]],
                "dtypes": ["torch.float32"] * 2,
                "shapes": [[1], [1]],
            },
        ]
        + [
            {"keys": ["x", key], "dtypes": ["torch.float32"] * 2, "shapes": [[1], [1]]}
            for key in (1, None, [], ["x", 1], ["x", []])
        ]
        + [
            {
                "keys": ["x", "y"],
                "dtypes": ["torch.float32", dtype],
                "shapes": [[1], [1]],
            }
            for dtype in ("torch.empty", "torch.not_a_dtype", None, [])
        ]
        + [
            {
                "keys": ["x", "y"],
                "dtypes": ["torch.float32"] * 2,
                "shapes": [[1], shape],
            }
            for shape in ([-1], [True], [1.5], ["1"], None, 1)
        ],
    )
    def test_uncollated_invalid_header_before_payload(self, header, monkeypatch):
        encoded = json.dumps(header).encode()
        ep = _FakeEndpoint()
        ep._buffers = [struct.pack("<Q", len(encoded)), encoded]

        def unexpected_allocation(*args, **kwargs):
            pytest.fail("allocated a payload before validating the entire header")

        monkeypatch.setattr(torch, "empty", unexpected_allocation)
        monkeypatch.setattr(torch, "tensor", unexpected_allocation)
        with pytest.raises(ValueError, match="uncollated"):
            asyncio.run(TensorDictPipe(ep, consolidated=False).arecv())
        assert ep._recv_idx == 2

    @pytest.mark.parametrize(
        "dtype",
        [
            torch.float32,
            torch.int64,
            torch.bool,
            torch.bfloat16,
            torch.complex64,
            torch.float8_e4m3fn,
            torch.float8_e5m2,
        ],
    )
    @pytest.mark.parametrize("shape", [(2,), (0, 2)])
    def test_uncollated_roundtrip(self, dtype, shape):
        td = TensorDict({"x": torch.ones(shape).to(dtype)}, batch_size=[])
        ep = _FakeEndpoint()
        asyncio.run(TensorDictPipe(ep, consolidated=False).asend(td))
        result = asyncio.run(TensorDictPipe(ep, consolidated=False).arecv())
        assert result["x"].dtype == dtype
        assert result["x"].shape == shape
        torch.testing.assert_close(result["x"].float(), td["x"].float())

    @pytest.mark.parametrize("sender", ["pipe", "send_tensordict"])
    def test_uncollated_roundtrip_nested(self, sender):
        # https://github.com/pytorch/tensordict/issues/2966
        td = TensorDict(
            a=torch.ones(2),
            nested=TensorDict(
                b=torch.zeros(3),
                c=TensorDict(d=torch.arange(4), e=torch.tensor(1.0)),
            ),
        )
        ep = _FakeEndpoint()
        if sender == "pipe":
            asyncio.run(TensorDictPipe(ep, consolidated=False).asend(td))
        else:
            asyncio.run(send_tensordict.__wrapped__(ep, td, consolidated=False))
        result = asyncio.run(TensorDictPipe(ep, consolidated=False).arecv())
        assert set(result.keys(True, True)) == set(td.keys(True, True))
        for key, value in td.items(True, True):
            assert result[key].dtype == value.dtype
            torch.testing.assert_close(result[key], value)

    @pytest.mark.parametrize("key", ["", ["x"], ["nested", "x"], ["nested", ["x"]]])
    def test_uncollated_nested_keys(self, key):
        header = {"keys": [key], "dtypes": ["torch.float32"], "shapes": [[2]]}
        encoded = json.dumps(header).encode()
        ep = _FakeEndpoint()
        ep._buffers = [
            struct.pack("<Q", len(encoded)),
            encoded,
            np.ones(2, dtype=np.float32).tobytes(),
        ]
        result = asyncio.run(TensorDictPipe(ep, consolidated=False).arecv())
        expected_key = "" if key == "" else "x" if key == ["x"] else ("nested", "x")
        torch.testing.assert_close(result[expected_key], torch.ones(2))

    def test_uncollated_sibling_keys(self):
        header = {
            "keys": [["a", "b"], ["a", "c"]],
            "dtypes": ["torch.float32"] * 2,
            "shapes": [[1], [1]],
        }
        encoded = json.dumps(header).encode()
        ep = _FakeEndpoint()
        ep._buffers = [
            struct.pack("<Q", len(encoded)),
            encoded,
            struct.pack("<f", 2.0),
            struct.pack("<f", 3.0),
        ]
        result = asyncio.run(TensorDictPipe(ep, consolidated=False).arecv())
        assert set(result.keys(True, True)) == {("a", "b"), ("a", "c")}
        torch.testing.assert_close(result["a", "b"], torch.tensor([2.0]))
        torch.testing.assert_close(result["a", "c"], torch.tensor([3.0]))

    def test_uncollated_scalar(self):
        header = {"keys": ["x"], "dtypes": ["torch.float32"], "shapes": [[]]}
        encoded = json.dumps(header).encode()
        ep = _FakeEndpoint()
        ep._buffers = [struct.pack("<Q", len(encoded)), encoded, struct.pack("<f", 2.0)]
        result = asyncio.run(TensorDictPipe(ep, consolidated=False).arecv())
        torch.testing.assert_close(result["x"], torch.tensor(2.0))

    def test_uncollated_empty(self):
        ep = _FakeEndpoint()
        asyncio.run(TensorDictPipe(ep, consolidated=False).asend(TensorDict({}, [])))
        result = asyncio.run(TensorDictPipe(ep, consolidated=False).arecv())
        assert not result.keys()

    def test_consolidated_roundtrip(self):
        td = TensorDict(
            {"a": torch.randn(3, 4), "b": torch.randn(3, 2)},
            batch_size=[3],
        )

        ep = _FakeEndpoint()
        sender = TensorDictPipe(ep, consolidated=True)
        asyncio.run(sender.asend(td))

        ep._recv_idx = 0
        receiver = TensorDictPipe(ep, consolidated=True)
        td_recv = asyncio.run(receiver.arecv())

        assert set(td_recv.keys()) == {"a", "b"}
        torch.testing.assert_close(td_recv["a"], td["a"])
        torch.testing.assert_close(td_recv["b"], td["b"])

    def test_consolidated_steady_state_inplace(self):
        td = TensorDict(
            {"a": torch.ones(4), "b": torch.zeros(4)},
            batch_size=[4],
        )

        ep = _FakeEndpoint()
        sender = TensorDictPipe(ep, consolidated=True)

        # First send
        asyncio.run(sender.asend(td))
        ep._recv_idx = 0
        receiver = TensorDictPipe(ep, consolidated=True)
        td_recv = asyncio.run(receiver.arecv())
        torch.testing.assert_close(td_recv["a"], torch.ones(4))

        # Second send with updated values
        td_updated = TensorDict(
            {"a": torch.full((4,), 42.0), "b": torch.full((4,), 99.0)},
            batch_size=[4],
        )
        ep2 = _FakeEndpoint()
        sender2 = TensorDictPipe(ep2, consolidated=True)
        sender2._send_schema_hash = sender._send_schema_hash
        asyncio.run(sender2.asend(td_updated))

        ep2._recv_idx = 0
        receiver2 = TensorDictPipe(ep2, consolidated=True)
        receiver2._recv_schema_hash = receiver._recv_schema_hash
        receiver2._recv_td = td_recv
        td_recv2 = asyncio.run(receiver2.arecv(td_recv))

        assert td_recv2 is td_recv
        torch.testing.assert_close(td_recv["a"], torch.full((4,), 42.0))
        torch.testing.assert_close(td_recv["b"], torch.full((4,), 99.0))

    def test_nested_tensordict_roundtrip(self):
        td = TensorDict(
            {
                "obs": torch.randn(2, 3),
                "info": TensorDict(
                    {"reward": torch.tensor([1.0, 2.0])},
                    batch_size=[2],
                ),
            },
            batch_size=[2],
        )

        ep = _FakeEndpoint()
        sender = TensorDictPipe(ep, consolidated=True)
        asyncio.run(sender.asend(td))

        ep._recv_idx = 0
        receiver = TensorDictPipe(ep, consolidated=True)
        td_recv = asyncio.run(receiver.arecv())

        torch.testing.assert_close(td_recv["obs"], td["obs"])
        torch.testing.assert_close(td_recv["info", "reward"], td["info", "reward"])


class TestSyncWrappersInRunningLoop:
    """The sync methods must not deadlock on the thread of a running loop (#2316)."""

    # Each scenario runs in a subprocess under a timeout, because before the
    # fix it hangs forever.
    _PRELUDE = textwrap.dedent(
        """
        import asyncio, gc, warnings
        import torch
        from tensordict import TensorDict
        from tensordict._ucxx import TensorDictPipe
        warnings.simplefilter("always")

        class Endpoint:
            closed = 0
            async def send(self, buf):
                pass
            async def recv(self, buf):
                pass
            async def close(self):
                self.closed += 1
        """
    )

    def _run(self, code):
        result = subprocess.run(
            [sys.executable, "-c", self._PRELUDE + textwrap.dedent(code)],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, result.stderr
        # No "coroutine ... was never awaited" or errors ignored in __del__.
        assert "never awaited" not in result.stderr, result.stderr
        assert "Exception ignored" not in result.stderr, result.stderr
        return result.stdout

    def test_del_inside_running_loop(self):
        out = self._run(
            """
            async def main():
                ep = Endpoint()
                pipe = TensorDictPipe(ep)
                del pipe
                gc.collect()
                await asyncio.sleep(0)
                print("closed", ep.closed)

            asyncio.run(main())
            """
        )
        assert out == "closed 1\n"

    def test_sync_methods_inside_running_loop_raise(self):
        out = self._run(
            """
            async def main():
                ep = Endpoint()
                pipe = TensorDictPipe(ep)
                td = TensorDict(a=torch.ones(3), batch_size=[3])
                calls = {
                    "aclose": pipe.close,
                    "asend": lambda: pipe.send(td),
                    "arecv": pipe.recv,
                }
                for name, call in calls.items():
                    try:
                        call()
                    except RuntimeError as err:
                        assert f"await pipe.{name}()" in str(err), str(err)
                        print(name, "raised")
                print("closed", ep.closed, pipe.closed)
                await pipe.aclose()

            asyncio.run(main())
            """
        )
        assert out == "aclose raised\nasend raised\narecv raised\nclosed 0 False\n"

    def test_close_without_running_loop(self):
        ep = _FakeEndpoint()
        pipe = TensorDictPipe(ep)
        pipe.close()
        assert pipe.closed

        # From another thread while a loop runs in this one.
        async def main():
            pipe = TensorDictPipe(_FakeEndpoint())
            thread = threading.Thread(target=pipe.close)
            thread.start()
            thread.join(timeout=60)
            assert not thread.is_alive()
            return pipe.closed

        assert asyncio.run(main())


class TestTensorDictSendRecvOverload:
    """Test that td.send(pipe) and td.recv(pipe) dispatch to TensorDictPipe."""

    def test_send_dispatches_to_pipe(self):
        td = TensorDict({"a": torch.randn(4)}, batch_size=[4])
        ep = _FakeEndpoint()
        pipe = TensorDictPipe(ep, consolidated=True)

        td.send(pipe)

        assert len(ep._buffers) > 0
        assert ep._buffers[0] == _NEW_SCHEMA_FLAG

    def test_recv_dispatches_to_pipe(self):
        td_src = TensorDict({"a": torch.randn(4)}, batch_size=[4])
        ep = _FakeEndpoint()
        pipe_send = TensorDictPipe(ep, consolidated=True)
        pipe_send.send(td_src)

        ep._recv_idx = 0
        pipe_recv = TensorDictPipe(ep, consolidated=True)
        td_dest = TensorDict({"a": torch.zeros(4)}, batch_size=[4])
        result = td_dest.recv(pipe_recv)

        torch.testing.assert_close(result["a"], td_src["a"])

    def test_recv_first_message_into_consolidated_td(self):
        # https://github.com/pytorch/tensordict/issues/2244
        ep = _FakeEndpoint()
        pipe_send = TensorDictPipe(ep, consolidated=True)
        pipe_send.send(TensorDict({"a": torch.ones(4)}, batch_size=[4]))
        pipe_send.send(TensorDict({"a": torch.full((4,), 2.0)}, batch_size=[4]))

        pipe_recv = TensorDictPipe(ep, consolidated=True)
        td_dest = TensorDict({"a": torch.zeros(4)}, batch_size=[4]).consolidate(
            metadata=True
        )
        result = td_dest.recv(pipe_recv)
        assert result is td_dest
        torch.testing.assert_close(td_dest["a"], torch.ones(4))

        result = td_dest.recv(pipe_recv)
        assert result is td_dest
        torch.testing.assert_close(td_dest["a"], torch.full((4,), 2.0))

    @pytest.mark.parametrize(
        "td_dest",
        [
            TensorDict({"a": torch.zeros(4)}, batch_size=[4]),
            TensorDict({"a": torch.zeros(5)}, batch_size=[5]).consolidate(
                metadata=True
            ),
            TensorDict({"b": torch.zeros(4)}, batch_size=[4]).consolidate(
                metadata=True
            ),
            TensorDict({"a": torch.zeros(4)}, batch_size=[]).consolidate(metadata=True),
        ],
        ids=["not-consolidated", "shape", "key", "batch-size"],
    )
    def test_recv_first_message_layout_mismatch(self, td_dest):
        ep = _FakeEndpoint()
        pipe_send = TensorDictPipe(ep, consolidated=True)
        td_src = TensorDict({"a": torch.ones(4)}, batch_size=[4])
        pipe_send.send(td_src)

        pipe_recv = TensorDictPipe(ep, consolidated=True)
        expected = td_dest.to_dict()
        result = td_dest.recv(pipe_recv)
        assert result is not td_dest
        torch.testing.assert_close(result["a"], td_src["a"])
        torch.testing.assert_close(td_dest.to_dict(), expected)

    @pytest.mark.parametrize(
        "td_dest",
        [
            TensorDict({"a": torch.zeros(4)}, batch_size=[4]),
            TensorDict({"a": torch.zeros(5)}, batch_size=[5]).consolidate(
                metadata=True
            ),
            # Another key and dtype, with the storage size of the message
            TensorDict(
                {"b": torch.zeros(4, dtype=torch.int32)}, batch_size=[4]
            ).consolidate(metadata=True),
        ],
        ids=["not-consolidated", "shape", "key-dtype"],
    )
    def test_recv_same_schema_layout_mismatch(self, td_dest):
        # https://github.com/pytorch/tensordict/issues/2244
        # https://github.com/pytorch/tensordict/issues/2377
        ep = _FakeEndpoint()
        pipe_send = TensorDictPipe(ep, consolidated=True)
        pipe_send.send(TensorDict({"a": torch.ones(4)}, batch_size=[4]))
        pipe_send.send(TensorDict({"a": torch.full((4,), 2.0)}, batch_size=[4]))

        pipe_recv = TensorDictPipe(ep, consolidated=True)
        expected = td_dest.to_dict()
        first = td_dest.recv(pipe_recv)
        result = td_dest.recv(pipe_recv)
        assert result is first
        assert result is not td_dest
        torch.testing.assert_close(result["a"], torch.full((4,), 2.0))
        torch.testing.assert_close(td_dest.to_dict(), expected)


# ---------------------------------------------------------------------------
# Integration tests — require ucxx
# ---------------------------------------------------------------------------


@requires_ucxx
class TestUCXXIntegration:
    """End-to-end tests using real UCXX endpoints on localhost."""

    @pytest.fixture
    def port(self):
        import random

        return random.randint(20000, 40000)

    def test_pipe_connect_listen(self, port):
        td = TensorDict(
            {"x": torch.randn(10), "y": torch.randn(5, 3)},
            batch_size=[],
        )

        async def _run():
            listener_pipe = await TensorDictPipe.listen(port)
            client_pipe = await TensorDictPipe.connect("127.0.0.1", port)

            await client_pipe.asend(td)
            td_recv = await listener_pipe.arecv()

            torch.testing.assert_close(td_recv["x"], td["x"])
            torch.testing.assert_close(td_recv["y"], td["y"])

            await client_pipe.aclose()
            await listener_pipe.aclose()

        asyncio.run(_run())

    def test_pipe_steady_state(self, port):
        async def _run():
            listener_pipe = await TensorDictPipe.listen(port)
            client_pipe = await TensorDictPipe.connect("127.0.0.1", port)

            td1 = TensorDict({"v": torch.ones(8)}, batch_size=[8])
            await client_pipe.asend(td1)
            td_recv = await listener_pipe.arecv()
            torch.testing.assert_close(td_recv["v"], torch.ones(8))

            td2 = TensorDict({"v": torch.full((8,), 7.0)}, batch_size=[8])
            await client_pipe.asend(td2)
            td_recv2 = await listener_pipe.arecv()

            assert td_recv2 is td_recv
            torch.testing.assert_close(td_recv["v"], torch.full((8,), 7.0))

            await client_pipe.aclose()
            await listener_pipe.aclose()

        asyncio.run(_run())

    def test_context_manager(self, port):
        td = TensorDict({"a": torch.tensor([1.0, 2.0, 3.0])}, batch_size=[3])

        async def _run():
            async with await TensorDictPipe.listen(port) as listener:
                async with await TensorDictPipe.connect("127.0.0.1", port) as client:
                    await client.asend(td)
                    td_recv = await listener.arecv()
                    torch.testing.assert_close(td_recv["a"], td["a"])

        asyncio.run(_run())

    def test_td_send_recv_overload(self, port):
        td = TensorDict({"a": torch.randn(4)}, batch_size=[4])

        async def _run():
            listener_pipe = await TensorDictPipe.listen(port)
            client_pipe = await TensorDictPipe.connect("127.0.0.1", port)

            await td.asend(client_pipe)
            td_recv = await TensorDict({}, []).arecv(listener_pipe)

            torch.testing.assert_close(td_recv["a"], td["a"])

            await client_pipe.aclose()
            await listener_pipe.aclose()

        asyncio.run(_run())


@requires_ucxx
@requires_cuda
class TestUCXXCUDA:
    """Tests for CUDA-direct transfers via UCXX."""

    @pytest.fixture
    def port(self):
        import random

        return random.randint(40000, 50000)

    def test_cuda_tensor_roundtrip(self, port):
        td = TensorDict(
            {"a": torch.randn(10, device="cuda"), "b": torch.ones(5, device="cuda")},
            batch_size=[],
        )

        async def _run():
            listener_pipe = await TensorDictPipe.listen(port)
            client_pipe = await TensorDictPipe.connect("127.0.0.1", port)

            await client_pipe.asend(td)
            td_recv = await listener_pipe.arecv(device="cuda")

            assert td_recv["a"].is_cuda
            assert td_recv["b"].is_cuda
            torch.testing.assert_close(td_recv["a"], td["a"])
            torch.testing.assert_close(td_recv["b"], td["b"])

            await client_pipe.aclose()
            await listener_pipe.aclose()

        asyncio.run(_run())


@requires_ucxx
class TestTensorDictServer:
    @pytest.fixture
    def port(self):
        import random

        return random.randint(50000, 60000)

    def test_serve_multiple_clients(self, port):
        from tensordict._ucxx import TensorDictServer

        async def _run():
            server = TensorDictServer(port)
            received = []

            async def handler(pipe):
                td = await pipe.arecv()
                received.append(td)

            server_task = asyncio.create_task(server.serve(handler))

            pipe1 = await TensorDictPipe.connect("127.0.0.1", port)
            pipe2 = await TensorDictPipe.connect("127.0.0.1", port)

            td1 = TensorDict({"a": torch.tensor([1.0])}, batch_size=[])
            td2 = TensorDict({"a": torch.tensor([2.0])}, batch_size=[])

            await pipe1.asend(td1)
            await pipe2.asend(td2)

            await asyncio.sleep(1.0)

            assert len(received) == 2
            values = sorted([r["a"].item() for r in received])
            assert values == [1.0, 2.0]

            server.close()
            server_task.cancel()

            await pipe1.aclose()
            await pipe2.aclose()

        asyncio.run(_run())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
