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

import pytest
import torch

from tensordict import TensorDict

# Protocol helpers and classes are importable even without ucxx installed
from tensordict._ucxx import (
    _compute_total_bytes,
    _metadata_hash,
    _NEW_SCHEMA_FLAG,
    _SAME_SCHEMA_FLAG,
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
        meta = {"leaves": {"a": ("torch.float32", [2, 3], 0, 24, 0)}, "cls": "TensorDict", "non_tensors": {}, "cls_metadata": {"batch_size": [2], "device": None, "names": None}}
        h1 = _metadata_hash(meta)
        h2 = _metadata_hash(meta)
        assert h1 == h2

    def test_different_metadata_different_hash(self):
        meta1 = {"leaves": {"a": ("torch.float32", [2, 3], 0, 24, 0)}, "cls": "TensorDict", "non_tensors": {}, "cls_metadata": {"batch_size": [2], "device": None, "names": None}}
        meta2 = {"leaves": {"b": ("torch.float64", [4], 0, 32, 0)}, "cls": "TensorDict", "non_tensors": {}, "cls_metadata": {"batch_size": [4], "device": None, "names": None}}
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
        import numpy as np

        if isinstance(buf, np.ndarray):
            self._buffers.append(bytes(buf))
        elif isinstance(buf, torch.Tensor):
            self._buffers.append(bytes(buf.cpu().numpy()))
        else:
            self._buffers.append(bytes(buf))

    async def recv(self, buf):
        import numpy as np

        data = self._buffers[self._recv_idx]
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
