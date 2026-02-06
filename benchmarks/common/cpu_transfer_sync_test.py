# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Benchmark CPU transfer sync strategies for TensorDict.to('cpu').

Three strategies are compared:
  1. Blocking: non_blocking=False (each tensor.to() blocks individually).
  2. Global sync: non_blocking=True per tensor, then torch.cuda.synchronize().
  3. Event sync: non_blocking=True per tensor, then event.record() + event.synchronize().

Strategy (3) is the current default (.to("cpu") with non_blocking=None).
Strategy (2) was the previous default before the event-based sync change.
"""
from __future__ import annotations

import pytest
import torch

from tensordict import TensorDict
from tensordict.base import _sync_cuda_transfer
from tensordict.utils import logger as tensordict_logger


@pytest.fixture
def td_cuda_many_tensors():
    """TensorDict on CUDA with many small tensors to stress sync overhead."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    num_keys = 1000
    shape = (64, 64)
    td = TensorDict(
        {
            str(i): torch.randn(shape, device="cuda", dtype=torch.float32)
            for i in range(num_keys)
        },
        batch_size=[],
    )
    tensordict_logger.info(
        f"td_cuda_many_tensors: {num_keys} keys, shape={shape}, "
        f"size {td.bytes() / 1024 / 1024:.2f} Mb"
    )
    return td


def _to_cpu_global_sync(td):
    """Manual non_blocking + torch.cuda.synchronize() (old behaviour)."""
    td_cpu = td.to("cpu", non_blocking=True)
    torch.cuda.synchronize()
    return td_cpu


def _to_cpu_event_sync(td):
    """Manual non_blocking + event-based sync (new behaviour)."""
    td_cpu = td.to("cpu", non_blocking=True)
    _sync_cuda_transfer()
    return td_cpu


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device found")
class TestCpuTransferSync:
    """Compare blocking, global sync, and event sync .to('cpu')."""

    def test_to_cpu_blocking(self, benchmark, td_cuda_many_tensors):
        """Each tensor blocks on transfer (non_blocking=False)."""
        td = td_cuda_many_tensors

        def run():
            return td.to("cpu", non_blocking=False)

        for _ in range(3):
            run()
        benchmark(run)

    def test_to_cpu_global_sync(self, benchmark, td_cuda_many_tensors):
        """non_blocking copies + torch.cuda.synchronize() (old default)."""
        td = td_cuda_many_tensors

        def run():
            return _to_cpu_global_sync(td)

        for _ in range(3):
            run()
        benchmark(run)

    def test_to_cpu_event_sync(self, benchmark, td_cuda_many_tensors):
        """non_blocking copies + event sync (new default)."""
        td = td_cuda_many_tensors

        def run():
            return _to_cpu_event_sync(td)

        for _ in range(3):
            run()
        benchmark(run)

    def test_to_cpu_default(self, benchmark, td_cuda_many_tensors):
        """Default .to('cpu') (non_blocking=None, uses event sync internally)."""
        td = td_cuda_many_tensors

        def run():
            return td.to("cpu")

        for _ in range(3):
            run()
        benchmark(run)
