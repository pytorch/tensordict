"""Distributed TensorDict benchmarks: leaf-by-leaf vs consolidated transport.

Requires NCCL backend with CUDA tensors and a multi-rank environment
(typically launched via ``torchrun``).  When the distributed environment is
not initialised the tests are skipped automatically.

Usage (2-node cluster via torchrun)::

    torchrun --nproc_per_node=1 --nnodes=2 --node_rank=0 \
        --master_addr=$MASTER_ADDR --master_port=29500 \
        -m pytest benchmarks/test_distributed_benchmarks.py -v
"""

import pytest
import torch
import torch.distributed as dist

from tensordict import TensorDict

_DIST_AVAILABLE = dist.is_available()
_CUDA_AVAILABLE = torch.cuda.is_available()

pytestmark = pytest.mark.skipif(
    not (_DIST_AVAILABLE and dist.is_initialized()),
    reason="Requires an initialised torch.distributed process group (launch with torchrun)",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_td(num_tensors, tensor_size=1024, dtype=torch.float32, device="cuda"):
    d = {
        f"t{i}": torch.randn(tensor_size, dtype=dtype, device=device)
        for i in range(num_tensors)
    }
    return TensorDict(d, batch_size=[], device=device)


def total_bytes(td):
    total = 0
    for v in td.values(True, True):
        if isinstance(v, torch.Tensor):
            total += v.numel() * v.element_size()
    return total


def _sync():
    """Barrier + CUDA sync for accurate timing."""
    if _CUDA_AVAILABLE:
        torch.cuda.synchronize()
    dist.barrier()


# ---------------------------------------------------------------------------
# Per-operation micro-benchmarks
# ---------------------------------------------------------------------------

N_ITERS = 20
WARMUP = 3
CONFIGS = [
    (10, 1024),
    (50, 1024),
    (100, 1024),
    (500, 1024),
    (10, 1024 * 1024),
    (50, 1024 * 1024),
]


@pytest.fixture()
def rank():
    return dist.get_rank()


@pytest.fixture()
def device():
    import os

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if _CUDA_AVAILABLE:
        torch.cuda.set_device(local_rank)
        return torch.device(f"cuda:{local_rank}")
    return torch.device("cpu")


@pytest.mark.skipif(
    _DIST_AVAILABLE and dist.is_initialized() and dist.get_world_size() < 2,
    reason="Requires at least 2 ranks",
)
@pytest.mark.parametrize("num_tensors,tensor_size", CONFIGS)
class TestPointToPoint:
    def test_leaf_send_recv(self, num_tensors, tensor_size, rank, device, benchmark):
        td = make_td(num_tensors, tensor_size, device=device)
        td_recv = make_td(num_tensors, tensor_size, device=device)
        td_recv.zero_()

        work_td = td if rank == 0 else td_recv

        for _ in range(WARMUP):
            if rank == 0:
                work_td.send(dst=1)
            else:
                work_td.recv(src=0)

        def _bench():
            if rank == 0:
                work_td.send(dst=1)
            else:
                work_td.recv(src=0)

        benchmark.pedantic(_bench, rounds=N_ITERS, warmup_rounds=0)

    def test_consolidated_send_recv(
        self, num_tensors, tensor_size, rank, device, benchmark
    ):
        td = make_td(num_tensors, tensor_size, device=device)

        if rank == 0:
            td.init_remote(dst=1)
            td_c = td.consolidate(metadata=True)
        else:
            td_c = TensorDict.from_remote_init(src=0, device=device)

        for _ in range(WARMUP):
            if rank == 0:
                td_c.send(dst=1, consolidated=True)
            else:
                td_c.recv(src=0, consolidated=True)

        def _bench():
            if rank == 0:
                td_c.send(dst=1, consolidated=True)
            else:
                td_c.recv(src=0, consolidated=True)

        benchmark.pedantic(_bench, rounds=N_ITERS, warmup_rounds=0)

    def test_init_remote(self, num_tensors, tensor_size, rank, device, benchmark):
        td = make_td(num_tensors, tensor_size, device=device)

        for _ in range(WARMUP):
            if rank == 0:
                td.init_remote(dst=1)
            else:
                TensorDict.from_remote_init(src=0, device=device)

        def _bench():
            if rank == 0:
                td.init_remote(dst=1)
            else:
                TensorDict.from_remote_init(src=0, device=device)

        benchmark.pedantic(_bench, rounds=N_ITERS, warmup_rounds=0)


@pytest.mark.parametrize("num_tensors,tensor_size", CONFIGS)
class TestCollectives:
    def test_broadcast(self, num_tensors, tensor_size, rank, device, benchmark):
        td = (
            make_td(num_tensors, tensor_size, device=device)
            if rank == 0
            else TensorDict({}, device=device)
        )

        for _ in range(WARMUP):
            td.broadcast(src=0)

        benchmark.pedantic(lambda: td.broadcast(src=0), rounds=N_ITERS, warmup_rounds=0)

    def test_all_reduce(self, num_tensors, tensor_size, rank, device, benchmark):
        td = make_td(num_tensors, tensor_size, device=device)

        for _ in range(WARMUP):
            td.all_reduce()

        benchmark.pedantic(lambda: td.all_reduce(), rounds=N_ITERS, warmup_rounds=0)
