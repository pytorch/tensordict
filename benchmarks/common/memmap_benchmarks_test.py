import pytest
import torch

from tensordict import MemmapTensor


def get_available_devices():
    devices = [torch.device("cpu")]
    n_cuda = torch.cuda.device_count()
    if n_cuda > 0:
        for i in range(n_cuda):
            devices += [torch.device(f"cuda:{i}")]
    return devices


@pytest.fixture
def tensor():
    return torch.zeros(3, 4, 5)


@pytest.fixture(params=get_available_devices())
def memmap_tensor(request):
    return MemmapTensor(3, 4, 5, device=request.param)


@pytest.mark.parametrize("device", get_available_devices())
def test_creation(benchmark, device):
    benchmark.pedantic(
        MemmapTensor, args=(3, 4, 5), kwargs={"device": device}, rounds=10, iterations=1
    )


def test_creation_from_tensor(benchmark, tensor):
    benchmark.pedantic(
        MemmapTensor.from_tensor, args=(tensor,), rounds=10, iterations=1
    )


def test_add_one(benchmark, memmap_tensor):
    benchmark.pedantic(lambda: memmap_tensor + 1, rounds=100, iterations=100)


def test_contiguous(benchmark, memmap_tensor):
    benchmark.pedantic(lambda: memmap_tensor.contiguous(), rounds=100, iterations=100)


def test_stack(benchmark, memmap_tensor):
    benchmark.pedantic(
        torch.stack, args=([memmap_tensor] * 2, 0), rounds=10, iterations=1
    )
