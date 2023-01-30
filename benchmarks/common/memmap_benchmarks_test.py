import pytest
import torch

from _utils_internal import get_available_devices
from tensordict import MemmapTensor


@pytest.fixture
def tensor():
    return torch.zeros(3, 4, 5)


@pytest.fixture(params=get_available_devices())
def memmap_tensor(request):
    return MemmapTensor(3, 4, 5, device=request.param)


@pytest.mark.parametrize("device", get_available_devices())
def test_creation(benchmark, device):
    benchmark.pedantic(
        MemmapTensor, args=(3, 4, 5), kwargs={"device": device}, iterations=10
    )


def test_creation_from_tensor(benchmark, tensor):
    benchmark.pedantic(MemmapTensor.from_tensor, args=(tensor,), iterations=10)


def test_add_one(benchmark, memmap_tensor):
    benchmark.pedantic(lambda: memmap_tensor + 1, iterations=10_000)


def test_contiguous(benchmark, memmap_tensor):
    benchmark.pedantic(lambda: memmap_tensor.contiguous(), iterations=10_000)


def test_stack(benchmark, memmap_tensor):
    benchmark.pedantic(torch.stack, args=([memmap_tensor] * 2, 0), iterations=10)
