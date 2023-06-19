import argparse

import pytest
import torch

from tensordict import MemmapTensor, TensorDict


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


@pytest.fixture
def td_memmap():
    return TensorDict(
        {str(i): torch.zeros(3, 40) + i for i in range(30)}, [3, 40]
    ).memmap_()


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


def test_memmaptd_index(benchmark, td_memmap):
    benchmark.pedantic(
        lambda td: td[0],
        args=(td_memmap,),
        iterations=1,
        rounds=1000,
    )


def test_memmaptd_index_astensor(benchmark, td_memmap):
    benchmark.pedantic(
        lambda td: td[0].as_tensor(),
        args=(td_memmap,),
        iterations=1,
        rounds=1000,
    )


def test_memmaptd_index_op(benchmark, td_memmap):
    benchmark.pedantic(
        lambda td: td[0].apply(lambda x: x + 1),
        args=(td_memmap,),
        iterations=1,
        rounds=1000,
    )


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
