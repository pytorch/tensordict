import argparse
from pathlib import Path

import pytest
import torch

from tensordict import MemmapTensor, TensorDict
from torch import nn


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
    benchmark(MemmapTensor, 3, 4, 5, device=device)


def test_creation_from_tensor(benchmark, tensor):
    benchmark(
        MemmapTensor.from_tensor,
        tensor,
    )


def test_add_one(benchmark, memmap_tensor):
    benchmark(lambda: memmap_tensor + 1)


def test_contiguous(benchmark, memmap_tensor):
    benchmark(lambda: memmap_tensor.contiguous())


def test_stack(benchmark, memmap_tensor):
    benchmark(torch.stack, [memmap_tensor] * 2, 0)


def test_memmaptd_index(benchmark, td_memmap):
    benchmark(
        lambda td: td[0],
        td_memmap,
    )


def test_memmaptd_index_astensor(benchmark, td_memmap):
    benchmark(
        lambda td: td[0].as_tensor(),
        td_memmap,
    )


def test_memmaptd_index_op(benchmark, td_memmap):
    benchmark(
        lambda td: td[0].apply(lambda x: x + 1),
        td_memmap,
    )


def test_serialize_model(benchmark, tmpdir):
    """Tests efficiency of saving weights as memmap tensors, including TD construction."""
    with torch.device("cuda" if torch.cuda.device_count() else "cpu"):
        t = nn.Transformer()
    benchmark(lambda: TensorDict.from_module(t).memmap(tmpdir, num_threads=32))


def test_serialize_model_filesystem(benchmark):
    """Tests efficiency of saving weights as memmap tensors in file system, including TD construction."""
    with torch.device("cuda" if torch.cuda.device_count() else "cpu"):
        t = nn.Transformer()
    benchmark(lambda: TensorDict.from_module(t).memmap(num_threads=32))


def test_serialize_model_pickle(benchmark, tmpdir):
    """Tests efficiency of pickling a model state-dict, including state-dict construction."""
    with torch.device("cuda" if torch.cuda.device_count() else "cpu"):
        t = nn.Transformer()
    path = Path(tmpdir) / "file.t"
    benchmark(lambda: torch.save(t.state_dict(), path))


def test_serialize_weights(benchmark, tmpdir):
    """Tests efficiency of saving weights as memmap tensors."""
    with torch.device("cuda" if torch.cuda.device_count() else "cpu"):
        t = nn.Transformer()

    weights = TensorDict.from_module(t)
    benchmark(lambda: weights.memmap(tmpdir, num_threads=32))


def test_serialize_weights_filesystem(benchmark):
    """Tests efficiency of saving weights as memmap tensors."""
    with torch.device("cuda" if torch.cuda.device_count() else "cpu"):
        t = nn.Transformer()

    weights = TensorDict.from_module(t)
    benchmark(lambda: weights.memmap(num_threads=32))


def test_serialize_weights_pickle(benchmark, tmpdir):
    """Tests efficiency of pickling a model state-dict."""
    with torch.device("cuda" if torch.cuda.device_count() else "cpu"):
        t = nn.Transformer()
    path = Path(tmpdir) / "file.t"
    weights = t.state_dict()
    benchmark(lambda: torch.save(weights, path))


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
