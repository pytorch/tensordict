import argparse
import pathlib
import time
import uuid
from pathlib import Path

import pytest
import torch

from tensordict import MemoryMappedTensor, TensorDict
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


@pytest.fixture(params=[torch.device("cpu")])
def memmap_tensor(request):
    return MemoryMappedTensor.zeros((3, 4, 5))


@pytest.fixture
def td_memmap():
    return TensorDict(
        {str(i): torch.zeros(3, 40) + i for i in range(30)}, [3, 40]
    ).memmap_()


@pytest.mark.parametrize("device", [torch.device("cpu")])
def test_creation(benchmark, device):
    benchmark(MemoryMappedTensor.empty, (3, 4, 5))


def test_creation_from_tensor(benchmark, tensor):
    benchmark(
        MemoryMappedTensor.from_tensor,
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


@pytest.fixture(scope="function")
def pause_when_exit():
    yield None
    time.sleep(0.5)


def test_serialize_model(benchmark, tmpdir, pause_when_exit):
    """Tests efficiency of saving weights as memmap tensors, including TD construction."""
    has_cuda = torch.cuda.device_count()
    with torch.device("cuda" if has_cuda else "cpu"):
        t = nn.Transformer()

    def func(t=t, tmpdir=tmpdir):
        TensorDict.from_module(t).memmap(tmpdir, num_threads=32)

    benchmark(func)
    del t


def test_serialize_model_pickle(benchmark, tmpdir, pause_when_exit):
    """Tests efficiency of pickling a model state-dict, including state-dict construction."""
    has_cuda = torch.cuda.device_count()
    with torch.device("cuda" if has_cuda else "cpu"):
        t = nn.Transformer()
    path = Path(tmpdir) / "file.t"

    def func(t=t, path=path):
        torch.save(t.state_dict(), path)

    benchmark(func)
    del t


def test_serialize_weights(benchmark, tmpdir, pause_when_exit):
    """Tests efficiency of saving weights as memmap tensors."""
    has_cuda = torch.cuda.device_count()
    with torch.device("cuda" if has_cuda else "cpu"):
        t = nn.Transformer()

    weights = TensorDict.from_module(t)

    def func(weights=weights):
        weights.memmap(tmpdir, num_threads=32)

    benchmark(func)
    del t, weights


def test_serialize_weights_returnearly(benchmark, tmpdir, pause_when_exit):
    """Tests efficiency of saving weights as memmap tensors, before writing is completed."""
    has_cuda = torch.cuda.device_count()
    with torch.device("cuda" if has_cuda else "cpu"):
        t = nn.Transformer()

    datapath = pathlib.Path(tmpdir)
    weights = TensorDict.from_module(t)

    def func(weights=weights, datapath=datapath):
        weights.memmap(datapath / f"{uuid.uuid1()}", num_threads=32, return_early=True)

    benchmark(func)
    del t, weights


def test_serialize_weights_pickle(benchmark, tmpdir, pause_when_exit):
    """Tests efficiency of pickling a model state-dict."""
    has_cuda = torch.cuda.device_count()
    with torch.device("cuda" if has_cuda else "cpu"):
        t = nn.Transformer()

    path = Path(tmpdir) / "file.t"
    weights = t.state_dict()

    def func(path=path, weights=weights):
        torch.save(weights, path)

    benchmark(func)
    del t, weights


def test_serialize_weights_filesystem(benchmark, pause_when_exit):
    """Tests efficiency of saving weights as memmap tensors."""
    has_cuda = torch.cuda.device_count()
    if has_cuda:
        pytest.skip(
            "Multithreaded saving on filesystem with models on CUDA. "
            "These should be first cast on CPU for safety."
        )
    with torch.device("cuda" if has_cuda else "cpu"):
        t = nn.Transformer()

    weights = TensorDict.from_module(t)

    def func(weights=weights):
        weights.memmap(num_threads=32)

    benchmark(func)
    del t, weights


def test_serialize_model_filesystem(benchmark, pause_when_exit):
    """Tests efficiency of saving weights as memmap tensors in file system, including TD construction."""
    has_cuda = torch.cuda.device_count()
    if has_cuda:
        pytest.skip(
            "Multithreaded saving on filesystem with models on CUDA. "
            "These should be first cast on CPU for safety."
        )
    with torch.device("cuda" if has_cuda else "cpu"):
        t = nn.Transformer()

    def func(t=t):
        TensorDict.from_module(t).memmap(num_threads=32)

    benchmark(func)
    del t


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
