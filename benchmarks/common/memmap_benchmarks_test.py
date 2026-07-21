# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import argparse
import pathlib
import time
import uuid
from pathlib import Path

import pytest
import torch

from tensordict import MemoryMappedTensor, TensorDict
from torch import nn

try:
    import zarr

    _has_zarr = int(zarr.__version__.split(".")[0]) >= 3
except ImportError:
    _has_zarr = False


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


def test_serialize_model_archive(benchmark, tmpdir, pause_when_exit):
    """Tests efficiency of saving weights as a single-file memmap archive."""
    has_cuda = torch.cuda.device_count()
    with torch.device("cuda" if has_cuda else "cpu"):
        t = nn.Transformer()

    def func(t=t, tmpdir=tmpdir):
        TensorDict.from_module(t).save(Path(tmpdir) / "model.tdz", num_threads=32)

    benchmark(func)
    del t


def test_load_archive(benchmark, tmpdir, pause_when_exit):
    """Tests efficiency of loading a single-file memmap archive."""
    t = nn.Transformer()
    path = Path(tmpdir) / "model.tdz"
    TensorDict.from_module(t).save(path, num_threads=32)

    def func(path=path):
        TensorDict.load_memmap(path)

    benchmark(func)
    del t


@pytest.mark.skipif(not _has_zarr, reason="zarr>=3.0 not found.")
def test_serialize_model_zarr(benchmark, tmpdir, pause_when_exit):
    """Tests efficiency of saving weights as a zarr store."""
    import shutil

    has_cuda = torch.cuda.device_count()
    with torch.device("cuda" if has_cuda else "cpu"):
        t = nn.Transformer()
    path = Path(tmpdir) / "model.zarr"

    def func(t=t, path=path):
        if path.exists():
            shutil.rmtree(path)
        TensorDict.from_module(t).data.to_zarr(path)

    benchmark(func)
    del t


@pytest.mark.skipif(not _has_zarr, reason="zarr>=3.0 not found.")
def test_load_zarr(benchmark, tmpdir, pause_when_exit):
    """Tests efficiency of loading and materializing a zarr store."""
    t = nn.Transformer()
    path = Path(tmpdir) / "model.zarr"
    TensorDict.from_module(t).data.to_zarr(path)

    def func(path=path):
        TensorDict.from_zarr(path).to_tensordict()

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


@pytest.fixture(params=["few_large", "many_small"])
def td_consolidate(request):
    if request.param == "few_large":
        # 8 leaves x 8 MB
        td = TensorDict({f"k{i}": torch.randn(2 * 2**20) for i in range(8)})
    else:
        # 1024 leaves x 64 KB
        td = TensorDict({f"k{i}": torch.randn(2**14) for i in range(1024)})
    yield td
    del td


@pytest.mark.parametrize("num_threads", [0, 8])
def test_consolidate(benchmark, td_consolidate, num_threads):
    """Tests efficiency of consolidating a tensordict in memory."""
    benchmark(td_consolidate.consolidate, num_threads=num_threads)


@pytest.mark.parametrize("num_threads", [0, 8])
def test_consolidate_to_file(benchmark, td_consolidate, num_threads, tmpdir):
    """Tests efficiency of consolidating a tensordict in a memory-mapped file."""
    count = [0]

    def consolidate():
        count[0] += 1
        td_consolidate.consolidate(
            filename=Path(tmpdir) / f"file{count[0]}.td", num_threads=num_threads
        )

    benchmark(consolidate)


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
