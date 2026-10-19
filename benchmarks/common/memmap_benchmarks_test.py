# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import argparse
import pathlib
import shutil
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


# The serialization benchmarks write a model's weights in every round, and
# pytest-benchmark runs as many rounds as fit in a second (more under
# --benchmark-min-rounds). With nn.Transformer() (176 MB), 64 MB consolidations,
# and a new directory or file per round in some of them, one run of this file
# wrote about 40 GB. They use a model with the same 184 tensors at 1/16 of the
# size, a quarter of the consolidated data, and a fixed number of rounds; the
# rounds that write a new directory or file delete the previous one.
_ROUNDS = 5


def _transformer():
    """The 184 tensors of nn.Transformer(), at 1/16 of its 176 MB."""
    return nn.Transformer(d_model=128, nhead=8, dim_feedforward=512)


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


@pytest.mark.gpu
def test_serialize_model(benchmark, tmpdir, pause_when_exit):
    """Tests efficiency of saving weights as memmap tensors, including TD construction."""
    has_cuda = torch.cuda.device_count()
    with torch.device("cuda" if has_cuda else "cpu"):
        t = _transformer()

    def func(t=t, tmpdir=tmpdir):
        TensorDict.from_module(t).memmap(tmpdir, num_threads=32)

    benchmark.pedantic(func, rounds=_ROUNDS, warmup_rounds=1)
    del t


@pytest.mark.gpu
def test_serialize_model_archive(benchmark, tmpdir, pause_when_exit):
    """Tests efficiency of saving weights as a single-file memmap archive."""
    has_cuda = torch.cuda.device_count()
    with torch.device("cuda" if has_cuda else "cpu"):
        t = _transformer()

    def func(t=t, tmpdir=tmpdir):
        TensorDict.from_module(t).save(Path(tmpdir) / "model.tdz", num_threads=32)

    benchmark.pedantic(func, rounds=_ROUNDS, warmup_rounds=1)
    del t


def test_load_memmap(benchmark, tmpdir, pause_when_exit):
    """Tests efficiency of loading memmap tensors saved in a directory."""
    t = _transformer()
    TensorDict.from_module(t).memmap(tmpdir, num_threads=32)

    def func(path=tmpdir):
        TensorDict.load_memmap(path)

    benchmark(func)
    del t


def test_load_archive(benchmark, tmpdir, pause_when_exit):
    """Tests efficiency of loading a single-file memmap archive."""
    t = _transformer()
    path = Path(tmpdir) / "model.tdz"
    TensorDict.from_module(t).save(path, num_threads=32)

    def func(path=path):
        TensorDict.load_memmap(path)

    benchmark(func)
    del t


@pytest.mark.gpu
@pytest.mark.skipif(not _has_zarr, reason="zarr>=3.0 not found.")
def test_serialize_model_zarr(benchmark, tmpdir, pause_when_exit):
    """Tests efficiency of saving weights as a zarr store."""
    import shutil

    has_cuda = torch.cuda.device_count()
    with torch.device("cuda" if has_cuda else "cpu"):
        t = _transformer()
    path = Path(tmpdir) / "model.zarr"

    def setup(path=path):
        # outside the timing
        if path.exists():
            shutil.rmtree(path)

    def func(t=t, path=path):
        TensorDict.from_module(t).data.to_zarr(path)

    benchmark.pedantic(func, setup=setup, rounds=_ROUNDS, warmup_rounds=1)
    del t


@pytest.mark.skipif(not _has_zarr, reason="zarr>=3.0 not found.")
def test_load_zarr(benchmark, tmpdir, pause_when_exit):
    """Tests efficiency of loading and materializing a zarr store."""
    t = _transformer()
    path = Path(tmpdir) / "model.zarr"
    TensorDict.from_module(t).data.to_zarr(path)

    def func(path=path):
        TensorDict.from_zarr(path).to_tensordict()

    benchmark(func)
    del t


@pytest.mark.gpu
def test_serialize_model_pickle(benchmark, tmpdir, pause_when_exit):
    """Tests efficiency of pickling a model state-dict, including state-dict construction."""
    has_cuda = torch.cuda.device_count()
    with torch.device("cuda" if has_cuda else "cpu"):
        t = _transformer()
    path = Path(tmpdir) / "file.t"

    def func(t=t, path=path):
        torch.save(t.state_dict(), path)

    benchmark.pedantic(func, rounds=_ROUNDS, warmup_rounds=1)
    del t


@pytest.mark.gpu
def test_serialize_weights(benchmark, tmpdir, pause_when_exit):
    """Tests efficiency of saving weights as memmap tensors."""
    has_cuda = torch.cuda.device_count()
    with torch.device("cuda" if has_cuda else "cpu"):
        t = _transformer()

    weights = TensorDict.from_module(t)

    def func(weights=weights):
        weights.memmap(tmpdir, num_threads=32)

    benchmark.pedantic(func, rounds=_ROUNDS, warmup_rounds=1)
    del t, weights


@pytest.mark.gpu
def test_serialize_weights_returnearly(benchmark, tmpdir, pause_when_exit):
    """Tests efficiency of saving weights as memmap tensors, before writing is completed."""
    has_cuda = torch.cuda.device_count()
    with torch.device("cuda" if has_cuda else "cpu"):
        t = _transformer()

    datapath = pathlib.Path(tmpdir)
    weights = TensorDict.from_module(t)
    pending = []

    def setup():
        # outside the timing: let the previous round finish writing its new
        # directory, then delete it
        while pending:
            future, path = pending.pop()
            future.result()
            shutil.rmtree(path)

    def func(weights=weights, datapath=datapath):
        path = datapath / f"{uuid.uuid1()}"
        pending.append((weights.memmap(path, num_threads=32, return_early=True), path))

    benchmark.pedantic(func, setup=setup, rounds=_ROUNDS, warmup_rounds=1)
    setup()
    del t, weights


@pytest.mark.gpu
def test_serialize_weights_pickle(benchmark, tmpdir, pause_when_exit):
    """Tests efficiency of pickling a model state-dict."""
    has_cuda = torch.cuda.device_count()
    with torch.device("cuda" if has_cuda else "cpu"):
        t = _transformer()

    path = Path(tmpdir) / "file.t"
    weights = t.state_dict()

    def func(path=path, weights=weights):
        torch.save(weights, path)

    benchmark.pedantic(func, rounds=_ROUNDS, warmup_rounds=1)
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
        t = _transformer()

    weights = TensorDict.from_module(t)

    def func(weights=weights):
        weights.memmap(num_threads=32)

    benchmark.pedantic(func, rounds=_ROUNDS, warmup_rounds=1)
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
        t = _transformer()

    def func(t=t):
        TensorDict.from_module(t).memmap(num_threads=32)

    benchmark.pedantic(func, rounds=_ROUNDS, warmup_rounds=1)
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


@pytest.fixture(params=["few_large", "many_small"])
def td_consolidate_file(request):
    # A quarter of td_consolidate (16 MB), as every round writes it to a file
    if request.param == "few_large":
        # 8 leaves x 2 MB
        td = TensorDict({f"k{i}": torch.randn(2**19) for i in range(8)})
    else:
        # 1024 leaves x 16 KB
        td = TensorDict({f"k{i}": torch.randn(2**12) for i in range(1024)})
    yield td
    del td


@pytest.mark.parametrize("num_threads", [0, 8])
def test_consolidate(benchmark, td_consolidate, num_threads):
    """Tests efficiency of consolidating a tensordict in memory."""
    benchmark(td_consolidate.consolidate, num_threads=num_threads)


@pytest.mark.parametrize("num_threads", [0, 8])
def test_consolidate_to_file(benchmark, td_consolidate_file, num_threads, tmpdir):
    """Tests efficiency of consolidating a tensordict in a memory-mapped file."""
    filename = Path(tmpdir) / "file.td"

    def setup():
        # outside the timing: each round writes the file anew
        filename.unlink(missing_ok=True)

    def consolidate():
        td_consolidate_file.consolidate(filename=filename, num_threads=num_threads)

    benchmark.pedantic(consolidate, setup=setup, rounds=_ROUNDS, warmup_rounds=1)


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
