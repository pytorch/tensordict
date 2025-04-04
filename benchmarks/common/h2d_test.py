# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import argparse
import time
from typing import Any

import pytest
import torch
from packaging import version

from tensordict import tensorclass, TensorDict
from tensordict.utils import logger as tensordict_logger

TORCH_VERSION = version.parse(version.parse(torch.__version__).base_version)


@tensorclass
class NJT:
    _values: torch.Tensor
    _offsets: torch.Tensor
    _lengths: torch.Tensor
    njt_shape: Any = None

    @classmethod
    def from_njt(cls, njt_tensor):
        return cls(
            _values=njt_tensor._values,
            _offsets=njt_tensor._offsets,
            _lengths=njt_tensor._lengths,
            njt_shape=njt_tensor.size(0),
        ).clone()


@pytest.fixture(autouse=True, scope="function")
def empty_compiler_cache():
    torch.compiler.reset()
    yield


def _make_njt():
    lengths = torch.arange(24, 1, -1)
    offsets = torch.cat([lengths[:1] * 0, lengths]).cumsum(0)
    return torch.nested.nested_tensor_from_jagged(
        torch.arange(78, dtype=torch.float), offsets=offsets, lengths=lengths
    )


def _njt_td():
    return TensorDict(
        # {str(i): {str(j): _make_njt() for j in range(32)} for i in range(32)},
        {str(i): _make_njt() for i in range(32)},
        device="cpu",
    )


@pytest.fixture
def njt_td():
    return _njt_td()


@pytest.fixture
def td():
    njtd = _njt_td()
    for k0, v0 in njtd.items():
        njtd[k0] = NJT.from_njt(v0)
        # for k1, v1 in v0.items():
        #     njtd[k0, k1] = NJT.from_njt(v1)
    return njtd


@pytest.fixture
def default_device():
    if torch.cuda.is_available():
        yield torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        yield torch.device("mps:0")
    else:
        pytest.skip("CUDA/MPS is not available")


@pytest.mark.parametrize(
    "compile_mode,num_threads",
    [
        [False, None],
        # [False, 4],
        # [False, 16],
        ["default", None],
        ["reduce-overhead", None],
    ],
)
@pytest.mark.skipif(
    TORCH_VERSION < version.parse("2.5.0"), reason="requires torch>=2.5"
)
class TestConsolidate:
    def test_consolidate(
        self, benchmark, td, compile_mode, num_threads, default_device
    ):
        tensordict_logger.info(f"td size {td.bytes() / 1024 / 1024:.2f} Mb")

        # td = td.to(default_device)

        def consolidate(td, num_threads):
            return td.consolidate(num_threads=num_threads)

        if compile_mode:
            consolidate = torch.compile(
                consolidate, mode=compile_mode, dynamic=False, fullgraph=True
            )

        t0 = time.time()
        consolidate(td, num_threads=num_threads)
        elapsed = time.time() - t0
        tensordict_logger.info(f"elapsed time first call: {elapsed:.2f} sec")

        for _ in range(3):
            consolidate(td, num_threads=num_threads)

        benchmark(consolidate, td, num_threads)

    def test_consolidate_njt(self, benchmark, njt_td, compile_mode, num_threads):
        tensordict_logger.info(f"njtd size {njt_td.bytes() / 1024 / 1024 :.2f} Mb")

        def consolidate(td, num_threads):
            return td.consolidate(num_threads=num_threads)

        if compile_mode:
            pytest.skip(
                "Compiling NJTs consolidation currently triggers a RuntimeError."
            )
            # consolidate = torch.compile(consolidate, mode=compile_mode, dynamic=True)

        for _ in range(3):
            consolidate(njt_td, num_threads=num_threads)

        benchmark(consolidate, njt_td, num_threads)


@pytest.mark.parametrize(
    "consolidated,compile_mode,num_threads",
    [
        [False, False, None],
        [True, False, None],
        ["within", False, None],
        # [True, False, 4],
        # [True, False, 16],
        [True, "default", None],
    ],
)
@pytest.mark.skipif(
    TORCH_VERSION < version.parse("2.5.2"), reason="requires torch>=2.5"
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device found")
class TestTo:
    def test_to(
        self, benchmark, consolidated, td, default_device, compile_mode, num_threads
    ):
        tensordict_logger.info(f"td size {td.bytes() / 1024 / 1024:.2f} Mb")
        pin_mem = default_device.type == "cuda"
        if consolidated is True:
            td = td.consolidate(pin_memory=pin_mem)

        if consolidated == "within":

            def to(td, num_threads):
                return td.consolidate(pin_memory=pin_mem).to(
                    default_device, num_threads=num_threads
                )

        else:

            def to(td, num_threads):
                return td.to(default_device, num_threads=num_threads)

        if compile_mode:
            to = torch.compile(to, mode=compile_mode, dynamic=True)

        for _ in range(3):
            to(td, num_threads=num_threads)

        benchmark(to, td, num_threads)

    def test_to_njt(
        self, benchmark, consolidated, njt_td, default_device, compile_mode, num_threads
    ):
        if compile_mode:
            pytest.skip(
                "Compiling NJTs consolidation currently triggers a RuntimeError."
            )

        tensordict_logger.info(f"njtd size {njt_td.bytes() / 1024 / 1024 :.2f} Mb")
        pin_mem = default_device.type == "cuda"
        if consolidated is True:
            njt_td = njt_td.consolidate(pin_memory=pin_mem)

        if consolidated == "within":

            def to(td, num_threads):
                return td.consolidate(pin_memory=pin_mem).to(
                    default_device, num_threads=num_threads
                )

        else:

            def to(td, num_threads):
                return td.to(default_device, num_threads=num_threads)

        if compile_mode:
            to = torch.compile(to, mode=compile_mode, dynamic=True)

        for _ in range(3):
            to(njt_td, num_threads=num_threads)

        benchmark(to, njt_td, num_threads)


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main(
        [
            __file__,
            "--capture",
            "no",
            "--exitfirst",
            "--benchmark-group-by",
            "func",
            "-vvv",
        ]
        + unknown
    )
