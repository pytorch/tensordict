# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import pytest
from tensordict import TensorDict
import torch

_has_cuda = torch.cuda.is_available()
def _make_big_td() -> TensorDict:
    return TensorDict({str(i): torch.randn(1_000_000) for i in range(1000)})

@pytest.mark.skipif(not _has_cuda, reason="CUDA is not available")
@pytest.mark.parametrize("non_blocking", [True, False])
@pytest.mark.parametrize("pin_memory", [True, False])
@pytest.mark.parametrize("consolidate", [None, True, False])
def test_to_cuda(non_blocking, pin_memory, consolidate, benchmark):
    td = _make_big_td()
    if consolidate is None:
        pass
    elif not consolidate:
        td = td.consolidate(pin_memory=pin_memory)
    else:
        benchmark(lambda: td.consolidate(pin_memory=pin_memory, num_threads=8).to("cuda", non_blocking=non_blocking))
        return
    benchmark(lambda: td.to("cuda", non_blocking=non_blocking, non_blocking_pin=pin_memory))


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
