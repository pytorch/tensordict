# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse

import pytest
import torch
from tensordict.utils import index_keyedjaggedtensor

try:
    from torchrec import KeyedJaggedTensor

    _has_torchrec = True
except ImportError as err:
    _has_torchrec = False
    # TORCHREC_ERR = str(err)


@pytest.mark.skipif(not _has_torchrec, reason="torchrec not found.")
@pytest.mark.parametrize("index", [[0, 2], 2, torch.tensor([0, 2]), range(0, 3, 2)])
def test_kjt_indexing(index):
    values = torch.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0])
    weights = torch.Tensor([1.0, 0.5, 1.5, 1.0, 0.5, 1.0, 1.0, 1.5, 1.0, 1.0, 1.0])
    keys = ["index_0", "index_1", "index_2"]
    offsets = torch.IntTensor([0, 2, 2, 3, 4, 5, 8, 9, 10, 11])

    jag_tensor = KeyedJaggedTensor(
        values=values,
        keys=keys,
        offsets=offsets,
        weights=weights,
    )
    j0 = jag_tensor["index_0"]
    j1 = jag_tensor["index_1"]
    j2 = jag_tensor["index_2"]
    ikjt = index_keyedjaggedtensor(jag_tensor, index)
    assert (
        ikjt["index_0"].to_padded_dense(),
        j0.to_padded_dense() == j0.to_padded_dense()[:, index],
    ).all()
    assert (
        ikjt["index_1"].to_padded_dense(),
        j1.to_padded_dense() == j1.to_padded_dense()[:, index],
    ).all()
    assert (
        ikjt["index_2"].to_padded_dense(),
        j2.to_padded_dense() == j2.to_padded_dense()[:, index],
    ).all()


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
