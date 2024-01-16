# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import re

import pytest
import torch

from tensordict import TensorDict
from tensordict.utils import index_keyedjaggedtensor, setitem_keyedjaggedtensor

try:
    from torchrec import KeyedJaggedTensor

    _has_torchrec = True
except ImportError:
    _has_torchrec = False
    # TORCHREC_ERR = str(err)


def _get_kjt():
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
    return jag_tensor


@pytest.mark.skipif(not _has_torchrec, reason="torchrec not found.")
class TestKJT:
    @pytest.mark.parametrize("index", [[0, 2], torch.tensor([0, 2]), range(0, 3, 2)])
    def test_kjt_indexing(self, index):
        jag_tensor = _get_kjt()
        j0 = jag_tensor["index_0"]
        j1 = jag_tensor["index_1"]
        j2 = jag_tensor["index_2"]
        ikjt = index_keyedjaggedtensor(jag_tensor, index)
        assert (ikjt["index_0"].to_padded_dense() == j0.to_padded_dense()[index]).all()
        assert (ikjt["index_1"].to_padded_dense() == j1.to_padded_dense()[index]).all()
        assert (ikjt["index_2"].to_padded_dense() == j2.to_padded_dense()[index]).all()

    def test_td_build(self):
        jag_tensor = _get_kjt()
        _ = TensorDict({}, [])
        _ = TensorDict({"b": jag_tensor}, [])
        _ = TensorDict({"b": jag_tensor}, [3])

    def test_td_repr(self):
        jag_tensor = _get_kjt()
        td = TensorDict({"b": jag_tensor}, [])
        assert (
            str(td)
            == """TensorDict(
    fields={
        b: KeyedJaggedTensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False)},
    batch_size=torch.Size([]),
    device=None,
    is_shared=False)"""
        )

    def test_td_index(self):
        jag_tensor = _get_kjt()
        td = TensorDict({"b": jag_tensor}, [3])
        subtd = td[:2]
        assert subtd.shape == torch.Size([2])
        assert (
            subtd["b", "index_0"].to_padded_dense()
            == torch.tensor([[1.0, 2.0], [0.0, 0.0]])
        ).all()
        subtd = td[[0, 2]]
        assert subtd.shape == torch.Size([2])
        assert (
            subtd["b", "index_0"].to_padded_dense()
            == torch.tensor([[1.0, 2.0], [3.0, 0.0]])
        ).all()

    def test_td_change_batch_size(self):
        jag_tensor = _get_kjt()
        td = TensorDict({"b": jag_tensor}, [])
        td.batch_size = [3]
        with pytest.raises(
            RuntimeError,
            match=re.escape(
                "the tensor b has shape torch.Size([3]) which is incompatible with the new shape torch.Size([4])"
            ),
        ):
            td.batch_size = [4]

    def test_setindex(self):
        jag_tensor = _get_kjt()
        sub_jag_tensor = index_keyedjaggedtensor(jag_tensor, [0, 2])
        out = setitem_keyedjaggedtensor(jag_tensor, [0, 2], sub_jag_tensor)
        for f in (
            "_weights",
            "_values",
        ):
            assert (getattr(jag_tensor, f) == getattr(out, f)).all()

        keys = ["index_0", "index_1", "index_2"]
        lengths2 = torch.IntTensor([2, 4, 6, 4, 2, 1])
        values2 = torch.zeros(
            lengths2.sum(),
        )
        weights2 = -torch.ones(
            lengths2.sum(),
        )
        sub_jag_tensor = KeyedJaggedTensor(
            values=values2,
            keys=keys,
            lengths=lengths2,
            weights=weights2,
        )
        out = setitem_keyedjaggedtensor(jag_tensor, [0, 2], sub_jag_tensor)

    @pytest.mark.parametrize("index", [[0, 2], range(2)])
    def test_setindex_td_same(self, index):
        jag_tensor = _get_kjt()
        td = TensorDict({"b": jag_tensor}, [3])
        sub_jag_tensor = index_keyedjaggedtensor(jag_tensor, index)
        td.set_at_("b", sub_jag_tensor, index)
        out = td["b"]
        for f in (
            "_weights",
            "_values",
        ):
            assert (getattr(jag_tensor, f) == getattr(out, f)).all()

    def test_setindex_td(
        self,
    ):
        jag_tensor = _get_kjt()
        td = TensorDict({"b": jag_tensor}, [3])

        keys = ["index_0", "index_1", "index_2"]
        lengths2 = torch.IntTensor([2, 4, 6, 4, 2, 1])
        values2 = torch.zeros(
            lengths2.sum(),
        )
        weights2 = -torch.ones(
            lengths2.sum(),
        )
        sub_jag_tensor = KeyedJaggedTensor(
            values=values2,
            keys=keys,
            lengths=lengths2,
            weights=weights2,
        )
        td.set_at_("b", sub_jag_tensor, range(2))
        td["b"]


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
