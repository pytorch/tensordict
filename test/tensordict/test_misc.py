# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import argparse
import contextlib
import gc
import importlib.util
import os
import platform
import sys
import sysconfig
import warnings

import pytest
import torch
from packaging import version
from tensordict import (
    LazyStackedTensorDict,
    make_tensordict,
    PersistentTensorDict,
    tensorclass,
    TensorDict,
)
from tensordict._td import is_tensor_collection
from tensordict.base import TensorDictBase
from tensordict.memmap import MemoryMappedTensor
from tensordict.tensorclass import NonTensorData
from tensordict.utils import (
    assert_allclose_td,
    is_non_tensor,
    is_tensorclass,
    set_lazy_legacy,
)
from torch.func import hessian, jacfwd, jacrev

if os.getenv("PYTORCH_TEST_FBCODE"):
    IS_FB = True
    from pytorch.tensordict.test._utils_internal import (
        get_available_devices,
        is_npu_available,
    )
else:
    IS_FB = False
    from _utils_internal import get_available_devices, is_npu_available


_has_streaming = importlib.util.find_spec("streaming", None) is not None

try:
    import h5py  # noqa

    _has_h5py = True
except ImportError:
    _has_h5py = False
TORCH_VERSION = version.parse(version.parse(torch.__version__).base_version)

_has_onnx = importlib.util.find_spec("onnxruntime", None) is not None

_v2_5 = TORCH_VERSION >= version.parse("2.5.0")
PYTORCH_TEST_FBCODE = os.getenv("PYTORCH_TEST_FBCODE")

_IS_OSX = platform.system() == "Darwin"
_IS_WINDOWS = sys.platform == "win32"

TD_BATCH_SIZE = 4
HAS_NESTED_TENSOR = (
    getattr(torch, "_nested_compute_contiguous_strides_offsets", None) is not None
)

# Capture all warnings
pytestmark = [
    pytest.mark.filterwarnings("error"),
    pytest.mark.filterwarnings(
        "ignore:There is a performance drop because we have not yet implemented the batching rule"
    ),
    pytest.mark.filterwarnings(
        "ignore:A destination should be provided when cloning a PersistentTensorDict"
    ),
    pytest.mark.filterwarnings(
        "ignore:Replacing an array with another one is inefficient"
    ),
    pytest.mark.filterwarnings(
        "ignore:Indexing an h5py.Dataset object with a boolean mask that needs broadcasting does not work directly"
    ),
    pytest.mark.filterwarnings(
        "ignore:The PyTorch API of nested tensors is in prototype"
    ),
    pytest.mark.filterwarnings(
        "ignore:Lazy modules are a new feature under heavy development so changes to the API or functionality"
    ),
    pytest.mark.filterwarnings(
        "ignore:The content of the stacked NonTensorData objects matched in value but not identity"
    ),
    pytest.mark.filterwarnings(
        "ignore:No PYTORCH_KERNEL_CACHE_PATH or HOME environment variable set"
    ),
    pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning"),
]

mp_ctx = "spawn"
cur_device = "cpu"
npu_device_count = 0
if torch.cuda.is_available():
    cur_device = "cuda"
elif is_npu_available():
    cur_device = "npu"
    npu_device_count = torch.npu.device_count()


class TestPointwiseOps:
    def test_r_ops(self):
        td = TensorDict(a=1)
        # mul
        assert isinstance(0 * td, TensorDict)
        assert isinstance(torch.zeros(()) * td, TensorDict)
        # +
        assert isinstance(0 + td, TensorDict)
        assert isinstance(torch.zeros(()) + td, TensorDict)
        # -
        assert isinstance(0 - td, TensorDict)
        assert isinstance(torch.zeros(()) - td, TensorDict)
        # /
        assert isinstance(0 / td, TensorDict)
        assert isinstance(torch.zeros(()) / td, TensorDict)
        # **
        # assert isinstance(1 ** td, TensorDict)
        # assert isinstance(torch.ones(()) ** td, TensorDict)

        td = TensorDict(a=True)
        # |
        assert isinstance(False | td, TensorDict)
        assert isinstance(torch.zeros((), dtype=torch.bool) | td, TensorDict)
        # ^
        assert isinstance(False ^ td, TensorDict)
        assert isinstance(torch.zeros((), dtype=torch.bool) ^ td, TensorDict)

    def test_builtins(self):
        td_float = TensorDict(a=1.0)
        td_bool = TensorDict(a=True)
        ones = torch.ones(())
        bool_ones = torch.ones(()).to(torch.bool)
        assert ((-td_float) == (-ones)).all()
        # assert ((-td_bool) == (-bool_ones)).all()  # Not defined for bool
        assert (abs(td_float) == abs(ones)).all()
        # assert (abs(td_bool) == abs(bool_ones)).all()  # Not defined for bool
        # assert ((~td_float) == (~ones)).all()  # Not defined for float
        assert ((~td_bool) == (~bool_ones)).all()
        assert ((td_float != td_float) == (ones != ones)).all()
        assert ((td_bool != td_bool) == (bool_ones != bool_ones)).all()
        assert ((td_float == td_float) == (ones == ones)).all()
        assert ((td_bool == td_bool) == (bool_ones == bool_ones)).all()
        assert ((td_float < td_float) == (ones < ones)).all()
        assert ((td_bool < td_bool) == (bool_ones < bool_ones)).all()
        assert ((td_float <= td_float) == (ones <= ones)).all()
        assert ((td_bool <= td_bool) == (bool_ones <= bool_ones)).all()
        assert ((td_float > td_float) == (ones > ones)).all()
        assert ((td_bool > td_bool) == (bool_ones > bool_ones)).all()
        assert ((td_float >= td_float) == (ones >= ones)).all()
        assert ((td_bool >= td_bool) == (bool_ones >= bool_ones)).all()
        assert ((td_float + td_float) == (ones + ones)).all()
        # assert ((td_bool + td_bool) == (bool_ones + bool_ones)).all()  # Not defined for bool
        assert ((td_float - td_float) == (ones - ones)).all()
        # assert ((td_bool - td_bool) == (bool_ones - bool_ones)).all()  # Not defined for bool
        assert ((td_float * td_float) == (ones * ones)).all()
        # assert ((td_bool * td_bool) == (bool_ones * bool_ones)).all()  # Not defined for bool
        assert ((td_float / td_float) == (ones / ones)).all()
        # assert ((td_bool / td_bool) == (bool_ones / bool_ones)).all()  # Not defined for bool
        assert ((td_float**td_float) == (ones**ones)).all()
        # assert ((td_bool**td_bool) == (bool_ones**bool_ones)).all()  # Not defined for bool
        # assert ((td_float & td_float) == (ones & ones)).all()  # Not defined for float
        assert ((td_bool & td_bool) == (bool_ones & bool_ones)).all()
        # assert ((td_float ^ td_float) == (ones ^ ones)).all()  # Not defined for float
        assert ((td_bool ^ td_bool) == (bool_ones ^ bool_ones)).all()
        # assert ((td_float | td_float) == (ones | ones)).all()  # Not defined for float
        assert ((td_bool | td_bool) == (bool_ones | bool_ones)).all()

    @property
    def dummy_td_0(self):
        return TensorDict(
            {"a": torch.zeros(3, 4), "b": {"c": torch.zeros(3, 5, dtype=torch.int)}}
        )

    @property
    def dummy_td_1(self):
        return self.dummy_td_0.apply(lambda x: x + 1)

    @property
    def dummy_td_2(self):
        return self.dummy_td_0.apply(lambda x: x + 2)

    def test_ordering(self):
        x0 = TensorDict({"y": torch.zeros(3), "x": torch.ones(3)})

        x1 = TensorDict({"x": torch.ones(3), "y": torch.zeros(3)})
        assert ((x0 + x1)["x"] == 2).all()
        assert ((x0 * x1)["x"] == 1).all()
        assert ((x0 - x1)["x"] == 0).all()

    @pytest.mark.parametrize("locked", [True, False])
    def test_add(self, locked):
        td = self.dummy_td_0
        if locked:
            td.lock_()
        assert (td.add(1) == 1).all()
        other = self.dummy_td_1
        if locked:
            other.lock_()
        assert (td.add(other) == 1).all()

        td = self.dummy_td_0
        if locked:
            td.lock_()
        assert (td + 1 == 1).all()
        other = self.dummy_td_1
        if locked:
            other.lock_()
        r = td + other
        assert r.is_locked is locked
        assert (r == 1).all()

    def test_add_default(self):
        # Create two tds with different key sets
        td0 = TensorDict(a=1, b=1, c=1)
        td1 = TensorDict(b=2, c=2, d=2)
        with pytest.raises(KeyError):
            td0.add(td1)
        with pytest.raises(KeyError):
            td0.exclude("a").add(td1)
        with pytest.raises(KeyError):
            td0.add(td1.exclude("d"))
        tdadd = td0.add(td1, default=torch.tensor(3))
        assert tdadd["a"] == 4  # 1 + 3
        assert tdadd["d"] == 5  # 2 + 3
        tdadd = td0.add(td1, default="intersection")
        assert "a" not in tdadd
        assert "d" not in tdadd
        assert "b" in tdadd

        td0 = TensorDict(a=1, b=1, c=1, non_tensor="a string")
        td1 = TensorDict(b=2, c=2, d=2, non_tensor="a string")
        td = td0.add(td1, default=torch.zeros(()))
        assert td["non_tensor"] == "a string"

    def test_sub_default(self):
        # Create two tds with different key sets
        td0 = TensorDict(a=1, b=1, c=1)
        td1 = TensorDict(b=2, c=2, d=2)
        with pytest.raises(KeyError):
            td0.sub(td1)
        tdsub = td0.sub(td1, default=torch.tensor(3))
        assert tdsub["b"] == -1
        assert tdsub["a"] == -2
        assert tdsub["d"] == 1
        tdsub = td0.sub(td1, default="intersection")
        assert "a" not in tdsub
        assert "d" not in tdsub
        assert "b" in tdsub

    @pytest.mark.parametrize("locked", [True, False])
    def test_add_(self, locked):
        td = self.dummy_td_0
        if locked:
            td.lock_()
        assert (td.add_(1) == 1).all()
        assert td.add_(1) is td
        td = self.dummy_td_0
        other = self.dummy_td_1
        if locked:
            other.lock_()
        assert (td.add_(other) == 1).all()

        td = self.dummy_td_0
        if locked:
            td.lock_()
        td += 1
        assert (td == 1).all()
        td = self.dummy_td_0
        other = self.dummy_td_1
        if locked:
            other.lock_()
        td += other
        assert (td == 1).all()

    @pytest.mark.parametrize("locked", [True, False])
    def test_mul(self, locked):
        td = self.dummy_td_1
        if locked:
            td.lock_()
        assert (td.mul(0) == 0).all()
        other = self.dummy_td_0
        if locked:
            other.lock_()
        assert (td.mul(other) == 0).all()

        td = self.dummy_td_1
        if locked:
            td.lock_()
        td = td * 0
        assert (td == 0).all()
        other = self.dummy_td_0
        if locked:
            other.lock_()
        td = td * other
        assert td.is_locked is locked
        assert (td == 0).all()

    def test_mul_default(self):
        # Create two tds with different key sets
        td0 = TensorDict(a=1, b=1, c=1)
        td1 = TensorDict(b=4, c=4, d=4)
        with pytest.raises(KeyError):
            td0.mul(td1)
        tdmul = td0.mul(td1, default=torch.tensor(2))
        assert tdmul["a"] == 2
        assert tdmul["d"] == 8
        tdmul = td0.mul(td1, default="intersection")
        assert "a" not in tdmul
        assert "d" not in tdmul
        assert "b" in tdmul

    @pytest.mark.parametrize("locked", [True, False])
    def test_mul_(self, locked):
        td = self.dummy_td_1
        if locked:
            td.lock_()
        assert (td.mul_(0) == 0).all()
        assert td.mul_(0) is td
        td = self.dummy_td_1
        other = self.dummy_td_0
        if locked:
            other.lock_()
        assert (td.mul_(other) == 0).all()

        td = self.dummy_td_1
        if locked:
            td.lock_()
        td *= 0
        assert (td == 0).all()
        td = self.dummy_td_1
        other = self.dummy_td_0
        if locked:
            other.lock_()
        td *= other
        assert (td == 0).all()

    @pytest.mark.parametrize("locked", [True, False])
    def test_div(self, locked):
        td = self.dummy_td_2
        if locked:
            td.lock_()
        assert (td.div(2) == 1).all()
        other = self.dummy_td_2
        if locked:
            other.lock_()
        assert (td.div(other) == 1).all()

        td = self.dummy_td_2
        if locked:
            td.lock_()
        assert (td / 2 == 1).all()
        other = self.dummy_td_2
        if locked:
            other.lock_()
        r = td / other
        assert r.is_locked is locked
        assert (r == 1).all()

    def test_div_default(self):
        # Create two tds with different key sets
        td0 = TensorDict(a=1, b=1, c=1)
        td1 = TensorDict(b=4, c=4, d=4)
        with pytest.raises(KeyError):
            td0.div(td1)
        tddiv = td0.div(td1, default=torch.tensor(2))
        assert tddiv["a"] == 0.5
        assert tddiv["d"] == 0.5
        tddiv = td0.div(td1, default="intersection")
        assert "a" not in tddiv
        assert "d" not in tddiv
        assert "b" in tddiv

    @pytest.mark.parametrize("locked", [True, False])
    def test_div_(self, locked):
        td = self.dummy_td_2.float()
        if locked:
            td.lock_()
        assert (td.div_(2) == 1).all()
        assert td.div_(2) is td
        td = self.dummy_td_2.float()
        other = self.dummy_td_2.float()
        if locked:
            other.lock_()
        assert (td.div_(other) == 1).all()

        td = self.dummy_td_2.float()
        if locked:
            td.lock_()
        td /= 2
        assert (td == 1).all()
        td = self.dummy_td_2.float()
        other = self.dummy_td_2.float()
        if locked:
            other.lock_()
        td /= other
        assert (td == 1).all()

    @pytest.mark.parametrize("locked", [True, False])
    def test_pow(self, locked):
        td = self.dummy_td_2
        if locked:
            td.lock_()
        assert (td.pow(2) == 4).all()
        other = self.dummy_td_2
        if locked:
            other.lock_()
        assert (td.pow(other) == 4).all()

        td = self.dummy_td_2
        if locked:
            td.lock_()
        assert (td**2 == 4).all()
        other = self.dummy_td_2
        if locked:
            other.lock_()

        r = td**other
        assert r.is_locked is locked

        assert (r == 4).all()

    def test_pow_default(self):
        # Create two tds with different key sets
        td0 = TensorDict(a=2, b=2, c=2)
        td1 = TensorDict(b=3, c=3, d=3)
        with pytest.raises(KeyError):
            td0.pow(td1)
        tdpow = td0.pow(td1, default=torch.tensor(1))
        assert tdpow["a"] == 2
        assert tdpow["d"] == 1
        tdpow = td0.pow(td1, default="intersection")
        assert "a" not in tdpow
        assert "d" not in tdpow
        assert "b" in tdpow

    @pytest.mark.parametrize("locked", [True, False])
    def test_pow_(self, locked):
        td = self.dummy_td_2.float()
        if locked:
            td.lock_()
        assert (td.pow_(2) == 4).all()
        assert td.pow_(2) is td
        td = self.dummy_td_2.float()
        other = self.dummy_td_2.float()
        if locked:
            other.lock_()
        assert (td.pow_(other) == 4).all()

        td = self.dummy_td_2.float()
        if locked:
            td.lock_()
        td **= 2
        assert (td == 4).all()
        td = self.dummy_td_2.float()
        other = self.dummy_td_2.float()
        if locked:
            other.lock_()
        td **= other
        assert (td == 4).all()

    @property
    def _lazy_td(self):
        tensordict = LazyStackedTensorDict(
            TensorDict({"a": -2}), TensorDict({"a": -1, "b": -2}), stack_dim=0
        )
        return TensorDict({"super": tensordict})

    def test_lazy_td_pointwise(self):
        td = self._lazy_td
        td.abs_()
        assert (td > 0).all()
        td = self._lazy_td
        assert ((td + td) == td * 2).all()
        td = self._lazy_td
        td += self._lazy_td
        assert (td == self._lazy_td * 2).all()
        assert ((td.abs() ** 2).clamp_max(td) == td).all()

    def test_clamp_min_default(self):
        # Create two tds with different key sets
        td0 = TensorDict(a=2, b=2, c=2)
        td1 = TensorDict(b=3, c=3, d=3)
        with pytest.raises(KeyError):
            td0.clamp_min(td1)
        tdpow = td0.clamp_min(td1, default=torch.tensor(10))
        assert tdpow["a"] == 10
        assert tdpow["d"] == 10
        tdpow = td0.clamp_min(td1, default="intersection")
        assert "a" not in tdpow
        assert "d" not in tdpow
        assert "b" in tdpow

    def test_clamp_max_default(self):
        # Create two tds with different key sets
        td0 = TensorDict(a=2, b=2, c=2)
        td1 = TensorDict(b=3, c=3, d=3)
        with pytest.raises(KeyError):
            td0.clamp_max(td1)
        tdpow = td0.clamp_max(td1, default=torch.tensor(1))
        assert tdpow["a"] == 1
        assert tdpow["d"] == 1
        tdpow = td0.clamp_max(td1, default="intersection")
        assert "a" not in tdpow
        assert "d" not in tdpow
        assert "b" in tdpow

    @pytest.mark.parametrize("shape", [(4,), (3, 4), (2, 3, 4)])
    def test_broadcast_tensor(self, shape):
        torch.manual_seed(0)
        td = TensorDict(
            a=torch.randn(3, 4),
            b=torch.zeros(3, 4, 5),
            c=torch.ones(3, 4, 5, 6),
            batch_size=(3, 4),
        )
        broadcast_shape = torch.broadcast_shapes(shape, td.shape)
        td_mul = td * torch.ones(shape)
        assert td_mul.shape == broadcast_shape
        assert (td_mul == td).all()
        td_add = td + torch.ones(shape)
        assert td_add.shape == broadcast_shape
        assert (td_add == td + 1).all()
        td_sub = td - torch.ones(shape)
        assert td_sub.shape == broadcast_shape
        assert (td_sub == td - 1).all()
        td_div = td / torch.ones(shape)
        assert td_div.shape == broadcast_shape
        assert (td_div == td).all()
        td_max = td.maximum(torch.ones(shape))
        assert td_max.shape == broadcast_shape
        assert (td_max == td.maximum(torch.ones_like(td))).all()
        td_min = td.minimum(torch.ones(shape))
        assert td_min.shape == broadcast_shape
        assert (td_min == td.minimum(torch.ones_like(td))).all()
        td_max = td.clamp_max(torch.ones(shape))
        assert td_max.shape == broadcast_shape
        assert (td_max == td.clamp_max(torch.ones_like(td))).all()
        td_min = td.clamp_min(torch.ones(shape))
        assert td_min.shape == broadcast_shape
        assert (td_min == td.clamp_min(torch.ones_like(td))).all()

        td_clamp = td.clamp(-torch.ones(shape), torch.ones(shape))
        assert td_clamp.shape == broadcast_shape
        assert_allclose_td(
            td_clamp,
            td.clamp(-torch.ones_like(td), torch.ones_like(td)).expand(broadcast_shape),
        )
        td_clamp = td.clamp(None, torch.ones(shape))
        assert td_clamp.shape == broadcast_shape
        assert_allclose_td(
            td_clamp, td.clamp(None, torch.ones_like(td)).expand(broadcast_shape)
        )
        td_clamp = td.clamp(-torch.ones(shape), None)
        assert td_clamp.shape == broadcast_shape
        assert_allclose_td(
            td_clamp, td.clamp(-torch.ones_like(td), None).expand(broadcast_shape)
        )

        td_pow = td.pow(torch.ones(shape))
        assert td_pow.shape == broadcast_shape
        assert (td_pow == td.pow(torch.ones_like(td))).all()

        td_ba = td.bool().bitwise_and(torch.ones(shape, dtype=torch.bool))
        assert td_ba.shape == broadcast_shape
        assert (td_ba == td.bool().bitwise_and(torch.ones_like(td.bool()))).all()

        td_la = td.logical_and(torch.ones(shape))
        assert td_la.shape == broadcast_shape
        assert (td_la == td.logical_and(torch.ones_like(td))).all()

        td_lerp = td.lerp(-torch.ones(shape), torch.ones(shape))
        assert td_lerp.shape == broadcast_shape
        assert_allclose_td(
            td_lerp,
            td.lerp(-torch.ones_like(td), torch.ones_like(td)).expand(broadcast_shape),
        )

        td_addcdiv = td.addcdiv(-torch.ones(shape), torch.ones(shape))
        assert td_addcdiv.shape == broadcast_shape
        assert_allclose_td(
            td_addcdiv,
            td.addcdiv(-torch.ones_like(td), torch.ones_like(td)).expand(
                broadcast_shape
            ),
        )

        td_addcmul = td.addcmul(-torch.ones(shape), torch.ones(shape))
        assert td_addcmul.shape == broadcast_shape
        assert_allclose_td(
            td_addcmul,
            td.addcmul(-torch.ones_like(td), torch.ones_like(td)).expand(
                broadcast_shape
            ),
        )

    @pytest.mark.parametrize("shape", [(4,), (3, 4), (2, 3, 4)])
    def test_broadcast_tensordict(self, shape):
        torch.manual_seed(0)
        td = TensorDict(
            a=torch.randn(3, 4),
            b=torch.zeros(3, 4, 5),
            c=torch.ones(3, 4, 5, 6),
            batch_size=(3, 4),
        )
        td_mul = td * torch.ones(shape)
        td_mul = td * td.new_ones(shape)
        broadcast_shape = torch.broadcast_shapes(shape, td.shape)
        assert td_mul.shape == broadcast_shape
        assert (td_mul == td).all()
        td_add = td + td.new_ones(shape)
        assert td_add.shape == broadcast_shape
        assert (td_add == td + 1).all()
        td_sub = td - td.new_ones(shape)
        assert td_sub.shape == broadcast_shape
        assert (td_sub == td - 1).all()
        td_div = td / td.new_ones(shape)
        assert td_div.shape == broadcast_shape
        assert (td_div == td).all()
        td_max = td.maximum(td.new_ones(shape))
        assert td_max.shape == broadcast_shape
        assert (td_max == td.maximum(torch.ones_like(td))).all()
        td_min = td.minimum(td.new_ones(shape))
        assert td_min.shape == broadcast_shape
        assert (td_min == td.minimum(torch.ones_like(td))).all()
        td_max = td.clamp_max(td.new_ones(shape))
        assert td_max.shape == broadcast_shape
        assert (td_max == td.clamp_max(torch.ones_like(td))).all()
        td_min = td.clamp_min(td.new_ones(shape))
        assert td_min.shape == broadcast_shape
        assert (td_min == td.clamp_min(torch.ones_like(td))).all()

        td_clamp = td.clamp(-td.new_ones(shape), td.new_ones(shape))
        assert td_clamp.shape == broadcast_shape
        assert_allclose_td(
            td_clamp,
            td.clamp(-torch.ones_like(td), torch.ones_like(td)).expand(broadcast_shape),
        )
        td_clamp = td.clamp(None, td.new_ones(shape))
        assert td_clamp.shape == broadcast_shape
        assert_allclose_td(
            td_clamp, td.clamp(None, torch.ones_like(td)).expand(broadcast_shape)
        )
        td_clamp = td.clamp(-torch.ones(shape), None)
        assert td_clamp.shape == broadcast_shape
        assert_allclose_td(
            td_clamp, td.clamp(-torch.ones_like(td), None).expand(broadcast_shape)
        )

        td_pow = td.pow(td.new_ones(shape))
        assert td_pow.shape == broadcast_shape
        assert (td_pow == td.pow(torch.ones_like(td))).all()

        td_ba = td.bool().bitwise_and(td.new_ones(shape, dtype=torch.bool))
        assert td_ba.shape == broadcast_shape
        assert (td_ba == td.bool().bitwise_and(torch.ones_like(td.bool()))).all()

        td_la = td.logical_and(td.new_ones(shape))
        assert td_la.shape == broadcast_shape
        assert (td_la == td.logical_and(torch.ones_like(td))).all()

        td_lerp = td.lerp(-td.new_ones(shape), td.new_ones(shape))
        assert td_lerp.shape == broadcast_shape
        assert_allclose_td(
            td_lerp,
            td.lerp(-torch.ones_like(td), torch.ones_like(td)).expand(broadcast_shape),
        )

        td_addcdiv = td.addcdiv(-td.new_ones(shape), td.new_ones(shape))
        assert td_addcdiv.shape == broadcast_shape
        assert_allclose_td(
            td_addcdiv,
            td.addcdiv(-torch.ones_like(td), torch.ones_like(td)).expand(
                broadcast_shape
            ),
        )

        td_addcmul = td.addcmul(-td.new_ones(shape), td.new_ones(shape))
        assert td_addcmul.shape == broadcast_shape
        assert_allclose_td(
            td_addcmul,
            td.addcmul(-torch.ones_like(td), torch.ones_like(td)).expand(
                broadcast_shape
            ),
        )


@pytest.mark.parametrize(
    "batch_size,feature_size",
    [
        ([], 2),
        ([2], 2),
        ([2, 3], 2),
        ([2, 3, 4], 2),
    ],
    ids=["nobatch", "batch_1d", "batch_2d", "batch_3d"],
)
class TestJacobians:
    """Tests for torch.func.jacrev, jacfwd, and hessian with TensorDict.

    Tensors must have at least one feature dim beyond batch_size,
    otherwise the Jacobian basis leading dim is ambiguous with the batch dim.
    """

    @staticmethod
    def _make_td(batch_size, feature_size):
        return TensorDict(
            {
                "a": torch.randn(*batch_size, feature_size),
                "b": torch.randn(*batch_size, feature_size),
            },
            batch_size=list(batch_size),
        )

    def test_jacrev(self, batch_size, feature_size):
        td = self._make_td(batch_size, feature_size)

        def f(td):
            return TensorDict(
                {"x": td["a"] ** 2, "y": td["b"] ** 3}, batch_size=td.batch_size
            )

        J = jacrev(f)(td)
        assert J.batch_size == td.batch_size
        a_shape = td["a"].shape
        b_shape = td["b"].shape
        assert J["x", "a"].shape == (*a_shape, *a_shape)
        assert J["x", "b"].shape == (*a_shape, *b_shape)
        assert J["y", "a"].shape == (*b_shape, *a_shape)
        assert J["y", "b"].shape == (*b_shape, *b_shape)
        a_flat = td["a"].flatten()
        b_flat = td["b"].flatten()
        assert torch.allclose(
            J["x", "a"].flatten(-len(a_shape)).flatten(end_dim=-2),
            torch.diag(2 * a_flat),
        )
        assert torch.allclose(
            J["x", "b"].flatten(-len(b_shape)).flatten(end_dim=-2),
            torch.zeros(a_flat.numel(), b_flat.numel()),
        )

    def test_jacrev_different_shapes(self, batch_size, feature_size):
        td = TensorDict(
            {
                "a": torch.randn(*batch_size, 2),
                "b": torch.randn(*batch_size, 5),
            },
            batch_size=list(batch_size),
        )

        def f(td):
            return TensorDict(
                {"x": td["a"] ** 2, "y": td["b"] ** 3}, batch_size=td.batch_size
            )

        J = jacrev(f)(td)
        assert J.batch_size == td.batch_size
        assert J["x", "a"].shape == (*batch_size, 2, *batch_size, 2)
        assert J["x", "b"].shape == (*batch_size, 2, *batch_size, 5)
        assert J["y", "a"].shape == (*batch_size, 5, *batch_size, 2)
        assert J["y", "b"].shape == (*batch_size, 5, *batch_size, 5)

    def test_jacrev_chunk_size(self, batch_size, feature_size):
        td = self._make_td(batch_size, feature_size)

        def f(td):
            return TensorDict(
                {"x": td["a"] ** 2, "y": td["b"] ** 3}, batch_size=td.batch_size
            )

        J = jacrev(f, chunk_size=1)(td)
        assert J.batch_size == td.batch_size
        a_flat = td["a"].flatten()
        b_flat = td["b"].flatten()
        a_shape = td["a"].shape
        b_shape = td["b"].shape
        assert torch.allclose(
            J["x", "a"].flatten(-len(a_shape)).flatten(end_dim=-2),
            torch.diag(2 * a_flat),
        )
        assert torch.allclose(
            J["y", "b"].flatten(-len(b_shape)).flatten(end_dim=-2),
            torch.diag(3 * b_flat**2),
        )

    def test_jacfwd(self, batch_size, feature_size):
        td = self._make_td(batch_size, feature_size)

        def f(td):
            return TensorDict(
                {"x": td["a"] ** 2, "y": td["b"] ** 3}, batch_size=td.batch_size
            )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            J = jacfwd(f)(td)
        assert J.batch_size == td.batch_size
        a_flat = td["a"].flatten()
        b_flat = td["b"].flatten()
        a_shape = td["a"].shape
        b_shape = td["b"].shape
        assert torch.allclose(
            J["x", "a"].flatten(-len(a_shape)).flatten(end_dim=-2),
            torch.diag(2 * a_flat),
        )
        assert torch.allclose(
            J["x", "b"].flatten(-len(b_shape)).flatten(end_dim=-2),
            torch.zeros(a_flat.numel(), b_flat.numel()),
        )

    def test_hessian(self, batch_size, feature_size):
        td = self._make_td(batch_size, feature_size)

        def f(td):
            return (td["a"] ** 3).sum() + (td["b"] ** 2).sum()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            H = hessian(f)(td)
        assert H.batch_size == td.batch_size
        a_shape = td["a"].shape
        b_shape = td["b"].shape
        a_flat = td["a"].flatten()
        assert H["a", "a"].shape == (*a_shape, *a_shape)
        assert torch.allclose(
            H["a", "a"].flatten(-len(a_shape)).flatten(end_dim=-2),
            torch.diag(6 * a_flat),
        )
        assert torch.allclose(
            H["b", "b"].flatten(-len(b_shape)).flatten(end_dim=-2),
            2 * torch.eye(td["b"].numel()),
        )
        assert torch.allclose(
            H["a", "b"].flatten(-len(b_shape)).flatten(end_dim=-2),
            torch.zeros(td["a"].numel(), td["b"].numel()),
        )

    def test_jacrev_has_aux(self, batch_size, feature_size):
        td = self._make_td(batch_size, feature_size)

        def f(td):
            out = TensorDict({"x": td["a"] ** 2}, batch_size=td.batch_size)
            return out, td["a"].sum()

        J, aux = jacrev(f, has_aux=True)(td)
        assert J.batch_size == td.batch_size
        a_flat = td["a"].flatten()
        a_shape = td["a"].shape
        assert torch.allclose(
            J["x", "a"].flatten(-len(a_shape)).flatten(end_dim=-2),
            torch.diag(2 * a_flat),
        )
        assert torch.allclose(aux, td["a"].sum())

    def test_jacrev_argnums_tuple(self, batch_size, feature_size):
        td1 = TensorDict(
            {"a": torch.randn(*batch_size, feature_size)}, batch_size=list(batch_size)
        )
        td2 = TensorDict(
            {"b": torch.randn(*batch_size, feature_size)}, batch_size=list(batch_size)
        )

        def f(td1, td2):
            return TensorDict(
                {"x": td1["a"] + td2["b"] ** 2}, batch_size=td1.batch_size
            )

        J = jacrev(f, argnums=(0, 1))(td1, td2)
        J_x = J["x"]
        assert isinstance(J_x, tuple)
        n = td1["a"].numel()
        assert torch.allclose(
            J_x[0]["a"].flatten(-len(td1["a"].shape)).flatten(end_dim=-2),
            torch.eye(n),
        )
        assert torch.allclose(
            J_x[1]["b"].flatten(-len(td2["b"].shape)).flatten(end_dim=-2),
            torch.diag(2 * td2["b"].flatten()),
        )


@pytest.mark.parametrize(
    "td_name",
    [
        "td",
        "stacked_td",
        "sub_td",
        "idx_td",
        "unsqueezed_td",
        "td_reset_bs",
    ],
)
@pytest.mark.parametrize(
    "device",
    get_available_devices(),
)
class TestTensorDictsRequiresGrad:
    def idx_td(self, device):
        return self.td(device)[0]

    def stacked_td(self, device):
        return LazyStackedTensorDict.lazy_stack([self.td(device) for _ in range(2)], 0)

    def sub_td(self, device):
        return self.td(device)._get_sub_tensordict(0)

    def test_clone_td(self, td_name, device):
        torch.manual_seed(1)
        td = getattr(self, td_name)(device)
        assert torch.clone(td).get("b").requires_grad

    def test_expand(self, td_name, device):
        torch.manual_seed(1)
        td = getattr(self, td_name)(device)
        batch_size = td.batch_size
        new_td = td.expand(3, *batch_size)
        assert new_td.get("b").requires_grad
        assert new_td.batch_size == torch.Size([3, *batch_size])

    def test_squeeze(self, td_name, device, squeeze_dim=-1):
        torch.manual_seed(1)
        td = getattr(self, td_name)(device)
        is_lazy = td_name in (
            "sub_td",
            "sub_td2",
            "permute_td",
            "unsqueezed_td",
            "squeezed_td",
            "td_h5",
        )
        error_dec = (
            pytest.raises(RuntimeError, match="Make it dense")
            if is_lazy
            else contextlib.nullcontext()
        )
        with error_dec:
            assert torch.squeeze(td, dim=-1).get("b").requires_grad

    @set_lazy_legacy(False)
    def test_view(self, td_name, device):
        torch.manual_seed(1)
        td = getattr(self, td_name)(device)

        is_lazy = td_name in (
            "sub_td",
            "sub_td2",
            "permute_td",
            "unsqueezed_td",
            "squeezed_td",
            "td_h5",
        )
        error_dec = (
            pytest.raises(RuntimeError, match="Cannot call `view`")
            if is_lazy
            else contextlib.nullcontext()
        )
        with error_dec:
            td_view = td.view(-1)
        if not is_lazy:
            assert td_view.get("b").requires_grad

    def td(self, device):
        return TensorDict(
            source={
                "a": torch.randn(3, 1, 5, device=device),
                "b": torch.randn(3, 1, 10, device=device, requires_grad=True),
                "c": torch.randint(10, (3, 1, 3), device=device),
            },
            batch_size=[3, 1],
        )

    def td_reset_bs(self, device):
        td = self.td(device)
        td = td.unsqueeze(-1).to_tensordict(retain_none=True)
        td.batch_size = torch.Size([3, 1])
        return td

    @set_lazy_legacy(True)
    def unsqueezed_td(self, device):
        return self.td(device).unsqueeze(0)


class TestMakeTensorDict:
    def test_create_tensordict(self):
        tensordict = make_tensordict(a=torch.zeros(3, 4), auto_batch_size=True)
        assert (tensordict["a"] == torch.zeros(3, 4)).all()

    def test_nested(self):
        input_dict = {
            "a": {"b": torch.randn(3, 4), "c": torch.randn(3, 4, 5)},
            "d": torch.randn(3),
        }
        tensordict = make_tensordict(input_dict, auto_batch_size=True)
        assert tensordict.shape == torch.Size([3])
        assert tensordict["a"].shape == torch.Size([3, 4])
        input_tensordict = TensorDict(
            {
                "a": {"b": torch.randn(3, 4), "c": torch.randn(3, 4, 5)},
                "d": torch.randn(3),
            },
            [],
        )
        tensordict = make_tensordict(input_tensordict, auto_batch_size=True)
        assert tensordict.shape == torch.Size([3])
        assert tensordict["a"].shape == torch.Size([3, 4])
        input_dict = {
            ("a", "b"): torch.randn(3, 4),
            ("a", "c"): torch.randn(3, 4, 5),
            "d": torch.randn(3),
        }
        tensordict = make_tensordict(input_dict, auto_batch_size=True)
        assert tensordict.shape == torch.Size([3])
        assert tensordict["a"].shape == torch.Size([3, 4])

    def test_tensordict_batch_size(self):
        tensordict = make_tensordict(auto_batch_size=True)
        assert tensordict.batch_size == torch.Size([])

        tensordict = make_tensordict(a=torch.randn(3, 4), auto_batch_size=True)
        assert tensordict.batch_size == torch.Size([3, 4])

        tensordict = make_tensordict(
            a=torch.randn(3, 4), b=torch.randn(3, 4, 5), auto_batch_size=True
        )
        assert tensordict.batch_size == torch.Size([3, 4])

        nested_tensordict = make_tensordict(
            c=tensordict, d=torch.randn(3, 5), auto_batch_size=True
        )  # nested
        assert nested_tensordict.batch_size == torch.Size([3])

        nested_tensordict = make_tensordict(
            c=tensordict, d=torch.randn(4, 5), auto_batch_size=True
        )  # nested
        assert nested_tensordict.batch_size == torch.Size([])

        tensordict = make_tensordict(
            a=torch.randn(3, 4, 2), b=torch.randn(3, 4, 5), auto_batch_size=True
        )
        assert tensordict.batch_size == torch.Size([3, 4])

        tensordict = make_tensordict(
            a=torch.randn(3, 4), b=torch.randn(1), auto_batch_size=True
        )
        assert tensordict.batch_size == torch.Size([])

        tensordict = make_tensordict(
            a=torch.randn(3, 4), b=torch.randn(3, 4, 5), batch_size=[3]
        )
        assert tensordict.batch_size == torch.Size([3])

        tensordict = make_tensordict(
            a=torch.randn(3, 4), b=torch.randn(3, 4, 5), batch_size=[]
        )
        assert tensordict.batch_size == torch.Size([])

    @pytest.mark.parametrize("device", get_available_devices())
    def test_tensordict_device(self, device):
        tensordict = make_tensordict(
            a=torch.randn(3, 4),
            b=torch.randn(3, 4),
            device=device,
            auto_batch_size=True,
        )
        assert tensordict.device == device
        assert tensordict["a"].device == device
        assert tensordict["b"].device == device

        tensordict = make_tensordict(
            a=torch.randn(3, 4, device=device),
            b=torch.randn(3, 4),
            c=torch.randn(3, 4, device="cpu"),
            device=device,
            auto_batch_size=True,
        )
        assert tensordict.device == device
        assert tensordict["a"].device == device
        assert tensordict["b"].device == device
        assert tensordict["c"].device == device


class TestSubclassing:
    def test_td_inheritance(self):
        class SubTD(TensorDict): ...

        assert is_tensor_collection(SubTD)

    def test_tc_inheritance(self):
        @tensorclass
        class MyClass: ...

        assert is_tensor_collection(MyClass)
        assert is_tensorclass(MyClass)

        class SubTC(MyClass): ...

        assert is_tensor_collection(SubTC)
        assert is_tensorclass(SubTC)

    def test_nontensor_inheritance(self):
        class SubTC(NonTensorData): ...

        assert is_tensor_collection(SubTC)
        assert is_tensorclass(SubTC)
        assert is_non_tensor(SubTC(data=1, batch_size=[]))


class TestLikeConstructors:
    @pytest.fixture(scope="module")
    def td(self):
        yield TensorDict(
            a=torch.randn(3, 4),
            b=TensorDict(c=torch.randint(10, (3, 4, 5)), batch_size=(3, 4)),
            batch_size=(3,),
        )

    @pytest.mark.parametrize("device", [None, "cpu"])
    @pytest.mark.parametrize("dtype", [None, torch.int64])
    def test_zeros_like(self, device, dtype, td):
        tdnew = torch.zeros_like(td, device=device, dtype=dtype)
        assert (tdnew == 0).all()
        assert tdnew.dtype == dtype
        if device is not None:
            assert tdnew.device == torch.device(device)
        else:
            assert tdnew.device is None

    @pytest.mark.parametrize("device", [None, "cpu"])
    @pytest.mark.parametrize("dtype", [None, torch.int64])
    def test_ones_like(self, device, dtype, td):
        tdnew = torch.ones_like(td, device=device, dtype=dtype)
        assert (tdnew == 1).all()
        assert tdnew.dtype == dtype
        if device is not None:
            assert tdnew.device == torch.device(device)
        else:
            assert tdnew.device is None

    @pytest.mark.parametrize("device", [None, "cpu"])
    @pytest.mark.parametrize("dtype", [None, torch.int64])
    def test_empty_like(self, device, dtype, td):
        tdnew = torch.empty_like(td, device=device, dtype=dtype)
        assert tdnew.dtype == dtype
        if device is not None:
            assert tdnew.device == torch.device(device)
        else:
            assert tdnew.device is None

    @pytest.mark.parametrize("device", [None, "cpu"])
    @pytest.mark.parametrize("dtype", [None, torch.int64])
    def test_full_like(self, device, dtype, td):
        tdnew = torch.full_like(td, 2, device=device, dtype=dtype)
        assert (tdnew == 2).all()
        assert tdnew.dtype == dtype
        if device is not None:
            assert tdnew.device == torch.device(device)
        else:
            assert tdnew.device is None

    @pytest.mark.parametrize("device", [None, "cpu"])
    @pytest.mark.parametrize("dtype", [None, torch.double])
    def test_rand_like(self, device, dtype, td):
        td = td.float()
        tdnew = torch.rand_like(td, device=device, dtype=dtype)
        assert (tdnew != td).all()
        assert (tdnew <= 1).all()
        assert (tdnew >= 0).all()
        if dtype is not None:
            assert tdnew.dtype == dtype
        if device is not None:
            assert tdnew.device == torch.device(device)
        else:
            assert tdnew.device is None

    @pytest.mark.parametrize("device", [None, "cpu"])
    @pytest.mark.parametrize("dtype", [None, torch.double])
    def test_randn_like(self, device, dtype, td):
        td = td.float()
        tdnew = torch.randn_like(td, device=device, dtype=dtype)
        assert (tdnew != td).all()
        if dtype is not None:
            assert tdnew.dtype == dtype
        if device is not None:
            assert tdnew.device == torch.device(device)
        else:
            assert tdnew.device is None


class TestMemmap:
    @pytest.mark.parametrize("robust_key", [False, True])
    def test_memmap_robust_key_normal(self, robust_key, tmpdir):
        """Test robust_key parameter with normal keys."""
        td = TensorDict({"normal_key": torch.randn(3, 4)})

        # Save with robust_key setting
        td_mmap = td.memmap(tmpdir, robust_key=robust_key)
        assert td_mmap.is_memmap()

        # Load with same robust_key setting
        td_loaded = TensorDict.load_memmap(tmpdir, robust_key=robust_key)
        assert "normal_key" in td_loaded
        assert_allclose_td(td, td_loaded)

        # Check that file exists
        files = [f for f in os.listdir(tmpdir) if f.endswith(".memmap")]
        assert len(files) == 1
        assert files[0] == "normal_key.memmap"  # Normal keys unchanged

    def test_memmap_robust_key_pathlike(self, tmpdir):
        """Test robust_key=True solves path-like key issue."""
        td = TensorDict({"a/b/c": torch.randn(3, 4)})

        # This should work with robust_key=True
        td_mmap = td.memmap(tmpdir, robust_key=True)
        assert td_mmap.is_memmap()

        # Load it back
        td_loaded = TensorDict.load_memmap(tmpdir, robust_key=True)
        assert "a/b/c" in td_loaded
        assert_allclose_td(td, td_loaded)

        # Check that encoded file was created
        files = [f for f in os.listdir(tmpdir) if f.endswith(".memmap")]
        assert len(files) == 1
        assert files[0] == "a%2Fb%2Fc.memmap"  # Encoded filename

    def test_memmap_robust_key_pathlike_legacy_fails(self, tmpdir):
        """Test that path-like keys still fail with robust_key=False."""
        td = TensorDict({"a/b/c": torch.randn(3, 4)})

        # This should still fail with robust_key=False (legacy behavior)
        with pytest.raises(RuntimeError, match="No such file or directory"):
            td.memmap(tmpdir, robust_key=False)

    def test_memmap_robust_key_backward_compatibility(self, tmpdir):
        """Test backward compatibility: load legacy saves with robust_key=True."""
        td = TensorDict({"normal_key": torch.randn(3, 4)})

        # Save with legacy behavior
        td.memmap(tmpdir, robust_key=False)

        # Load with robust_key=True should work (fallback)
        td_loaded = TensorDict.load_memmap(tmpdir, robust_key=True)
        assert "normal_key" in td_loaded
        assert_allclose_td(td, td_loaded)

    def test_memmap_robust_key_default_no_warning(self, tmpdir):
        """Test that robust_key=None uses robust encoding without warnings."""
        import warnings

        td_normal = TensorDict({"normal_key": torch.randn(3, 4)})
        td_pathlike = TensorDict({"a/b/c": torch.randn(3, 4)})

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            td_normal.memmap(tmpdir / "normal", robust_key=None)
            TensorDict.load_memmap(tmpdir / "normal", robust_key=None)

            td_pathlike.memmap(tmpdir / "pathlike", robust_key=None)
            td_loaded = TensorDict.load_memmap(tmpdir / "pathlike", robust_key=None)

        assert "a/b/c" in td_loaded
        assert_allclose_td(td_pathlike, td_loaded)

    def test_memmap_robust_key_default_for_problematic_keys(self, tmpdir):
        """Test that the default robust encoding handles problematic keys."""
        import warnings

        problematic_keys = ["a/b/c", "key with spaces", "key:colon", "key*star"]

        for key in problematic_keys:
            td = TensorDict({key: torch.randn(2, 3)})
            path = (
                tmpdir
                / f"problematic_{key.replace('/', '_').replace(' ', '_').replace(':', '_').replace('*', '_')}"
            )
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                td.memmap(path, robust_key=None)
                td_loaded = TensorDict.load_memmap(path, robust_key=None)

            assert key in td_loaded
            assert_allclose_td(td, td_loaded)

    def test_memmap_robust_key_encoding_bijective(self):
        """Test that key encoding is bijective."""
        from tensordict.utils import (
            _decode_key_from_filesystem,
            _encode_key_for_filesystem,
        )

        test_keys = [
            "a/b/c",
            "path\\with\\backslashes",
            "key with spaces",
            "key:with:colons",
            "key*with*stars",
            "normal_key",
            "key%with%percent",
            "",
        ]

        for key in test_keys:
            encoded = _encode_key_for_filesystem(key, robust=True)
            decoded = _decode_key_from_filesystem(encoded)
            assert decoded == key, f"Failed: {key!r} -> {encoded!r} -> {decoded!r}"

            # Legacy encoding should return key unchanged
            legacy = _encode_key_for_filesystem(key, robust=False)
            assert legacy == key


class TestFreeThreading:
    """Tests for free-threading (GIL-less Python) compatibility."""

    @pytest.mark.skipif(
        not sysconfig.get_config_var("Py_GIL_DISABLED"),
        reason="Only runs on free-threading Python builds (3.13t+)",
    )
    def test_concurrent_gc_stress(self):
        """Regression test for free-threading race condition (PR #1481).

        This test exercises concurrent access to TensorDict instances while
        triggering garbage collection. On Python 3.14t with PYTHON_GIL=0,
        this would previously cause segfaults due to a race condition in
        the _validate_value_cached attribute.

        The test passes on all Python versions but only catches the actual
        race condition on free-threading builds.
        """
        import threading

        def gc_stress():
            for i in range(100):
                td = TensorDict(
                    {"a": torch.randn(5, 5), "b": {"c": torch.randn(5, 3)}},
                    batch_size=[5],
                )
                # Access _validate_value to trigger the code path that had the race
                _ = td._validate_value
                td = None
                # Call gc.collect() periodically rather than every iteration
                # to reduce overhead while still catching race conditions.
                # Calling gc.collect() in every iteration causes ~100x slowdown
                # due to thread synchronization overhead in free-threaded Python.
                if i % 10 == 0:
                    gc.collect()
            # Final gc to clean up
            gc.collect()

        threads = [threading.Thread(target=gc_stress) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()


class TestFromSchema:
    """Tests for TensorDictBase.from_schema with various storage backends."""

    SCHEMA = {
        "obs": ([4, 4], torch.float32),
        "action": ([2], torch.int64),
        "reward": ([], torch.float32),
    }
    BATCH = [8]

    def test_default_storage(self):
        td = TensorDict.from_schema(self.SCHEMA, batch_size=self.BATCH)
        assert isinstance(td, TensorDict)
        assert td.batch_size == torch.Size(self.BATCH)
        assert td["obs"].shape == torch.Size([8, 4, 4])
        assert td["obs"].dtype == torch.float32
        assert td["action"].shape == torch.Size([8, 2])
        assert td["action"].dtype == torch.int64
        assert td["reward"].shape == torch.Size([8])
        assert (td["obs"] == 0).all()
        assert (td["action"] == 0).all()
        assert (td["reward"] == 0).all()

    def test_default_storage_no_batch(self):
        td = TensorDict.from_schema({"x": ([3], torch.float32)})
        assert td.batch_size == torch.Size(())
        assert td["x"].shape == torch.Size([3])

    def test_default_storage_write_read(self):
        td = TensorDict.from_schema(self.SCHEMA, batch_size=self.BATCH)
        td[0] = TensorDict(
            obs=torch.ones(4, 4),
            action=torch.ones(2, dtype=torch.int64),
            reward=torch.tensor(1.0),
            batch_size=[],
        )
        assert (td[0]["obs"] == 1).all()
        assert (td[1]["obs"] == 0).all()

    def test_memmap_storage(self, tmp_path):
        td = TensorDict.from_schema(
            self.SCHEMA, batch_size=self.BATCH, storage="memmap", prefix=str(tmp_path)
        )
        assert td.is_memmap()
        assert td["obs"].shape == torch.Size([8, 4, 4])
        assert td["action"].shape == torch.Size([8, 2])
        assert td["reward"].shape == torch.Size([8])
        assert isinstance(td["obs"], MemoryMappedTensor)

    def test_memmap_storage_write_read(self, tmp_path):
        td = TensorDict.from_schema(
            self.SCHEMA, batch_size=self.BATCH, storage="memmap", prefix=str(tmp_path)
        )
        td[0] = TensorDict(
            obs=torch.ones(4, 4),
            action=torch.ones(2, dtype=torch.int64),
            reward=torch.tensor(1.0),
            batch_size=[],
        )
        assert (td[0]["obs"] == 1).all()
        assert (td[1]["obs"] == 0).all()

    @pytest.mark.skipif(not _has_h5py, reason="h5py not available")
    def test_h5_storage(self, tmp_path):
        filename = str(tmp_path / "test.h5")
        td = TensorDict.from_schema(
            self.SCHEMA, batch_size=self.BATCH, storage="h5", filename=filename
        )
        assert isinstance(td, PersistentTensorDict)
        assert td.batch_size == torch.Size(self.BATCH)
        assert td["obs"].shape == torch.Size([8, 4, 4])
        assert td["action"].shape == torch.Size([8, 2])
        assert td["reward"].shape == torch.Size([8])

    @pytest.mark.skipif(not _has_h5py, reason="h5py not available")
    def test_h5_storage_write_read(self, tmp_path):
        filename = str(tmp_path / "test.h5")
        td = TensorDict.from_schema(
            self.SCHEMA, batch_size=self.BATCH, storage="h5", filename=filename
        )
        td[0] = TensorDict(
            obs=torch.ones(4, 4),
            action=torch.ones(2, dtype=torch.int64),
            reward=torch.tensor(1.0),
            batch_size=[],
        )
        assert (td[0]["obs"] == 1).all()
        assert (td[1]["obs"] == 0).all()

    def test_shared_storage(self):
        td = TensorDict.from_schema(
            self.SCHEMA, batch_size=self.BATCH, storage="shared"
        )
        assert td.is_shared()
        assert td["obs"].shape == torch.Size([8, 4, 4])
        assert td["action"].shape == torch.Size([8, 2])
        assert td["reward"].shape == torch.Size([8])
        assert (td["obs"] == 0).all()

    def test_shared_storage_write_read(self):
        td = TensorDict.from_schema(
            self.SCHEMA, batch_size=self.BATCH, storage="shared"
        )
        td[0] = TensorDict(
            obs=torch.ones(4, 4),
            action=torch.ones(2, dtype=torch.int64),
            reward=torch.tensor(1.0),
            batch_size=[],
        )
        assert (td[0]["obs"] == 1).all()
        assert (td[1]["obs"] == 0).all()

    def test_unknown_storage_raises(self):
        with pytest.raises(ValueError, match="Unknown storage backend"):
            TensorDict.from_schema(self.SCHEMA, batch_size=self.BATCH, storage="foobar")

    def test_callable_from_base(self, tmp_path):
        td = TensorDictBase.from_schema(
            {"x": ([3], torch.float32)},
            batch_size=[4],
            storage="memmap",
            prefix=str(tmp_path),
        )
        assert td.is_memmap()
        assert td["x"].shape == torch.Size([4, 3])


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
