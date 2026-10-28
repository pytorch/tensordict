# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import argparse
import contextlib
import gc
import json
import os
import sysconfig
import warnings

import pytest
import torch
from tensordict import (
    LazyStackedTensorDict,
    make_tensordict,
    pack_memmap,
    PersistentTensorDict,
    tensorclass,
    TensorDict,
    TypedTensorDict,
    UnbatchedTensor,
)
from tensordict._lazy import _CustomOpTensorDict
from tensordict._td import is_tensor_collection
from tensordict._torch_func import _POINTWISE_FUNCTIONS
from tensordict.base import TensorDictBase
from tensordict.memmap import MemoryMappedTensor
from tensordict.nn import TensorDictParams
from tensordict.tensorclass import NonTensorData
from tensordict.utils import (
    _decode_key_from_filesystem,
    _encode_key_for_filesystem,
    assert_allclose_td,
    expand_as_right,
    is_non_tensor,
    is_tensorclass,
    set_lazy_legacy,
)
from torch.func import hessian, jacfwd, jacrev

if os.getenv("PYTORCH_TEST_FBCODE"):
    from pytorch.tensordict.test._utils_internal import (
        get_available_devices,
        legacy_lazy_mode,
    )
else:
    from _utils_internal import (
        get_available_devices,
        legacy_lazy_mode,
    )


try:
    import h5py  # noqa

    _has_h5py = True
except ImportError:
    _has_h5py = False

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
        "ignore:No PYTORCH_KERNEL_CACHE_PATH or HOME environment variable set"
    ),
    pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning"),
]


class TestPointwiseOps:
    @pytest.mark.parametrize("name", _POINTWISE_FUNCTIONS)
    def test_torch_pointwise_functions(self, name):
        # torch.<name>(td) computes torch.<name> on each leaf (#1926)
        torch.manual_seed(0)
        x = torch.rand(3, 4) * 0.8 + 0.1
        y = torch.rand(3, 4) * 0.8 + 0.1
        args = {
            "add": (y,),
            "addcdiv": (y, y),
            "addcmul": (y, y),
            "bitwise_and": (y > 0.5,),
            "clamp": (0.3, 0.6),
            "clamp_max": (0.5,),
            "clamp_max_": (0.5,),
            "clamp_min": (0.5,),
            "clamp_min_": (0.5,),
            "div": (y,),
            "lerp": (y, 0.5),
            "logical_and": (y > 0.5,),
            "minimum": (y,),
            "mul": (y,),
            "pow": (y,),
            "rsub": (y,),
            "sub": (y,),
        }.get(name, ())
        if name == "bitwise_and":
            x = x > 0.3
        func = getattr(torch, name)
        td = TensorDict(a=x.clone(), nested=TensorDict(b=x.clone()), batch_size=[3, 4])
        result = func(td, *args)
        expected = func(x.clone(), *args)
        if name.endswith("_"):
            assert result is td
        assert isinstance(result, TensorDict)
        torch.testing.assert_close(result["a"], expected)
        torch.testing.assert_close(result["nested", "b"], expected)

    def test_torch_pointwise_functions_kwargs(self):
        td = TensorDict(a=torch.arange(4.0), batch_size=[4])
        torch.testing.assert_close(
            torch.clamp(td, min=1.5)["a"], td["a"].clamp(min=1.5)
        )
        torch.testing.assert_close(
            torch.sub(td, td, alpha=2)["a"], torch.sub(td["a"], td["a"], alpha=2)
        )
        torch.testing.assert_close(
            torch.add(td, td, alpha=2)["a"], torch.add(td["a"], td["a"], alpha=2)
        )
        # a tensor or a number as first argument still raises
        with pytest.raises(TypeError, match="takes a tensordict"):
            torch.add(torch.ones(4), td)
        with pytest.raises(TypeError, match="takes a tensordict"):
            torch.pow(2, td)

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

    @pytest.mark.parametrize("nested", [False, True])
    @pytest.mark.parametrize(
        "name",
        (
            "abs acos asin atan ceil cos cosh erf erfc exp expm1 floor frac lgamma "
            "log log10 log1p log2 reciprocal round sigmoid sign sin sinh sqrt tan "
            "tanh trunc"
        ).split(),
    )
    def test_unary_no_tensor(self, name, nested):
        # Without tensor leaves, the structure comes back, as with -td.
        td = TensorDict({"a": TensorDict({}, [3])} if nested else {}, [3])
        result = getattr(td, name)()
        assert result is not td
        assert result.batch_size == td.batch_size
        assert list(result.keys(True)) == list(td.keys(True))
        assert getattr(td, name + "_")() is td

    @pytest.mark.parametrize("nested", [False, True])
    def test_binary_no_tensor(self, nested):
        # Without tensor leaves, the structure comes back, as with td + 1.
        td = TensorDict({"a": TensorDict({}, [3])} if nested else {}, [3])
        results = [td - 1, 1 - td, td / 2, 2 / td, td**2, td - td, td / td, td**td]
        results += [td.sub(1, alpha=2), td.rsub(1, alpha=2)]
        results += [td.lerp(td, 0.5), td.addcmul(td, td), td.addcdiv(td, td)]
        for result in results:
            assert result is not td
            assert result.batch_size == td.batch_size
            assert list(result.keys(True)) == list(td.keys(True))
        assert td.sub_(1) is td
        assert td.sub_(td, alpha=2) is td
        assert td.div_(2) is td
        assert td.pow_(td) is td
        assert td.lerp_(td, 0.5) is td
        assert td.addcmul_(td, td) is td
        assert td.addcdiv_(td, td) is td
        norm = td.norm()
        assert norm.batch_size == ()
        assert list(norm.keys(True)) == list(td.keys(True))

    @pytest.mark.parametrize("keyword", [False, True])
    @pytest.mark.parametrize("lazy", [False, True])
    @pytest.mark.parametrize("inplace", [False, True])
    @pytest.mark.parametrize("end_shape", [(3,), ()])
    def test_lerp_tensor_end(self, end_shape, inplace, lazy, keyword):
        # A tensor end with a number weight, as torch.lerp takes them. The
        # end is broadcast to the batch dims, also when passed by keyword:
        # leaf c would take it on its last dim otherwise.
        torch.manual_seed(0)
        td = TensorDict(
            a=torch.randn(3),
            c=torch.randn(3, 3),
            nested=TensorDict(b=torch.randn(3, 4), batch_size=[3]),
            batch_size=[3],
        )
        end = torch.randn(end_shape)
        expected = td.apply(lambda x: torch.lerp(x, expand_as_right(end, x), 0.5))
        if lazy:
            td = LazyStackedTensorDict(*td.unbind(0), stack_dim=0)
        kwargs = {"end": end, "weight": 0.5} if keyword else {}
        args = () if keyword else (end, 0.5)
        if inplace:
            result = td.lerp_(*args, **kwargs)
            assert result is td
        else:
            result = td.lerp(*args, **kwargs)
            assert_allclose_td(torch.lerp(td, *args, **kwargs), expected)
        assert type(result) is type(td)
        assert_allclose_td(result, expected)

    @pytest.mark.parametrize("lazy", [False, True])
    @pytest.mark.parametrize("tensor_first", [False, True])
    @pytest.mark.parametrize(
        "op", ["lerp", "lerp_", "addcmul", "addcmul_", "addcdiv", "addcdiv_"]
    )
    def test_tensordict_and_tensor_operands(self, op, tensor_first, lazy):
        # A tensordict operand is paired by key and a tensor operand is
        # broadcast to the batch dims, also when they come together (#2963)
        torch.manual_seed(0)
        td = TensorDict(
            a=torch.randn(3, 2),
            nested=TensorDict(b=torch.randn(3, 2, 4), batch_size=[3, 2]),
            batch_size=[3, 2],
        )
        other_td = td.apply(lambda x: torch.rand_like(x) + 1)
        other = torch.rand(3, 2) + 1
        other_as_td = td.apply(lambda x: expand_as_right(other, x))
        args = [other, other_td]
        expected_args = [other_as_td, other_td]
        if not tensor_first:
            args, expected_args = args[::-1], expected_args[::-1]
        expected = getattr(td.clone(), op)(*expected_args)
        if lazy:
            td = LazyStackedTensorDict(*td.unbind(1), stack_dim=1)
        result = getattr(td, op)(*args)
        if op.endswith("_"):
            assert result is td
        assert_allclose_td(result, expected)

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

    @pytest.mark.parametrize(
        "name", ["clamp_min", "clamp_max", "clamp_min_", "clamp_max_"]
    )
    def test_clamp_error_is_raised(self, name):
        td = TensorDict(a=-torch.ones(4, 3), batch_size=[4])
        other = TensorDict(a=torch.zeros(4, 5), batch_size=[4])
        with pytest.raises(RuntimeError, match="must match the size"):
            getattr(td, name)(other)

    @pytest.mark.parametrize("kind", ["dense", "lazy", "locked", "torch", "keyword"])
    @pytest.mark.parametrize(
        "bounds",
        [
            "number,number",
            "number,None",
            "None,number",
            "0d,0d",
            "batch,batch",
            "batch,None",
            "td,td",
            "None,td",
        ],
    )
    def test_clamp_out(self, bounds, kind):
        # clamp copies its result into the leaves of out and returns it (#2961)
        def make_td():
            return TensorDict(
                a=torch.tensor([[-2.0, 0.5], [1.0, 3.0]]),
                nested=TensorDict(b=torch.tensor([0.2, -1.0]), batch_size=[2]),
                batch_size=[2],
            )

        values = {
            "number": (0.0, 1.0),
            "None": (None, None),
            "0d": (torch.tensor(0.0), torch.tensor(1.0)),
            "batch": (torch.tensor([0.0, 0.5]), torch.tensor([1.0, 2.0])),
            "td": (make_td() * 0, make_td() * 0 + 1),
        }
        low, high = (values[name][i] for i, name in enumerate(bounds.split(",")))
        td = make_td()
        out = make_td().zero_()
        if kind == "lazy":
            td = LazyStackedTensorDict(*td.unbind(0), stack_dim=0)
            out = LazyStackedTensorDict(*out.unbind(0), stack_dim=0)
        args, kwargs = (low, high), {}
        if kind == "keyword":
            args, kwargs = (), {"min": low, "max": high}
        expected = td.clamp(*args, **kwargs)
        tds = out.tensordicts if kind == "lazy" else [out]
        data_ptrs = [x.data_ptr() for t in tds for x in t.values(True, True)]
        if kind == "locked":
            out.lock_()
        if kind == "torch":
            result = torch.clamp(td, *args, **kwargs, out=out)
        else:
            result = td.clamp(*args, **kwargs, out=out)
        assert result is out
        assert [x.data_ptr() for t in tds for x in t.values(True, True)] == data_ptrs
        assert_allclose_td(out, expected)

    def test_clamp_out_errors(self):
        td = TensorDict(a=torch.zeros(2), batch_size=[2])
        with pytest.raises(ValueError, match="isn't implemented"):
            td.clamp(None, None, out=td.clone())
        for bound in (1.0, torch.tensor(1.0), torch.ones(2)):
            with pytest.raises(ValueError, match="Mixed tensordict"):
                td.clamp(td, bound)
            with pytest.raises(ValueError, match="Mixed tensordict"):
                td.clamp(bound, td, out=td.clone())

    def test_clamp_out_checks_out(self):
        # out must have the keys and dtypes of the result: nothing is dropped
        # or cast (#2961)
        td = TensorDict(x=torch.arange(3.0), z=torch.arange(3.0), batch_size=[3])
        with pytest.raises(KeyError, match="lacks {'z'}"):
            td.clamp(0.5, 1.5, out=TensorDict(x=torch.zeros(3), batch_size=[3]))
        with pytest.raises(KeyError, match="extra keys {'y'}"):
            td.clamp(0.5, 1.5, out=td.clone().set("y", torch.zeros(3)))
        td = TensorDict(x=torch.tensor([1, 2, 3]), batch_size=[3])
        with pytest.raises(RuntimeError, match="Found dtype torch.int64"):
            td.clamp(0.4, 2.5, out=td.clone())
        # torch.clamp raises for the same leaves
        with pytest.raises(RuntimeError, match="Found dtype Long"):
            torch.clamp(td["x"], 0.4, 2.5, out=td["x"].clone())
        out = td.clone()
        assert td.clamp(1, 2, out=out) is out
        assert (out["x"] == torch.tensor([1, 2, 2])).all()

    def test_clamp_out_empty(self):
        # an empty tensordict returns out, also with a tensor bound (#2961)
        td = TensorDict(batch_size=[2])
        out = TensorDict(batch_size=[2])
        assert td.clamp(torch.tensor([0.0, 1.0]), None, out=out) is out
        assert td.clamp(0.0, 1.0, out=out) is out

    def test_torch_clamp_out_tensorclass(self):
        @tensorclass
        class TC:
            x: torch.Tensor

        tc = TC(x=torch.arange(3.0), batch_size=[3])
        out = tc.clone()
        assert torch.clamp(tc, 0.5, 2.0, out=out) is out
        assert (out.x == torch.tensor([0.5, 1.0, 2.0])).all()

    @pytest.mark.parametrize(
        "op", ["lerp", "lerp_", "addcmul", "addcmul_", "addcdiv", "addcdiv_"]
    )
    @pytest.mark.parametrize("tensor_first", [False, True])
    def test_nested_lazy_and_tensor_operands(self, op, tensor_first):
        # With a lazy stack nested in the tensordict, each member gets its
        # slice of a batch-shaped tensor, as for a dense tensordict (#3005)
        def make_td(lazy):
            n = TensorDict(a=torch.arange(6.0).view(2, 3) + 1, batch_size=[2, 3])
            if lazy:
                n = LazyStackedTensorDict(*n.unbind(0))
            return TensorDict(n=n, batch_size=[2, 3])

        def make_args(lazy):
            args = [make_td(lazy) + 10, torch.arange(6.0).view(2, 3) / 6]
            return args[::-1] if tensor_first else args

        td = make_td(True)
        result = getattr(td, op)(*make_args(True))
        expected = getattr(make_td(False), op)(*make_args(False))
        assert isinstance(result["n"], LazyStackedTensorDict)
        torch.testing.assert_close(result["n", "a"], expected["n", "a"])
        if op.endswith("_"):
            assert result is td

    @pytest.mark.parametrize("op", ["maximum", "minimum"])
    @pytest.mark.parametrize("other_type", ["tensordict", "float", "tensor"])
    def test_maximum_minimum_unbatched(self, op, other_type):
        def make_td():
            return TensorDict(
                a=torch.tensor([[0.0, 1.0, 2.0]]),
                unbatched=UnbatchedTensor(torch.tensor([0.0, 1.0, 2.0, 3.0])),
                batch_size=[1],
            )

        if other_type == "tensordict":
            other = make_td().apply(lambda x: torch.full_like(x, 1.5))
        elif other_type == "float":
            other = 1.5
        else:
            other = torch.tensor(1.5)
        torch_op = getattr(torch, op)
        expected = make_td().apply(lambda x: torch_op(x, torch.tensor(1.5)))

        td = make_td()
        result = getattr(td, op)(other)
        assert isinstance(result.get("unbatched"), UnbatchedTensor)
        assert (result == expected).all()

        data_ptrs = {key: td.get(key).data_ptr() for key in td.keys()}
        result = getattr(td, f"{op}_")(other)
        assert result is td
        assert isinstance(td.get("unbatched"), UnbatchedTensor)
        assert {key: td.get(key).data_ptr() for key in td.keys()} == data_ptrs
        assert (td == expected).all()

    @pytest.mark.parametrize("lazy", [False, True])
    @pytest.mark.parametrize("op", ["maximum", "minimum", "clamp_min", "clamp_max"])
    def test_inplace_broadcast_tensor_max_min(self, op, lazy):
        # The in-place ops broadcast a tensor to the batch dims, as the
        # out-of-place ones do (#2962)
        torch.manual_seed(0)
        td = TensorDict(
            a=torch.randn(2, 3),
            nested=TensorDict(b=torch.randn(2, 3, 4), batch_size=[2, 3]),
            batch_size=[2, 3],
        )
        other = torch.randn(3)
        expected = getattr(td, op)(other)
        if lazy:
            td = LazyStackedTensorDict(*td.unbind(1), stack_dim=1)
        tds = td.tensordicts if lazy else [td]
        data_ptrs = [x.data_ptr() for t in tds for x in t.values(True, True)]
        assert getattr(td, op + "_")(other) is td
        assert [x.data_ptr() for t in tds for x in t.values(True, True)] == data_ptrs
        assert_allclose_td(td, expected)

    @pytest.mark.parametrize("op", ["maximum_", "minimum_", "clamp_min_", "clamp_max_"])
    def test_inplace_tensor_empty_and_unbatched(self, op):
        # An empty tensordict and an UnbatchedTensor leaf keep their own
        # paths for a tensor with dims (#2962 changes neither)
        td = TensorDict(batch_size=[2])
        assert getattr(td, op)(torch.ones(3)) is td
        other = torch.tensor([-1.0, 1.0])
        td = TensorDict(
            a=torch.zeros(2, 2), u=UnbatchedTensor(torch.zeros(3, 2)), batch_size=[2]
        )
        # every leaf meets the tensor as torch broadcasts it, from the right
        torch_op = {"maximum_": "clamp_min", "minimum_": "clamp_max"}.get(op, op[:-1])
        assert getattr(td, op)(other) is td
        assert (td["a"] == getattr(torch.zeros(2, 2), torch_op)(other)).all()
        assert (td["u"] == getattr(torch.zeros(3, 2), torch_op)(other)).all()

    @pytest.mark.parametrize("op", ["lerp", "addcmul", "addcdiv", "clamp"])
    def test_empty_tensordict_and_tensor_operands(self, op):
        # An empty tensordict gives an empty result, whatever the keys of a
        # tensordict operand (#2963 changes only tensordicts with leaves)
        td = TensorDict(batch_size=[2])
        result = getattr(td, op)(
            TensorDict(a=torch.ones(2), batch_size=[2]), torch.ones(2)
        )
        assert result.is_empty()
        assert result.batch_size == td.batch_size

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
        td_mod = td % torch.full(shape, 2.0)
        assert td_mod.shape == broadcast_shape
        assert (td_mod == td % 2.0).all()
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
    @pytest.mark.parametrize("alpha", [None, 2])
    def test_rsub_broadcast_tensor(self, shape, alpha):
        # torch.Tensor has no rsub method, so the leaves go through torch.rsub
        torch.manual_seed(0)
        td = TensorDict(
            a=torch.randn(3, 4), b={"c": torch.randn(3, 4, 5)}, batch_size=(3, 4)
        )
        other = torch.randn(shape)
        kwargs = {} if alpha is None else {"alpha": alpha}
        results = [td.rsub(other, **kwargs)]
        if alpha is None:
            results.append(other - td)
        for result in results:
            assert result.shape == torch.broadcast_shapes(shape, td.shape)
            for key in ("a", ("b", "c")):
                leaf = td[key]
                other_leaf = other[(..., *[None] * (leaf.ndim - td.ndim))]
                torch.testing.assert_close(
                    result[key], torch.rsub(leaf, other_leaf, **kwargs)
                )

    @pytest.mark.parametrize("alpha", [None, 1, 2, -0.3])
    @pytest.mark.parametrize("other_kind", ["tensordict", "number", "0-dim tensor"])
    def test_rsub_alpha_scales_self(self, other_kind, alpha):
        # Like torch.rsub, rsub computes other - alpha * self (#1950)
        torch.manual_seed(0)
        td = TensorDict(
            a=torch.randn(3, 4), b={"c": torch.randn(3, 4, 5)}, batch_size=(3, 4)
        )
        if other_kind == "tensordict":
            other = td.apply(torch.randn_like)
        elif other_kind == "number":
            other = 1.5
        else:
            other = torch.tensor(1.5)
        kwargs = {} if alpha is None else {"alpha": alpha}
        result = td.rsub(other, **kwargs)
        for key in ("a", ("b", "c")):
            other_leaf = other[key] if other_kind == "tensordict" else other
            torch.testing.assert_close(
                result[key], torch.rsub(td[key], other_leaf, **kwargs)
            )

    @pytest.mark.parametrize("alpha", [1, 2, -0.3])
    @pytest.mark.parametrize("op", ["add", "add_", "sub", "sub_"])
    def test_add_sub_alpha_number(self, op, alpha):
        # Like torch, add and sub scale a number other by alpha (#2495)
        torch.manual_seed(0)
        td = TensorDict(
            a=torch.randn(3, 4), b={"c": torch.randn(3, 4, 5)}, batch_size=(3, 4)
        )
        expected = td.apply(lambda x: getattr(x, op.rstrip("_"))(1.5, alpha=alpha))
        result = getattr(td, op)(1.5, alpha=alpha)
        if op.endswith("_"):
            assert result is td
        for key in ("a", ("b", "c")):
            torch.testing.assert_close(result[key], expected[key])

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
        td_mod = td % td.new_full(shape, 2.0)
        assert td_mod.shape == broadcast_shape
        assert (td_mod == td % 2.0).all()
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

    @pytest.mark.parametrize("locked", [True, False])
    @pytest.mark.parametrize("shape", [(4,), (3, 4)])
    def test_inplace_broadcast_tensor(self, shape, locked):
        # In-place ops with a tensor that broadcasts to the batch return self
        td = TensorDict(
            a=torch.ones(3, 4), b={"c": torch.ones(3, 4, 5)}, batch_size=(3, 4)
        )
        if locked:
            td.lock_()
        other = torch.full(shape, 2.0)
        expected = td.clone()
        for name, args in [
            ("add_", (other,)),
            ("sub_", (other,)),
            ("mul_", (other,)),
            ("div_", (other,)),
            ("pow_", (other,)),
            ("addcmul_", (other, other)),
            ("addcdiv_", (other, other)),
            ("maximum_", (other,)),
            ("minimum_", (other,)),
            ("clamp_min_", (other,)),
            ("clamp_max_", (other,)),
        ]:
            assert getattr(td, name)(*args) is td
            expected = getattr(expected, name[:-1])(*args)
            assert_allclose_td(td, expected)
        alias = td
        td += other
        td -= other
        td *= other
        td /= other
        td **= other
        assert td is alias
        assert td.is_locked is locked
        assert_allclose_td(td, ((expected + other - other) * other / other) ** other)
        # Out-of-place ops and comparisons still return a new tensordict
        assert td.add(other) is not td
        assert (td == other) is not td

    def test_inplace_broadcast_tensor_keeps_class(self):
        other = torch.full((3, 4), 2.0)
        params = TensorDictParams(TensorDict(a=torch.ones(3, 4), batch_size=(3, 4)))
        alias = params
        with torch.no_grad():
            params *= other
        assert params is alias
        assert (params["a"] == 2).all()

        @tensorclass
        class MyClass:
            x: torch.Tensor

        data = MyClass(x=torch.ones(3, 4), batch_size=(3, 4))
        alias = data
        data *= other
        assert data is alias
        assert (data.x == 2).all()

    def test_inplace_pairs_operands_by_key(self):
        # lerp_, addcdiv_ and addcmul_ pair the leaves of a tensordict operand
        # by key, as add_ does, also when its keys come in another order
        td = TensorDict(a=torch.zeros(3), n={"b": torch.zeros(3)}, batch_size=[3])
        other = TensorDict(
            n={"b": torch.full((3,), 2.0)}, a=torch.ones(3), batch_size=[3]
        )
        ones = other.apply(torch.ones_like)
        for name, args, kwargs in [
            ("lerp_", (other, 1.0), {}),
            ("addcmul_", (other, ones), {"value": 1.0}),
            ("addcdiv_", (other, ones), {"value": 1.0}),
        ]:
            result = getattr(td.clone(), name)(*args, **kwargs)
            assert (result["a"] == 1).all(), name
            assert (result["n", "b"] == 2).all(), name

    @pytest.mark.parametrize("op", ["lerp", "addcmul", "addcdiv"])
    @pytest.mark.parametrize("container", ["td", "lazy_stack"])
    def test_pairs_operands_by_key(self, op, container):
        # lerp, addcmul and addcdiv pair the leaves of every tensordict operand
        # by key, also when its keys come in another order (#1980)
        torch.manual_seed(0)

        def make(*, reverse):
            if reverse:
                return TensorDict(n={"b": torch.rand(3) + 1}, a=torch.rand(3) + 1)
            return TensorDict(a=torch.rand(3) + 1, n={"b": torch.rand(3) + 1})

        if container == "lazy_stack":
            # the members of each stack differ in key order
            td = LazyStackedTensorDict(make(reverse=False), make(reverse=True))
            other1 = LazyStackedTensorDict(make(reverse=True), make(reverse=False))
            other2 = LazyStackedTensorDict(make(reverse=True), make(reverse=False))
        else:
            td = make(reverse=False)
            other1 = make(reverse=True)
            other2 = make(reverse=True)
        if op == "lerp":
            # a weight below 1 so that the result depends on the pairing
            other2 = other2 / 4
        torch_op = getattr(torch, op)
        result = getattr(td, op)(other1, other2)
        for key in td.keys(True, True):
            torch.testing.assert_close(
                result[key], torch_op(td[key], other1[key], other2[key])
            )

        # an operand that lacks a key of td raises instead of pairing by position
        other_keys = TensorDict(a=torch.rand(3), d={"b": torch.rand(3)})
        if container == "lazy_stack":
            other_keys = LazyStackedTensorDict(other_keys, other_keys.clone())
        with pytest.raises(KeyError):
            getattr(td, op)(other_keys, other2)
        with pytest.raises(KeyError):
            getattr(td, op)(other1, other_keys)

    @pytest.mark.skipif(not _has_h5py, reason="h5py not available")
    def test_inplace_broadcast_tensor_h5(self, tmp_path):
        # A PersistentTensorDict hands out copies of its datasets: the in-place
        # op writes its result back to the file and returns self (#2217)
        td = TensorDict(a=torch.ones(3, 4), batch_size=(3, 4))
        h5 = PersistentTensorDict.from_dict(td, filename=str(tmp_path / "td.h5"))
        out = h5.add_(torch.full((3, 4), 2.0))
        assert out is h5
        assert (h5["a"] == 3).all()

    def test_inplace_broadcast_tensor_lazy_hetero(self):
        # The tensors of this lazy stack differ in shape, so they cannot be
        # stacked: the in-place op still writes into the members
        def make_stack():
            return LazyStackedTensorDict(
                *[
                    TensorDict(x=torch.zeros(2, 3, i), batch_size=[2])
                    for i in (1, 2, 3)
                ],
                stack_dim=1,
            )

        class TypedX(TypedTensorDict):
            x: torch.Tensor

        stack, source, inner_source = make_stack(), make_stack(), make_stack()
        nested = TensorDict(a=torch.zeros(2, 3), s=make_stack(), batch_size=[2, 3])
        typed = TypedX.from_tensordict(source)
        typed_twice = TypedX.from_tensordict(TypedX.from_tensordict(inner_source))
        for td in (stack, nested, typed, typed_twice):
            td.add_(torch.ones(2, 3))
        for members in (
            stack.tensordicts,
            nested["s"].tensordicts,
            source.tensordicts,
            inner_source.tensordicts,
        ):
            assert all((member["x"] == 1).all() for member in members)
        assert (nested["a"] == 1).all()

    @staticmethod
    def _typed_lazy_stack(kind, sizes, wrappers, tmp_path):
        # #3392: a lazy stack of the given kind of members, whose leaves x
        # (and y) have the last dimension in sizes and hold 2.0, under
        # wrappers TypedTensorDicts. Returns the wrapper, the members that
        # hold the leaves with their keys and shapes, and the keys that only
        # one member holds.
        class TypedX(TypedTensorDict):
            x: torch.Tensor

        class TypedXY(TypedTensorDict):
            x: torch.Tensor
            y: torch.Tensor

        cls, fields = (TypedXY, ["x", "y"]) if kind == "two_fields" else (TypedX, ["x"])
        leaves = []
        exclusive = set()

        def member(i):
            td = TensorDict(
                {field: torch.full((2, 3, i), 2.0) for field in fields},
                batch_size=[2],
            )
            if kind == "extra_keys":
                key = f"e{len(leaves)}"
                td[key] = torch.full((2, 3), 2.0)
                exclusive.add(key)
            elif kind == "nested":
                td["n"] = TensorDict(z=torch.full((2, 3, i), 2.0), batch_size=[2, 3])
            elif kind == "memmap":
                td = td.memmap(tmp_path / str(len(leaves)))
            elif kind == "h5":
                td = PersistentTensorDict.from_dict(
                    td, filename=str(tmp_path / f"{len(leaves)}.h5")
                )
            leaves.append((td, {key: td[key].shape for key in td.keys(True, True)}))
            return td

        if kind == "stack_of_stacks":
            members = [
                LazyStackedTensorDict(member(i), member(i), stack_dim=1) for i in sizes
            ]
        else:
            members = [member(i) for i in sizes]
        stack = LazyStackedTensorDict(*members, stack_dim=1)
        if kind == "locked":
            stack.lock_()
        td = stack
        for _ in range(wrappers):
            td = cls.from_tensordict(td)
        return td, leaves, exclusive

    _typed_lazy_kinds = [
        "dense",
        "locked",
        "memmap",
        pytest.param(
            "h5",
            marks=pytest.mark.skipif(not _has_h5py, reason="h5py not available"),
        ),
        "two_fields",
        "extra_keys",
        "nested",
        "stack_of_stacks",
    ]

    @pytest.mark.parametrize(
        "op,args,expected",
        [
            ("add_", ("other",), 5.0),
            ("sub_", ("other",), -1.0),
            ("mul_", ("other",), 6.0),
            ("pow_", (2,), 4.0),
            ("lerp_", ("other", 0.5), 2.5),
            ("zero_", (), 0.0),
        ],
    )
    @pytest.mark.parametrize("wrappers", [1, 2, 3])
    @pytest.mark.parametrize("sizes", [(1, 2, 3), (2, 2, 2)], ids=["hetero", "same"])
    @pytest.mark.parametrize("kind", _typed_lazy_kinds)
    def test_inplace_typed_lazy_keys(
        self, op, args, expected, wrappers, sizes, kind, tmp_path
    ):
        # #3392: a TypedTensorDict that wraps another one over a lazy stack
        # wrote entries such as ("0", "x") into every member, or raised
        # partway through the write, whether or not the members differ in
        # shape. Each member keeps only its own keys, with any number of
        # wrappers, for every kind of member and for the in-place methods
        # that read copies of the leaves: unary, binary with a tensor
        # operand, and ternary. The keys that only one member holds are left
        # unchanged, as with one wrapper (#2739).
        td, leaves, exclusive = self._typed_lazy_stack(kind, sizes, wrappers, tmp_path)
        other = torch.full(td.batch_size, 3.0)
        args = [other if arg == "other" else arg for arg in args]
        assert getattr(td, op)(*args) is td
        for leaf, shapes in leaves:
            assert sorted(leaf.keys(True, True), key=str) == sorted(shapes, key=str)
            for key, shape in shapes.items():
                assert leaf[key].shape == shape
                assert (leaf[key] == (2.0 if key in exclusive else expected)).all()

    @pytest.mark.parametrize("op", ["update_", "update_keys", "copy_"])
    @pytest.mark.parametrize("wrappers", [0, 1, 2, 3])
    @pytest.mark.parametrize("sizes", [(1, 2, 3), (2, 2, 2)], ids=["hetero", "same"])
    @pytest.mark.parametrize("kind", _typed_lazy_kinds)
    def test_update_typed_lazy_keys(self, op, wrappers, sizes, kind, tmp_path):
        # #3392: update_ and copy_ on a TypedTensorDict that wraps another
        # one over a lazy stack wrote entries such as ("0", "x") into every
        # member, raised partway through the write, or, with keys_to_update,
        # left the values unchanged. Each member keeps only its own keys and
        # takes the new values, with no wrapper, one, or more. Through a
        # wrapper, the keys that only one member holds are left unchanged
        # (#2739).
        td, leaves, exclusive = self._typed_lazy_stack(kind, sizes, wrappers, tmp_path)
        other = td.clone().apply(lambda x: torch.full_like(x, 3.0))
        if op == "update_keys":
            updated = {"x"}
            assert td.update_(other, keys_to_update=["x"]) is td
        else:
            updated = {key for _, shapes in leaves for key in shapes}
            if wrappers:
                updated -= exclusive
            assert getattr(td, op)(other) is td
        for leaf, shapes in leaves:
            assert sorted(leaf.keys(True, True), key=str) == sorted(shapes, key=str)
            for key, shape in shapes.items():
                assert leaf[key].shape == shape
                assert (leaf[key] == (3.0 if key in updated else 2.0)).all()


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
            warnings.filterwarnings(
                "ignore", message=r"`torch\.jit\.script`.*", category=FutureWarning
            )
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
            warnings.filterwarnings(
                "ignore", message=r"`torch\.jit\.script`.*", category=FutureWarning
            )
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
            pytest.raises(RuntimeError, match="to_tensordict")
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

    @legacy_lazy_mode()
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
    @classmethod
    def td(cls):
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

    @pytest.mark.parametrize(
        "func,args",
        [
            (torch.zeros_like, ()),
            (torch.ones_like, ()),
            (torch.empty_like, ()),
            (torch.full_like, (2.0,)),
            (torch.rand_like, ()),
            (torch.randn_like, ()),
        ],
    )
    def test_like_kwargs(self, func, args, td):
        # requires_grad and memory_format are passed on to every leaf
        td = td.float()
        tdnew = func(td, *args, requires_grad=True)
        assert tdnew.batch_size == td.batch_size
        for key, leaf in tdnew.items(True, True):
            assert leaf.shape == td[key].shape
            assert leaf.requires_grad and leaf.is_leaf
        td_t = TensorDict(a=torch.randn(4, 3).T, batch_size=(3,))
        assert not func(td_t, *args)["a"].is_contiguous()
        tdnew = func(td_t, *args, memory_format=torch.contiguous_format)
        assert tdnew["a"].is_contiguous()
        # torch rejects unknown keywords before calling the override
        with pytest.raises(TypeError):
            func(td, *args, foo=1)


class TestMemmap:
    @pytest.mark.parametrize("archive", [False, True])
    @pytest.mark.parametrize("device", ["cpu", "meta"])
    @pytest.mark.parametrize(
        "op,kwargs",
        [
            ("squeeze", {"dim": -2}),
            ("squeeze", {}),
            ("unsqueeze", {"dim": -1}),
            ("view", {"size": [2, 3]}),
            ("transpose", {"dim0": -3, "dim1": -1}),
            ("permute", {"dims": [-1, -3, -2]}),
        ],
    )
    @legacy_lazy_mode()
    def test_saved_view_operation_roundtrip(
        self, tmp_path, monkeypatch, archive, device, op, kwargs
    ):
        # The stdlib backend serializes the torch.Size emitted by legacy view().
        monkeypatch.setattr("tensordict._utils_key_json._JSON_BACKEND", "json")
        source = TensorDict(
            {"x": torch.arange(24).reshape(2, 1, 3, 4)}, batch_size=[2, 1, 3]
        )
        view = getattr(source, op)(**kwargs)
        assert isinstance(view, _CustomOpTensorDict)
        path = tmp_path / "view"
        view.memmap(path)
        if archive:
            path = pack_memmap(path, tmp_path / "view.tdz")
        loaded = TensorDict.load_memmap(path, device=device, allow_pickle=False)
        assert type(loaded) is type(view)
        assert loaded.batch_size == view.batch_size
        assert loaded["x"].shape == view["x"].shape
        assert loaded["x"].device == torch.device(device)
        if device == "cpu":
            torch.testing.assert_close(loaded["x"], view["x"])
            loaded.set_("x", torch.zeros_like(view["x"]))
            assert not loaded._source["x"].any()

    @pytest.mark.parametrize("archive", [False, True])
    @pytest.mark.parametrize(
        "shape,op,kwargs",
        [
            ([1, 2, 1], "squeeze", {"dim": (0, -1)}),
            ([1, 2, 1], "squeeze", {"dim": ()}),
            ([1, 2, 1], "squeeze", {}),
            ([], "unsqueeze", {"dim": -1}),
            ([1], "view", {"size": ()}),
            ([2, 3], "view", {"size": (-1,)}),
            ([0, 2], "view", {"size": (0,)}),
            ([], "permute", {"dims": ()}),
            ([2, 3], "permute", {"dims": (-1, -2)}),
            ([2, 3], "transpose", {"dim0": -1, "dim1": -2}),
        ],
    )
    def test_saved_view_operation_without_inverse(
        self, tmp_path, archive, shape, op, kwargs
    ):
        source = TensorDict({"x": torch.ones(shape)}, batch_size=shape)
        view = _CustomOpTensorDict(source, op, custom_op_kwargs=kwargs)
        path = tmp_path / "view"
        view.memmap(path)
        if archive:
            path = pack_memmap(path, tmp_path / "view.tdz")
        loaded = TensorDict.load_memmap(path, allow_pickle=False)
        assert type(loaded) is _CustomOpTensorDict
        assert loaded.inv_op is None
        assert loaded.batch_size == view.batch_size
        torch.testing.assert_close(loaded["x"], view["x"])

    @pytest.mark.parametrize("archive", [False, True])
    @legacy_lazy_mode()
    def test_saved_view_operation_pickle_policy(self, tmp_path, archive):
        source = TensorDict({"x": torch.ones(2), "value": {1, 2}}, batch_size=[2])
        view = source.unsqueeze(0)
        path = tmp_path / "view"
        view.memmap(path)
        if archive:
            path = pack_memmap(path, tmp_path / "view.tdz")
        with pytest.raises(RuntimeError, match="allow_pickle=True"):
            TensorDict.load_memmap(path, allow_pickle=False)
        loaded = TensorDict.load_memmap(path, allow_pickle=True)
        assert loaded._source["value"] == {1, 2}
        torch.testing.assert_close(loaded["x"], view["x"])

    @pytest.mark.parametrize("archive", [False, True])
    @pytest.mark.parametrize(
        "field,op",
        [
            (field, op)
            for field in ("custom_op", "inv_op")
            for op in ("clone", "zero_", "__getattribute__", "", [])
        ]
        + [("custom_op", None)],
    )
    @legacy_lazy_mode()
    def test_saved_view_operation_rejects_name(
        self, tmp_path, monkeypatch, archive, field, op
    ):
        path = tmp_path / "view"
        TensorDict({"x": torch.ones(2)}, [2]).unsqueeze(0).memmap(path)
        meta_path = path / "meta.json"
        metadata = json.loads(meta_path.read_text())
        metadata[field] = op
        meta_path.write_text(json.dumps(metadata))
        if archive:
            path = pack_memmap(path, tmp_path / "view.tdz")

        def unexpected_source_load(*args, **kwargs):
            pytest.fail("Source loaded before saved view operation validation")

        monkeypatch.setattr(TensorDict, "_load_memmap", unexpected_source_load)
        with pytest.raises(ValueError, match="saved view operation"):
            TensorDict.load_memmap(path)

    @pytest.mark.parametrize("archive", [False, True])
    @pytest.mark.parametrize("field", ["custom_op", "inv_op"])
    @pytest.mark.parametrize(
        "op,kwargs",
        [
            ("unsqueeze", {}),
            ("unsqueeze", {"dim": True}),
            ("squeeze", {"dim": None}),
            ("squeeze", {"dim": [0, "1"]}),
            ("view", {"size": [2.0]}),
            ("view", {"dtype": "float32"}),
            ("transpose", {"dim0": 0}),
            ("transpose", {"dim0": 0, "dim1": 1, "other": 2}),
            ("permute", {"dims": 0}),
            ("permute", []),
        ],
    )
    @legacy_lazy_mode()
    def test_saved_view_operation_rejects_kwargs(
        self, tmp_path, monkeypatch, archive, field, op, kwargs
    ):
        path = tmp_path / "view"
        TensorDict({"x": torch.ones(2)}, [2]).unsqueeze(0).memmap(path)
        meta_path = path / "meta.json"
        metadata = json.loads(meta_path.read_text())
        metadata[field] = op
        metadata[field + "_kwargs"] = kwargs
        meta_path.write_text(json.dumps(metadata))
        if archive:
            path = pack_memmap(path, tmp_path / "view.tdz")

        def unexpected_source_load(*args, **kwargs):
            pytest.fail("Source loaded before saved view operation validation")

        monkeypatch.setattr(TensorDict, "_load_memmap", unexpected_source_load)
        with pytest.raises(ValueError, match="saved view operation"):
            TensorDict.load_memmap(path)

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

    def test_memmap_robust_key_nested_pathlike(self, tmp_path):
        key = "../outside"
        td = TensorDict({key: {"x": torch.randn(3)}}, batch_size=[])
        prefix = tmp_path / "saved"

        td.memmap(prefix, robust_key=True)

        encoded = _encode_key_for_filesystem(key)
        assert (prefix / encoded / "meta.json").is_file()
        assert not (tmp_path / "outside").exists()
        loaded = TensorDict.load_memmap(prefix)
        assert_allclose_td(td, loaded)
        loaded_subtree = TensorDict.load_memmap(prefix, subpath=(key,))
        assert_allclose_td(td[key], loaded_subtree)

    def test_memmap_nested_collection_symlink_is_rejected(self, tmp_path):
        prefix = tmp_path / "saved"
        outside = tmp_path / "outside"
        prefix.mkdir()
        outside.mkdir()
        (prefix / "nested").symlink_to(outside, target_is_directory=True)
        td = TensorDict({"nested": {"x": torch.randn(3)}}, batch_size=[])

        with pytest.raises(RuntimeError, match="TensorDict through symlink"):
            td.memmap(prefix)

        assert not list(outside.iterdir())

    def test_memmap_robust_load_does_not_traverse_legacy_path(self, tmp_path):
        key = "../outside"
        td = TensorDict({key: torch.randn(3)}, batch_size=[])
        prefix = tmp_path / "saved"
        td.memmap(prefix, robust_key=True)

        encoded_file = prefix / f"{_encode_key_for_filesystem(key)}.memmap"
        encoded_file.replace(tmp_path / "outside.memmap")

        loaded = TensorDict.load_memmap(prefix, robust_key=True)
        assert key not in loaded

    def test_memmap_load_stats_each_path_once(self, tmp_path, monkeypatch):
        # Loading tests each leaf file and nested directory for existence once:
        # on a network file system every stat is a metadata-server round trip.
        td = TensorDict(
            a=torch.randn(3),
            empty=torch.zeros(0),
            nested=TensorDict(b=torch.randn(3), deeper=TensorDict(c=torch.randn(3))),
        )
        td.memmap(tmp_path)
        n_leaves = len(list(td.keys(True, True)))
        n_nested = 2
        stats = []
        stat = os.stat

        def counted_stat(path, *args, **kwargs):
            stats.append(path)
            return stat(path, *args, **kwargs)

        monkeypatch.setattr(os, "stat", counted_stat)
        loaded = TensorDict.load_memmap(tmp_path)
        monkeypatch.undo()
        assert len(stats) <= n_leaves + n_nested, [str(p) for p in stats]
        assert_allclose_td(loaded, td)

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
        test_keys = [
            "a/b/c",
            "path\\with\\backslashes",
            "key with spaces",
            "key:with:colons",
            "key*with*stars",
            "normal_key",
            "key%with%percent",
            "",
            ".",
            "..",
        ]

        encodings = set()
        for key in test_keys:
            encoded = _encode_key_for_filesystem(key, robust=True)
            assert encoded not in encodings
            encodings.add(encoded)
            decoded = _decode_key_from_filesystem(encoded)
            assert decoded == key, f"Failed: {key!r} -> {encoded!r} -> {decoded!r}"

            # Legacy encoding should return key unchanged
            legacy = _encode_key_for_filesystem(key, robust=False)
            assert legacy == key

    @pytest.mark.parametrize("fmt", ["h5", "zarr", "tdz"])
    def test_save_unbatched_warns(self, tmp_path, fmt):
        # #3264: h5, zarr and .tdz files have no place for the UnbatchedTensor
        # class, so the leaves are saved as plain tensors. Saving warns once
        # per call site and writes the same file as before.
        if fmt == "h5":
            pytest.importorskip("h5py")
        elif fmt == "zarr":
            zarr = pytest.importorskip("zarr")
            if int(zarr.__version__.split(".")[0]) < 3:
                pytest.skip("tensordict's zarr support needs zarr>=3")
        save, load = {
            "h5": (TensorDict.to_h5, PersistentTensorDict.from_h5),
            "zarr": (TensorDict.to_zarr, PersistentTensorDict.from_zarr),
            "tdz": (TensorDict.memmap, TensorDict.load_memmap),
        }[fmt]
        data = torch.arange(12.0).view(3, 4)
        td = TensorDict(x=torch.zeros(3), u=UnbatchedTensor(data), batch_size=[3])
        td["nested"] = TensorDict(v=UnbatchedTensor(data), batch_size=[3])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("default")
            save(td, tmp_path / f"unbatched.{fmt}")
        caught = [w for w in caught if "UnbatchedTensor leaves" in str(w.message)]
        assert len(caught) == 1
        assert caught[0].category is UserWarning
        assert caught[0].filename == __file__
        loaded = load(tmp_path / f"unbatched.{fmt}")
        for key in ("u", ("nested", "v")):
            assert not isinstance(loaded.get(key), UnbatchedTensor)
            torch.testing.assert_close(loaded.get(key), data)

        plain = TensorDict(x=torch.zeros(3), u=data, batch_size=[3])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            save(plain, tmp_path / f"plain.{fmt}")
        assert not [w for w in caught if "UnbatchedTensor leaves" in str(w.message)]


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
