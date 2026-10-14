# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Operations on every tensordict type, checked against its dense copy.

Each tensordict type of the pool of :class:`TestTensorDictsBase` (lazy stacks,
sub-tensordicts, lazy views, memory-mapped and h5 tensordicts, parameters,
tensorclasses, typed tensordicts, ...) must behave like its dense copy
``td.to_tensordict()``. For each operation of :data:`_OPS`, the tests run the
operation on a new tensordict of the pool and on its dense copy, and compare the
two results with :func:`_difference`: tensordicts and tensorclasses through
``to_tensordict()`` (keys, batch size, names, device and values), tensors with
:func:`torch.testing.assert_close` and non-tensor data through ``tolist()``. An
operation that raises the same exception type on both sides agrees. An in-place
operation must also agree on whether it returns ``self`` and on the state that
it leaves.

The cases that fail today are listed in ``conformance_known_failures.json``.
Each test compares its failing cases with that list: a failure that is not
listed is a regression, and a listed case that passes must be removed from the
list. Run with ``TENSORDICT_UPDATE_KNOWN_FAILURES=1`` to rewrite the list.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import pickle

import pytest
import torch
from _utils_internal import TestTensorDictsBase
from tensordict import is_tensor_collection, TensorDict, UnbatchedTensor
from tensordict.base import _is_leaf_nontensor
from tensordict.utils import is_non_tensor

_KNOWN_FAILURES_FILE = pathlib.Path(__file__).with_name(
    "conformance_known_failures.json"
)
_KNOWN_FAILURES = json.loads(_KNOWN_FAILURES_FILE.read_text())
_UPDATE_KNOWN_FAILURES = bool(os.environ.get("TENSORDICT_UPDATE_KNOWN_FAILURES"))

pytestmark = pytest.mark.filterwarnings(
    "ignore:A destination should be provided when cloning a PersistentTensorDict"
)

_CONTAINERS = list(
    dict.fromkeys(
        name
        for name, device in TestTensorDictsBase.TYPES_DEVICES
        if torch.device(device).type == "cpu"
    )
)

# Every tensordict of the pool has batch size [4, 3, 2, 1] and the keys "a"
# (float), "b" (float) and "c" (int).
_MASK = (torch.arange(24) % 3 == 0).view(4, 3, 2, 1)


def _inplace(method):
    """Wrap an in-place ``method`` so that its result shows what it did to its input."""

    def op(td):
        with torch.no_grad():
            result = method(td)
        return {"returns self": result is td, "state": td}

    return op


def _entry_class_mismatches(td):
    return [
        key for key in td.keys(True) if td.entry_class(key) is not type(td.get(key))
    ]


def _pickle_round_trip(td):
    return pickle.loads(pickle.dumps(td))


_OPS = {
    "keys": {
        "keys()": lambda td: list(td.keys()),
        "keys(True)": lambda td: list(td.keys(True)),
        "keys(True, True)": lambda td: list(td.keys(True, True)),
        "keys(True, True, is_leaf=_is_leaf_nontensor)": lambda td: list(
            td.keys(True, True, is_leaf=_is_leaf_nontensor)
        ),
        "items(True, True)": lambda td: list(td.items(True, True)),
        "values(True, True)": lambda td: list(td.values(True, True)),
        "len()": len,
        "batch_size, names, device": lambda td: [td.batch_size, td.names, td.device],
        "is_contiguous()": lambda td: td.is_contiguous(),
        "clear_device_()": _inplace(lambda td: td.clear_device_()),
        "get('a')": lambda td: td.get("a"),
        "get('missing', None)": lambda td: td.get("missing", None),
        "entry_class(key) is type(get(key))": _entry_class_mismatches,
    },
    "select": {
        "select('a')": lambda td: td.select("a"),
        "select('a', 'b')": lambda td: td.select("a", "b"),
        "select('missing')": lambda td: td.select("missing"),
        "select('a', 'missing', strict=False)": lambda td: td.select(
            "a", "missing", strict=False
        ),
        "select(('my_nested_td', 'inner'), strict=False)": lambda td: td.select(
            ("my_nested_td", "inner"), strict=False
        ),
        "exclude('a')": lambda td: td.exclude("a"),
        "exclude('a', 'missing')": lambda td: td.exclude("a", "missing"),
        "exclude(('my_nested_td', 'inner'))": lambda td: td.exclude(
            ("my_nested_td", "inner")
        ),
        "clone()": lambda td: td.clone(),
        "clone(recurse=False)": lambda td: td.clone(recurse=False),
        "contiguous()": lambda td: td.contiguous(),
        "contiguous(canonical=True)": lambda td: td.contiguous(canonical=True),
    },
    "pointwise": {
        "abs()": lambda td: td.abs(),
        "add(1)": lambda td: td.add(1),
        "mul(2)": lambda td: td.mul(2),
        "add(td)": lambda td: td.add(td),
        "mul(td)": lambda td: td.mul(td),
        "td == td": lambda td: td == td,
        "td != td": lambda td: td != td,
        "td < td": lambda td: td < td,
        "(td > 0) | (td < 1)": lambda td: (td > 0) | (td < 1),
        "(td > 0) ^ (td < 1)": lambda td: (td > 0) ^ (td < 1),
        "clamp(-1, 1)": lambda td: td.clamp(-1, 1),
        "abs_()": _inplace(lambda td: td.abs_()),
        "add_(1)": _inplace(lambda td: td.add_(1)),
        "mul_(2)": _inplace(lambda td: td.mul_(2)),
        "add_(td)": _inplace(lambda td: td.add_(td)),
        "mul_(td)": _inplace(lambda td: td.mul_(td)),
        "mul_(tensor(2))": _inplace(lambda td: td.mul_(torch.tensor(2))),
        "clamp_min_(-1)": _inplace(lambda td: td.clamp_min_(-1)),
        "clamp_max_(1)": _inplace(lambda td: td.clamp_max_(1)),
    },
    "reductions": {
        "sum()": lambda td: td.sum(),
        "sum(0)": lambda td: td.sum(0),
        "sum(-1, keepdim=True)": lambda td: td.sum(-1, keepdim=True),
        "sum((0, 2))": lambda td: td.sum((0, 2)),
        "mean(1, dtype=float32)": lambda td: td.mean(1, dtype=torch.float32),
        "mean((1, 2), keepdim=True, dtype=float32)": lambda td: td.mean(
            (1, 2), keepdim=True, dtype=torch.float32
        ),
        "amax(0)": lambda td: td.amax(0),
        "amax(1, keepdim=True)": lambda td: td.amax(1, keepdim=True),
        "prod(0, keepdim=True)": lambda td: td.prod(0, keepdim=True),
        "sum('feature')": lambda td: td.sum("feature"),
        "sum(reduce=True)": lambda td: td.sum(reduce=True),
        "amax(reduce=True)": lambda td: td.amax(reduce=True),
        "all(0)": lambda td: td.all(0),
        "any()": lambda td: td.any(),
    },
    "shape": {
        "reshape(-1)": lambda td: td.reshape(-1),
        "reshape(2, -1)": lambda td: td.reshape(2, -1),
        "view(-1)": lambda td: td.view(-1),
        "flatten()": lambda td: td.flatten(),
        "flatten(1, 2)": lambda td: td.flatten(1, 2),
        "unflatten(0, (2, 2))": lambda td: td.unflatten(0, (2, 2)),
        "squeeze()": lambda td: td.squeeze(),
        "squeeze(-1)": lambda td: td.squeeze(-1),
        "unsqueeze(0)": lambda td: td.unsqueeze(0),
        "unsqueeze(-1)": lambda td: td.unsqueeze(-1),
        "permute(3, 2, 1, 0)": lambda td: td.permute(3, 2, 1, 0),
        "transpose(0, 1)": lambda td: td.transpose(0, 1),
        "expand(2, 4, 3, 2, 1)": lambda td: td.expand(2, 4, 3, 2, 1),
        "chunk(2, 0)": lambda td: td.chunk(2, 0),
        "split(2, 0)": lambda td: td.split(2, 0),
        "split([1, 3], 0)": lambda td: td.split([1, 3], 0),
        "unbind(0)": lambda td: td.unbind(0),
        "unbind(1)": lambda td: td.unbind(1),
        "torch.stack([td, td], 1)": lambda td: torch.stack([td, td], 1),
        "torch.cat([td, td], 1)": lambda td: torch.cat([td, td], 1),
        "flip(0)": lambda td: td.flip(0),
        "roll(1, 0)": lambda td: td.roll(1, 0),
        "rot90(1, (0, 1))": lambda td: td.rot90(1, (0, 1)),
        "tile((2, 1, 1, 1))": lambda td: td.tile((2, 1, 1, 1)),
        "narrow(1, 0, 2)": lambda td: td.narrow(1, 0, 2),
        "broadcast_to((2, 4, 3, 2, 1))": lambda td: td.broadcast_to((2, 4, 3, 2, 1)),
        "repeat_interleave(2, 0)": lambda td: td.repeat_interleave(2, 0),
        "repeat_interleave(2, 2)": lambda td: td.repeat_interleave(2, 2),
        "densify()": lambda td: td.densify(),
        "get_item_shape('a')": lambda td: td.get_item_shape("a"),
    },
    "masking": {
        "where(mask, 0)": lambda td: td.where(_MASK, 0),
        "where(mask, zeros_like)": lambda td: td.where(
            _MASK, td.apply(torch.zeros_like)
        ),
        "torch.where(mask, td, 0)": lambda td: torch.where(_MASK, td, 0),
        "masked_fill(mask, -1)": lambda td: td.masked_fill(_MASK, -1),
        "masked_select(mask)": lambda td: td.masked_select(_MASK),
        "masked_fill_(mask, -1)": _inplace(lambda td: td.masked_fill_(_MASK, -1)),
        "masked_fill_(mask[:, 0, 0, 0], -1)": _inplace(
            lambda td: td.masked_fill_(_MASK[:, 0, 0, 0], -1)
        ),
        "masked_fill_(tensor(True), -1)": _inplace(
            lambda td: td.masked_fill_(torch.tensor(True), -1)
        ),
    },
    "filling": {
        "fill_('a', 2.0)": _inplace(lambda td: td.fill_("a", 2.0)),
        "fill_('my_nested_td', 2.0)": _inplace(
            lambda td: td.fill_("my_nested_td", 2.0)
        ),
        "fill_('missing', 2.0)": _inplace(lambda td: td.fill_("missing", 2.0)),
        "zero_()": _inplace(lambda td: td.zero_()),
        "detach_()": _inplace(lambda td: td.detach_()),
    },
    "update": {
        "update({'a': zeros})": _inplace(
            lambda td: td.update({"a": torch.zeros(4, 3, 2, 1, 5)})
        ),
        "update({'new': zeros})": _inplace(
            lambda td: td.update({"new": torch.zeros(4, 3, 2, 1)})
        ),
        "update(new=zeros)": _inplace(
            lambda td: td.update(new=torch.zeros(4, 3, 2, 1))
        ),
        "update(dense + 1)": _inplace(lambda td: td.update(td.to_tensordict() + 1)),
        "update(dense + 1, inplace=True)": _inplace(
            lambda td: td.update(td.to_tensordict() + 1, inplace=True)
        ),
        "update(dense + 1, keys_to_update=['a'])": _inplace(
            lambda td: td.update(td.to_tensordict() + 1, keys_to_update=["a"])
        ),
        "update({('n', 'x'): zeros})": _inplace(
            lambda td: td.update({("n", "x"): torch.zeros(4, 3, 2, 1)})
        ),
        "update(dense[0], update_batch_size=True)": _inplace(
            lambda td: td.update(td.to_tensordict()[0], update_batch_size=True)
        ),
        "update_({'a': zeros})": _inplace(
            lambda td: td.update_({"a": torch.zeros(4, 3, 2, 1, 5)})
        ),
        "update_(dense * 0)": _inplace(lambda td: td.update_(td.to_tensordict() * 0)),
        "update_({'a': tensor(0.)})": _inplace(
            lambda td: td.update_({"a": torch.tensor(0.0)})
        ),
        "update_(dense * 0, keys_to_update=['a'])": _inplace(
            lambda td: td.update_(td.to_tensordict() * 0, keys_to_update=["a"])
        ),
        "update_({'missing': zeros})": _inplace(
            lambda td: td.update_({"missing": torch.zeros(4, 3, 2, 1)})
        ),
        "update_at_({'a': zeros}, 0)": _inplace(
            lambda td: td.update_at_({"a": torch.zeros(3, 2, 1, 5)}, 0)
        ),
        "update_at_(dense[1:3] * 0, slice(1, 3))": _inplace(
            lambda td: td.update_at_(td.to_tensordict()[1:3] * 0, slice(1, 3))
        ),
        "update_at_({'a': tensor(0.)}, (..., 0))": _inplace(
            lambda td: td.update_at_({"a": torch.tensor(0.0)}, (Ellipsis, 0))
        ),
        "update_at_(dense[0] * 0, 0, keys_to_update=['a'])": _inplace(
            lambda td: td.update_at_(td.to_tensordict()[0] * 0, 0, keys_to_update=["a"])
        ),
    },
    "apply": {
        "apply(x + 1)": lambda td: td.apply(lambda x: x + 1),
        "apply(x + y, td)": lambda td: td.apply(lambda x, y: x + y, td),
        "apply_(x + 1)": _inplace(lambda td: td.apply_(lambda x: x + 1)),
        "apply_(None)": _inplace(lambda td: td.apply_(lambda x: None)),
        "apply_(x + 1, propagate_lock=True)": _inplace(
            lambda td: td.apply_(lambda x: x + 1, propagate_lock=True)
        ),
    },
    "serialization": {
        "to_dict()": lambda td: td.to_dict(),
        "TensorDict.from_dict(to_dict())": lambda td: TensorDict.from_dict(
            td.to_dict(), batch_size=td.batch_size
        ),
        "pickle": _pickle_round_trip,
        "state_dict()": lambda td: td.state_dict(),
        "to_tensordict()": lambda td: td.to_tensordict(),
        "to(float64)": lambda td: td.to(torch.float64),
    },
}


def _normalize(value):
    """Return ``value`` as plain lists, dicts, tensors and Python values."""
    if is_non_tensor(value):
        return {"non-tensor": value.tolist()}
    if is_tensor_collection(value):
        td = value.to_tensordict()
        return {
            "batch size": list(td.batch_size),
            "names": list(td.names),
            "device": td.device,
            "locked": value.is_locked,
            "leaves": {
                key: _normalize(leaf)
                for key, leaf in td.items(True, True, is_leaf=_is_leaf_nontensor)
            },
        }
    if isinstance(value, UnbatchedTensor):
        # an UnbatchedTensor and a plain tensor compare equal in assert_close
        return {"unbatched": value}
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, dict):
        return {key: _normalize(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)) and not _is_nested_key(value):
        return [_normalize(val) for val in value]
    return value


def _is_nested_key(value):
    return isinstance(value, tuple) and all(isinstance(key, str) for key in value)


def _show(value):
    if isinstance(value, (dict, list, torch.Tensor)):
        return type(value).__name__
    return repr(value)


def _difference(result, expected):
    """Describe how the normalized ``result`` differs from ``expected``, or return ``None``."""
    if isinstance(expected, torch.Tensor) and isinstance(result, torch.Tensor):
        try:
            torch.testing.assert_close(result, expected, equal_nan=True)
        except AssertionError as err:
            return str(err).strip().splitlines()[0]
        return None
    if type(result) is not type(expected):
        return f"{_show(result)}, dense {_show(expected)}"
    if isinstance(expected, dict):
        extra = sorted(set(result).difference(expected), key=str)
        missing = sorted(set(expected).difference(result), key=str)
        if extra or missing:
            return f"extra keys {extra}, missing keys {missing}"
        for key, val in expected.items():
            difference = _difference(result[key], val)
            if difference is not None:
                return f"{key}: {difference}"
        return None
    if isinstance(expected, list):
        if len(result) != len(expected):
            return f"length {len(result)}, dense {len(expected)}"
        for i, (res, exp) in enumerate(zip(result, expected)):
            difference = _difference(res, exp)
            if difference is not None:
                return f"[{i}]: {difference}"
        return None
    return None if result == expected else f"{result!r}, dense {expected!r}"


def _describe(err):
    return f"raises {type(err).__name__}: {str(err).splitlines()[0][:80] if str(err) else ''}"


def _run(op, td):
    """Return ``(normalized result, None)``, or ``(None, error)`` if ``op`` raises."""
    try:
        return _normalize(op(td)), None
    except Exception as err:
        return None, err


def _compare(make, op):
    """Run ``op`` on a new tensordict of ``make`` and on its dense copy; describe a difference."""
    torch.manual_seed(0)
    td = make("cpu")
    dense = td.to_tensordict()
    if td.is_locked:
        dense.lock_()
    expected, dense_error = _run(op, dense)
    result, error = _run(op, td)
    if dense_error is not None:
        if error is None:
            return f"returns, dense {_describe(dense_error)}"
        if type(error) is not type(dense_error):
            return f"{_describe(error)}; dense {_describe(dense_error)}"
        return None
    if error is not None:
        return _describe(error)
    difference = _difference(result, expected)
    return None if difference is None else difference[:160]


def _check_known_failures(key, failures):
    """Compare ``failures`` ({case: reason}) with the known failures of ``key``."""
    if _UPDATE_KNOWN_FAILURES:
        known = json.loads(_KNOWN_FAILURES_FILE.read_text())
        known[key] = sorted(failures)
        _KNOWN_FAILURES_FILE.write_text(
            json.dumps(dict(sorted(known.items())), indent=1) + "\n"
        )
        return
    known = set(_KNOWN_FAILURES.get(key, ()))
    new = {case: reason for case, reason in failures.items() if case not in known}
    fixed = sorted(known.difference(failures))
    message = []
    if new:
        message.append(
            f"{len(new)} cases fail and are not known failures:\n"
            + "\n".join(f"  {case}: {reason}" for case, reason in sorted(new.items()))
        )
    if fixed:
        message.append(
            f"{len(fixed)} known failures pass now; remove them from "
            f"{_KNOWN_FAILURES_FILE.name} under {key!r}:\n"
            + "\n".join(f"  {case}" for case in fixed)
        )
    assert not message, "\n".join(message)


class TestContainersMatchDense:
    """Every tensordict type gives what its dense copy gives (batch size [4, 3, 2, 1])."""

    @pytest.mark.parametrize("group", list(_OPS))
    @pytest.mark.parametrize("td_name", _CONTAINERS)
    def test_ops(self, td_name, group):
        make = getattr(TestTensorDictsBase, td_name)
        failures = {}
        for case, op in _OPS[group].items():
            difference = _compare(make, op)
            if difference is not None:
                failures[case] = difference
        _check_known_failures(f"{td_name} {group}", failures)


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
