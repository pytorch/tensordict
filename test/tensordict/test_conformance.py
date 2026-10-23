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

:class:`TestContainersCompile` runs the same operations through
``torch.compile(fullgraph=True, dynamic=True)`` and compares each result with
the eager result on the same type. Its known failures
are in ``conformance_compile_known_failures.json``. They change with the torch
version, so these tests run only with ``TENSORDICT_COMPILE_CONFORMANCE=1``,
which the CI job with the newest stable torch sets. Rewrite that list with the
torch of that job.

:class:`TestOneImplementation` checks that the subclasses of
``TensorDictBase`` override its public methods only where
:data:`_ALLOWED_OVERRIDES` gives a reason, and that the private methods they
override have a docstring.

:class:`TestSaveLoadRoundTrip` saves each tensordict of :data:`_ZOO` through
each save/load path of :data:`_SAVE_LOAD`, and checks that it loads back
unchanged, or that saving it raises as :data:`_ROUND_TRIP_OUTCOMES` says.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import pathlib
import pickle

import pytest
import tensordict
import torch
from _utils_internal import TestTensorDictsBase
from tensordict import (
    is_tensor_collection,
    LazyStackedTensorDict,
    NonTensorStack,
    tensorclass,
    TensorDict,
    UnbatchedTensor,
)
from tensordict.base import _is_leaf_nontensor, TensorDictBase
from tensordict.utils import _is_tensorclass, is_non_tensor

_KNOWN_FAILURES_FILE = pathlib.Path(__file__).with_name(
    "conformance_known_failures.json"
)
_COMPILE_KNOWN_FAILURES_FILE = pathlib.Path(__file__).with_name(
    "conformance_compile_known_failures.json"
)
_KNOWN_FAILURES = {
    path: json.loads(path.read_text())
    for path in (_KNOWN_FAILURES_FILE, _COMPILE_KNOWN_FAILURES_FILE)
}
_UPDATE_KNOWN_FAILURES = bool(os.environ.get("TENSORDICT_UPDATE_KNOWN_FAILURES"))
_COMPILE_CONFORMANCE = bool(os.environ.get("TENSORDICT_COMPILE_CONFORMANCE"))

pytestmark = pytest.mark.filterwarnings(
    "ignore:A destination should be provided when cloning a PersistentTensorDict"
)

_CONTAINERS = list(
    dict.fromkeys(
        name
        for name, device in TestTensorDictsBase.TYPES_DEVICES
        if torch.device(device).type == "cpu"
    )
) + ["typed_lazy"]

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
        "list(td)": list,
    },
    "key_edits": {
        "popitem()": lambda td: [*td.popitem(), td],
        "del_('a')": _inplace(lambda td: td.del_("a")),
        "del_('missing')": _inplace(lambda td: td.del_("missing")),
        "del_(())": _inplace(lambda td: td.del_(())),
        "del_(('a', 'x'))": _inplace(lambda td: td.del_(("a", "x"))),
        "rename_key_('a', 'z')": _inplace(lambda td: td.rename_key_("a", "z")),
        "rename_key_('a', ())": _inplace(lambda td: td.rename_key_("a", ())),
        "rename_keys({'a': 'b', 'b': 'a'})": lambda td: td.rename_keys(
            {"a": "b", "b": "a"}
        ),
        "rename_keys({'a': ('n', 'x'), 'missing': 'z'}, strict=False)": lambda td: (
            td.rename_keys({"a": ("n", "x"), "missing": "z"}, strict=False)
        ),
        "rename_keys({'a': 'b'})": lambda td: td.rename_keys({"a": "b"}),
        "flatten_keys()": lambda td: td.flatten_keys(),
        "flatten_keys(inplace=True)": _inplace(
            lambda td: td.flatten_keys(inplace=True)
        ),
        "unflatten_keys()": lambda td: td.unflatten_keys(),
        "get_at('a', 0)": lambda td: td.get_at("a", 0),
        "get_at('missing', 0, None)": lambda td: td.get_at("missing", 0, None),
        "get_at(('a', 'x'), 0)": lambda td: td.get_at(("a", "x"), 0),
        "get(('a', 'x'), None)": lambda td: td.get(("a", "x"), None),
        "is_empty()": lambda td: td.is_empty(),
        "non_tensor_items(True)": lambda td: [
            key for key, _ in td.non_tensor_items(True)
        ],
        "'a' in td": lambda td: "a" in td,
        "('a', 'x') in td": lambda td: ("a", "x") in td,
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
        "contiguous(inplace=True)": _inplace(lambda td: td.contiguous(inplace=True)),
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
        "transpose(0, 2)": lambda td: td.transpose(0, 2),
        "transpose(1, 3)": lambda td: td.transpose(1, 3),
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
        "where(mask, 0, out=dense)": lambda td: (
            lambda out: [td.where(_MASK, 0, out=out) is out, out]
        )(td.to_tensordict().zero_()),
        "where(mask, dense.exclude('b'), pad=0)": lambda td: td.where(
            _MASK, td.to_tensordict().exclude("b"), pad=0
        ),
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
        "share_memory_()": _inplace(lambda td: td.share_memory_()),
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
        "copy_(dense * 0)": _inplace(lambda td: td.copy_(td.to_tensordict() * 0)),
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
    "setitem": {
        "td['a'] = zeros": _inplace(
            lambda td: td.__setitem__("a", torch.zeros(4, 3, 2, 1, 5))
        ),
        "td['new'] = zeros": _inplace(
            lambda td: td.__setitem__("new", torch.zeros(4, 3, 2, 1))
        ),
        "td[('n', 'x')] = zeros": _inplace(
            lambda td: td.__setitem__(("n", "x"), torch.zeros(4, 3, 2, 1))
        ),
        "td[0] = dense[0] * 0": _inplace(
            lambda td: td.__setitem__(0, td.to_tensordict()[0] * 0)
        ),
        "td[1:3] = {'a': zeros}": _inplace(
            lambda td: td.__setitem__(slice(1, 3), {"a": torch.zeros(2, 3, 2, 1, 5)})
        ),
        "td[0] = {'new': ones}": _inplace(
            lambda td: td.__setitem__(0, {"new": torch.ones(3, 2, 1)})
        ),
        "td[0] = 0": _inplace(lambda td: td.__setitem__(0, 0)),
        "td[mask] = 0": _inplace(lambda td: td.__setitem__(_MASK, 0)),
        "td[..., 0] = 0": _inplace(lambda td: td.__setitem__((Ellipsis, 0), 0)),
        "td[1, ...] = -1": _inplace(lambda td: td.__setitem__((1, Ellipsis), -1)),
        "td[...] = 1": _inplace(lambda td: td.__setitem__(Ellipsis, 1)),
    },
    "empty": {
        "empty()": lambda td: td.empty(),
        "empty(recurse=True)": lambda td: td.empty(recurse=True),
        "empty(batch_size=[4, 3])": lambda td: td.empty(batch_size=[4, 3]),
        "empty(recurse=True, batch_size=[4, 3, 2, 1])": lambda td: td.empty(
            recurse=True, batch_size=[4, 3, 2, 1]
        ),
        "empty(batch_size=[4, 3], names=['w', 'x'])": lambda td: td.empty(
            batch_size=[4, 3], names=["w", "x"]
        ),
        "empty(names=None)": lambda td: td.empty(names=None),
        "empty(device=None)": lambda td: td.empty(device=None),
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


def _check_known_failures(key, failures, path=_KNOWN_FAILURES_FILE, hint=None):
    """Compare ``failures`` ({case: reason}) with the known failures of ``key`` in ``path``.

    ``hint`` ends the message of a failure.
    """
    if _UPDATE_KNOWN_FAILURES:
        known = json.loads(path.read_text())
        known[key] = sorted(failures)
        path.write_text(json.dumps(dict(sorted(known.items())), indent=1) + "\n")
        return
    known = set(_KNOWN_FAILURES[path].get(key, ()))
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
            f"{path.name} under {key!r}:\n" + "\n".join(f"  {case}" for case in fixed)
        )
    if message and hint:
        message.append(hint)
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


# The types and operations that TestContainersCompile leaves out, with the reason.
_NOT_COMPILED_CONTAINERS = {
    "td_h5": "Dynamo does not trace the h5py calls that read and write the file",
    "permute_td": "legacy lazy view, removed in 0.17",
    "squeezed_td": "legacy lazy view, removed in 0.17",
    "unsqueezed_td": "legacy lazy view, removed in 0.17",
}
_NOT_COMPILED_OPS = {
    "any()": "returns a Python bool, so it needs the values of the tensors",
    "pickle": "Dynamo does not trace pickle",
    "share_memory_()": "Dynamo does not trace Tensor.share_memory_",
}
_COMPILE_CONTAINERS = [
    name for name in _CONTAINERS if name not in _NOT_COMPILED_CONTAINERS
]
_COMPILE_HINT = (
    "The known failures under torch.compile change with the torch version "
    f"(this is torch {torch.__version__}). If the torch of the CI job that runs "
    "them changed, rewrite them with TENSORDICT_UPDATE_KNOWN_FAILURES=1 "
    "TENSORDICT_COMPILE_CONFORMANCE=1 under that torch."
)


def _compare_compiled(make, op):
    """Run ``op`` compiled and in eager mode on new tensordicts of ``make``; describe a difference.

    ``op`` compiles with the default backend, and with ``dynamic=True``, which
    traces the sizes as symbols, as Dynamo does once a size changes between
    calls.
    """
    # The compiled lambda below has the same code for every case: without a
    # reset, its recompilations would reach Dynamo's limit, an error with
    # fullgraph=True.
    torch._dynamo.reset()
    torch.manual_seed(0)
    expected, eager_error = _run(op, make("cpu"))
    # The lambda makes Dynamo trace the operations that are builtins, such as len.
    compiled = torch.compile(lambda td: op(td), fullgraph=True, dynamic=True)
    torch.manual_seed(0)
    result, error = _run(compiled, make("cpu"))
    if eager_error is not None:
        # With fullgraph=True, Dynamo raises its own error for an exception
        # that leaves the compiled function.
        return None if error is not None else f"returns, eager {_describe(eager_error)}"
    if error is not None:
        return _describe(error)
    difference = _difference(result, expected)
    return None if difference is None else difference[:160]


@pytest.mark.skipif(
    not _COMPILE_CONFORMANCE,
    reason="the known failures under torch.compile depend on the torch version; "
    "set TENSORDICT_COMPILE_CONFORMANCE=1 to run them",
)
class TestContainersCompile:
    """Each operation compiles with fullgraph=True on every tensordict type, and gives its eager result."""

    @pytest.mark.parametrize("group", list(_OPS))
    @pytest.mark.parametrize("td_name", _COMPILE_CONTAINERS)
    def test_ops(self, td_name, group):
        make = getattr(TestTensorDictsBase, td_name)
        failures = {}
        for case, op in _OPS[group].items():
            if case in _NOT_COMPILED_OPS:
                continue
            difference = _compare_compiled(make, op)
            if difference is not None:
                failures[case] = difference
        _check_known_failures(
            f"{td_name} {group}",
            failures,
            _COMPILE_KNOWN_FAILURES_FILE,
            hint=_COMPILE_HINT,
        )


# The modules that define subclasses of TensorDictBase, besides tensordict.
_SUBCLASS_MODULES = ("tensordict.nn", "tensordict.persistent", "tensordict.store")

_DESCRIPTORS = (property, classmethod, staticmethod)

# Class attributes that are not operations.
_NOT_OPERATIONS = {
    "__init__",
    "__init_subclass__",
    "__class_getitem__",
    "__annotate__",
    "__annotate_func__",
}

# The public methods of TensorDictBase that each class still overrides, with
# the reason (see TestOneImplementation). Remove an entry with its override.
_DENSE_FAST_PATH = "dense fast path over the dict of entries"
_DENSE_MEMMAP = "only dense storage can hold the new memory-mapped tensor"
_FLATTEN_COPY = "one to_tensordict() read, without the cache of a locked tensordict"
_TD_PARAMS = "an nn.Module that forwards to the tensordict it wraps"
_LEGACY_VIEW = "legacy lazy view, removed in 0.17"
_ALLOWED_OVERRIDES = {
    "TensorDict": {
        "del_": "dense fast path: a top-level key is deleted without the call to _del",
        "empty": "fast path for the dense root, which many operations call",
        "from_module": "the implementation, which TensorDictBase.from_module calls",
        "is_empty": _DENSE_FAST_PATH,
        "items": _DENSE_FAST_PATH,
        "make_memmap": _DENSE_MEMMAP,
        "make_memmap_from_storage": _DENSE_MEMMAP,
        "make_memmap_from_tensor": _DENSE_MEMMAP,
        "popitem": "O(1) on the dict of entries",
        "values": _DENSE_FAST_PATH,
    },
    "_SubTensorDict": {
        "expand": "accepts, and ignores, the inplace keyword it had",
        "names": "the names of the source at the index of the view",
        "rename_key_": "the source checks its lock and the keys, and renames",
        "unlock_": "raises: only the source can be unlocked",
    },
    "LazyStackedTensorDict": {
        "__getitem__": "indexes the members, so the result shares their storage",
        "__getitems__": "indexes the members, so the result shares their storage",
        "__setstate__": "reads the pickles of TensorDict 0.14 and earlier",
        "__torch_function__": "dispatches the torch functions that lazy stacks implement",
        "device": "the members' common device",
        "from_list": "stacks one tensordict per element (#1955)",
        "get_item_shape": "-1 for the dims where the shapes of the members differ",
        "is_locked": "locked when all the members are",
        "is_memmap": "computed from the members",
        "is_shared": "computed from the members",
        "lazy_stack": "the constructor that the method of TensorDictBase calls",
        "maybe_dense_stack": "the constructor that the method of TensorDictBase calls",
        "names": "the names of the members and of the stack dim",
        "non_tensor_items": "stacks the non-tensor leaves that every member holds",
        "pop": "with a default, a deep key that keys(True) lacks is not read",
        "rename_key_": "renames in each member, which checks its lock and the keys",
        "zero_": "each member zeros itself: a file member writes zeros without reading",
    },
    "NonTensorStack": {
        "__getitem__": "an advanced index copies the entries, as it does for tensors",
        "__setitem__": "wraps a plain value as non-tensor data; saves a memmap stack again",
        "contiguous": "a NonTensorStack is its own materialized form",
        "data": "the value that every element holds",
        "from_list": "builds the stack from plain values, not from dicts",
        "is_empty": "a NonTensorStack holds a value",
        "to_dict": "returns the values as a list, as NonTensorData.to_dict does",
        "tolist": "lists the stored values, not dicts of the members",
        "update": "the source is a NonTensorData or a NonTensorStack, not a mapping",
        "update_": "the source is a NonTensorData or a NonTensorStack, not a mapping",
    },
    "PersistentTensorDict": {
        "__getitem__": "an index gives a writable view of the file",
        "__getitems__": "a view of the file; in a DataLoader worker, the rows in memory",
        "__getstate__": "replaces the open file with what its backend can pickle",
        "__setstate__": "reopens the file, and reads the pickles of TensorDict 0.14",
        "batch_size": "checks the new batch size against the file",
        "flatten_keys": _FLATTEN_COPY,
        "unflatten_keys": _FLATTEN_COPY,
    },
    "TensorDictStore": {
        "device": "the device can be set, and is persisted on the server",
        "flatten_keys": _FLATTEN_COPY,
        "from_schema": "takes the connection arguments of the store",
        "unflatten_keys": _FLATTEN_COPY,
    },
    "LazyStackedTensorDictStore": {
        "__getitem__": "an integer on the stack dim gives a view that writes through",
        "device": "the device can be set; reads move the entries to it",
        "flatten_keys": _FLATTEN_COPY,
        "unflatten_keys": _FLATTEN_COPY,
    },
    "_StoreStackElementView": {
        "device": "settable, but leaf reads ignore it (FutureWarning until 0.17)",
        "flatten_keys": "works on an in-memory copy",
        "unflatten_keys": "works on an in-memory copy",
    },
    "_StateFromSource": {
        "device": "a view reports the device of its source",
        "is_locked": "a view reports the lock of its source",
    },
    "_StoreConnection": {
        "__getstate__": "the event loop, its thread and the client cannot be pickled",
        "__setstate__": "opens a new connection to the same server",
    },
    "TypedTensorDict": {
        "__setitem__": "forwards to the source; through _setitem_index a key write is 16-28% slower",
        "all": "forwards to the source; the base method is 11-14% slower with a dim",
        "any": "forwards to the source; the base method is 11-14% slower with a dim",
        "batch_size": "the batch size of the source",
        "contiguous": "forwards to the source and wraps the result",
        "del_": "forwards to the source; through _del it is 60-80% slower",
        "from_dict_instance": "builds from the source and wraps the result",
        "is_contiguous": "forwards to the source",
        "lock_": "locks the source; the base method builds the wrong lock graph",
        "make_memmap": "the memory-mapped tensor goes into the source",
        "make_memmap_from_storage": "the memory-mapped tensor goes into the source",
        "make_memmap_from_tensor": "the memory-mapped tensor goes into the source",
        "masked_fill": "forwards to the source; the base copies an h5 source to a new file",
        "masked_select": "forwards to the source and wraps the result",
        "names": "the names of the source",
        "popitem": "pops from the source",
        "rename_key_": "the source renames: a lazy stack renames in each member",
        "repeat_interleave": "forwards to the source; in place, the base method is about 50% slower",
        "reshape": "forwards to the source; in place, the base method is about 50% slower",
        "unlock_": "unlocks the source; the base method builds the wrong lock graph",
    },
    "TensorDictParams": dict.fromkeys(
        (
            "__getitem__",
            "__getitems__",
            "__hash__",
            "__repr__",
            "__setitem__",
            "__torch_function__",
            "all",
            "any",
            "apply",
            "apply_",
            "auto_device_",
            "batch_size",
            "contiguous",
            "cpu",
            "create_nested",
            "cuda",
            "data",
            "del_",
            "device",
            "fill_",
            "flatten_keys",
            "from_dict_instance",
            "get",
            "grad",
            "is_contiguous",
            "is_locked",
            "is_memmap",
            "is_shared",
            "items",
            "load_state_dict",
            "map",
            "masked_fill",
            "masked_select",
            "memmap_",
            "memmap_like",
            "named_apply",
            "names",
            "pin_memory",
            "pop",
            "rename_key_",
            "repeat_interleave",
            "reshape",
            "set",
            "set_",
            "set_at_",
            "shape",
            "state_dict",
            "to",
            "to_h5",
            "to_tensordict",
            "to_zarr",
            "unflatten_keys",
            "update",
            "update_",
            "update_at_",
            "values",
            "zero_",
        ),
        _TD_PARAMS,
    ),
    "_CustomOpTensorDict": dict.fromkeys(
        (
            "__repr__",
            "batch_size",
            "contiguous",
            "del_",
            "detach_",
            "device",
            "is_empty",
            "is_locked",
            "is_memmap",
            "is_shared",
            "lock_",
            "make_memmap",
            "make_memmap_from_storage",
            "make_memmap_from_tensor",
            "masked_fill_",
            "pin_memory",
            "popitem",
            "rename_key_",
            "share_memory_",
            "sorted_keys",
            "to",
            "unlock_",
        ),
        _LEGACY_VIEW,
    ),
    "_PermutedTensorDict": {"names": _LEGACY_VIEW},
    "_SqueezedTensorDict": {"names": _LEGACY_VIEW},
    "_TransposedTensorDict": {"names": _LEGACY_VIEW},
    "_UnsqueezedTensorDict": {"names": _LEGACY_VIEW},
    "_ViewedTensorDict": {"names": _LEGACY_VIEW},
}


# The private methods of TensorDictBase that a subclass overrides, but that have
# no docstring stating their contract. Remove a name once it has one.
_UNDOCUMENTED_HOOKS = {
    **dict.fromkeys(
        (
            "_apply_nest",
            "_batch_size_setter",
            "_cast_reduction",
            "_change_batch_size",
            "_check_batch_size",
            "_check_device",
            "_check_is_shared",
            "_check_new_batch_size",
            "_convert_inplace",
            "_convert_to_tensordict",
            "_create_nested_str",
            "_erase_cache",
            "_exclude",
            "_get_names_idx",
            "_get_non_tensor",
            "_has_names",
            "_index_tensordict",
            "_irecv",
            "_isend",
            "_load_memmap",
            "_lock_parents_weakrefs",
            "_multithread_apply_flat",
            "_multithread_rebuild",
            "_new_impl",
            "_propagate_unlock",
            "_recv",
            "_reduce_get_metadata",
            "_repeat",
            "_select",
            "_send",
            "_set_device",
            "_set_names",
            "_set_non_tensor",
            "_to_module",
            "_unbind",
            "_update_at_fast",
        ),
        "older than the one-implementation series; document it when it changes",
    ),
    **dict.fromkeys(
        (
            "_legacy_permute",
            "_legacy_squeeze",
            "_legacy_transpose",
            "_legacy_unsqueeze",
            "_legacy_view",
        ),
        "used by the legacy lazy views, removed in 0.17",
    ),
}


def _in_library(cls):
    return cls.__module__.split(".")[0] == "tensordict"


def _library_classes():
    """The subclasses of TensorDictBase that tensordict defines, and their mixins.

    Returns them by name, and whether every module of :data:`_SUBCLASS_MODULES`
    imported.
    """
    complete = True
    for module in _SUBCLASS_MODULES:
        try:
            importlib.import_module(module)
        except ImportError:
            # An optional backend: its classes are skipped.
            complete = False
    classes = {}
    pending = [TensorDictBase]
    while pending:
        for subclass in pending.pop().__subclasses__():
            pending.append(subclass)
            for cls in subclass.__mro__:
                if _in_library(cls) and cls not in TensorDictBase.__mro__:
                    classes[cls.__qualname__] = cls
    return classes, complete


def _operations():
    """The public names of TensorDictBase and its dunders, without the abstract ones."""
    names = {name for name in dir(TensorDictBase) if not name.startswith("_")}
    for cls in TensorDictBase.__mro__:
        if not _in_library(cls):
            continue
        for name, value in vars(cls).items():
            if (
                name.startswith("__")
                and name.endswith("__")
                and (
                    # TensorDictBase sets __hash__ to None.
                    value is None or callable(value) or isinstance(value, _DESCRIPTORS)
                )
            ):
                names.add(name)
    return names - _NOT_OPERATIONS - TensorDictBase.__abstractmethods__


def _overridden_hooks(classes):
    """The private methods of TensorDictBase that one of ``classes`` overrides.

    Returns them by name, with their definition on TensorDictBase or the first
    of its mixins, without the abstract storage methods.
    """
    overridden = set().union(*(vars(cls) for cls in classes.values()))
    hooks = {}
    for cls in TensorDictBase.__mro__:
        if not _in_library(cls):
            continue
        for name, value in vars(cls).items():
            if (
                name.startswith("_")
                and not name.startswith("__")
                and name in overridden
                and name not in TensorDictBase.__abstractmethods__
                and (callable(value) or isinstance(value, _DESCRIPTORS))
            ):
                hooks.setdefault(name, value)
    return hooks


def _docstring(value):
    if isinstance(value, property):
        value = value.fget
    elif isinstance(value, (classmethod, staticmethod)):
        value = value.__func__
    return (value.__doc__ or "").strip()


class TestOneImplementation:
    """Each public method of TensorDictBase has one implementation.

    A subclass defines the storage protocol and private hooks, and overrides a
    public method of TensorDictBase only for a reason given in
    :data:`_ALLOWED_OVERRIDES`. Each private method of TensorDictBase that a
    subclass overrides states its contract in a docstring. See "The
    TensorDictBase contract" in ARCHITECTURE.md.
    """

    def test_public_overrides(self):
        classes, complete = _library_classes()
        operations = _operations()
        new, gone = [], []
        for name, cls in sorted(classes.items()):
            overrides = operations.intersection(vars(cls))
            allowed = _ALLOWED_OVERRIDES.get(name, {})
            new += [f"{name}.{op}" for op in sorted(overrides - allowed.keys())]
            gone += [f"{name}.{op}" for op in sorted(allowed.keys() - overrides)]
        if complete:
            gone += [
                f"{name} (no such class)"
                for name in sorted(_ALLOWED_OVERRIDES.keys() - classes.keys())
            ]
        message = []
        if new:
            message.append(
                f"{len(new)} overrides of public TensorDictBase methods are not in "
                "_ALLOWED_OVERRIDES:\n"
                + "\n".join(f"  {override}" for override in new)
                + "\nWrite the part that depends on the storage as a private hook "
                'that the public method calls (see "The TensorDictBase contract" '
                "in ARCHITECTURE.md), or add the override to _ALLOWED_OVERRIDES "
                "with the reason a hook does not do."
            )
        if gone:
            message.append(
                f"{len(gone)} allowed overrides are gone; remove them from "
                "_ALLOWED_OVERRIDES:\n"
                + "\n".join(f"  {override}" for override in gone)
            )
        assert not message, "\n".join(message)

    def test_overridden_hooks_are_documented(self):
        classes, complete = _library_classes()
        hooks = _overridden_hooks(classes)
        undocumented = sorted(
            name
            for name, value in hooks.items()
            if not _docstring(value) and name not in _UNDOCUMENTED_HOOKS
        )
        stale = sorted(
            name
            for name in _UNDOCUMENTED_HOOKS
            if (name in hooks and _docstring(hooks[name]))
            or (complete and name not in hooks)
        )
        message = []
        if undocumented:
            message.append(
                f"{len(undocumented)} private methods of TensorDictBase that "
                "subclasses override have no docstring:\n"
                + "\n".join(f"  {name}" for name in undocumented)
                + "\nState the contract of each one in its docstring: what the "
                "public method has done before the call, what the hook must do, "
                "and what it returns."
            )
        if stale:
            message.append(
                f"{len(stale)} names of _UNDOCUMENTED_HOOKS have a docstring now, "
                "or no override; remove them:\n"
                + "\n".join(f"  {name}" for name in stale)
            )
        assert not message, "\n".join(message)


# Round trips through every save/load path (#3235). A save/load pair must give
# back the tensordict it saved (classes, values, batch sizes, dim names,
# UnbatchedTensor leaves and non-tensor data), or raise a clear error when it
# saves. _ROUND_TRIP_OUTCOMES lists the pairs that do something else.


@tensorclass
class _SavedLeaf:
    x: torch.Tensor
    label: str


@tensorclass
class _SavedOuter:
    y: torch.Tensor
    inner: _SavedLeaf


def _saved_leaf(label="leaf"):
    return _SavedLeaf(x=torch.arange(6.0).view(3, 2), label=label, batch_size=[3])


def _stack_member(value, width=4):
    return TensorDict(
        a=torch.full((3, width), float(value)),
        nested={"b": torch.full((3,), value)},
        batch_size=[3],
    )


def _zoo_unbatched():
    td = TensorDict(
        x=torch.zeros(3), u=UnbatchedTensor(torch.arange(5.0)), batch_size=[3]
    )
    td["nested"] = TensorDict(
        v=UnbatchedTensor(torch.arange(12.0).view(3, 4)), batch_size=[3]
    )
    return td


def _zoo_non_tensor():
    td = TensorDict(x=torch.zeros(3), batch_size=[3])
    td.set_non_tensor("s", "hello")
    td.set_non_tensor(("nested", "n"), 7)
    return td


def _zoo_shared_storage():
    base = torch.arange(12.0)
    return TensorDict(
        whole=base.view(3, 4),
        same=base.view(3, 4),
        half=base[:6].view(3, 2),
        batch_size=[3],
    )


def _zoo_non_contiguous():
    base = torch.arange(24.0).view(4, 6)
    return TensorDict(t=base.t()[:3], strided=base[:3, ::2], batch_size=[3])


# Each entry builds a new tensordict: the tests compare what a path loads with a
# second, untouched copy.
_ZOO = {
    "dense_nested": lambda: TensorDict(
        a=torch.arange(12.0).view(3, 4),
        nested={"b": torch.arange(6).view(3, 2), "deep": {"c": torch.ones(3, 1, 2)}},
        batch_size=[3],
    ),
    "unbatched": _zoo_unbatched,
    # Its leading dim equals the batch size, so it also fits as a batched leaf.
    "unbatched_batch_shaped": lambda: TensorDict(
        x=torch.zeros(3),
        u=UnbatchedTensor(torch.arange(12.0).view(3, 4)),
        batch_size=[3],
    ),
    "non_tensor": _zoo_non_tensor,
    "non_tensor_stack": lambda: TensorDict(
        x=torch.zeros(3), s=NonTensorStack("a", "b", "c"), batch_size=[3]
    ),
    "lazy_stack_dim0": lambda: LazyStackedTensorDict.lazy_stack(
        [_stack_member(0), _stack_member(1)], 0
    ),
    "lazy_stack_dim1": lambda: LazyStackedTensorDict.lazy_stack(
        [_stack_member(0), _stack_member(1)], 1
    ),
    "lazy_stack_ragged": lambda: LazyStackedTensorDict.lazy_stack(
        [_stack_member(0, width=4), _stack_member(1, width=5)], 0
    ),
    "tensorclass": lambda: _saved_leaf("flat"),
    "tensorclass_nested": lambda: _SavedOuter(
        y=torch.arange(3.0), inner=_saved_leaf(), batch_size=[3]
    ),
    "tensorclass_in_td": lambda: TensorDict(
        x=torch.zeros(3), tc=_saved_leaf(), batch_size=[3]
    ),
    "dim_names": lambda: TensorDict(
        a=torch.zeros(3, 4, 2),
        nested={"b": torch.ones(3, 4)},
        batch_size=[3, 4],
        names=["time", "env"],
    ),
    "empty": lambda: TensorDict(),
    "empty_batched": lambda: TensorDict(batch_size=[3]),
    "zero_dim_batch": lambda: TensorDict(
        a=torch.tensor(1.0), nested={"b": torch.tensor([1, 2])}, batch_size=[]
    ),
    "zero_length_batch": lambda: TensorDict(a=torch.zeros(0, 2), batch_size=[0]),
    "jagged": lambda: TensorDict(
        j=torch.nested.nested_tensor(
            [torch.arange(2.0), torch.arange(3.0)], layout=torch.jagged
        ),
        batch_size=[2],
    ),
    "dtypes": lambda: TensorDict(
        bf16=torch.arange(3, dtype=torch.bfloat16),
        bool=torch.tensor([True, False, True]),
        complex=torch.tensor([1 + 2j, 3 - 1j, 0j]),
        batch_size=[3],
    ),
    "shared_storage": _zoo_shared_storage,
    "non_contiguous": _zoo_non_contiguous,
}


def _save_to_file(write, suffix=""):
    def save(td, path):
        path = path.with_suffix(suffix)
        write(td, path)
        return path

    return save


def _append_memmap(td, path):
    # Saves the first row, then appends the others. Without batch dims,
    # appends the whole tensordict, which append_memmap refuses.
    if not td.batch_dims:
        td.memmap(path)
        tensordict.append_memmap(path, td)
        return path
    td[:1].memmap(path)
    tensordict.append_memmap(path, td[1:])
    return path


def _load_weights_only(path):
    # PyTorch 2.16 no longer allowlists nested jagged tensors for weights_only
    # loads (pytorch/pytorch#194380), and asks users to opt in, as here. The
    # path then tests what tensordict allowlists, on every torch version.
    from torch.nested._internal.nested_tensor import _rebuild_njt, NestedTensor

    with torch.serialization.safe_globals([_rebuild_njt, NestedTensor]):
        return torch.load(path, weights_only=True)


def _from_dict(data):
    td, as_dict = data
    if _is_tensorclass(type(td)):
        return type(td).from_dict(as_dict, batch_size=td.batch_size)
    return TensorDict.from_dict(
        as_dict,
        batch_size=td.batch_size,
        names=td.names if td._has_names() else None,
    )


# Each path is (save, load): ``save(td, path)`` writes ``td`` under ``path`` and
# returns what ``load`` reads it from.
_SAVE_LOAD = {
    "memmap": (
        _save_to_file(lambda td, path: td.memmap(path)),
        tensordict.load_memmap,
    ),
    "memmap_atomic": (
        _save_to_file(lambda td, path: td.memmap(path, atomic=True)),
        tensordict.load_memmap,
    ),
    "memmap_archive": (
        _save_to_file(lambda td, path: td.memmap(path), ".tdz"),
        tensordict.load_memmap,
    ),
    "save": (_save_to_file(lambda td, path: td.save(path)), tensordict.load),
    "append_memmap": (_append_memmap, tensordict.load_memmap),
    "consolidate_filename": (
        _save_to_file(lambda td, path: td.consolidate(filename=path)),
        tensordict.from_consolidated,
    ),
    "consolidate_pickle": (
        lambda td, path: pickle.dumps(td.consolidate()),
        pickle.loads,
    ),
    "safetensors": (
        _save_to_file(tensordict.to_safetensors),
        tensordict.from_safetensors,
    ),
    "h5": (
        _save_to_file(lambda td, path: td.to_h5(path), ".h5"),
        tensordict.from_h5,
    ),
    "zarr": (
        _save_to_file(lambda td, path: td.to_zarr(path), ".zarr"),
        tensordict.from_zarr,
    ),
    "pickle": (lambda td, path: pickle.dumps(td), pickle.loads),
    "torch_save": (
        _save_to_file(lambda td, path: torch.save(td, path)),
        lambda path: torch.load(path, weights_only=False),
    ),
    "torch_save_weights_only": (
        _save_to_file(lambda td, path: torch.save(td, path)),
        _load_weights_only,
    ),
    "to_dict": (lambda td, path: (td, td.to_dict()), _from_dict),
}

# The module each path needs, and its minimum version: tensordict's zarr
# support needs zarr 3.
_PATH_REQUIRES = {
    "safetensors": ("safetensors", None),
    "h5": ("h5py", None),
    "zarr": ("zarr", "3"),
}


class _Raises:
    """The pair raises ``error`` with a message that matches ``match``, ``when`` it saves or loads."""

    def __init__(self, error, match, when="save", reason=""):
        self.error = error
        self.match = match
        self.when = when
        self.reason = reason


# The changes that the docs of a path describe, as {kind of difference: where
# the docs say so}. The kinds are those of _round_trip_differences: "class"
# (of a tensordict or tensorclass), "leaf" (the type of a leaf), "batch size",
# "names", "keys" and "value".
_PATH_CHANGES = {
    "consolidate_filename": {"class": "from_consolidated returns a TensorDict"},
    "h5": {"class": "from_h5 returns a PersistentTensorDict"},
    "zarr": {"class": "from_zarr returns a PersistentTensorDict"},
    "to_dict": {"class": "to_dict turns tensordicts and tensorclasses into dicts"},
}

_MEMMAP_PATHS = ("memmap", "memmap_atomic", "memmap_archive", "save", "append_memmap")

# The pairs that raise, all documented or with a clear message. Every other
# pair must round trip exactly, apart from the changes of _PATH_CHANGES.
_ROUND_TRIP_OUTCOMES = {
    **{
        (member, path): _Raises(
            NotImplementedError, "UnbatchedTensor entries cannot be memory-mapped"
        )
        for member in ("unbatched", "unbatched_batch_shaped")
        for path in ("memmap", "memmap_atomic", "save", "append_memmap")
    },
    **{
        (member, path): _Raises(
            NotImplementedError, "UnbatchedTensor entries cannot be consolidated"
        )
        for member in ("unbatched", "unbatched_batch_shaped")
        for path in ("consolidate_filename", "consolidate_pickle")
    },
    ("non_tensor_stack", "append_memmap"): _Raises(
        ValueError, "a stack of non-tensor values cannot grow"
    ),
    ("non_tensor_stack", "safetensors"): _Raises(TypeError, "non-tensor stack"),
    **{
        (member, "append_memmap"): _Raises(
            ValueError, "Cannot append a LazyStackedTensorDict"
        )
        for member in ("lazy_stack_dim0", "lazy_stack_dim1", "lazy_stack_ragged")
    },
    **{
        ("lazy_stack_ragged", path): _Raises(RuntimeError, "Failed to stack tensors")
        for path in ("safetensors", "h5", "zarr", "to_dict")
    },
    **{
        (member, "append_memmap"): _Raises(
            ValueError, "appends along the first batch dimension"
        )
        for member in ("empty", "zero_dim_batch")
    },
    **{
        ("jagged", path): _Raises(
            NotImplementedError, "does not support nested tensors with the torch.jagged"
        )
        for path in _MEMMAP_PATHS
    },
    ("jagged", "safetensors"): _Raises(TypeError, "only dense tensors"),
    **{
        (member, "torch_save_weights_only"): _Raises(
            pickle.UnpicklingError,
            "Unsupported global",
            when="load",
            reason="the allowlist of weights_only=True leaves out tensorclasses "
            '(saving.rst, "torch.save and torch.load")',
        )
        for member in ("tensorclass", "tensorclass_nested", "tensorclass_in_td")
    },
    ("dtypes", "h5"): _Raises(TypeError, "unsupported ScalarType BFloat16"),
    ("dtypes", "zarr"): _Raises(TypeError, "unsupported ScalarType BFloat16"),
}

# The pairs that filed bugs still break, as {(member, path): xfail reason}.
_ROUND_TRIP_BUGS = {
    **{
        (member, "h5"): "#3263: from_h5 infers the batch sizes and loses the dim names"
        for member in (
            "dense_nested",
            "unbatched",
            "non_tensor",
            "tensorclass",
            "tensorclass_nested",
            "tensorclass_in_td",
            "dim_names",
            "empty_batched",
            "zero_dim_batch",
            "zero_length_batch",
        )
    },
    **{
        pair: "#3264: h5, zarr and .tdz files lose the UnbatchedTensor class"
        for pair in (
            ("unbatched", "zarr"),
            ("unbatched", "memmap_archive"),
            ("unbatched_batch_shaped", "h5"),
            ("unbatched_batch_shaped", "zarr"),
            ("unbatched_batch_shaped", "memmap_archive"),
        )
    },
    **{
        (member, "torch_save_weights_only"): (
            "#3264: torch.load(weights_only=True) rejects UnbatchedTensor"
        )
        for member in ("unbatched", "unbatched_batch_shaped")
    },
    **{
        (
            "jagged",
            path,
        ): "#3265: saving a jagged leaf fails with an error that names no key"
        for path in ("h5", "zarr")
    },
}


def _round_trip_params():
    for member in _ZOO:
        for path in _SAVE_LOAD:
            reason = _ROUND_TRIP_BUGS.get((member, path))
            marks = [pytest.mark.xfail(strict=True, reason=reason)] if reason else []
            yield pytest.param(member, path, marks=marks, id=f"{member}-{path}")


def _kind(value):
    if isinstance(value, NonTensorStack):
        return "NonTensorStack"
    if is_non_tensor(value):
        return "NonTensorData"
    if _is_tensorclass(type(value)):
        return type(value).__name__
    if isinstance(value, LazyStackedTensorDict):
        return f"lazy stack along dim {value.stack_dim}"
    if is_tensor_collection(value):
        return "tensordict"
    if isinstance(value, UnbatchedTensor):
        return "UnbatchedTensor"
    if isinstance(value, torch.Tensor):
        return "nested tensor" if value.is_nested else "tensor"
    return type(value).__name__


def _non_tensor_value(value):
    if is_non_tensor(value) and not isinstance(value, NonTensorStack):
        return value.data
    return value


def _round_trip_differences(saved, loaded, key=()):
    """Returns ``[(kind, description)]`` for each way ``loaded`` differs from ``saved``."""
    where = "the root" if not key else repr(key)
    # A tensorclass reads a non-tensor field as its value, and a tensordict as a
    # NonTensorData: compare the values.
    saved, loaded = _non_tensor_value(saved), _non_tensor_value(loaded)
    saved_kind, loaded_kind = _kind(saved), _kind(loaded)
    differences = []
    if saved_kind != loaded_kind:
        # "class" is the class of a tensordict or tensorclass, "leaf" the type
        # of a leaf.
        containers = all(
            is_tensor_collection(value) and not is_non_tensor(value)
            for value in (saved, loaded)
        )
        differences.append(
            (
                "class" if containers else "leaf",
                f"{where}: saved a {saved_kind}, loaded a {loaded_kind}",
            )
        )
    if is_tensor_collection(saved) != is_tensor_collection(loaded):
        return differences
    if is_tensor_collection(saved):
        if saved.batch_size != loaded.batch_size:
            differences.append(
                (
                    "batch size",
                    f"{where}: saved batch size {list(saved.batch_size)}, loaded "
                    f"{list(loaded.batch_size)}",
                )
            )
            return differences
        if list(saved.names) != list(loaded.names):
            differences.append(
                (
                    "names",
                    f"{where}: saved dim names {saved.names}, loaded {loaded.names}",
                )
            )
        if is_non_tensor(saved) or is_non_tensor(loaded):
            saved_data, loaded_data = saved.tolist(), loaded.tolist()
            if saved_data != loaded_data or type(saved_data) is not type(loaded_data):
                differences.append(
                    (
                        "value",
                        f"{where}: saved the non-tensor data {saved_data!r}, "
                        f"loaded {loaded_data!r}",
                    )
                )
            return differences
        if isinstance(saved, LazyStackedTensorDict):
            # Compare the stacked tensordicts one by one, which also works for
            # entries whose shapes differ and for a loaded dense tensordict.
            loaded_parts = (
                loaded.tensordicts
                if isinstance(loaded, LazyStackedTensorDict)
                and loaded.stack_dim == saved.stack_dim
                else loaded.unbind(saved.stack_dim)
            )
            for i, (part, loaded_part) in enumerate(
                zip(saved.tensordicts, loaded_parts)
            ):
                differences += _round_trip_differences(
                    part, loaded_part, key + (f"<stacked {i}>",)
                )
            return differences
        saved_keys, loaded_keys = set(saved.keys()), set(loaded.keys())
        if saved_keys != loaded_keys:
            differences.append(
                (
                    "keys",
                    f"{where}: saved the keys {sorted(saved_keys)}, loaded "
                    f"{sorted(loaded_keys)}",
                )
            )
        for name in sorted(saved_keys & loaded_keys):
            differences += _round_trip_differences(
                saved.get(name), loaded.get(name), key + (name,)
            )
        return differences
    if not isinstance(saved, torch.Tensor) or not isinstance(loaded, torch.Tensor):
        # A non-tensor field of a tensorclass reads as its Python value.
        if type(saved) is not type(loaded) or saved != loaded:
            differences.append(
                ("value", f"{where}: saved {saved!r}, loaded {loaded!r}")
            )
        return differences
    if saved.is_nested or loaded.is_nested:
        # The shape of a jagged tensor has a symbolic dim: compare the parts.
        saved, loaded = (
            list(value.unbind()) if value.is_nested else [value]
            for value in (saved, loaded)
        )
        if len(saved) != len(loaded):
            differences.append(("value", f"{where}: the number of parts differs"))
            return differences
        for i, (part, loaded_part) in enumerate(zip(saved, loaded)):
            differences += _round_trip_differences(
                part, loaded_part, key + (f"<part {i}>",)
            )
    elif saved.dtype != loaded.dtype or saved.shape != loaded.shape:
        differences.append(
            (
                "value",
                f"{where}: saved a {saved.dtype} tensor of shape "
                f"{list(saved.shape)}, loaded a {loaded.dtype} tensor of shape "
                f"{list(loaded.shape)}",
            )
        )
    elif saved.tolist() != loaded.tolist():
        differences.append(("value", f"{where}: the values differ"))
    return differences


class TestSaveLoadRoundTrip:
    """Each tensordict of :data:`_ZOO` through each save/load path of :data:`_SAVE_LOAD`."""

    @pytest.mark.filterwarnings("ignore::FutureWarning")
    @pytest.mark.parametrize(("member", "path"), list(_round_trip_params()))
    def test_round_trip(self, member, path, tmp_path):
        if path in _PATH_REQUIRES:
            module, minversion = _PATH_REQUIRES[path]
            pytest.importorskip(module, minversion=minversion)
        save, load = _SAVE_LOAD[path]
        expected = _ROUND_TRIP_OUTCOMES.get((member, path))
        td = _ZOO[member]()
        if isinstance(expected, _Raises) and expected.when == "save":
            with pytest.raises(expected.error, match=expected.match):
                save(td, tmp_path / "saved")
            return
        try:
            saved = save(td, tmp_path / "saved")
        except Exception as err:
            pytest.fail(
                f"saving {member} through {path} raised {type(err).__name__}: {err}"
            )
        if isinstance(expected, _Raises):
            with pytest.raises(expected.error, match=expected.match):
                load(saved)
            return
        try:
            loaded = load(saved)
        except Exception as err:
            pytest.fail(
                f"{member} saved through {path}, but loading it raised "
                f"{type(err).__name__}: {err}"
            )
        allowed = _PATH_CHANGES.get(path, {})
        differences = [
            description
            for kind, description in _round_trip_differences(_ZOO[member](), loaded)
            if kind not in allowed
        ]
        assert not differences, (
            f"{member} through {path} came back changed:\n  " + "\n  ".join(differences)
        )
        differences = _round_trip_differences(_ZOO[member](), td)
        assert not differences, (
            f"saving {member} through {path} changed it:\n  "
            + "\n  ".join(description for _, description in differences)
        )

    def test_outcomes_are_used(self):
        stale = sorted(
            pair
            for pair in _ROUND_TRIP_OUTCOMES
            if pair[0] not in _ZOO or pair[1] not in _SAVE_LOAD
        )
        assert not stale, f"_ROUND_TRIP_OUTCOMES names unknown pairs: {stale}"


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
