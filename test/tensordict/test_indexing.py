# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Indexing of every tensordict type, checked against torch.

Torch defines what an index selects. These tests check that:

* a :class:`~tensordict.TensorDict` gives the batch size, the values and the
  writes that torch gives for the same index on a tensor, and dim names that
  follow the rule of :func:`_expected_names`;
* the other tensordict types (lazy stacks, sub-tensordicts, memory-mapped and
  h5 tensordicts, parameters, tensorclasses, stores, ...) give the same results
  as their dense copy.

The indices come from :func:`index_pool`, a fixed set built from named index
elements. The cases that fail today are listed in ``indexing_known_failures.json``.
Each test compares its failing cases with that list: a failure that is not
listed is a regression, and a listed case that passes must be removed from the
list. Run with ``TENSORDICT_UPDATE_KNOWN_FAILURES=1`` to rewrite the list.
"""

from __future__ import annotations

import argparse
import functools
import itertools
import json
import pathlib
import re
import warnings

import numpy as np
import pytest
import tensordict.base
import torch
from _utils_internal import _describe, check_known_failures, TestTensorDictsBase
from tensordict import lazy_stack, TensorDict, UnbatchedTensor
from tensordict._lazy import LazyStackedTensorDict
from tensordict.base import is_tensor_collection
from tensordict.memmap import MemoryMappedTensor
from tensordict.nn import TensorDictParams
from tensordict.store import LazyStackedTensorDictStore, TensorDictStore
from tensordict.store._store import _has_redis
from tensordict.tensorclass import NonTensorData, NonTensorStack, tensorclass
from torch.utils.data import DataLoader

_KNOWN_FAILURES_FILE = pathlib.Path(__file__).with_name("indexing_known_failures.json")
_KNOWN_FAILURES = json.loads(_KNOWN_FAILURES_FILE.read_text())
# Indices that torch rejects for the reference tensor are not part of a pool.
_TORCH_INDEX_ERRORS = (IndexError, RuntimeError, TypeError, ValueError)


def _mask(size, offset=0):
    return [(i + offset) % 2 == 0 for i in range(size)]


def _elements(size):
    """Named index elements for one dim of ``size`` elements."""
    mask = _mask(size)
    return {
        "0": 0,
        "-1": -1,
        "t(1)": torch.tensor(1),
        ":": slice(None),
        "1:": slice(1, None),
        "::2": slice(None, None, 2),
        "1:1": slice(1, 1),
        "[0,-1]": [0, -1],
        "[[0],[-1]]": [[0], [-1]],
        "t[0,-1]": torch.tensor([0, -1]),
        "t[[0,-1]]": torch.tensor([[0, -1]]),
        "np[0,-1]": np.array([0, -1]),
        "range": range(0, size, 2),
        "m": torch.tensor(mask),
        "m.list": mask,
        "m.np_bool": [np.bool_(b) for b in mask],
        "m.np": np.array(mask),
        "m.none": torch.zeros(size, dtype=torch.bool),
    }


# Index elements that use no dim of their own.
_SCALARS = {
    "None": None,
    "...": Ellipsis,
    "True": True,
    "False": False,
    "t(True)": torch.tensor(True),
}


def index_pool(batch_size):
    """Return ``{name: index}`` for named indices of ``batch_size`` that torch accepts.

    The pool has every element alone (bare and in a 1-tuple), N-D masks, pairs of
    elements for the two leading dims, and indices with advanced indices
    separated by a slice or ``None``.

    Some indices are left out on purpose, because torch reads them in a way that
    tensordict does not follow: indices with several ``Ellipsis``, and N-D masks
    given as nested lists or ndarrays next to other indices.
    """
    batch_size = tuple(batch_size)
    candidates = {}
    firsts = {**_elements(batch_size[0]), **_SCALARS}
    for name, element in firsts.items():
        candidates[name] = element
        candidates[f"({name},)"] = (element,)
    if len(batch_size) >= 2:
        mask_2d = [_mask(batch_size[1], k) for k in range(batch_size[0])]
        candidates["m2"] = torch.tensor(mask_2d)
        candidates["m2.list"] = mask_2d
        candidates["m2.np"] = np.array(mask_2d)
        candidates["(m2, ...)"] = (torch.tensor(mask_2d), Ellipsis)
        candidates["(..., m2)"] = (
            Ellipsis,
            torch.ones(batch_size[-2:], dtype=torch.bool),
        )
        seconds = {**_elements(batch_size[1]), **_SCALARS}
        for (name0, index0), (name1, index1) in itertools.product(
            firsts.items(), seconds.items()
        ):
            if index0 is Ellipsis and index1 is Ellipsis:
                continue
            candidates[f"({name0}, {name1})"] = (index0, index1)
    if len(batch_size) >= 3:
        one = torch.tensor([1])
        mask = torch.tensor(_mask(batch_size[0]))
        candidates["(:, [1], None, t[1])"] = (slice(None), [1], None, one)
        candidates["([0,1], :, t[1])"] = ([0, 1], slice(None), one)
        candidates["(m, :, t[1])"] = (mask, slice(None), one)
        candidates["(True, :, [0,1])"] = (True, slice(None), [0, 1])
        candidates["(t(1), :, [0,1])"] = (torch.tensor(1), slice(None), [0, 1])
    reference = torch.zeros(batch_size)
    pool = {}
    for name, index in candidates.items():
        try:
            reference[_torch_index(index)]
        except _TORCH_INDEX_ERRORS:
            continue
        pool[name] = index
    return pool


def _torch_index(index):
    # tensordict reads a bare index as a 1-tuple; torch reads a bare nested
    # list as a tuple of per-dim indices (deprecated)
    return index if isinstance(index, tuple) else (index,)


def _check_known_failures(key, failures):
    message = check_known_failures(key, failures, _KNOWN_FAILURES_FILE, _KNOWN_FAILURES)
    assert not message, "\n".join(message)


def _expected_names(batch_size, index, names):
    """Names that indexing a tensordict named ``names`` with ``index`` should give.

    An output dim keeps the name of the input dim that it depends on, if it
    depends on exactly one input dim, and is unnamed otherwise: a slice or a 1-D
    index keeps the name of its dim, while ``None``, a scalar bool, an N-D mask
    and advanced indices broadcast together give unnamed dims. The dependency is
    traced by indexing a tensor of coordinates along each input dim. A dim of
    size 0 or 1 does not show the dependency and is returned as ``Ellipsis``.
    A name that several output dims would take names none of them.
    """
    torch_index = _torch_index(index)
    coords = [
        torch.arange(size)
        .view([-1 if dim == i else 1 for dim in range(len(batch_size))])
        .expand(batch_size)[torch_index]
        for i, size in enumerate(batch_size)
    ]
    shape = coords[0].shape
    expected = []
    for out_dim, size in enumerate(shape):
        if size < 2 or coords[0].numel() == 0:
            expected.append(Ellipsis)
            continue
        depends = [
            i
            for i, coord in enumerate(coords)
            if (coord.amax(out_dim) != coord.amin(out_dim)).any()
        ]
        expected.append(names[depends[0]] if len(depends) == 1 else None)
    # names are unique: one that several output dims would take names none
    return [
        name if name in (None, Ellipsis) or expected.count(name) == 1 else None
        for name in expected
    ]


class TestTensorDictMatchesTorch:
    """A TensorDict gives what torch gives for the same index on a tensor."""

    BATCH_SIZES = [(3, 4), (3, 4, 5)]

    @staticmethod
    def _reference(batch_size):
        return torch.arange(float(np.prod(batch_size))).view(batch_size)

    @pytest.mark.parametrize("batch_size", BATCH_SIZES, ids=str)
    def test_getitem(self, batch_size):
        reference = self._reference(batch_size)
        failures = {}
        for case, index in index_pool(batch_size).items():
            expected = reference[_torch_index(index)]
            td = TensorDict(
                {
                    "x": reference.unsqueeze(-1).expand(*batch_size, 2),
                    "nested": {"y": reference},
                },
                batch_size,
            )
            try:
                result = td[index]
            except Exception as err:
                failures[case] = _describe(err)
                continue
            if result.batch_size != expected.shape:
                failures[case] = (
                    f"batch size {tuple(result.batch_size)}, torch {tuple(expected.shape)}"
                )
            elif result["nested"].batch_size != expected.shape:
                failures[case] = "nested batch size"
            elif not (
                torch.equal(result["x"][..., 0], expected)
                and torch.equal(result["nested", "y"], expected)
            ):
                failures[case] = "values"
        _check_known_failures(f"TensorDict getitem {batch_size}", failures)

    @pytest.mark.parametrize("batch_size", BATCH_SIZES, ids=str)
    def test_setitem(self, batch_size):
        reference = self._reference(batch_size)
        failures = {}
        for case, index in index_pool(batch_size).items():
            torch_index = _torch_index(index)
            written = reference.clone()
            written[torch_index] = -1
            new_key = torch.zeros(batch_size)
            new_key[torch_index] = -1
            value_shape = reference[torch_index].shape
            value = -torch.ones(value_shape)
            try:
                # a tensordict with an existing key and a new key
                td = TensorDict({"x": reference.clone()}, batch_size)
                td[index] = TensorDict({"x": value, "new": value}, value_shape)
                if not (
                    torch.equal(td["x"], written) and torch.equal(td["new"], new_key)
                ):
                    failures[case] = "values"
                    continue
                td = TensorDict({"x": reference.clone()}, batch_size)
                td[index] = -1.0
                if not torch.equal(td["x"], written):
                    failures[case] = "values of a scalar write"
            except Exception as err:
                failures[case] = _describe(err)
        _check_known_failures(f"TensorDict setitem {batch_size}", failures)

    @pytest.mark.parametrize("batch_size", BATCH_SIZES, ids=str)
    def test_names(self, batch_size):
        names = [f"d{i}" for i in range(len(batch_size))]
        reference = self._reference(batch_size)
        failures = {}
        for case, index in index_pool(batch_size).items():
            expected = _expected_names(batch_size, index, names)
            td = TensorDict({"x": reference}, batch_size, names=names)
            try:
                result = td[index].names
            except Exception as err:
                failures[case] = _describe(err)
                continue
            if len(result) != len(expected) or any(
                want is not Ellipsis and got != want
                for got, want in zip(result, expected)
            ):
                shown = [n if n is not Ellipsis else "?" for n in expected]
                failures[case] = f"names {result}, expected {shown}"
        _check_known_failures(f"TensorDict names {batch_size}", failures)


@tensorclass
class _ShortcutTC:
    x: torch.Tensor


def _shortcut_nested():
    r = torch.randn
    return TensorDict(
        {
            "a": r(4, 3, 2),
            "b": torch.randint(5, (4, 3)),
            "n": {"c": r(4, 3, 2, 2), "d": {"e": r(4, 3)}},
            "z": r(4, 3, 1),
        },
        [4, 3],
    )


# tensordicts with each type of entry, for the exact-Tensor shortcut of #3458
_SHORTCUT_TDS = {
    "nested": _shortcut_nested,
    "1-d": lambda: TensorDict(a=torch.randn(4), b=torch.randn(4, 2), batch_size=[4]),
    "0-d": lambda: TensorDict(a=torch.randn(2)),
    "names": lambda: TensorDict(
        {"a": torch.randn(4, 3), "n": {"c": torch.randn(4, 3)}},
        [4, 3],
        names=["x", "y"],
    ),
    "device": lambda: TensorDict(
        {"a": torch.randn(4, 3), "n": {"c": torch.randn(4, 3)}}, [4, 3], device="cpu"
    ),
    "non-tensor": lambda: TensorDict(
        {
            "a": torch.randn(4, 3),
            "s": NonTensorData("x", batch_size=[4, 3]),
            "t": NonTensorStack(*[NonTensorStack(*"abc") for _ in range(4)]),
        },
        [4, 3],
    ),
    "unbatched": lambda: TensorDict(
        a=torch.randn(4, 3), u=UnbatchedTensor(torch.randn(5)), batch_size=[4, 3]
    ),
    "parameter": lambda: TensorDict(
        a=torch.nn.Parameter(torch.randn(4, 3)), b=torch.randn(4, 3), batch_size=[4, 3]
    ),
    "requires grad": lambda: TensorDict(
        a=torch.randn(4, 3, requires_grad=True) * 2,
        b=torch.randn(4, 3),
        batch_size=[4, 3],
    ),
    "lazy stack": lambda: TensorDict(
        a=torch.randn(4, 3),
        l=lazy_stack(
            [TensorDict(x=torch.randn(3, 2), batch_size=[3]) for _ in range(4)]
        ),
        batch_size=[4, 3],
    ),
    "tensorclass": lambda: TensorDict(
        a=torch.randn(4, 3),
        t=_ShortcutTC(x=torch.randn(4, 3, 2), batch_size=[4, 3]),
        batch_size=[4, 3],
    ),
    "jagged": lambda: TensorDict(
        j=torch.nested.nested_tensor(
            [torch.randn(2), torch.randn(3), torch.randn(1), torch.randn(4)],
            layout=torch.jagged,
        ),
        a=torch.randn(4),
        batch_size=[4],
    ),
    "locked": lambda: _shortcut_nested().lock_(),
    "shared": lambda: _shortcut_nested().share_memory_(),
    "TensorDictParams": lambda: TensorDictParams(_shortcut_nested()),
}
# more tensordicts for the list indices of #3568
_LIST_INDEX_TDS = {
    "one leaf": lambda: TensorDict(a=torch.randn(4, 3), batch_size=[4]),
    "one nested": lambda: TensorDict(n=_shortcut_nested(), batch_size=[4]),
    "empty batch": lambda: TensorDict(
        a=torch.randn(0, 3), b=torch.randn(0), batch_size=[0]
    ),
    "memmap": lambda: _shortcut_nested().memmap_(),
    "memmap view": lambda: _shortcut_nested().memmap_()[:, 1:],
    "memmap and tensors": lambda: TensorDict(
        a=torch.randn(4, 3),
        m=MemoryMappedTensor.from_tensor(torch.randn(4, 3)),
        s=NonTensorData("x", batch_size=[4]),
        n={"m": MemoryMappedTensor.from_tensor(torch.randn(4, 3))},
        batch_size=[4],
    ),
}
# the shortest list of ints that is turned into an index tensor (#3568)
_LIST_MIN = 64
_LIST_INDICES = {
    "ascending": sorted([0, 2, 3] * _LIST_MIN),
    "repeated": [1, 1, 0, 1] * _LIST_MIN,
    "negative": [-1, 0, -3] * _LIST_MIN,
    "shortest converted": ([3, 0, 1, 1] * _LIST_MIN)[:_LIST_MIN],
    "longest kept": ([3, 0, 1, 1] * _LIST_MIN)[: _LIST_MIN - 1],
    "short": [0, 2, 3],
    "length 1": [2],
    "in a tuple": ([0, 2] * _LIST_MIN,),
    "out of range": [0, 10] * _LIST_MIN,
    "short out of range": [0, 10],
    "negative out of range": [-5] * _LIST_MIN,
    "short negative out of range": [-5],
    "empty": [],
    "bools": [True, False, True, False],
    "int and bool": [0, True] * _LIST_MIN,
    "short int and bool": [0, True],
    "NumPy ints": [np.int64(0), np.int64(2)] * _LIST_MIN,
    "int and NumPy int": [0, np.int64(2)] * _LIST_MIN,
    "nested": [[0, 1], [2, 0]],
    "0-d tensors": [torch.tensor(0), torch.tensor(2)],
    "int and 0-d tensor": [0] * _LIST_MIN + [torch.tensor(2)],
    "beyond int64": [0] * _LIST_MIN + [2**70],
    "short beyond int64": [2**70],
    "int64 max": [2**63 - 1] * _LIST_MIN,
    "short int64 max": [2**63 - 1],
    "floats": [0.0, 1.0],
    "range": range(2),
    "list and slice": ([0, 2], slice(None)),
}
# indices that raise or warn, next to the pools of indices that torch takes
_SHORTCUT_EXTRA_INDICES = {
    "out of range": 10,
    "past the batch dims": (0, 0, 0, 0, 0),
    "slices past the batch dims": (slice(None),) * 4 + (0,),
    "nested list of bools": [[True, False, True], [False] * 3, [True] * 3, [False] * 3],
    "np.int64": np.int64(1),
    "()": (),
}


def _shortcut_describe(x, storages):
    """A comparable description of the result ``x`` of an index."""
    if isinstance(x, torch.Tensor):
        desc = [type(x), x.dtype, x.device, x.requires_grad, x.is_nested]
        desc.append(type(x.grad_fn))
        if x.is_nested:
            return desc + [[t.tolist() for t in x.unbind()]]
        desc += [x.shape, x.stride()]
        if x.device.type == "meta":
            return desc
        return desc + [
            x.untyped_storage().data_ptr() in storages,
            x.detach().tolist(),
        ]
    if not is_tensor_collection(x):
        return [type(x), repr(x)]
    desc = [type(x), x.batch_size, x.device, x.is_locked]
    desc += [getattr(x, "_is_shared", None), getattr(x, "_is_memmap", None)]
    desc.append(x.names if x._has_names() else None)
    if isinstance(x, LazyStackedTensorDict):
        return desc + [
            x.stack_dim,
            [_shortcut_describe(td, storages) for td in x.tensordicts],
        ]
    entries = getattr(x, "_tensordict", None)
    if not isinstance(entries, dict):
        entries = dict(x.items())
    return desc + [
        [(key, _shortcut_describe(value, storages)) for key, value in entries.items()]
    ]


def _shortcut_leaves(td):
    # the caller keeps the leaves alive: the leaves of a lazy stack are new
    # tensors, whose storage a result could reuse once they are freed
    return [
        value
        for value in td.values(True, True)
        if isinstance(value, torch.Tensor)
        and not value.is_nested
        and value.device.type != "meta"
    ]


def _shortcut_outcomes(make, indices):
    outcomes = {}
    for case, index in indices.items():
        torch.manual_seed(0)
        td = make()
        leaves = _shortcut_leaves(td)
        storages = {leaf.untyped_storage().data_ptr() for leaf in leaves}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                outcome = _shortcut_describe(td[index], storages)
            except Exception as err:
                # the ragged dims of jagged tensors are numbered per process
                outcome = (type(err), re.sub(r"\bj\d+\b", "j", str(err)))
        outcomes[case] = outcome, [(w.category, str(w.message)) for w in caught]
    return outcomes


def _check_shortcut_matches_generic(make, indices, monkeypatch):
    shortcut = _shortcut_outcomes(make, indices)
    # without Dynamo's frame callback, as on a torch that lacks it, indexing
    # takes the generic path
    monkeypatch.setattr(tensordict.base, "_get_eval_frame_callback", None)
    generic = _shortcut_outcomes(make, indices)
    differ = [case for case in indices if shortcut[case] != generic[case]]
    assert not differ, differ


_CONTAINERS = list(
    dict.fromkeys(
        name
        for name, device in TestTensorDictsBase.TYPES_DEVICES
        if torch.device(device).type == "cpu"
    )
)


def _tensor_keys(td):
    """Keys of the batched tensor leaves of ``td``."""
    return [
        key
        for key, value in td.items(include_nested=True, leaves_only=True)
        if isinstance(value, torch.Tensor) and not isinstance(value, UnbatchedTensor)
    ]


class TestContainersMatchDense:
    """Every tensordict type gives what its dense copy gives (batch size [4, 3, 2, 1])."""

    BATCH_SIZE = (4, 3, 2, 1)

    @pytest.mark.parametrize("td_name", _CONTAINERS)
    def test_getitem(self, td_name):
        container = getattr(TestTensorDictsBase, td_name)("cpu")
        dense = container.to_tensordict()
        failures = {}
        for case, index in index_pool(self.BATCH_SIZE).items():
            try:
                expected = dense[index]
            except Exception:
                # the dense result is checked against torch above
                continue
            try:
                result = container[index]
                if result.batch_size != expected.batch_size:
                    failures[case] = (
                        f"batch size {tuple(result.batch_size)}, "
                        f"dense {tuple(expected.batch_size)}"
                    )
                elif not (result.to_tensordict() == expected).all():
                    failures[case] = "values"
            except Exception as err:
                failures[case] = _describe(err)
        _check_known_failures(f"{td_name} getitem", failures)

    @pytest.mark.parametrize(
        # h5py does not take every selection that torch takes
        "td_name",
        [name for name in _CONTAINERS if name != "td_h5"],
    )
    def test_setitem(self, td_name):
        make = getattr(TestTensorDictsBase, td_name)
        failures = {}
        for case, index in index_pool(self.BATCH_SIZE).items():
            container = make("cpu")
            dense = container.to_tensordict()
            keys = _tensor_keys(dense)
            with torch.no_grad():
                try:
                    value = (
                        dense[index]
                        .select(*keys)
                        .apply(lambda x: torch.full_like(x, -1))
                    )
                    dense[index] = value
                except Exception:
                    continue
                try:
                    container[index] = value
                    if not (
                        container.to_tensordict().select(*keys) == dense.select(*keys)
                    ).all():
                        failures[case] = "values"
                except Exception as err:
                    failures[case] = _describe(err)
        _check_known_failures(f"{td_name} setitem", failures)

    @pytest.mark.parametrize("td_name", _CONTAINERS)
    def test_getitem_exact_tensor_shortcut(self, td_name, monkeypatch):
        # #3458: indexing sends exact Tensor leaves straight to _get_item; the
        # results, errors and warnings are those of the generic path
        make = functools.partial(getattr(TestTensorDictsBase, td_name), "cpu")
        indices = {**index_pool(self.BATCH_SIZE), **_SHORTCUT_EXTRA_INDICES}
        _check_shortcut_matches_generic(make, indices, monkeypatch)

    @pytest.mark.parametrize("td_name", list(_SHORTCUT_TDS))
    def test_getitem_exact_tensor_shortcut_entries(self, td_name, monkeypatch):
        # #3458: the same, for each type of entry next to exact Tensors
        make = _SHORTCUT_TDS[td_name]
        batch_size = make().batch_size
        indices = dict(_SHORTCUT_EXTRA_INDICES)
        if batch_size:
            indices.update(index_pool(batch_size))
        else:
            indices.update(_SCALARS)
        _check_shortcut_matches_generic(make, indices, monkeypatch)

    def test_getitem_exact_tensor_shortcut_taken(self, monkeypatch):
        # #3458: exact Tensor leaves skip the _is_unbatched test, other leaves
        # take it
        calls = []
        is_unbatched = tensordict.base._is_unbatched

        def counting(item):
            calls.append(type(item))
            return is_unbatched(item)

        monkeypatch.setattr(tensordict.base, "_is_unbatched", counting)
        td = TensorDict(
            a=torch.zeros(4, 3),
            p=torch.nn.Parameter(torch.zeros(4)),
            n={"b": torch.zeros(4)},
            batch_size=[4],
        )
        result = td[1]
        assert torch.Tensor not in calls
        assert torch.nn.Parameter in calls
        assert result.batch_size == torch.Size([])
        assert set(result.keys(True, True)) == {"a", "p", ("n", "b")}
        # a torch without the frame callback getter takes the generic path
        calls.clear()
        monkeypatch.setattr(tensordict.base, "_get_eval_frame_callback", None)
        assert (td[1] == result).all()
        assert calls.count(torch.Tensor) == 2

    @pytest.mark.parametrize("td_name", [*_SHORTCUT_TDS, *_LIST_INDEX_TDS])
    def test_getitem_int_list(self, td_name, monkeypatch):
        # #3568: a list of ints is turned into an index tensor once, for the
        # plain and the memory-mapped tensors; the results, errors and warnings
        # are those of the generic path, where every leaf takes the list
        make = {**_SHORTCUT_TDS, **_LIST_INDEX_TDS}[td_name]
        _check_shortcut_matches_generic(make, _LIST_INDICES, monkeypatch)

    @pytest.mark.parametrize("td_name", ["nested", "names", "memmap", "memmap view"])
    @pytest.mark.parametrize(
        "lst",
        [
            sorted([0, 2, 3] * _LIST_MIN),
            [1, 1, 0, 1] * _LIST_MIN,
            [-1, 0] * _LIST_MIN,
            ([3, 0] * _LIST_MIN)[:_LIST_MIN],
            ([3, 0] * _LIST_MIN)[: _LIST_MIN - 1],
            [0, 2, 3],
            [2],
        ],
        ids=[
            "ascending",
            "repeated",
            "negative",
            "shortest converted",
            "longest kept",
            "short",
            "length 1",
        ],
    )
    def test_getitem_int_list_matches_tensor(self, td_name, lst):
        # #3568: td[list] is td[tensor], and each leaf is leaf[list], for the
        # lists that are converted and for the shorter ones, which are not
        torch.manual_seed(0)
        td = {**_SHORTCUT_TDS, **_LIST_INDEX_TDS}[td_name]()
        result = td[lst]
        expected = td[torch.tensor(lst)]
        assert type(result) is type(expected)
        assert result.batch_size == expected.batch_size
        assert result.names == expected.names
        assert result.device == expected.device
        assert result.is_locked == expected.is_locked
        assert result.is_memmap() == expected.is_memmap()
        assert list(result.keys(True, True)) == list(td.keys(True, True))
        for key, leaf in td.items(True, True):
            assert type(result[key]) is torch.Tensor
            assert torch.equal(result[key], leaf[lst])
            assert torch.equal(result[key], expected[key])
        # the list is left as it is, and the result is a copy
        assert lst == list(lst)
        result.zero_()
        assert not (td[lst] == 0).all()

    def test_getitem_int_list_converted_once(self, monkeypatch, tmp_path):
        # #3568: for a list of at least _LIST_MIN ints, the plain tensors, the
        # memory-mapped tensors and the nested TensorDicts take one index
        # tensor; the other entries, the shorter lists and the lists that
        # torch does not read as an int64 index tensor keep the list
        calls = []
        get_item = tensordict.base._get_item

        def recording(item, index):
            calls.append((type(item), index))
            return get_item(item, index)

        monkeypatch.setattr(tensordict.base, "_get_item", recording)
        td = TensorDict(
            a=torch.zeros(4, 3),
            m=MemoryMappedTensor.from_tensor(torch.arange(4)),
            p=torch.nn.Parameter(torch.zeros(4)),
            s=NonTensorData("x", batch_size=[4]),
            n={"b": torch.arange(4), "q": torch.nn.Parameter(torch.zeros(4))},
            batch_size=[4],
        )
        assert tensordict._indexing._LIST_INDEX_MIN_LEN == _LIST_MIN
        lst = ([0, 2] * _LIST_MIN)[:_LIST_MIN]
        result = td[lst]
        assert result.batch_size == torch.Size([len(lst)])
        assert result["n", "b"].tolist() == lst
        indices = dict(calls)
        converted = [index for cls, index in calls if type(index) is torch.Tensor]
        assert [cls for cls, index in calls if type(index) is torch.Tensor] == [
            torch.Tensor,
            MemoryMappedTensor,
            torch.Tensor,
        ]
        assert all(index is converted[0] for index in converted)
        assert converted[0].dtype is torch.int64 and converted[0].tolist() == lst
        assert indices[torch.nn.Parameter] == (lst,)
        assert indices[NonTensorData] == (lst,)
        # the lists that keep the generic path
        kept = {
            "longest kept": lst[:-1],
            "short": [0, 2],
            "length 1": [2],
            "empty": [],
            "bools": [True, False, True, False],
            "int and bool": [0, True] * _LIST_MIN,
            "NumPy ints": [np.int64(0), np.int64(2)] * _LIST_MIN,
            "int and NumPy int": [0, np.int64(2)] * _LIST_MIN,
            "nested": [[0, 1], [2, 0]],
            "0-d tensors": [torch.tensor(0), torch.tensor(2)],
            "beyond int64": [0] * _LIST_MIN + [2**70],
        }
        for case, kept_lst in kept.items():
            calls.clear()
            try:
                td[kept_lst]
            except (ValueError, RuntimeError, OverflowError):
                assert case == "beyond int64"
            assert calls, case
            assert all(index == (kept_lst,) for _, index in calls), case
        # the list is converted only if the first two entries are plain or
        # memory-mapped tensors: where a single leaf reads the list, and where
        # another kind of entry comes first or second, every leaf takes the list
        one_leaf = {
            "one tensor": TensorDict(a=torch.arange(4), batch_size=[4]),
            "non-tensor": TensorDict(
                a=torch.arange(4), s=NonTensorData("x", batch_size=[4]), batch_size=[4]
            ),
            "parameter": TensorDict(
                a=torch.arange(4), p=torch.nn.Parameter(torch.zeros(4)), batch_size=[4]
            ),
            "one nested": TensorDict(n={"a": torch.arange(4)}, batch_size=[4]),
            "lazy stack": TensorDict(
                a=torch.arange(4),
                l=lazy_stack([TensorDict(b=torch.arange(4)) for _ in range(4)]),
                batch_size=[4],
            ),
            "nested second": TensorDict(
                a=torch.arange(4),
                n={"b": torch.arange(4)},
                c=torch.arange(4),
                batch_size=[4],
            ),
            "nested first": TensorDict(
                n={"b": torch.arange(4)},
                a=torch.arange(4),
                c=torch.arange(4),
                batch_size=[4],
            ),
            "non-tensor first": TensorDict(
                s=NonTensorData("x", batch_size=[4]),
                a=torch.arange(4),
                c=torch.arange(4),
                batch_size=[4],
            ),
            "tensorclass second": TensorDict(
                a=torch.arange(4),
                t=_ShortcutTC(x=torch.arange(4), batch_size=[4]),
                c=torch.arange(4),
                batch_size=[4],
            ),
        }
        for case, td_one in one_leaf.items():
            calls.clear()
            assert td_one[lst].batch_size == (len(lst),), case
            assert calls, case
            assert all(index == (lst,) for _, index in calls), case
        # the leaves of a nested TensorDict share the tensor of the outer one
        calls.clear()
        TensorDict(
            a=torch.arange(4),
            b=torch.arange(4),
            n={"c": torch.arange(4)},
            batch_size=[4],
        )[lst]
        assert [type(index) for _, index in calls] == [torch.Tensor] * 3
        assert all(index is calls[0][1] for _, index in calls)
        # a memory-mapped tensor without elements records its index
        calls.clear()
        empty = TensorDict(
            a=torch.arange(4), b=torch.arange(4), e=torch.zeros(4, 0), batch_size=[4]
        )
        empty = empty.memmap(tmp_path)
        assert type(empty["e"]) is MemoryMappedTensor
        result_empty = empty[lst]
        assert [type(index) for _, index in calls] == [
            torch.Tensor,
            torch.Tensor,
            tuple,
        ]
        assert result_empty["e"].shape == (len(lst), 0)
        assert result_empty["a"].tolist() == lst
        # also when it is one of the first two entries
        calls.clear()
        result_empty = TensorDict(e=empty["e"], a=torch.arange(4), batch_size=[4])[lst]
        assert [type(index) for _, index in calls] == [tuple, torch.Tensor]
        assert result_empty["e"].shape == (len(lst), 0)
        # in a frame that Dynamo runs, every leaf takes the list
        calls.clear()
        monkeypatch.setattr(tensordict.base, "_get_eval_frame_callback", None)
        assert (td[lst]["n", "b"] == result["n", "b"]).all()
        assert all(type(index) is tuple for _, index in calls)

    def test_getitem_int_list_errors(self):
        # #3568: the errors of a list index are those of torch
        td = _shortcut_nested()
        for lst in (
            [0, 10] * _LIST_MIN,
            [-5] * _LIST_MIN,
            [2**63 - 1] * _LIST_MIN,
            [0] * _LIST_MIN + [2**70],
            [0, 10],
            [-5],
            [2**63 - 1],
            [2**70],
        ):
            with pytest.raises(Exception) as torch_err:
                td["a"][lst]
            with pytest.raises(type(torch_err.value)) as err:
                td[lst]
            assert str(err.value) == str(torch_err.value)
        # an int beyond int64 raises only if a leaf reads it
        assert TensorDict(batch_size=[4])[[2**70]].batch_size == torch.Size([1])

    def test_getitem_int_list_default_device(self):
        # #3568: the index tensor is on the CPU whatever the default device
        td = _shortcut_nested()
        lst = [0, 2] * _LIST_MIN
        expected = td[lst]
        with torch.device("meta"):
            result = td[lst]
        assert (result == expected).all()
        assert result["a"].device.type == "cpu"

    @pytest.mark.parametrize(
        "device", ["meta"] + (["cuda"] if torch.cuda.is_available() else [])
    )
    def test_getitem_int_list_other_device(self, device, monkeypatch):
        # #3568: the list is converted only if the first two entries are on the
        # CPU; the leaves on another device that come after them take the same
        # index tensor, which is on the CPU (torch moves it, as it moves the
        # one it builds from the list), and give what leaf[list] gives
        calls = []
        get_item = tensordict.base._get_item

        def recording(item, index):
            calls.append(index)
            return get_item(item, index)

        monkeypatch.setattr(tensordict.base, "_get_item", recording)

        def cpu():
            return torch.arange(4)

        def other():
            return torch.arange(4, device=device)

        lst = [0, 2] * _LIST_MIN
        tds = {
            "tensordict device": TensorDict(
                a=other(), b=other(), c=other(), batch_size=[4], device=device
            ),
            "no tensordict device": TensorDict(a=other(), b=other(), batch_size=[4]),
            "other device first": TensorDict(
                a=other(), b=cpu(), c=cpu(), batch_size=[4]
            ),
            "other device second": TensorDict(
                a=cpu(), b=other(), c=cpu(), batch_size=[4]
            ),
            "other device later": TensorDict(
                a=cpu(),
                b=cpu(),
                c=other(),
                n={"d": other(), "e": cpu()},
                batch_size=[4],
            ),
        }
        for case, td in tds.items():
            calls.clear()
            result = td[lst]
            assert len(calls) == len(list(td.keys(True, True))), case
            if case == "other device later":
                assert all(index is calls[0] for index in calls), case
                assert type(calls[0]) is torch.Tensor, case
                assert calls[0].device.type == "cpu", case
            else:
                assert all(index == (lst,) for index in calls), case
            assert result.device == td.device, case
            for key, leaf in td.items(True, True):
                assert result[key].device == leaf.device, case
                assert result[key].shape == leaf[lst].shape, case
                if device != "meta":
                    assert torch.equal(result[key], leaf[lst]), case
            # a shorter list is read by every leaf
            short = lst[: _LIST_MIN - 1]
            calls.clear()
            td[short]
            assert calls and all(index == (short,) for index in calls), case

    @pytest.mark.parametrize("td_name", ["nested", "memmap"])
    def test_dataloader_int_list(self, td_name):
        # #3568: a DataLoader reads each batch with a list of ints
        torch.manual_seed(0)
        td = torch.cat([_shortcut_nested()] * _LIST_MIN)
        rows = 4 * _LIST_MIN
        td = TensorDict(dict(td.items()), batch_size=[rows])
        if td_name == "memmap":
            td = td.memmap_()
        loader = DataLoader(td, batch_size=3 * _LIST_MIN, collate_fn=lambda x: x)  # noqa: TOR401
        batches = list(loader)
        assert [batch.batch_size for batch in batches] == [
            (3 * _LIST_MIN,),
            (_LIST_MIN,),
        ]
        for batch, index in zip(batches, torch.arange(rows).split(3 * _LIST_MIN)):
            expected = td[index]
            assert type(batch) is type(expected)
            for key, value in expected.items(True, True):
                assert torch.equal(batch[key], value)

    def test_getitem_int_list_jit_trace(self, monkeypatch):
        # #3568: under torch.jit.trace every leaf takes the list: torch.tensor
        # warns there, and reading the list in the index does not
        calls = []
        get_item = tensordict.base._get_item

        def recording(item, index):
            calls.append(index)
            return get_item(item, index)

        monkeypatch.setattr(tensordict.base, "_get_item", recording)
        td = _shortcut_nested()
        lst = [0, 2] * _LIST_MIN
        expected = td[lst]
        calls.clear()

        # the newest torch has no torch.jit.trace: patch the tracer test
        monkeypatch.setattr(tensordict._indexing, "_is_jit_tracing", lambda: True)
        result = td[lst]
        assert calls and all(index == (lst,) for index in calls)
        assert torch.equal(result["a"], expected["a"])
        assert torch.equal(result["n", "d", "e"], expected["n", "d", "e"])

    def test_compiled_int_list(self):
        # #3568: a compiled td[list] takes the generic path, in one graph
        td = _shortcut_nested()
        lst = [0, 2] * _LIST_MIN
        expected = td[lst]
        result = torch.compile(lambda td: td[lst], backend="eager", fullgraph=True)(td)
        assert result.batch_size == expected.batch_size
        assert (result == expected).all()


@functools.cache
def _redis_port():
    """The port of a reachable Redis-protocol server, or None."""
    if not _has_redis:
        return None
    import redis

    for port in (6379, 6380):
        try:
            with redis.Redis(port=port, socket_connect_timeout=2) as client:
                client.ping()
            return port
        except (redis.ConnectionError, OSError):
            continue
    return None


class TestStoresMatchDense:
    """Stores give what the dense tensordict that they store gives."""

    CASES = {
        "TensorDictStore (5,)": (TensorDictStore, (5,)),
        "TensorDictStore (5, 4)": (TensorDictStore, (5, 4)),
        "LazyStackedTensorDictStore (5, 4)": (LazyStackedTensorDictStore, (5, 4)),
    }

    @pytest.fixture
    def store_kwargs(self):
        port = _redis_port()
        if port is None:
            pytest.skip("no Redis-protocol server on localhost:6379 or :6380")
        return {"backend": "redis", "port": port, "db": 15}

    @staticmethod
    def _dense(batch_size):
        numel = int(np.prod(batch_size))
        return TensorDict(
            {
                "x": torch.arange(float(numel)).view(batch_size),
                "nested": {"y": torch.arange(numel).view(batch_size)},
            },
            batch_size,
        )

    def _store(self, store_type, dense, store_kwargs):
        if store_type is TensorDictStore:
            return TensorDictStore.from_tensordict(dense, **store_kwargs)
        return LazyStackedTensorDictStore.from_lazy_stack(
            lazy_stack(list(dense.unbind(0))), **store_kwargs
        )

    @pytest.mark.parametrize("case_name", CASES)
    def test_getitem(self, case_name, store_kwargs):
        store_type, batch_size = self.CASES[case_name]
        dense = self._dense(batch_size)
        store = self._store(store_type, dense, store_kwargs)
        getitem_failures, get_at_failures = {}, {}
        try:
            for case, index in index_pool(batch_size).items():
                try:
                    expected = dense[index]
                except Exception:
                    continue
                try:
                    result = store[index]
                    if result.batch_size != expected.batch_size:
                        getitem_failures[case] = (
                            f"batch size {tuple(result.batch_size)}, "
                            f"dense {tuple(expected.batch_size)}"
                        )
                    elif not (result.to_tensordict() == expected).all():
                        getitem_failures[case] = "values"
                except Exception as err:
                    getitem_failures[case] = _describe(err)
                try:
                    result = store.get_at("x", index)
                    if not torch.equal(result, expected["x"]):
                        get_at_failures[case] = "values"
                except Exception as err:
                    get_at_failures[case] = _describe(err)
        finally:
            store.clear_redis()
            store.close()
        _check_known_failures(f"{case_name} getitem", getitem_failures)
        _check_known_failures(f"{case_name} get_at", get_at_failures)

    @pytest.mark.parametrize("case_name", CASES)
    def test_setitem(self, case_name, store_kwargs):
        store_type, batch_size = self.CASES[case_name]
        original = self._dense(batch_size)
        store = self._store(store_type, original, store_kwargs)
        failures = {}
        try:
            for case, index in index_pool(batch_size).items():
                dense = original.clone()
                try:
                    value = dense[index].apply(lambda x: torch.full_like(x, -1))
                    dense[index] = value
                except Exception:
                    continue
                store.update(original)
                try:
                    store[index] = value
                    if not (store.to_tensordict() == dense).all():
                        failures[case] = "values"
                except Exception as err:
                    failures[case] = _describe(err)
        finally:
            store.clear_redis()
            store.close()
        _check_known_failures(f"{case_name} setitem", failures)


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
