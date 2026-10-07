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
import os
import pathlib

import numpy as np
import pytest
import torch

from _utils_internal import TestTensorDictsBase

from tensordict import lazy_stack, TensorDict, UnbatchedTensor
from tensordict.store import LazyStackedTensorDictStore, TensorDictStore
from tensordict.store._store import _has_redis

_KNOWN_FAILURES_FILE = pathlib.Path(__file__).with_name("indexing_known_failures.json")
_KNOWN_FAILURES = json.loads(_KNOWN_FAILURES_FILE.read_text())
_UPDATE_KNOWN_FAILURES = bool(os.environ.get("TENSORDICT_UPDATE_KNOWN_FAILURES"))
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


def _describe(err):
    return f"raises {type(err).__name__}: {str(err).splitlines()[0][:80] if str(err) else ''}"


def _expected_names(batch_size, index, names):
    """Names that indexing a tensordict named ``names`` with ``index`` should give.

    An output dim keeps the name of the input dim that it depends on, if it
    depends on exactly one input dim, and is unnamed otherwise: a slice or a 1-D
    index keeps the name of its dim, while ``None``, a scalar bool, an N-D mask
    and advanced indices broadcast together give unnamed dims. The dependency is
    traced by indexing a tensor of coordinates along each input dim. A dim of
    size 0 or 1 does not show the dependency and is returned as ``Ellipsis``.
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
    return expected


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
                # reset one key at a time: LazyStackedTensorDictStore.update()
                # does not write nested keys
                for key in _tensor_keys(original):
                    store[key] = original[key]
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
