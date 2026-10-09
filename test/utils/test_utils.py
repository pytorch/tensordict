# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import ast
import collections
import importlib
import inspect
import pkgutil
import random
import re
import sys
import types
import warnings
from pathlib import Path

import numpy as np
import pytest
import pyvers
import tensordict
import tensordict._td
import tensordict.tensordict
import torch
from _utils_internal import get_available_devices
from tensordict import (
    _deprecation,
    lazy_stack,
    tensorclass,
    TensorDict,
    UnbatchedTensor,
    unravel_key,
    unravel_key_list,
)
from tensordict._indexing import convert_ellipsis_to_idx
from tensordict.utils import (
    _check_recursive_properties,
    _get_shared_executor,
    _getitem_batch_size,
    _make_cache_key,
    _TensorDictPropertyError,
    _unravel_key_to_tuple,
    _unravel_keys,
    isin,
    parse_tensor_dict_string,
    remove_duplicates,
)


@pytest.mark.parametrize("tensor", [torch.rand(2, 3, 4, 5), torch.rand(2, 3, 4, 5, 6)])
@pytest.mark.parametrize(
    "index1",
    [
        slice(None),
        slice(0, 1),
        0,
        [0],
        [0, 1],
        np.arange(2),
        torch.arange(2),
        [True, True],
        Ellipsis,
    ],
)
@pytest.mark.parametrize(
    "index2",
    [
        slice(None),
        slice(1, 3, 1),
        slice(-3, -1),
        0,
        [0],
        [0, 1],
        np.arange(0, 1),
        torch.arange(2),
        range(2),
        torch.tensor([[0, 1], [0, 1]]),
        [True, False, True],
        Ellipsis,
    ],
)
@pytest.mark.parametrize(
    "index3",
    [
        slice(None),
        slice(1, 3, 1),
        slice(-3, -1),
        0,
        [0],
        [0, 1],
        np.arange(1, 3),
        torch.arange(2),
        range(2),
        torch.tensor([[0, 1], [0, 1]]),
        [True, False, True, False],
        Ellipsis,
    ],
)
@pytest.mark.parametrize(
    "index4",
    [
        slice(None),
        slice(0, 4, 2),
        slice(-4, -2),
        0,
        [0],
        [0, 1],
        np.arange(0, 4, 2),
        torch.arange(2),
        range(2),
        torch.tensor([[0, 1], [0, 1]]),
        [True, False, False, False, True],
        Ellipsis,
    ],
)
def test_getitem_batch_size(tensor, index1, index2, index3, index4):
    # cannot have 2 ellipsis
    if (index1 is Ellipsis) + (index2 is Ellipsis) + (index3 is Ellipsis) + (
        index4 is Ellipsis
    ) > 1:
        pytest.skip("cannot have more than one ellipsis in an index.")
    if (index1 is Ellipsis) + (index2 is Ellipsis) + (index3 is Ellipsis) + (
        index4 is Ellipsis
    ) == 1 and tensor.ndim == 5:
        pytest.skip("index possibly incompatible with tensor shape.")
    index = (index1, index2, index3, index4)
    index = convert_ellipsis_to_idx(index, tensor.shape)
    assert tensor[index].shape == _getitem_batch_size(tensor.shape, index), index


@pytest.mark.parametrize("tensor", [torch.rand(2, 3, 4, 5), torch.rand(2, 3, 4, 5, 6)])
@pytest.mark.parametrize("idx", range(3))
@pytest.mark.parametrize("ndim", range(1, 4))
@pytest.mark.parametrize("slice_leading_dims", [True, False])
def test_getitem_batch_size_mask(tensor, idx, ndim, slice_leading_dims):
    # test n-dimensional boolean masks are handled correctly
    if idx + ndim > 4:
        pytest.skip(
            "Not enough dimensions in test tensor for this combination of parameters"
        )
    mask_shape = (2, 3, 4, 5)[idx : idx + ndim]
    mask = torch.randint(2, mask_shape, dtype=torch.bool)
    if slice_leading_dims:
        index = (slice(None),) * idx + (mask,)
    else:
        index = (0,) * idx + (mask,)
    index = convert_ellipsis_to_idx(index, tensor.shape)
    assert tensor[index].shape == _getitem_batch_size(tensor.shape, index), index


@pytest.mark.parametrize(
    "index",
    [
        [np.True_, np.False_, np.True_],
        np.array([True, False, True]),
        (slice(None), [[0, 1], [1, 2]]),
        ([[True, False, True, False], [False, True, False, True], [True] * 4],),
        True,
        False,
        (slice(None), False),
        (True, [0, 2]),
        (True, slice(None), [0, 1]),
        (torch.ones(3, 4, dtype=torch.bool), Ellipsis),
        (Ellipsis, torch.ones(4, 5, dtype=torch.bool)),
        (slice(None), [2], None, torch.tensor([2])),
        # torch reads a uint8 tensor as a mask (deprecated)
        torch.tensor([1, 0, 2], dtype=torch.uint8),
        (slice(None), torch.tensor([[0, 1, 1, 0, 1]] * 4, dtype=torch.uint8)),
        (Ellipsis, torch.tensor(1, dtype=torch.uint8)),
        (torch.ones(3, 4, dtype=torch.uint8), Ellipsis),
        # torch reads a 0-d integer index as an int, not as an advanced index
        (torch.tensor(1), slice(None), [0, 1]),
        (np.array(1), slice(None), [0, 1]),
        ([0, 1], slice(None), torch.tensor(1)),
    ],
)
def test_getitem_batch_size_index_types(index):
    tensor = torch.zeros(3, 4, 5)
    expected = tensor[index].shape
    index = convert_ellipsis_to_idx(index, tensor.shape)
    assert _getitem_batch_size(tensor.shape, index) == expected


@pytest.mark.parametrize("index", [np.True_, (slice(None), np.False_)])
def test_getitem_batch_size_numpy_bool_scalar(index):
    with pytest.raises(IndexError, match="NumPy bool"):
        _getitem_batch_size(torch.Size([3, 4]), index)


@pytest.mark.parametrize(
    "index",
    [
        Ellipsis,
        (Ellipsis, 0),
        (None, Ellipsis, None),
        (torch.ones(3, 4, dtype=torch.bool), Ellipsis),
        ([0, 1], Ellipsis, [0, 1]),
    ],
)
def test_getitem_batch_size_ellipsis(index):
    # the Ellipsis does not need to be converted first
    tensor = torch.zeros(3, 4, 5)
    assert _getitem_batch_size(tensor.shape, index) == tensor[index].shape


@pytest.fixture
def check_invariants(monkeypatch):
    # what the TD_CHECK_INVARIANTS environment variable turns on
    monkeypatch.setattr(tensordict._td, "_CHECK_INVARIANTS", True)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        (
            {"source": {"a": torch.zeros(3)}, "batch_size": torch.Size([4])},
            "does not start with the batch size",
        ),
        (
            {
                "source": {"a": TensorDict(batch_size=[3])},
                "batch_size": torch.Size([4]),
            },
            "does not start with the batch size",
        ),
        (
            {
                "source": {"a": torch.zeros(4)},
                "batch_size": torch.Size([4]),
                "names": ["x", "y"],
            },
            "dim names",
        ),
        ({"source": {"a": torch.zeros(4)}, "batch_size": [4]}, "not a torch.Size"),
    ],
)
def test_check_invariants_raises(check_invariants, kwargs, match):
    with pytest.raises(AssertionError, match=match):
        TensorDict._new_unsafe(**kwargs)


def test_check_invariants_valid(check_invariants):
    td = TensorDict(
        a=torch.zeros(4, 3),
        nested=TensorDict(b=torch.zeros(4, 3, 2), batch_size=[4, 3, 2]),
        batch_size=[4, 3],
        names=["x", "y"],
    )
    # these are built with _new_unsafe
    td[0], td[:, [0, 2]], td.clone(), td.apply(lambda x: x + 1)
    # unbatched entries and values without a shape are not checked
    TensorDict._new_unsafe(
        {"u": UnbatchedTensor(torch.zeros(5)), "t": (torch.zeros(4),)},
        batch_size=torch.Size([4]),
    )


@pytest.mark.parametrize("name", ["min", "max", "cummin", "cummax"])
@pytest.mark.parametrize("dim", [0, 1])
def test_check_invariants_reduction_with_indices(check_invariants, name, dim):
    td = TensorDict(a=torch.randn(4, 3, 2), batch_size=[4, 3])
    out = getattr(td, name)(dim=dim, return_indices=True)
    expected = getattr(td["a"], name)(dim=dim)
    assert (out.values["a"] == expected.values).all()
    assert (out.indices["a"] == expected.indices).all()


def test_make_cache_key():
    Q = torch.rand(3)
    V = torch.zeros(2)
    args = (1, (2, 3), Q)
    kwargs = {"a": V, "b": "c", "d": ("e", "f")}
    assert _make_cache_key(args, kwargs) == (
        (1, (2, 3), id(Q)),
        (("a", id(V)), ("b", "c"), ("d", ("e", "f"))),
    )


@tensorclass
class _RecursivePropertiesTensorClass:
    tensor: torch.Tensor
    nested: TensorDict
    metadata: str


def _valid_nested_recursive_properties_tree():
    td = TensorDict(
        {
            "tensor": torch.zeros(3, 2),
            "nested": TensorDict({"value": torch.ones(3, 4)}, batch_size=[3]),
        },
        batch_size=[3],
    )
    td["metadata"] = "value"
    return td


def _valid_lazy_recursive_properties_tree():
    left = _valid_nested_recursive_properties_tree()
    right = _valid_nested_recursive_properties_tree()
    return lazy_stack([left, right], 0)


def _valid_tensorclass_recursive_properties_tree():
    return _RecursivePropertiesTensorClass(
        tensor=torch.zeros(3, 2),
        nested=TensorDict({"value": torch.ones(3, 4)}, batch_size=[3]),
        metadata="value",
        batch_size=[3],
    )


@pytest.mark.parametrize(
    "make_tree",
    [
        _valid_nested_recursive_properties_tree,
        _valid_lazy_recursive_properties_tree,
        _valid_tensorclass_recursive_properties_tree,
    ],
    ids=["nested", "lazy-stack", "tensorclass"],
)
def test_check_recursive_properties_valid_tree(make_tree):
    assert _check_recursive_properties(make_tree())


def test_check_recursive_properties_stale_unbatched_metadata():
    td = TensorDict(
        {
            "tensor": torch.zeros(3),
        },
        batch_size=[3],
    )
    td._set_str(
        "unbatched",
        UnbatchedTensor(torch.ones(())),
        inplace=False,
        validated=True,
    )

    assert not _check_recursive_properties(td, raise_exception=False)
    with pytest.raises(
        _TensorDictPropertyError,
        match=r"Batch-size mismatch.*\('unbatched',\)",
    ):
        _check_recursive_properties(td)


def test_check_recursive_properties_device_mismatch():
    td = TensorDict({"tensor": torch.zeros(3)}, batch_size=[3])
    td._device = torch.device("meta")

    with pytest.raises(
        _TensorDictPropertyError,
        match="Device mismatch: parent device=meta, leaf device=cpu",
    ):
        _check_recursive_properties(td)


def test_check_recursive_properties_lock_mismatch():
    nested = TensorDict({"value": torch.ones(3)}, batch_size=[3])
    td = TensorDict({"nested": nested}, batch_size=[3]).lock_()
    nested._is_locked = False

    with pytest.raises(
        _TensorDictPropertyError,
        match=r"Lock mismatch.*\('nested',\)",
    ):
        _check_recursive_properties(td)


def test_check_recursive_properties_names_mismatch():
    td = TensorDict(
        {
            "tensor": torch.zeros(3),
            "nested": TensorDict({"value": torch.ones(3)}, batch_size=[3]),
        },
        batch_size=[3],
        names=["batch"],
    )
    td["nested"].rename_("other")

    with pytest.raises(
        _TensorDictPropertyError,
        match=r"Names mismatch.*\('nested',\)",
    ):
        _check_recursive_properties(td)


def test_check_recursive_properties_lazy_stack_unbatched_metadata():
    data = torch.ones(())
    td = TensorDict(
        {
            "tensor": torch.zeros(3),
            "unbatched": UnbatchedTensor(data, batch_size=[3]),
        },
        batch_size=[3],
    )
    result = lazy_stack([td, td], 0)

    assert result["unbatched"].batch_size == result.batch_size
    assert _check_recursive_properties(result)


# (key, unravel_key(key))
_VALID_KEYS = [
    ("a", "a"),
    (("a",), "a"),
    (("a", "b"), ("a", "b")),
    ((("a", "b"), "c"), ("a", "b", "c")),
    (("a", ("b", ("c",))), ("a", "b", "c")),
    ((("a",),), "a"),
]
# Tuples with a part that is neither a str nor a tuple of str unravel to ().
# TorchRL's Composite.__getitem__ relies on index tuples such as
# (slice(None), 0) unravelling to ().
_INVALID_TUPLE_KEYS = [
    ("a", 1),
    (("a", 1), "b"),
    ("a", ()),
    (),
    ((),),
    ("a", (1,), ("b",)),
    ("a", (slice(None),), ("b",)),
    (slice(None), 0),
    (0, Ellipsis),
]
_NON_TUPLE_INVALID_KEYS = [1, None, ["a"]]


@pytest.mark.parametrize("listtype", (list, tuple))
def test_unravel_key_list(listtype):
    keys_in = listtype(["a0", ("b0",), ("c0", ("d",))])
    keys_out = unravel_key_list(keys_in)
    assert keys_out == ["a0", "b0", ("c0", "d")]


@pytest.mark.parametrize("key", _INVALID_TUPLE_KEYS + _NON_TUPLE_INVALID_KEYS)
def test_unravel_key_list_invalid(key):
    with pytest.raises(RuntimeError, match="key should be a Sequence<NestedKey>"):
        unravel_key_list(["a", key])


@pytest.mark.parametrize("key,expected", _VALID_KEYS)
def test_unravel_key(key, expected):
    assert unravel_key(key) == expected


@pytest.mark.parametrize("key", _INVALID_TUPLE_KEYS)
def test_unravel_key_invalid_tuple(key):
    assert unravel_key(key) == ()


@pytest.mark.parametrize("key", _NON_TUPLE_INVALID_KEYS)
def test_unravel_key_invalid(key):
    with pytest.raises(RuntimeError, match="key should be a Sequence<NestedKey>"):
        unravel_key(key)


def test_unravel_keys():
    assert _unravel_keys(("a",)) == "a"
    assert _unravel_keys("a", ("b", ("c",)), ("d",)) == ("a", ("b", "c"), "d")


@pytest.mark.parametrize("key,expected", _VALID_KEYS)
def test_unravel_key_to_tuple(key, expected):
    expected = (expected,) if isinstance(expected, str) else expected
    assert _unravel_key_to_tuple(key) == expected


@pytest.mark.parametrize("key", _INVALID_TUPLE_KEYS + _NON_TUPLE_INVALID_KEYS)
def test_unravel_key_to_tuple_invalid(key):
    assert _unravel_key_to_tuple(key) == ()


def _reference_unravel_key_to_tuple(key):
    # What _unravel_key_to_tuple computes, written for clarity rather than speed.
    if isinstance(key, str):
        return (key,)
    if not isinstance(key, tuple):
        return ()
    parts = []
    for subkey in key:
        subkey = _reference_unravel_key_to_tuple(subkey)
        if not subkey:
            return ()
        parts.extend(subkey)
    return tuple(parts)


class _StrKey(str):
    pass


_KeyPair = collections.namedtuple("_KeyPair", ["first", "second"])
_VALID_LEAVES = ["a", "b", "", _StrKey("c")]
_INVALID_LEAVES = [0, None, slice(None), Ellipsis, (), ["a"]]


def _random_key(rng, depth=0):
    if depth == 3 or rng.random() < 0.4:
        if rng.random() < 0.85:
            return rng.choice(_VALID_LEAVES)
        return rng.choice(_INVALID_LEAVES)
    parts = tuple(_random_key(rng, depth + 1) for _ in range(rng.randint(1, 3)))
    if len(parts) == 2 and rng.random() < 0.2:
        return _KeyPair(*parts)
    return parts


def test_unravel_key_matches_reference():
    rng = random.Random(0)
    for _ in range(3000):
        key = _random_key(rng)
        expected = _reference_unravel_key_to_tuple(key)
        result = _unravel_key_to_tuple(key)
        assert result == expected, key
        assert type(result) is tuple, key
        # str subclasses are kept, as the C++ extension kept them
        assert [type(part) for part in result] == [type(part) for part in expected]
        if isinstance(key, (str, tuple)):
            expected_key = expected[0] if len(expected) == 1 else expected
            if isinstance(key, str):
                expected_key = key
            assert unravel_key(key) == expected_key, key
        else:
            with pytest.raises(RuntimeError, match="Sequence<NestedKey>"):
                unravel_key(key)


def test_unravel_key_tuple_subclass():
    key = _KeyPair("a", ("b", "c"))
    assert _unravel_key_to_tuple(key) == ("a", "b", "c")
    flat = _KeyPair("a", "b")
    assert _unravel_key_to_tuple(flat) == ("a", "b")
    assert type(_unravel_key_to_tuple(flat)) is tuple


@pytest.mark.parametrize("keys", ["ab", iter(["a"]), {"a"}], ids=["str", "iter", "set"])
def test_unravel_key_list_rejects_non_sequences(keys):
    # "incompatible function arguments" is the C++ binding's wording, which
    # TorchRL's tests match.
    with pytest.raises(
        TypeError, match="incompatible function arguments.*list or a tuple of keys"
    ):
        unravel_key_list(keys)


def test_C_module_is_deprecated():
    sys.modules.pop("tensordict._C", None)
    with pytest.warns(DeprecationWarning, match="tensordict._C is deprecated"):
        import tensordict._C as _C
    assert _C.unravel_key is unravel_key
    assert _C.unravel_key_list is unravel_key_list
    assert _C._unravel_key_to_tuple is _unravel_key_to_tuple
    # the C++ binding took a single key
    assert _C.unravel_keys(("a", ("b",))) == ("a", "b")


_NESTED_KEY_REPLACEMENT = (
    "isinstance(key, tensordict.NestedKey) (which rejects lists and accepts "
    "nested tuples)"
)
_SEQ_OF_NESTED_KEY_REPLACEMENT = (
    "isinstance(key, tensordict.NestedKey) on each key (which rejects lists "
    "and accepts nested tuples)"
)
# The deprecated names of tensordict.utils, with the private object that each
# one returns and what to use instead.
_DEPRECATED_UTILS_NAMES = {
    "BufferLegacy": ("_BufferLegacy", None),
    "KeyDependentDefaultDict": ("_KeyDependentDefaultDict", None),
    "NESTED_TENSOR_ERR": ("_NESTED_TENSOR_ERR", None),
    "NUMPY_TO_TORCH_DTYPE_DICT": ("_NUMPY_TO_TORCH_DTYPE_DICT", None),
    "TORCH_TO_NUMPY_DTYPE_DICT": ("_TORCH_TO_NUMPY_DTYPE_DICT", None),
    "cache": ("_cache_while_locked", None),
    "erase_cache": ("_erase_cache_first", None),
    "infer_size_impl": ("_infer_size_impl", None),
    "int_generator": ("_int_generator", None),
    "is_namedtuple": ("_is_namedtuple", None),
    "is_namedtuple_class": ("_is_namedtuple_class", None),
    "is_nested_key": ("_is_nested_key", _NESTED_KEY_REPLACEMENT),
    "is_seq_of_nested_key": ("_is_seq_of_nested_key", _SEQ_OF_NESTED_KEY_REPLACEMENT),
    "lock_blocked": ("_lock_blocked", None),
    "prod": ("_prod", "math.prod"),
    "strtobool": ("_strtobool", None),
    "unravel_keys": ("_unravel_keys", "unravel_key or unravel_key_list"),
}


def _deprecated_message_pattern(what, replacement):
    message = f"{what} is deprecated and will be removed in TensorDict 0.17."
    if replacement is not None:
        message += f" Use {replacement} instead."
    return "^" + re.escape(message) + "$"


@pytest.mark.parametrize(
    "name,private,replacement",
    [(name, *value) for name, value in _DEPRECATED_UTILS_NAMES.items()]
    + [
        (
            "convert_ellipsis_to_idx",
            tensordict._indexing.convert_ellipsis_to_idx,
            None,
        ),
        ("get_json_backend", tensordict._utils_key_json.get_json_backend, None),
        (
            "json_dumps",
            tensordict._utils_key_json.json_dumps,
            "json.dumps or orjson.dumps",
        ),
        ("set_json_backend", tensordict._utils_key_json.set_json_backend, None),
    ],
)
def test_utils_deprecated_names(name, private, replacement):
    if isinstance(private, str):
        private = getattr(tensordict.utils, private)
    with pytest.warns(
        DeprecationWarning,
        match=_deprecated_message_pattern(f"tensordict.utils.{name}", replacement),
    ) as record:
        value = getattr(tensordict.utils, name)
    assert record[0].filename == __file__
    assert value is private


def test_utils_deprecated_names_behave_as_before():
    with pytest.warns(DeprecationWarning, match="tensordict.utils.prod"):
        from tensordict.utils import prod
    assert prod([2, 3, 4]) == 24
    assert prod(torch.Size([2, 3])) == 6
    assert prod([]) == 1
    with pytest.warns(DeprecationWarning, match="tensordict.utils.strtobool"):
        from tensordict.utils import strtobool
    assert strtobool("Yes") == 1
    assert strtobool("off") == 0
    with pytest.raises(ValueError, match="invalid truth value"):
        strtobool("maybe")
    with pytest.warns(DeprecationWarning, match="tensordict.utils.infer_size_impl"):
        from tensordict.utils import infer_size_impl
    assert infer_size_impl([2, -1], 6) == [2, 3]
    with pytest.warns(DeprecationWarning, match="tensordict.utils.is_nested_key"):
        from tensordict.utils import is_nested_key
    with pytest.warns(
        DeprecationWarning, match="tensordict.utils.is_seq_of_nested_key"
    ):
        from tensordict.utils import is_seq_of_nested_key
    # Unlike NestedKey, lists are nested keys and nested tuples are not.
    assert is_nested_key(["a", "b"])
    assert not is_nested_key(("a", ("b",)))
    assert is_seq_of_nested_key([("a", "b"), "c"])


def test_tensordict_module_deprecated_names():
    legacy_module = tensordict.tensordict
    # Importing the legacy module, and the names that it still exports, does not warn.
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        importlib.reload(legacy_module)
        from tensordict.tensordict import (  # noqa: F401
            assert_allclose_td,
            dense_stack_tds,
            expand_as_right,
            expand_right,
            is_tensor_collection,
            is_tensorclass,
            LazyStackedTensorDict,
            make_tensordict,
            MemoryMappedTensor,
            merge_tensordicts,
            NestedKey,
            NO_DEFAULT,
            pad,
            pad_sequence,
            TensorDict,
            TensorDictBase,
        )
    assert legacy_module.TensorDict is TensorDict

    deprecated = {
        name: value
        for name, value in _DEPRECATED_UTILS_NAMES.items()
        if name
        in (
            "cache",
            "erase_cache",
            "infer_size_impl",
            "int_generator",
            "is_nested_key",
            "is_seq_of_nested_key",
            "lock_blocked",
        )
    }
    deprecated["convert_ellipsis_to_idx"] = (
        tensordict._indexing.convert_ellipsis_to_idx,
        None,
    )
    deprecated["implement_for"] = (pyvers.implement_for, "pyvers.implement_for")
    for name, (private, replacement) in deprecated.items():
        if isinstance(private, str):
            private = getattr(tensordict.utils, private)
        with pytest.warns(
            DeprecationWarning,
            match=_deprecated_message_pattern(
                f"tensordict.tensordict.{name}", replacement
            ),
        ) as record:
            value = getattr(legacy_module, name)
        assert record[0].filename == __file__
        assert value is private


class TestDeprecationHelpers:
    def test_warn_deprecated(self):
        with pytest.warns(
            DeprecationWarning,
            match=r"^old\(\) is deprecated and will be removed in TensorDict 0\.17\. "
            r"Use new\(\) instead\.$",
        ) as record:
            _deprecation.warn_deprecated(
                "old()", removal="0.17", replacement="new()", stacklevel=1
            )
        assert record[0].filename == __file__

    def test_deprecated(self):
        @_deprecation.deprecated("old()", removal="0.17")
        def old(x):
            """Adds one."""
            return x + 1

        assert old.__name__ == "old"
        assert old.__doc__ == "Adds one."
        with pytest.warns(
            DeprecationWarning,
            match=r"^old\(\) is deprecated and will be removed in TensorDict 0\.17\.$",
        ) as record:
            assert old(1) == 2
        # The warning points to the caller of the deprecated function.
        assert record[0].filename == __file__

    def test_deprecated_attributes(self):
        module = types.ModuleType("mod")
        module.__getattr__ = _deprecation.deprecated_attributes(
            "mod", {"old": (1, "mod.new")}, removal="0.17"
        )
        with pytest.warns(
            DeprecationWarning,
            match=r"^mod\.old is deprecated and will be removed in TensorDict 0\.17\. "
            r"Use mod\.new instead\.$",
        ) as record:
            assert module.old == 1
        assert record[0].filename == __file__
        with pytest.raises(
            AttributeError, match="module 'mod' has no attribute 'other'"
        ):
            module.other


def _version_tuple(version):
    return tuple(int(part) for part in version.split(".")[:2])


def test_deprecation_deadlines():
    # Every deprecation names the release that removes it, either as the
    # removal= argument of a tensordict._deprecation helper or as "removed in
    # TensorDict X.Y" in a message or docstring. Once version.txt reaches that
    # release, the deprecated code has to go.
    package = Path(tensordict.__file__).parent
    version_file = package.parent / "version.txt"
    if not version_file.exists():
        pytest.skip("version.txt is only available in a source checkout")
    current = _version_tuple(version_file.read_text().strip())
    overdue = []
    for path in sorted(package.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if (
                isinstance(node, ast.keyword)
                and node.arg == "removal"
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
            ):
                versions = [node.value.value]
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                versions = re.findall(r"removed in TensorDict (\d+\.\d+)", node.value)
            else:
                continue
            overdue.extend(
                f"{path.relative_to(package.parent)}:{node.lineno}: {version}"
                for version in versions
                if _version_tuple(version) <= current
            )
    assert not overdue, (
        f"version.txt is {'.'.join(map(str, current))}; remove these deprecations:\n"
        + "\n".join(overdue)
    )


@pytest.mark.parametrize("key", ("tensor1", "tensor3"))
@pytest.mark.parametrize("dim", (0, 1, -1, -2))
def test_isin_1dim(key, dim):
    td = TensorDict(
        {
            "tensor1": torch.tensor([[1, 2, 3], [4, 5, 6], [1, 2, 3], [7, 8, 9]]),
            "tensor2": torch.tensor([[10, 20], [30, 40], [40, 50], [50, 60]]),
        },
        batch_size=[4],
    )
    td_ref = TensorDict(
        {
            "tensor1": torch.tensor([[1, 2, 3], [4, 5, 6], [10, 11, 12]]),
            "tensor2": torch.tensor([[10, 20], [30, 40], [50, 60]]),
        },
        batch_size=[3],
    )

    if key == "tensor3":
        with pytest.raises(
            KeyError, match=f"Key '{key}' not found in input or not a tensor."
        ):
            isin(td, td_ref, key, dim)
    elif dim in (1, -2):
        with pytest.raises(
            ValueError,
            match=f"The specified dimension '{dim}' is invalid for an input TensorDict with batch size .*.",
        ):
            isin(td, td_ref, key, dim)
    else:
        in_ref = isin(td, td_ref, key, dim)
        expected_in_ref = torch.tensor([True, True, True, False])
        torch.testing.assert_close(in_ref, expected_in_ref)

        with pytest.raises(
            ValueError,
            match="The number of dimensions in the batch size of the input and reference must be the same.",
        ):
            td.batch_size = []
            isin(td, td_ref, key, dim)


@pytest.mark.parametrize("dim", (0, 1, -1, -2))
def test_isin_2dim(dim):
    key = "tensor1"
    input = TensorDict(
        {
            "tensor1": torch.ones(4, 4),
            "tensor2": torch.ones(4, 4),
        },
        batch_size=[4, 4],
    )
    td_ref = TensorDict(
        {
            "tensor1": torch.ones(4, 4),
            "tensor2": torch.ones(4, 4),
        },
        batch_size=[4, 4],
    )

    positive_dim = dim if dim >= 0 else dim + 2
    input[key][(slice(None),) * positive_dim + (0,)] = 2
    in_ref = isin(input, td_ref, key, dim)
    expected_in_ref = torch.tensor([False, True, True, True])
    torch.testing.assert_close(in_ref, expected_in_ref)


@pytest.mark.parametrize("key", ("tensor1", "tensor3", "next"))
@pytest.mark.parametrize("dim", (0, 1, -1, -2))
@pytest.mark.parametrize("device", get_available_devices())
def test_remove_duplicates_1dim(key, dim, device):
    input_tensordict = TensorDict(
        {
            "tensor1": torch.tensor([[1, 2, 3], [4, 5, 6], [1, 2, 3], [7, 8, 9]]),
            "tensor2": torch.tensor([[10, 20], [30, 40], [40, 50], [50, 60]]),
            "next": TensorDict(
                {
                    "tensor3": torch.tensor([[0], [1], [2], [3]]),
                    "tensor4": torch.tensor([[10], [11], [12], [13]]),
                },
                batch_size=[4],
                device=device,
            ),
        },
        batch_size=[4],
        device=device,
    )

    # Test for non-existent key
    if key == "tensor3":
        with pytest.raises(
            KeyError, match=f"The key '{key}' does not exist in the TensorDict."
        ):
            remove_duplicates(input_tensordict, key, dim)

    # Test for non-leaf key
    elif key == "next":
        with pytest.raises(
            KeyError,
            match=f"The key '{key}' does not point to a tensor in the TensorDict.",
        ):
            remove_duplicates(input_tensordict, key, dim)

    # Test for invalid dimension
    elif dim in (1, -2):
        with pytest.raises(
            ValueError,
            match=f"The specified dimension '{dim}' is invalid for a TensorDict with batch size .*.",
        ):
            remove_duplicates(input_tensordict, key, dim)
    else:
        output_tensordict = remove_duplicates(input_tensordict, key, dim)
        expected_output = TensorDict(
            {
                "tensor1": torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
                "tensor2": torch.tensor([[10, 20], [30, 40], [50, 60]]),
                "next": TensorDict(
                    {
                        "tensor3": torch.tensor([[0], [1], [3]]),
                        "tensor4": torch.tensor([[10], [11], [13]]),
                    },
                    batch_size=[3],
                    device=device,
                ),
            },
            batch_size=[3],
            device=device,
        )
        assert (output_tensordict == expected_output).all()


@pytest.mark.parametrize("dim", (0, 1, 2, -1, -2, -3))
@pytest.mark.parametrize("device", get_available_devices())
def test_remove_duplicates_2dims(dim, device):
    key = "tensor1"
    input_tensordict = TensorDict(
        {
            "tensor1": torch.ones(4, 4),
            "tensor2": torch.ones(4, 4),
        },
        batch_size=[4, 4],
        device=device,
    )

    if dim in (2, -3):
        with pytest.raises(
            ValueError,
            match=f"The specified dimension '{dim}' is invalid for a TensorDict with batch size .*.",
        ):
            remove_duplicates(input_tensordict, key, dim)

    else:
        output_tensordict = remove_duplicates(input_tensordict, key, dim)
        if dim in (0, -2):
            expected_output = TensorDict(
                {
                    "tensor1": torch.ones(1, 4),
                    "tensor2": torch.ones(1, 4),
                },
                batch_size=[1, 4],
                device=device,
            )
        else:
            expected_output = TensorDict(
                {
                    "tensor1": torch.ones(4, 1),
                    "tensor2": torch.ones(4, 1),
                },
                batch_size=[4, 1],
                device=device,
            )
        assert (output_tensordict == expected_output).all()


def test_parse_tensor_dict_string():
    td = TensorDict(a=0)
    assert str(td) == str(parse_tensor_dict_string(str(td)))
    td = TensorDict(a=[0], batch_size=[1])
    assert str(td) == str(parse_tensor_dict_string(str(td)))
    td = TensorDict(a=[[0] * 2], batch_size=[1, 2])
    assert str(td) == str(parse_tensor_dict_string(str(td)))
    td = TensorDict(
        a=[[0] * 2], b=TensorDict(c=[[1] * 2], batch_size=[1, 2]), batch_size=[1]
    )
    assert str(td) == str(parse_tensor_dict_string(str(td)))

    td = TensorDict(
        a=[[0] * 2],
        b=TensorDict(c=[[1] * 2], batch_size=[1, 2]),
        batch_size=[1],
        device="cpu",
    )
    assert str(td) == str(parse_tensor_dict_string(str(td)))
    td = TensorDict(
        a=[[0] * 2],
        b=TensorDict(c=[[1] * 2], batch_size=[1, 2], device="cpu"),
        batch_size=[1],
    )
    assert str(td) == str(parse_tensor_dict_string(str(td)))


def test_get_shared_executor():
    executor = _get_shared_executor(2)
    # repeated calls with the same thread count reuse the pool
    assert _get_shared_executor(2) is executor
    # distinct thread counts get distinct pools
    assert _get_shared_executor(3) is not executor
    assert executor.submit(lambda: 1).result() == 1
    # the pool survives (i.e. is not shut down by) a threaded save
    td = TensorDict({"a": torch.zeros(2), "b": torch.ones(3)})
    td.consolidate(num_threads=2)
    assert executor.submit(lambda: 2).result() == 2


# Modules that another change turns into deprecated shims over private modules.
_NOT_PUBLIC_MODULES = {"tensordict.tabular", "tensordict.testing"}

# Public functions and classes that other changes of the API cleanup remove or
# make private, and so are left out of ``__all__``. Remove the entries as those
# changes land; entries for names that no longer exist are ignored.
_NOT_IN_ALL_PENDING = {
    "tensordict.base": {"from_list"},
    "tensordict.memmap": {"implements_for_memmap"},
    "tensordict.nn.distributions.continuous": {"NormalParamWrapper"},
    "tensordict.nn.functional_modules": {
        "extract_weights_and_buffers",
        "get_functional",
        "is_functional",
        "make_functional",
        "repopulate_module",
        "set_tensor",
        "set_tensor_dict",
    },
    "tensordict.nn.params": {"implements_for_tdparam"},
    "tensordict.nn.utils": {"StrEnum"},
}

# Public functions and classes that stay out of ``__all__`` on purpose.
_NOT_IN_ALL = {
    # Python < 3.11 has no typing.dataclass_transform, so these modules define a
    # fallback with that name. On Python >= 3.11 the name is imported from typing.
    "tensordict.tensorclass": {"dataclass_transform"},
    "tensordict.typedtensordict": {"dataclass_transform"},
    # The names in ``discrete.__all__`` are the classes of
    # ``tensordict.nn.distributions.distributions_maps``. rand_one_hot is
    # public through ``tensordict.nn``.
    "tensordict.nn.distributions.discrete": {"rand_one_hot"},
}


def _public_module_names(package=tensordict):
    yield package.__name__
    for info in pkgutil.iter_modules(package.__path__, package.__name__ + "."):
        if info.name.rsplit(".", 1)[-1].startswith("_"):
            continue
        if info.name in _NOT_PUBLIC_MODULES:
            continue
        if info.ispkg:
            yield from _public_module_names(importlib.import_module(info.name))
        else:
            yield info.name


@pytest.mark.parametrize("module_name", sorted(_public_module_names()))
def test_public_module_all(module_name):
    # Import by name: ``tensordict.memmap`` and ``tensordict.tensorclass`` are
    # a function and a decorator as attributes of the package.
    module = importlib.import_module(module_name)
    namespace = vars(module)
    assert "__all__" in namespace, f"{module_name} does not define __all__"
    module_all = namespace["__all__"]
    assert isinstance(module_all, (list, tuple))
    assert all(isinstance(name, str) for name in module_all)
    assert len(set(module_all)) == len(module_all), "__all__ has duplicates"
    # A name served by a module __getattr__ is not in vars(module).
    undefined = [name for name in module_all if name not in namespace]
    assert not undefined, f"{module_name}.__all__ lists undefined names: {undefined}"
    exempt = _NOT_IN_ALL_PENDING.get(module_name, set()) | _NOT_IN_ALL.get(
        module_name, set()
    )
    missing = sorted(
        name
        for name, obj in namespace.items()
        if not name.startswith("_")
        and (inspect.isfunction(obj) or inspect.isclass(obj))
        and obj.__module__ == module_name
        and name not in module_all
        and name not in exempt
    )
    assert not missing, f"{module_name}.__all__ misses public names: {missing}"


def test_public_module_names():
    module_names = set(_public_module_names())
    for module_name in (
        "tensordict",
        "tensordict.memmap",
        "tensordict.nn.distributions.truncated_normal",
        "tensordict.prototype.fx",
        "tensordict.tensorclass",
    ):
        assert module_name in module_names
    assert not any(
        part.startswith("_") for name in module_names for part in name.split(".")
    )


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
