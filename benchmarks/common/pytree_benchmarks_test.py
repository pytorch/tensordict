import pytest
import torch

from tensordict import TensorDict
from torch.utils._pytree import tree_map


@pytest.fixture
def nested_dict():
    return {
        "a": {"b": torch.randn(3, 4, 1), "c": {"d": torch.rand(3, 4, 5, 6)}},
        "c": torch.rand(3, 4, 1),
    }


@pytest.fixture
def nested_td(nested_dict):
    return TensorDict(nested_dict, [3, 4])


# reshape
def test_reshape_pytree(benchmark, nested_dict):
    benchmark.pedantic(
        tree_map,
        args=(lambda x: x.reshape(12, *x.shape[2:]), nested_dict),
    )


def test_reshape_td(benchmark, nested_td):
    benchmark.pedantic(
        nested_td.reshape,
        args=(12,),
    )


# view
def test_view_pytree(benchmark, nested_dict):
    benchmark.pedantic(
        tree_map,
        args=(lambda x: x.view(12, *x.shape[2:]), nested_dict),
    )


def test_view_td(benchmark, nested_td):
    benchmark.pedantic(
        nested_td.view,
        args=(12,),
    )


# unbind
def test_unbind_pytree(benchmark, nested_dict):
    benchmark.pedantic(
        tree_map,
        args=(lambda x: x.unbind(0), nested_dict),
    )


def test_unbind_td(benchmark, nested_td):
    benchmark.pedantic(
        nested_td.unbind,
        args=(0,),
    )


# split
def test_split_pytree(benchmark, nested_dict):
    benchmark.pedantic(
        tree_map,
        args=(lambda x: x.split([1, 2], 0), nested_dict),
    )


def test_split_td(benchmark, nested_td):
    benchmark.pedantic(
        nested_td.split,
        args=([1, 2], 0),
    )


# add
def test_add_pytree(benchmark, nested_dict):
    benchmark.pedantic(
        tree_map,
        args=(lambda x: x + 1, nested_dict),
    )


def test_add_td(benchmark, nested_td):
    benchmark.pedantic(
        nested_td.apply,
        args=(lambda x: x + 1,),
    )
