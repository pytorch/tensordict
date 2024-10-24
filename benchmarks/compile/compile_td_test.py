# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import argparse

import pytest
import torch
from packaging import version
from tensordict import LazyStackedTensorDict, tensorclass, TensorDict
from torch.utils._pytree import tree_map

TORCH_VERSION = version.parse(version.parse(torch.__version__).base_version)


@tensorclass
class MyTensorClass:
    a: torch.Tensor
    b: torch.Tensor
    c: torch.Tensor
    d: torch.Tensor
    e: torch.Tensor
    f: torch.Tensor


@pytest.fixture(autouse=True, scope="function")
def empty_compiler_cache():
    torch._dynamo.reset_code_caches()
    yield


# Functions
def add_one(td):
    return td + 1


def add_one_pytree(td):
    return tree_map(lambda x: x + 1, td)


def add_self(td):
    return td + td


def add_self_pytree(td):
    return tree_map(lambda x: x + x, td)


def copy(td):
    return td.copy()


def copy_pytree(td):
    return tree_map(lambda x: x, td)


def assign_and_add(td, k):
    for i in range(k, k + 100):
        td[str(i)] = i
    return td + 1


def assign_and_add_pytree(td, k, device):
    for i in range(k, k + 100):
        td[str(i)] = torch.tensor(i, device=device)
    return tree_map(lambda x: x + 1, td)


def assign_and_add_stack(td, k):
    for i in range(k, k + 100):
        td[str(i)] = torch.full((2,), i, device=td.device)
    return td + 1


def index(td, idx):
    return td[idx]


def index_pytree(td, idx):
    return tree_map(lambda x: x[idx], td)


def get_nested_td():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    d = {}
    _d = d
    for i in range(10):
        _d["a"] = torch.ones((), device=device)
        _d[str(i)] = {}
        _d = _d[str(i)]
    _d["a"] = torch.ones((), device=device)
    return TensorDict(d, device=device)


def get_flat_td():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return TensorDict(
        {str(i): torch.full((), i, device=device) for i in range(50)}, device=device
    )


def get_flat_tc():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return MyTensorClass(
        a=torch.ones((15,), device=device),
        b=torch.ones((15,), device=device),
        c=torch.ones((15,), device=device),
        d=torch.ones((15,), device=device),
        e=torch.ones((15,), device=device),
        f=torch.ones((15,), device=device),
        device=device,
    )


# Tests runtime of a simple arithmetic op over a highly nested tensordict
@pytest.mark.skipif(
    TORCH_VERSION < version.parse("2.4.0"), reason="requires torch>=2.4"
)
@pytest.mark.parametrize("mode", ["compile", "eager"])
@pytest.mark.parametrize("dict_type", ["tensordict", "pytree"])
def test_compile_add_one_nested(mode, dict_type, benchmark):
    if dict_type == "tensordict":
        if mode == "compile":
            func = torch.compile(add_one, fullgraph=True, mode="reduce-overhead")
        else:
            func = add_one
        td = get_nested_td()
    else:
        if mode == "compile":
            func = torch.compile(add_one_pytree, fullgraph=True, mode="reduce-overhead")
        else:
            func = add_one_pytree
        td = get_nested_td().to_dict()
    func(td)
    func(td)
    benchmark(func, td)


# Tests the speed of copying a nested tensordict
@pytest.mark.skipif(
    TORCH_VERSION < version.parse("2.4.0"), reason="requires torch>=2.4"
)
@pytest.mark.parametrize("mode", ["compile", "eager"])
@pytest.mark.parametrize("dict_type", ["tensordict", "pytree"])
def test_compile_copy_nested(mode, dict_type, benchmark):
    if dict_type == "tensordict":
        if mode == "compile":
            func = torch.compile(copy, fullgraph=True, mode="reduce-overhead")
        else:
            func = copy
        td = get_nested_td()
    else:
        if mode == "compile":
            func = torch.compile(copy_pytree, fullgraph=True, mode="reduce-overhead")
        else:
            func = copy_pytree
        td = get_nested_td().to_dict()
    func(td)
    func(td)
    benchmark(func, td)


# Tests runtime of a simple arithmetic op over a flat tensordict
@pytest.mark.skipif(
    TORCH_VERSION < version.parse("2.4.0"), reason="requires torch>=2.4"
)
@pytest.mark.parametrize("mode", ["compile", "eager"])
@pytest.mark.parametrize("dict_type", ["tensordict", "tensorclass", "pytree"])
def test_compile_add_one_flat(mode, dict_type, benchmark):
    if dict_type == "tensordict":
        if mode == "compile":
            func = torch.compile(add_one, fullgraph=True, mode="reduce-overhead")
        else:
            func = add_one
        td = get_flat_td()
    elif dict_type == "tensorclass":
        if mode == "compile":
            func = torch.compile(add_one, fullgraph=True, mode="reduce-overhead")
        else:
            func = add_one
        td = get_flat_tc()
    else:
        if mode == "compile":
            func = torch.compile(add_one_pytree, fullgraph=True, mode="reduce-overhead")
        else:
            func = add_one_pytree
        td = get_flat_td().to_dict()
    func(td)
    func(td)
    benchmark(func, td)


@pytest.mark.skipif(
    TORCH_VERSION < version.parse("2.4.0"), reason="requires torch>=2.4"
)
@pytest.mark.parametrize("mode", ["eager", "compile"])
@pytest.mark.parametrize("dict_type", ["tensordict", "tensorclass", "pytree"])
def test_compile_add_self_flat(mode, dict_type, benchmark):
    if dict_type == "tensordict":
        if mode == "compile":
            func = torch.compile(add_self, fullgraph=True, mode="reduce-overhead")
        else:
            func = add_self
        td = get_flat_td()
    elif dict_type == "tensorclass":
        if mode == "compile":
            func = torch.compile(add_self, fullgraph=True, mode="reduce-overhead")
        else:
            func = add_self
        td = get_flat_tc()
    else:
        if mode == "compile":
            func = torch.compile(
                add_self_pytree, fullgraph=True, mode="reduce-overhead"
            )
        else:
            func = add_self_pytree
        td = get_flat_td().to_dict()
    func(td)
    func(td)
    benchmark(func, td)


# Tests the speed of copying a flat tensordict
@pytest.mark.skipif(
    TORCH_VERSION < version.parse("2.4.0"), reason="requires torch>=2.4"
)
@pytest.mark.parametrize("mode", ["compile", "eager"])
@pytest.mark.parametrize("dict_type", ["tensordict", "pytree"])
def test_compile_copy_flat(mode, dict_type, benchmark):
    if dict_type == "tensordict":
        if mode == "compile":
            func = torch.compile(copy, fullgraph=True, mode="reduce-overhead")
        else:
            func = copy
        td = get_flat_td()
    elif dict_type == "tensorclass":
        if mode == "compile":
            func = torch.compile(copy, fullgraph=True, mode="reduce-overhead")
        else:
            func = copy
        td = get_flat_tc()
    else:
        if mode == "compile":
            func = torch.compile(copy_pytree, fullgraph=True, mode="reduce-overhead")
        else:
            func = copy_pytree
        td = get_flat_td().to_dict()
    func(td)
    func(td)
    benchmark(func, td)


# Tests the speed of assigning entries to an empty tensordict
@pytest.mark.skipif(
    TORCH_VERSION < version.parse("2.4.0"), reason="requires torch>=2.4"
)
@pytest.mark.parametrize("mode", ["compile", "eager"])
@pytest.mark.parametrize("dict_type", ["tensordict", "pytree"])
def test_compile_assign_and_add(mode, dict_type, benchmark):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    td = TensorDict(device=device)
    if dict_type == "tensordict":
        if mode == "compile":
            func = torch.compile(assign_and_add, fullgraph=True, mode="reduce-overhead")
        else:
            func = assign_and_add
        kwargs = {}
    else:
        if mode == "compile":
            func = torch.compile(
                assign_and_add_pytree, fullgraph=True, mode="reduce-overhead"
            )
        else:
            func = assign_and_add_pytree
        td = td.to_dict()
        kwargs = {"device": device}
    func(td, 5, **kwargs)
    func(td, 5, **kwargs)
    benchmark(func, td, 5, **kwargs)


# Tests the speed of assigning entries to a lazy stacked tensordict


@pytest.mark.skipif(
    TORCH_VERSION < version.parse("2.4.0"), reason="requires torch>=2.4"
)
@pytest.mark.skipif(
    torch.cuda.is_available(), reason="max recursion depth error with cuda"
)
@pytest.mark.parametrize("mode", ["compile", "eager"])
def test_compile_assign_and_add_stack(mode, benchmark):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    td = LazyStackedTensorDict(TensorDict(device=device), TensorDict(device=device))
    if mode == "compile":
        func = torch.compile(
            assign_and_add_stack, fullgraph=True, mode="reduce-overhead"
        )
    else:
        func = assign_and_add_stack
    kwargs = {}
    func(td, 5, **kwargs)
    func(td, 5, **kwargs)
    benchmark(func, td, 5, **kwargs)


# Tests indexing speed
@pytest.mark.skipif(
    TORCH_VERSION < version.parse("2.4.0"), reason="requires torch>=2.4"
)
@pytest.mark.parametrize("mode", ["compile", "eager"])
@pytest.mark.parametrize("dict_type", ["tensordict", "tensorclass", "pytree"])
@pytest.mark.parametrize("index_type", ["tensor", "slice", "int"])
def test_compile_indexing(mode, dict_type, index_type, benchmark):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    td = TensorDict(
        {"a": torch.arange(100), "b": {"c": torch.arange(100)}},
        batch_size=[100],
        device=device,
    )
    if dict_type == "tensordict":
        if mode == "compile":
            func = torch.compile(index, fullgraph=True, mode="reduce-overhead")
        else:
            func = index
    else:
        if mode == "compile":
            func = torch.compile(index_pytree, fullgraph=True, mode="reduce-overhead")
        else:
            func = index_pytree
        td = td.to_dict()
    if index_type == int:
        idx = 5
    else:
        idx = slice(None, None, 2)
    if index_type == "tensor":
        idx = torch.tensor(range(*idx.indices(10)))

    func(td, idx)
    func(td, idx)
    benchmark(func, td, idx)


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
