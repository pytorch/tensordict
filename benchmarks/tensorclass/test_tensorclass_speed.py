# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import argparse
from dataclasses import make_dataclass

import pytest
import torch

from tensordict import tensorclass, TensorClass, TensorDict, TypedTensorDict


@tensorclass
class MyData:
    a: torch.Tensor
    b: torch.Tensor
    c: str
    d: "MyData" = None


class MyDataTensorOnly(TensorClass["tensor_only"]):
    a: torch.Tensor
    b: torch.Tensor
    c: torch.Tensor | None


class MyTypedTensorDict(TypedTensorDict):
    a: torch.Tensor
    b: torch.Tensor


def test_tc_init(benchmark):
    z = torch.zeros(())
    o = torch.ones(())
    benchmark(lambda: MyData(a=z, b=o, c="a string", d=None))


def test_tc_init_tensor_only(benchmark):
    z = torch.zeros(())
    o = torch.ones(())
    benchmark(lambda: MyDataTensorOnly(a=z, b=o, c=None))


def test_ttd_init(benchmark):
    z = torch.zeros(())
    o = torch.ones(())

    def make_ttd():
        return MyTypedTensorDict(a=z, b=o, batch_size=[])

    benchmark(make_ttd)


def test_td_init_reference(benchmark):
    z = torch.zeros(())
    o = torch.ones(())

    def make_td():
        return TensorDict({"a": z, "b": o}, batch_size=[])

    benchmark(make_td)


def test_tc_init_nested(benchmark):
    z = torch.zeros(())
    o = torch.ones(())
    benchmark(
        lambda: MyData(a=z, b=o, c="a string", d=MyData(a=z, b=o, c="a string", d=None))
    )


class ManyFieldsTC(TensorClass["tensor_only"]):
    f0: torch.Tensor
    f1: torch.Tensor
    f2: torch.Tensor
    f3: torch.Tensor
    f4: torch.Tensor
    f5: torch.Tensor
    f6: torch.Tensor
    f7: torch.Tensor
    f8: torch.Tensor
    f9: torch.Tensor


@pytest.mark.parametrize("num_fields", [4, 16, 64, 256])
@pytest.mark.parametrize("tensor_only", [False, True])
@pytest.mark.parametrize("batch_size", [(), (32,)])
@pytest.mark.parametrize("device", [None, "cpu"])
def test_tc_init_tensor_schema(benchmark, num_fields, tensor_only, batch_size, device):
    cls = tensorclass(tensor_only=tensor_only)(
        make_dataclass(
            "TensorSchema", [(f"f{i}", torch.Tensor) for i in range(num_fields)]
        )
    )
    source = {f"f{i}": torch.empty(32, 8) for i in range(num_fields)}
    benchmark(cls, **source, batch_size=batch_size, device=device)


@pytest.mark.parametrize("num_children", [4, 64])
@pytest.mark.parametrize("tensor_only", [False, True])
@pytest.mark.parametrize("ready_children", [False, True])
def test_tc_init_nested_tensor_schema(
    benchmark, num_children, tensor_only, ready_children
):
    child_cls = tensorclass(tensor_only=tensor_only)(
        make_dataclass("Child", [(f"f{i}", torch.Tensor) for i in range(4)])
    )
    parent_cls = tensorclass(tensor_only=tensor_only)(
        make_dataclass("Parent", [(f"g{i}", child_cls) for i in range(num_children)])
    )
    source = {
        f"g{i}": {f"f{j}": torch.empty(32, 8) for j in range(4)}
        for i in range(num_children)
    }
    if ready_children:
        source = {
            key: child_cls(**value, batch_size=[32], device="cpu")
            for key, value in source.items()
        }
        benchmark(parent_cls, **source, batch_size=[32], device="cpu")
    else:

        def build():
            children = {
                key: child_cls(**value, batch_size=[32], device="cpu")
                for key, value in source.items()
            }
            return parent_cls(**children, batch_size=[32], device="cpu")

        benchmark(build)


def test_tc_init_many_fields(benchmark):
    z = torch.zeros(())
    kwargs = {f"f{i}": z for i in range(10)}
    benchmark(lambda: ManyFieldsTC(**kwargs))


def test_tc_first_layer_tensor(benchmark):
    d = MyData(a=0, b=1, c="a string", d=MyData(None, None, None))

    def get():
        return d.a

    benchmark(get)


def test_tc_first_layer_tensor_only(benchmark):
    z = torch.zeros(())
    o = torch.ones(())
    d = MyDataTensorOnly(a=z, b=o, c=None)

    def get():
        return d.a

    benchmark(get)


def test_ttd_first_layer_tensor(benchmark):
    d = MyTypedTensorDict(a=torch.zeros(()), b=torch.ones(()), batch_size=[])

    def get():
        return d.a

    benchmark(get)


def test_td_first_layer_tensor_reference(benchmark):
    d = TensorDict({"a": torch.zeros(()), "b": torch.ones(())}, batch_size=[])

    def get():
        return d["a"]

    benchmark(get)


def test_tc_first_layer_tensor_set(benchmark):
    d = MyData(a=0, b=1, c="a string", d=MyData(None, None, None))
    z = torch.zeros(())

    def set(d=d, z=z):
        d.a = z

    benchmark(set)


def test_tc_first_layer_tensor_only_set(benchmark):
    z = torch.zeros(())
    o = torch.ones(())
    d = MyDataTensorOnly(a=z, b=o, c=None)

    def set(d=d, z=z):
        d.a = z

    benchmark(set)


def test_tc_first_layer_nontensor(benchmark):
    d = MyData(a=0, b=1, c="a string", d=MyData(None, None, None))
    benchmark(lambda: d.c)


def test_tc_second_layer_tensor(benchmark):
    d = MyData(a=0, b=1, c="a string", d=MyData(torch.zeros(()), None, None))
    benchmark(lambda: d.d.a)


def test_tc_second_layer_nontensor(benchmark):
    d = MyData(a=0, b=1, c="a string", d=MyData(torch.zeros(()), None, "a string"))
    benchmark(lambda: d.d.c)


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
