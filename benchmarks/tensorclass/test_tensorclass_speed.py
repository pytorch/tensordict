# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import argparse

import pytest
import torch

from tensordict import tensorclass, TensorClass


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


def test_tc_init(benchmark):
    z = torch.zeros(())
    o = torch.ones(())
    benchmark(lambda: MyData(a=z, b=o, c="a string", d=None))


def test_tc_init_tensor_only(benchmark):
    z = torch.zeros(())
    o = torch.ones(())
    benchmark(lambda: MyDataTensorOnly(a=z, b=o, c=None))


def test_tc_init_nested(benchmark):
    z = torch.zeros(())
    o = torch.ones(())
    benchmark(
        lambda: MyData(a=z, b=o, c="a string", d=MyData(a=z, b=o, c="a string", d=None))
    )


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
