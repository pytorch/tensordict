import pytest
import torch

from tensordict.prototype import tensorclass


@tensorclass
class MyData:
    a: torch.Tensor
    b: torch.Tensor
    other: str
    nested: "MyData" = None


@pytest.fixture
def a():
    return torch.zeros(300, 400, 50)


@pytest.fixture
def b():
    return torch.zeros(300, 400, 50)


@pytest.fixture
def tc(a, b):
    return MyData(
        a=a,
        b=b,
        other="hello",
        nested=MyData(
            a=a.clone(), b=b.clone(), other="goodbye", batch_size=[300, 400, 50]
        ),
        batch_size=[300, 400],
    )


def test_unbind(benchmark, tc):
    benchmark.pedantic(torch.unbind, args=(tc, 0), iterations=10, rounds=10)


def test_full_like(benchmark, tc):
    benchmark.pedantic(torch.full_like, args=(tc, 2.0), iterations=10, rounds=10)


def test_zeros_like(benchmark, tc):
    benchmark.pedantic(torch.zeros_like, args=(tc,), iterations=10, rounds=10)


def test_ones_like(benchmark, tc):
    benchmark.pedantic(torch.ones_like, args=(tc,), iterations=10, rounds=10)


def test_clone(benchmark, tc):
    benchmark.pedantic(torch.clone, args=(tc,), iterations=10, rounds=10)


def test_squeeze(benchmark, tc):
    benchmark.pedantic(torch.squeeze, args=(tc,), iterations=10, rounds=10)


def test_unsqueeze(benchmark, tc):
    benchmark.pedantic(torch.unsqueeze, args=(tc, 0), iterations=10, rounds=10)


def test_split(benchmark, tc):
    benchmark.pedantic(torch.split, args=(tc, [200, 100]), iterations=10, rounds=10)


def test_permute(benchmark, tc):
    benchmark.pedantic(torch.permute, args=(tc, [1, 0]), iterations=10, rounds=10)


def test_stack(benchmark, tc):
    benchmark.pedantic(torch.stack, args=([tc] * 3, 0), iterations=10, rounds=10)


def test_cat(benchmark, tc):
    benchmark.pedantic(torch.cat, args=([tc] * 3, 0), iterations=10, rounds=10)
