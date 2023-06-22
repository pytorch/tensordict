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
    benchmark(torch.unbind, tc, 0)


def test_full_like(benchmark, tc):
    benchmark(torch.full_like, tc, 2.0)


def test_zeros_like(benchmark, tc):
    benchmark(
        torch.zeros_like,
        tc,
    )


def test_ones_like(benchmark, tc):
    benchmark(
        torch.ones_like,
        tc,
    )


def test_clone(benchmark, tc):
    benchmark(
        torch.clone,
        tc,
    )


def test_squeeze(benchmark, tc):
    benchmark(
        torch.squeeze,
        tc,
    )


def test_unsqueeze(benchmark, tc):
    benchmark(torch.unsqueeze, tc, 0)


def test_split(benchmark, tc):
    benchmark(torch.split, tc, [200, 100])


def test_permute(benchmark, tc):
    benchmark(torch.permute, tc, [1, 0])


def test_stack(benchmark, tc):
    benchmark(torch.stack, [tc] * 3, 0)


def test_cat(benchmark, tc):
    benchmark(torch.cat, [tc] * 3, 0)
