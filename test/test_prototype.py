import torch
import pytest

from _utils_internal import get_available_devices
from tensordict import TensorMap


@pytest.mark.parametrize("device", get_available_devices())
def test_tensormap_simple_set_tensor(device):
    m = TensorMap()
    x = torch.ones(3)
    m['a'] = x
    assert m['a'] is x

    m['a'] = torch.zeros(3)
    assert (m['a'] == 0).all()

    m['b'] = m['a']
    assert m['b'] is m['a']


@pytest.mark.parametrize("device", get_available_devices())
def test_tensormap_simple_set_map(device):
    m1 = TensorMap()
    m2 = TensorMap()

    m1['a'] = m2
    assert m1['a'] is m2  # Failing - Need to fix!


@pytest.mark.parametrize("device", get_available_devices())
def test_tensormap_nested_set(device):
    m1 = TensorMap()
    m2 = TensorMap()
    x = torch.rand(3)
    y = torch.rand(3)

    m1['a'] = torch.ones(3)
    m1['b'] = m2
    m2['c'] = x

    assert x is m1['b']['c']
    assert m1['b']['c'] is m1['b', 'c']
    assert m1['b', 'c'] is m2['c']

    m1['b', 'c'] = y
    assert m1['b']['c'] is y
    assert m1['b', 'c'] is y
    assert m2['c'] is y


@pytest.mark.parametrize("device", get_available_devices())
def test_tensormap_nested_overrite(device):
    m1 = TensorMap()

    m1['a'] = torch.ones(3)
    assert (m1['a'] == 1).all()
    m1['a', 'b', 'c'] = torch.ones(3)
    assert type(m1['a']) is TensorMap
    assert type(m1['a', 'b']) is TensorMap
    assert m1['a', 'b', 'c'] is m1['a', 'b']['c']
    assert (m1['a', 'b', 'c'] == 1).all()

    m2 = TensorMap()
    m2['x'] = m1['a', 'b']
    assert m2['x', 'c'] is m1['a', 'b', 'c']
