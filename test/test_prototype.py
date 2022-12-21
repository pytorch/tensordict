import os
import torch
import pytest

from _utils_internal import get_available_devices
from tensordict import TensorMap


# Region: Testing simple get, set feature of TensorMap

@pytest.mark.parametrize("device", get_available_devices())
def test_tensormap_simple_set_tensor(device):
    m = TensorMap([])
    x = torch.ones(3)
    m['a'] = x
    assert m['a'] is x

    m['a'] = torch.zeros(3)
    assert (m['a'] == 0).all()

    m['b'] = m['a']
    assert m['b'] is m['a']


@pytest.mark.skip(reason="TensorMap ref not working for now") # parametrize("device", get_available_devices())
def test_tensormap_simple_set_map(device):
    m1 = TensorMap([])
    m2 = TensorMap([])

    m1['a'] = m2
    assert m1['a'] is m2  # Failing - Need to fix!


# Region: Testing Nested{Set,Get} feature of TensorMap
# In this part we test multiple combinations of set and get for tensors with on nested maps
# TODO: add test for nested maps ref check when issue is fixed.

@pytest.mark.parametrize("device", get_available_devices())
def test_tensormap_nested_set(device):
    m1 = TensorMap([])
    m2 = TensorMap([])
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
    m1 = TensorMap([])

    m1['a'] = torch.ones(3)
    assert (m1['a'] == 1).all()
    m1['a', 'b', 'c'] = torch.ones(3)
    assert type(m1['a']) is TensorMap
    assert type(m1['a', 'b']) is TensorMap
    assert m1['a', 'b', 'c'] is m1['a', 'b']['c']
    assert (m1['a', 'b', 'c'] == 1).all()

    m2 = TensorMap([])
    m2['x'] = m1['a', 'b']
    assert m2['x', 'c'] is m1['a', 'b', 'c']


# Region: Testing batchsize constraint of TensorMap
# In this part we test the constraint first on tensors then on nested dicts

@pytest.mark.parametrize("device", get_available_devices())
def test_tensormap_batch_constraint(device):
    m = TensorMap([3])
    m['a'] = torch.ones(3, 4)
    with pytest.raises(Exception):
        m['b'] = torch.ones(4, 4)

    m = TensorMap([4, 5, 6])
    m['a'] = torch.ones(4, 5, 6)
    assert m['a'].size() == torch.Size([4, 5, 6, 1])
    m['a'] = torch.ones(4, 5, 6, 3)
    with pytest.raises(Exception):
        m['b'] = torch.ones(2, 5, 6, 3)
        m['b'] = torch.ones(4)
        m['b'] = torch.ones(4, 5)


# Region: Testing keys() feature of TensorMap
# In this part we test the keys method with the all cominations of include_nested and leaves_only

@pytest.mark.parametrize("device", get_available_devices())
def test_tensormap_get_keys(device):
    m = TensorMap([])
    m['a'] = torch.ones(3)
    m['b'] = torch.zeros(3)

    expected_keys = {'a', 'b'}
    assert_equal_sets(expected_keys, m.keys())

    m['c', 'd'] = torch.ones(3)
    m['a', 'x', 'y'] = torch.zeros(3)
    expected_keys = {'a', ('a', 'x'), ('a', 'x', 'y'), 'b', 'c', ('c', 'd')}
    assert_equal_sets(expected_keys, m.keys(True))
    expected_keys = {'a', 'b', 'c'}
    assert_equal_sets(expected_keys, m.keys())

    m['a', 'z'] = torch.rand(3)
    expected_keys = {'a', ('a', 'z'), ('a', 'x'), ('a', 'x', 'y'), 'b', 'c', ('c', 'd')}
    assert_equal_sets(expected_keys, m.keys(True))
    expected_keys = {'a', 'b', 'c'}
    assert_equal_sets(expected_keys, m.keys())

    m['c', 'd'] = m['a']
    expected_keys = {'a', ('a', 'z'), ('a', 'x'), ('a', 'x', 'y'), 'b', 'c', ('c', 'd'), ('c', 'd', 'z'), ('c', 'd', 'x'), ('c', 'd', 'x', 'y')}
    assert_equal_sets(expected_keys, m.keys(True))
    expected_keys = {'a', 'b', 'c'}
    assert_equal_sets(expected_keys, m.keys())

    m['c'] = torch.ones(3)
    expected_keys = {'a', ('a', 'z'), ('a', 'x'), ('a', 'x', 'y'), 'b', 'c'}
    assert_equal_sets(expected_keys, m.keys(True))
    expected_keys = {'a', 'b', 'c'}
    assert_equal_sets(expected_keys, m.keys())


@pytest.mark.parametrize("device", get_available_devices())
def test_tensormap_get_keys_leaves_only(device):
    m = TensorMap([])
    m['a'] = torch.ones(3)
    m['b'] = torch.zeros(3)

    expected_keys = {'a', 'b'}
    assert_equal_sets(expected_keys, m.keys(False, True))

    m['c', 'd'] = torch.ones(3)
    m['a', 'x', 'y'] = torch.zeros(3)
    expected_keys = {('a', 'x', 'y'), 'b', ('c', 'd')}
    assert_equal_sets(expected_keys, m.keys(True, True))
    expected_keys = {'b'}
    assert_equal_sets(expected_keys, m.keys(False, True))

    m['a', 'z'] = torch.rand(3)
    expected_keys = {('a', 'z'), ('a', 'x', 'y'), 'b', ('c', 'd')}
    assert_equal_sets(expected_keys, m.keys(True, True))
    expected_keys = {'b'}
    assert_equal_sets(expected_keys, m.keys(False, True))

    m['c', 'd'] = m['a']
    expected_keys = {('a', 'z'), ('a', 'x', 'y'), 'b', ('c', 'd', 'z'), ('c', 'd', 'x', 'y')}
    assert_equal_sets(expected_keys, m.keys(True, True))
    expected_keys = {'b'}
    assert_equal_sets(expected_keys, m.keys(False, True))

    m['c'] = torch.ones(3)
    expected_keys = {('a', 'z'), ('a', 'x', 'y'), 'b', 'c'}
    assert_equal_sets(expected_keys, m.keys(True, True))
    expected_keys = {'b', 'c'}
    assert_equal_sets(expected_keys, m.keys(False, True))


@pytest.mark.parametrize("device", get_available_devices())
def test_tensormap_in_keys(device):
    m = TensorMap([])
    m['a', 'b', 'c'] = torch.zeros(3)
    m['d'] = torch.zeros(3)
    m['a', 'x'] = torch.zeros(3)

    keys = m.keys(True, False)
    assert 'd' in keys
    assert ('a', 'b', 'c') in keys
    assert ('a', 'b') in keys
    assert 'a' in keys
    assert ('a', 'x') in keys


def assert_equal_sets(expected: set, actual: set):
    assert len(actual) == len(expected)
    assert len(expected.difference(actual)) == 0
    assert len(actual.difference(expected)) == 0
