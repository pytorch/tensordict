import torch
from tensor_map_cpp import TensorMap
import pytest
from _utils_internal import get_available_devices


@pytest.mark.parametrize("device", get_available_devices())
def test_tensordict_set(device):
    torch.manual_seed(1)
    td = TensorMap()
    val = torch.randn(4, 5)
    td.set("key1", val)
    assert (td.get("key1") == val).all()

    td.set(["key2", "key3"], torch.ones(4, 5))
    assert (td.get("key2").get("key3") == 1).all()


def test_tensordict_ref():
    m = TensorMap()
    x = torch.randn(3)
    m.set('a', x)
    assert m.get('a') is x
    n = TensorMap()
    m.set('b', n)
    y = torch.randn(3)
    n.set('c', y)
    assert m.get('b') is n
    assert m.get('b').get('c') is m.get(['b', 'c'])
    assert m.get(['b', 'c']) is y
