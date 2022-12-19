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
