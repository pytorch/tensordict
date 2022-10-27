import pytest
import torch
from tensordict import TensorDict

try:
    from functorch.dim import dims

    _has_torchdim = True
except ImportError:
    _has_torchdim = False


@pytest.mark.skipif(not _has_torchdim, reason="functorch.dim not found")
def test_batch_size():
    td = TensorDict({}, [3, 4, 5])
    d = dims(1)
    td_dim = td[:, d]
    assert td_dim.batch_size == torch.Size([3, 5])


@pytest.mark.skipif(not _has_torchdim, reason="functorch.dim not found")
def test_items():
    td = TensorDict({"a": torch.rand(3, 4, 5, 6)}, [3, 4, 5])
    d1, d2 = dims(2)
    td_dim = td[d1, :, d2]

    assert td_dim.batch_size == torch.Size([4])

    assert td_dim["a"].shape == torch.Size([4, 6])
    assert td_dim["a"].ndim == 2
    assert td_dim["a"].dims == (d1, d2)
