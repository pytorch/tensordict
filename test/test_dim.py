import pytest
import torch
from tensordict import TensorDict

try:
    from functorch.dim import Dim, dims
except ImportError:
    pytest.skip(reason="functorch.dim not found")


def test_batch_size():
    td = TensorDict({}, [3, 4, 5])
    d = dims(1)
    td_dim = td[:, d]

    assert td_dim.batch_size == torch.Size([3, 5])
    assert td_dim.dim() == 2


def test_items():
    td = TensorDict({"a": torch.rand(3, 4, 5, 6)}, [3, 4, 5])
    d1, d2 = dims(2)
    td_dim = td[d1, :, d2]

    assert td_dim.batch_size == torch.Size([4])

    assert td_dim["a"].shape == torch.Size([4, 6])
    assert td_dim["a"].ndim == 2
    assert td_dim["a"].dims == (d1, d2)


def test_levels():
    # test repeated slicing results in same levels behaviour as functorch.dim.Tensor
    td = TensorDict(
        {"a": torch.rand(3, 4, 5, 6), "b": torch.rand(3, 4, 5, 6, 7)}, [3, 4, 5]
    )

    d1, d2, d3 = dims(3)

    td_dim = td[:, d1]
    td_dim2 = td_dim[:, d2]
    td_dim3 = td_dim2[d3]

    assert td_dim3._levels == td_dim3["a"]._levels[:3] == td_dim3["b"]._levels[:3]


def test_view():
    td = TensorDict({"a": torch.rand(3, 4, 5, 6)}, [3, 4, 5])

    d = dims(1)
    td_dim = td[d]

    # FIXME: equality comparison of functorch.dim.Tensor with torch.Tensor causes
    # an exception, so compare to underlying tensor for now.
    assert (td_dim.view(20)["a"]._tensor == td["a"].view(3, 20, 6)).all()


def test_reshape():
    td = TensorDict({"a": torch.rand(2, 3, 4)}, [2, 3])

    d = dims(1)
    td_dim = td[d]

    # FIXME: equality comparison of functorch.dim.Tensor with torch.Tensor causes
    # an exception, so compare to underlying tensor for now.
    assert (td_dim.reshape(3, 1)["a"]._tensor == td["a"].reshape(2, 3, 1, -1)).all()
