import torch

from tensordict import TensorDict
from tensordict.prototype import tensordictclass


@tensordictclass
class MyData:
    X: torch.tensor
    y: torch.tensor
    batch_size: list

    def stuff(self):
        return self.X + self.y

def test_type():

    data = MyData(
        X=torch.ones(3, 4, 5),
        y=torch.zeros(3, 4, 5, dtype=torch.bool),
        batch_size=[3, 4]
    )
    assert isinstance(data, MyData)

def test_attributes():

    X = torch.ones(3, 4, 5)
    y = torch.zeros(3, 4, 5, dtype=torch.bool)
    batch_size = [3, 4]
    tensordict = TensorDict({"X": X, "y": y,}, batch_size=[3, 4])

    data = MyData(X=X, y=y, batch_size=batch_size)

    equality_tensordict = (data.tensordict == tensordict)

    assert torch.equal(data.X, X)
    assert torch.equal(data.y, y)
    assert data.batch_size == batch_size
    assert equality_tensordict["X"].all()
    assert equality_tensordict["y"].all()
    assert equality_tensordict.batch_size == torch.Size(batch_size)

