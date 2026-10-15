# expand_as_right

tensordict.utils.expand_as_right(*tensor: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*, *dest: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*) → [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase)

Expand a tensor on the right to match another tensor shape.

Parameters:

- **tensor** - tensor to be expanded
- **dest** - tensor providing the target shape

Returns:

a tensor with shape matching the dest input tensor shape.

Examples

```
>>> import torch
>>> from tensordict.utils import expand_as_right
>>> tensor = torch.zeros(3,4)
>>> dest = torch.zeros(3,4,5)
>>> print(expand_as_right(tensor, dest).shape)
torch.Size([3, 4, 5])
```