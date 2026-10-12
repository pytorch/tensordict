# expand_right

tensordict.utils.expand_right(*tensor: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*, *shape: [Sequence](https://docs.python.org/3/library/typing.html#typing.Sequence)[[int](https://docs.python.org/3/builtins/functions.html#int)]*) → [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

Expand a tensor on the right to match a desired shape.

Parameters:

- **tensor** - tensor to be expanded
- **shape** - target shape

Returns:

a tensor with shape matching the target shape.

Examples

```
>>> import torch
>>> from tensordict.utils import expand_right
>>> tensor = torch.zeros(3,4)
>>> shape = (3,4,5)
>>> print(expand_right(tensor, shape).shape)
torch.Size([3, 4, 5])
```