# expand_right

tensordict.utils.expand_right(*tensor: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*, *shape: [Sequence](https://docs.python.org/3/library/typing.html#typing.Sequence)[[int](https://docs.python.org/3/builtins/functions.html#int)]*) → [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

Expands a tensor on the right to `shape`.

Broadcasting adds dimensions on the left; this function adds them on the
right instead, then expands every dimension of size one. The leading
dimensions of `tensor` must match `shape` or be of size one. The
result is a view of `tensor`, as with [`torch.Tensor.expand()`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.expand.html#torch.Tensor.expand).

Parameters:

- **tensor** ([*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) - the tensor to expand.
- **shape** ([*Size*](https://docs.pytorch.org/docs/stable/size.html#torch.Size)*or**sequence**of*[*int*](https://docs.python.org/3/builtins/functions.html#int)) - the shape of the result.

Returns:

A view of `tensor` with shape `shape`.

Examples

```
>>> import torch
>>> from tensordict.utils import expand_right
>>> x = torch.arange(3)
>>> expand_right(x, (3, 2))
tensor([[0, 0],
 [1, 1],
 [2, 2]])
>>> expand_right(torch.zeros(3, 4), (3, 4, 5)).shape
torch.Size([3, 4, 5])
```