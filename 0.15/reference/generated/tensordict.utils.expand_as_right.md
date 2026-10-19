# expand_as_right

tensordict.utils.expand_as_right(*tensor: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*, *dest: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*) → [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase)

Expands a tensor on the right to the shape of `dest`.

Broadcasting adds dimensions on the left; this function adds them on the
right instead, then expands every dimension of size one. The leading
dimensions of `tensor` must match those of `dest` or be of size one.
The result is a view of `tensor`, as with [`torch.Tensor.expand()`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.expand.html#torch.Tensor.expand).

Parameters:

- **tensor** ([*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*or*[*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)) - the tensor to expand.
- **dest** ([*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*or*[*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)) - the tensor whose shape the
result takes. For a tensordict, the shape is its batch size.

Returns:

A view of `tensor` with the shape of `dest`.

Raises:

[**RuntimeError**](https://docs.python.org/3/builtins/exceptions.html#RuntimeError) - if `dest` has fewer dimensions than `tensor`, or if
 their leading dimensions do not match.

Examples

```
>>> import torch
>>> from tensordict.utils import expand_as_right
>>> x = torch.arange(3)
>>> expand_as_right(x, torch.zeros(3, 2))
tensor([[0, 0],
 [1, 1],
 [2, 2]])
```