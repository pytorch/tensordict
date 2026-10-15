# TensorDictBase.permute

TensorDictBase.permute(**args*, ***kwargs*)

Returns a view of a tensordict with the batch dimensions permuted according to dims.

Parameters:

- ***dims** ([*int*](https://docs.python.org/3/builtins/functions.html#int)) - the new ordering of the batch dims of the tensordict. Alternatively,
a single iterable of integers can be provided.
- **dims** ([*list*](https://docs.python.org/3/builtins/stdtypes.html#list)*of*[*int*](https://docs.python.org/3/builtins/functions.html#int)) - alternative way of calling permute(...).

Returns:

a new tensordict with the batch dimensions in the desired order.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> tensordict = TensorDict({"a": torch.randn(3, 4, 5)}, [3, 4])
>>> print(tensordict.permute([1, 0]))
TensorDict(
 fields={
 a: Tensor(shape=torch.Size([4, 3, 5]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([4, 3]),
 device=None,
 is_shared=False)
>>> print(tensordict.permute(1, 0))
TensorDict(
 fields={
 a: Tensor(shape=torch.Size([4, 3, 5]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([4, 3]),
 device=None,
 is_shared=False)
>>> print(tensordict.permute(dims=[1, 0]))
TensorDict(
 fields={
 a: Tensor(shape=torch.Size([4, 3, 5]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([4, 3]),
 device=None,
 is_shared=False)
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).

## Tutorials that use `permute`

![](../../_images/sphx_glr_tensordict_shapes_thumb.png)

[Changing the shape of a TensorDict](../../tutorials/tensordict_shapes.html)

Changing the shape of a TensorDict