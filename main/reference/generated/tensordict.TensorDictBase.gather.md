# TensorDictBase.gather

TensorDictBase.gather(*dim: [int](https://docs.python.org/3/builtins/functions.html#int)*, *index: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*, *out: T | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, ***, *inplace: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*) → Self

Gathers values along an axis specified by dim.

Parameters:

- **dim** ([*int*](https://docs.python.org/3/builtins/functions.html#int)) - the dimension along which collect the elements
- **index** ([*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) - a long tensor which number of dimension matches
the one of the tensordict with only one dimension differing between
the two (the gathering dimension). Its elements refer to the
index to be gathered along the required dimension.
- **out** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*,**optional*) - a destination tensordict. It must
have the same shape as the index.

Keyword Arguments:

**inplace** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)*,**optional*) - If `True`, this tensordict's identity
and key set are preserved; each leaf storage is replaced by
its gathered counterpart one leaf at a time, keeping peak
memory close to `1x`. Requires `index.ndim ==
self.batch_dims` (the result must have the same number of
batch dims as the input). Mutually exclusive with `out`.
Not supported on [`LazyStackedTensorDict`](tensordict.LazyStackedTensorDict.html#tensordict.LazyStackedTensorDict).
Defaults to `False`.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict(
... {"a": torch.randn(3, 4, 5),
... "b": TensorDict({"c": torch.zeros(3, 4, 5)}, [3, 4, 5])},
... [3, 4])
>>> index = torch.randint(4, (3, 2))
>>> td_gather = td.gather(dim=1, index=index)
>>> print(td_gather)
TensorDict(
 fields={
 a: Tensor(shape=torch.Size([3, 2, 5]), device=cpu, dtype=torch.float32, is_shared=False),
 b: TensorDict(
 fields={
 c: Tensor(shape=torch.Size([3, 2, 5]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([3, 2, 5]),
 device=None,
 is_shared=False)},
 batch_size=torch.Size([3, 2]),
 device=None,
 is_shared=False)
```

Gather keeps the dimension names.

Examples

```
>>> td.names = ["a", "b"]
>>> td_gather = td.gather(dim=1, index=index)
>>> td_gather.names
['a', 'b']
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).

## Tutorials that use `gather`

![](../../_images/sphx_glr_tensordict_shapes_thumb.png)

[Changing the shape of a TensorDict](../../tutorials/tensordict_shapes.html)

Changing the shape of a TensorDict