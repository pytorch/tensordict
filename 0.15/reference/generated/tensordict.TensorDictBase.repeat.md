# TensorDictBase.repeat

TensorDictBase.repeat(**repeats: [int](https://docs.python.org/3/builtins/functions.html#int)*, *inplace: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*) → Self

Repeats this tensor along the specified dimensions.

Unlike [`expand()`](tensordict.TensorDictBase.expand.html#tensordict.TensorDictBase.expand), this function copies the tensor's data.

Warning

`repeat()` behaves differently from [`repeat()`](https://numpy.org/doc/stable/reference/generated/numpy.repeat.html#numpy.repeat), but is more similar to
[`numpy.tile()`](https://numpy.org/doc/stable/reference/generated/numpy.tile.html#numpy.tile). For the operator similar to [`numpy.repeat()`](https://numpy.org/doc/stable/reference/generated/numpy.repeat.html#numpy.repeat), see [`repeat_interleave()`](tensordict.TensorDictBase.repeat_interleave.html#tensordict.TensorDictBase.repeat_interleave).

Parameters:

**repeats** ([*Size*](https://docs.pytorch.org/docs/stable/size.html#torch.Size)*,*[*int*](https://docs.python.org/3/builtins/functions.html#int)*...**,*[*tuple*](https://docs.python.org/3/builtins/stdtypes.html#tuple)*of*[*int*](https://docs.python.org/3/builtins/functions.html#int)*or*[*list*](https://docs.python.org/3/builtins/stdtypes.html#list)*of*[*int*](https://docs.python.org/3/builtins/functions.html#int)) - The number of times to repeat this tensor along
each dimension.

Keyword Arguments:

**inplace** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)*,**optional*) - If `True`, this tensordict's identity
and key set are preserved; each leaf storage is replaced by
its repeated counterpart one leaf at a time, keeping peak
memory close to `1x`. `LazyStackedTensorDict` is not
supported in this mode (the stack dim's repeat factor would
rebuild the stack's constituents list) -- call
`to_tensordict()` first. Defaults to `False`.

Examples

```
>>> import torch
>>>
>>> from tensordict import TensorDict
>>>
>>> td = TensorDict(
... {
... "a": torch.randn(3, 4, 5),
... "b": TensorDict({
... "c": torch.randn(3, 4, 10, 1),
... "a string": "a string!",
... }, batch_size=[3, 4, 10])
... }, batch_size=[3, 4],
... )
>>> print(td.repeat(1, 2))
TensorDict(
 fields={
 a: Tensor(shape=torch.Size([3, 8, 5]), device=cpu, dtype=torch.float32, is_shared=False),
 b: TensorDict(
 fields={
 a string: NonTensorData(data=a string!, batch_size=torch.Size([3, 8, 10]), device=None),
 c: Tensor(shape=torch.Size([3, 8, 10, 1]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([3, 8, 10]),
 device=None,
 is_shared=False)},
 batch_size=torch.Size([3, 8]),
 device=None,
 is_shared=False)
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).