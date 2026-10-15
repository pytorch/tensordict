# TensorDictBase.repeat_interleave

TensorDictBase.repeat_interleave(*repeats: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | [int](https://docs.python.org/3/builtins/functions.html#int)*, *dim: [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, ***, *output_size: [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *inplace: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*) → Self

Repeat elements of a TensorDict.

Warning

This is different from [`repeat()`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.repeat.html#torch.Tensor.repeat) but similar to [`numpy.repeat()`](https://numpy.org/doc/stable/reference/generated/numpy.repeat.html#numpy.repeat).

Parameters:

- **repeats** ([*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*or*[*int*](https://docs.python.org/3/builtins/functions.html#int)) - The number of repetitions for each element. repeats is broadcast to fit
the shape of the given axis.
- **dim** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - The dimension along which to repeat values. By default, use the flattened input
array, and return a flat output array. A `dim` past the batch dims warns, and is used on the
dims of the leaves. Starting with TensorDict 0.17, the default will change to raising an
`IndexError`, as for a tensor.

Keyword Arguments:

- **output_size** ([*int*](tensordict.TensorDictBase.int.html#tensordict.TensorDictBase.int)*,**optional*) - Total output size for the given axis (e.g. sum of repeats). If given, it
will avoid stream synchronization needed to calculate output shape of the tensordict.
- **inplace** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)*,**optional*) - If `True`, this tensordict's identity
and key set are preserved; each leaf storage is replaced by
its repeated counterpart one leaf at a time. Not supported on
[`LazyStackedTensorDict`](tensordict.LazyStackedTensorDict.html#tensordict.LazyStackedTensorDict) (call
`to_tensordict()` first). Defaults to `False`.

Returns:

Repeated TensorDict which has the same shape as input, except along the given axis.

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
>>> print(td.repeat_interleave(2, dim=0))
TensorDict(
 fields={
 a: Tensor(shape=torch.Size([6, 4, 5]), device=cpu, dtype=torch.float32, is_shared=False),
 b: TensorDict(
 fields={
 a string: NonTensorData(data=a string!, batch_size=torch.Size([6, 4, 10]), device=None),
 c: Tensor(shape=torch.Size([6, 4, 10, 1]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([6, 4, 10]),
 device=None,
 is_shared=False)},
 batch_size=torch.Size([6, 4]),
 device=None,
 is_shared=False)
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).