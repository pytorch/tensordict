# isin

tensordict.utils.isin(*input: [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*, *reference: [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*, *key: [NestedKey](tensordict.NestedKey.html#tensordict.NestedKey)*, *dim: [int](https://docs.python.org/3/builtins/functions.html#int) = 0*) → Tensor

Returns which elements of `input` along `dim` are also in `reference`.

The entry `key` is cut into slices along `dim`. An element of
`input` is in `reference` if its slice is equal to one of the slices
of `reference[key]`. The two tensordicts must have the same number of
batch dimensions, and the shapes of the two entries must match except
along `dim`.

Parameters:

- **input** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)) - the tensordict to test.
- **reference** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)) - the tensordict to look in.
- **key** ([*NestedKey*](tensordict.NestedKey.html#tensordict.NestedKey)) - the key of the tensor to compare. It must hold a
tensor in both tensordicts.
- **dim** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the batch dimension along which to compare.
Defaults to `0`.

Returns:

A boolean tensor of length `input.batch_size[dim]`.

Raises:

- [**KeyError**](https://docs.python.org/3/builtins/exceptions.html#KeyError) - if `key` is missing or is not a tensor in either
 tensordict.
- [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) - if the numbers of batch dimensions differ, or if `dim`
 is not a batch dimension.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> from tensordict.utils import isin
>>> td = TensorDict(
... x=torch.tensor([[1, 2], [3, 4], [1, 2], [5, 6]]),
... y=torch.arange(4),
... batch_size=[4],
... )
>>> reference = TensorDict(x=torch.tensor([[1, 2], [5, 6]]), batch_size=[2])
>>> isin(td, reference, key="x")
tensor([ True, False, True, True])
```