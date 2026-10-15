# TensorDictBase.flatten

TensorDictBase.flatten(*start_dim: [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *end_dim: [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, ***, *inplace: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*)

Flattens all the tensors of a tensordict.

Parameters:

- **start_dim** ([*int*](https://docs.python.org/3/builtins/functions.html#int)) - the first dim to flatten
- **end_dim** ([*int*](https://docs.python.org/3/builtins/functions.html#int)) - the last dim to flatten

A `start_dim` or `end_dim` outside the batch dims warns, and is
used on the dims of the leaves. Starting with TensorDict 0.17, the
default will change to raising an `IndexError`, as for a tensor.

Keyword Arguments:

**inplace** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)*,**optional*) - If `True`, this tensordict's identity
and key set are preserved; each leaf is flattened one at a
time. `torch.flatten` may return a view when the flattened
range is contiguous in memory; in that case leaves share
storage with the originals and the memory benefit does not
materialize. Not supported on
[`LazyStackedTensorDict`](tensordict.LazyStackedTensorDict.html#tensordict.LazyStackedTensorDict). Defaults to
`False`.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict({
... "a": torch.arange(60).view(3, 4, 5),
... "b": torch.arange(12).view(3, 4)}, batch_size=[3, 4])
>>> td_flat = td.flatten(0, 1)
>>> td_flat.batch_size
torch.Size([12])
>>> td_flat["a"]
tensor([[ 0, 1, 2, 3, 4],
 [ 5, 6, 7, 8, 9],
 [10, 11, 12, 13, 14],
 [15, 16, 17, 18, 19],
 [20, 21, 22, 23, 24],
 [25, 26, 27, 28, 29],
 [30, 31, 32, 33, 34],
 [35, 36, 37, 38, 39],
 [40, 41, 42, 43, 44],
 [45, 46, 47, 48, 49],
 [50, 51, 52, 53, 54],
 [55, 56, 57, 58, 59]])
>>> td_flat["b"]
tensor([ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).