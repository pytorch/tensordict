# TensorDictBase.repeat_interleave

TensorDictBase.repeat_interleave(*repeats: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | [int](https://docs.python.org/3/builtins/functions.html#int)*, *dim: [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, ***, *output_size: [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *inplace: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*) → Self

Repeats each element of the batch along `dim`, as [`torch.Tensor.repeat_interleave()`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.repeat_interleave.html#torch.Tensor.repeat_interleave) does.

Each element is repeated next to itself, as with [`numpy.repeat()`](https://numpy.org/doc/stable/reference/generated/numpy.repeat.html#numpy.repeat).
To repeat the whole batch instead, as with [`numpy.tile()`](https://numpy.org/doc/stable/reference/generated/numpy.tile.html#numpy.tile), use
[`repeat()`](tensordict.TensorDictBase.repeat.html#tensordict.TensorDictBase.repeat). The data is copied. With a `dim`, the names are
kept.

Parameters:

- **repeats** ([*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*or*[*int*](https://docs.python.org/3/builtins/functions.html#int)) - the number of repetitions of each
element, broadcast to the size of `dim`.
- **dim** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the batch dimension to repeat along. Defaults
to `None`, which flattens the batch dimensions first, so the
result has one batch dimension. A `dim` past the batch
dimensions warns, and is used on the dimensions of the leaves.
Starting with TensorDict 0.17, the default will change to
raising an `IndexError`, as for a tensor.

Keyword Arguments:

- **output_size** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the size of the result along `dim`
(the sum of `repeats`). Passing it avoids the device
synchronization that computes it. Defaults to `None`, which
computes it.
- **inplace** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, the tensordict keeps its
identity and keys, its leaves are replaced one by one, and the
call returns `self`. It needs a `dim` and at least one
batch dimension. A lazy stack, an indexed view (such as a
slice of a [`PersistentTensorDict`](tensordict.PersistentTensorDict.html#tensordict.PersistentTensorDict)) or a
legacy lazy view (`set_lazy_legacy(True)`), or a tensordict
that holds one, raises a `NotImplementedError`: call
[`to_tensordict()`](tensordict.TensorDictBase.to_tensordict.html#tensordict.TensorDictBase.to_tensordict) first. See [In-place operations](../../concepts/operations.html#inplace-conventions).
Defaults to `False`.

Returns:

A tensordict with the same batch size, except along `dim`, or
`self` with `inplace=True`.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict(a=torch.tensor([[0, 1], [2, 3]]), batch_size=[2, 2])
>>> td.repeat_interleave(2, dim=0)["a"]
tensor([[0, 1],
 [0, 1],
 [2, 3],
 [2, 3]])
>>> td.repeat_interleave(torch.tensor([1, 2]), dim=1)["a"]
tensor([[0, 1, 1],
 [2, 3, 3]])
>>> td.repeat_interleave(2).batch_size
torch.Size([8])
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).