# TensorDictBase.flatten

TensorDictBase.flatten(*start_dim: [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *end_dim: [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, ***, *inplace: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*)

Returns a tensordict whose batch dimensions `start_dim` to `end_dim` are flattened into one.

The leaves keep their feature dimensions. As for [`torch.flatten()`](https://docs.pytorch.org/docs/stable/generated/torch.flatten.html#torch.flatten),
a leaf of the result is a view of the original leaf when the memory
layout allows it, and a copy otherwise. The flattened dimension has no
name. A tensordict without batch dimensions gets one batch dimension
of size 1.

A `start_dim` or `end_dim` outside the batch dimensions warns, and
is used on the dimensions of the leaves. Starting with TensorDict 0.17,
the default will change to raising an `IndexError`, as for a tensor.

It can be used as a context manager, as [`unsqueeze()`](tensordict.TensorDictBase.unsqueeze.html#tensordict.TensorDictBase.unsqueeze) can: when
the `with` block exits, the entries of the block are written back to
the tensordict with the inverse operation.

Parameters:

- **start_dim** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the first batch dimension to flatten.
Defaults to `None`, which is `0`.
- **end_dim** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the last batch dimension to flatten.
Defaults to `None`, which is `-1`.

Keyword Arguments:

**inplace** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, the tensordict keeps its
identity and keys, its leaves are replaced one by one, and the
call returns `self`. A leaf keeps its storage when the
flatten can return a view of it. A lazy stack, or a tensordict
that holds one, raises a `NotImplementedError`: call
[`to_tensordict()`](tensordict.TensorDictBase.to_tensordict.html#tensordict.TensorDictBase.to_tensordict) first. See [In-place operations](../../concepts/operations.html#inplace-conventions).
Defaults to `False`.

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
>>> td_flat["a"].shape
torch.Size([12, 5])
>>> td_flat["b"]
tensor([ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).