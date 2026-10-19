# TensorDictBase.tensor_split

TensorDictBase.tensor_split(*indices_or_sections: [int](https://docs.python.org/3/builtins/functions.html#int) | [list](https://docs.python.org/3/builtins/stdtypes.html#list)[[int](https://docs.python.org/3/builtins/functions.html#int)] | [tuple](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[int](https://docs.python.org/3/builtins/functions.html#int), ...] | [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*, *dim=0*) → [tuple](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase), ...]

Splits the tensordict along a batch dimension, as [`torch.tensor_split()`](https://docs.pytorch.org/docs/stable/generated/torch.tensor_split.html#torch.tensor_split) does.

The tensordicts of the result hold views of the original tensors.

Parameters:

- **indices_or_sections** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,*[*list*](https://docs.python.org/3/builtins/stdtypes.html#list)*of*[*int*](https://docs.python.org/3/builtins/functions.html#int)*,*[*tuple*](https://docs.python.org/3/builtins/stdtypes.html#tuple)*of*[*int*](https://docs.python.org/3/builtins/functions.html#int)*or*[*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) - an integer `n` (or a 0-d long tensor) splits the tensordict
into `n` sections. When `n` does not divide
`batch_size[dim]`, the first `batch_size[dim] % n`
sections have one more element than the others. A list, a tuple
or a 1-d long tensor of indices splits it before each index:
`[2, 3]` with `dim=0` gives `td[:2]`, `td[2:3]` and
`td[3:]`. As with [`torch.tensor_split()`](https://docs.pytorch.org/docs/stable/generated/torch.tensor_split.html#torch.tensor_split), negative
indices count from the end, indices past the end are clamped,
and an index smaller than the one before it gives an empty or
an overlapping section. A tensor must be on the CPU.
- **dim** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the batch dimension to split. Defaults to
`0`.

Returns:

A tuple of tensordicts.

Raises:

- [**IndexError**](https://docs.python.org/3/builtins/exceptions.html#IndexError) - if `dim` is not a batch dimension, for example when
 the tensordict has no batch dimension.
- [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) - if the number of sections is smaller than 1, or if a
 tensor `indices_or_sections` is not a 0-d or 1-d long tensor
 on the CPU.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict(x=torch.arange(5), batch_size=[5])
>>> [t["x"] for t in td.tensor_split(2)]
[tensor([0, 1, 2]), tensor([3, 4])]
>>> [t["x"] for t in td.tensor_split([1, 3])]
[tensor([0]), tensor([1, 2]), tensor([3, 4])]
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).