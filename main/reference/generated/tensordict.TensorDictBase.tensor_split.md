# TensorDictBase.tensor_split

TensorDictBase.tensor_split(*indices_or_sections: [int](https://docs.python.org/3/builtins/functions.html#int) | [list](https://docs.python.org/3/builtins/stdtypes.html#list)[[int](https://docs.python.org/3/builtins/functions.html#int)] | [tuple](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[int](https://docs.python.org/3/builtins/functions.html#int), ...] | [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*, *dim=0*) → [tuple](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase), ...]

Splits a TensorDict into multiple sub-tensordicts, all of which are views of input, along dimension dim according to the indices or number of sections specified by indices_or_sections.

Parameters:

- **indices_or_sections** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*or**List**(*[*int*](https://docs.python.org/3/builtins/functions.html#int)*) or*[*tuple*](https://docs.python.org/3/builtins/stdtypes.html#tuple)*(*[*int*](https://docs.python.org/3/builtins/functions.html#int)*) or**1D tensor**of**ints*) - If indices_or_sections is an integer
n or a zero dimensional long tensordict with value n, input is split into n sections along dimension dim.
If input is divisible by n along dimension dim, each section will be of equal size, input.size(dim) / n.
If input is not divisible by n, the sizes of the first int(input.size(dim) % n) sections will have
size int(input.size(dim) / n) + 1, and the rest will have size int(input.size(dim) / n).
If indices_or_sections is a list or tuple of ints, or a one-dimensional long tensor, then input is split
along dimension dim at each of the indices in the list, tuple or tensor.
For instance, indices_or_sections=[2, 3] and dim=0 would result in the tensors input[:2], input[2:3], and input[3:].
As with [`torch.tensor_split()`](https://docs.pytorch.org/docs/stable/generated/torch.tensor_split.html#torch.tensor_split), negative indices count from the end, indices past the end are clamped,
and an index smaller than the one before it gives an empty or an overlapping section.
If indices_or_sections is a tensor, it must be a zero-dimensional or one-dimensional long tensor on the CPU.
- **dim** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - dimension along which to split the tensor. Default: 0

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict({
... 'x': torch.arange(24).reshape(3, 4, 2),
... }, batch_size=[3, 4])
>>> td0, td1 = td.tensor_split(dim=-1, indices_or_sections=2)
>>> td0['x']
tensor([[[ 0, 1],
 [ 2, 3]],
 [[ 8, 9],
 [10, 11]],
 [[16, 17],
 [18, 19]]])
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).