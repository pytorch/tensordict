# stack

tensordict.stack(*input: Sequence['T']*, *dim: [int](https://docs.python.org/3/builtins/functions.html#int) = 0*, ***, *out=None*) → T

Stacks tensordicts into one dense tensordict along a new dimension `dim`.

Same as `torch.stack(input, dim)`. Every input must have the same keys,
entry shapes and batch size. If they do not, this function raises a
`RuntimeError`. To fall back to a lazy stack when the inputs differ, use
[`maybe_dense_stack()`](tensordict.maybe_dense_stack.html#tensordict.maybe_dense_stack). To always get a lazy stack, use
[`lazy_stack()`](tensordict.lazy_stack.html#tensordict.lazy_stack). [Stacking and variable-size data](../../concepts/stacking.html#stacking) compares these functions.

Parameters:

- **input** (*sequence**of*[*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*or**tensorclass instances*) - the
tensordicts to stack.
- **dim** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the new dimension. Defaults to `0`. A `dim` outside
`[-batch_dims - 1, batch_dims]` warns, and is used on the dimensions
of the leaves. Starting with TensorDict 0.17, the default will
change to raising an `IndexError`, as for a tensor.

Keyword Arguments:

**out** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*,**optional*) - a tensordict to write the result
into. Its batch size must be the stacked batch size. Defaults to
`None`.

Returns:

A [`TensorDict`](tensordict.TensorDict.html#tensordict.TensorDict) whose batch size is
`batch_size[:dim] + [len(input)] + batch_size[dim:]`. Tensorclass
inputs give an instance of their class. If every input is a
[`LazyStackedTensorDict`](tensordict.LazyStackedTensorDict.html#tensordict.LazyStackedTensorDict), the result is a
[`LazyStackedTensorDict`](tensordict.LazyStackedTensorDict.html#tensordict.LazyStackedTensorDict).

Examples

```
>>> import torch
>>> import tensordict
>>> from tensordict import TensorDict
>>> tds = [TensorDict(a=torch.zeros(3), batch_size=[3]) for _ in range(2)]
>>> td = tensordict.stack(tds, dim=1)
>>> td.batch_size
torch.Size([3, 2])
>>> type(td).__name__
'TensorDict'
```