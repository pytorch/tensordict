# maybe_dense_stack

tensordict.maybe_dense_stack(*input: Sequence['T']*, *dim: [int](https://docs.python.org/3/builtins/functions.html#int) = 0*, ***, *out=None*, ***kwargs*) → T

Stacks tensordicts densely if possible, or into a lazy stack otherwise.

Use it when the inputs usually match, but can differ. If every input has
the same keys, entry shapes and batch size, the result is the dense
stack that [`stack()`](tensordict.stack.html#tensordict.stack) returns. Otherwise, the result is the
lazy stack that [`lazy_stack()`](tensordict.lazy_stack.html#tensordict.lazy_stack) returns. [Stacking and variable-size data](../../concepts/stacking.html#stacking)
compares these functions. Same as
[`tensordict.LazyStackedTensorDict.maybe_dense_stack()`](tensordict.LazyStackedTensorDict.html#tensordict.LazyStackedTensorDict.maybe_dense_stack).

Parameters:

- **input** (*sequence**of*[*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*or**tensorclass instances*) - the
tensordicts to stack.
- **dim** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the new dimension. Defaults to `0`. A `dim` outside
`[-batch_dims - 1, batch_dims]` warns, and is used on the dimensions
of the leaves. Starting with TensorDict 0.17, the default will
change to raising an `IndexError`, as for a tensor.

Keyword Arguments:

- **out** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*,**optional*) - a tensordict to write the result
into. Defaults to `None`.
- **strict** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - used only with `out`. If `True`, `out`
must have the same keys as the inputs. Defaults to `False`.

Returns:

A [`TensorDict`](tensordict.TensorDict.html#tensordict.TensorDict) (or the tensorclass of the inputs)
if the inputs can be stacked densely, a
[`LazyStackedTensorDict`](tensordict.LazyStackedTensorDict.html#tensordict.LazyStackedTensorDict) otherwise.

Examples

```
>>> import torch
>>> import tensordict
>>> from tensordict import TensorDict
>>> same = [TensorDict(a=torch.zeros(3)), TensorDict(a=torch.ones(3))]
>>> type(tensordict.maybe_dense_stack(same)).__name__
'TensorDict'
>>> different = [TensorDict(a=torch.zeros(3)), TensorDict(a=torch.ones(4))]
>>> type(tensordict.maybe_dense_stack(different)).__name__
'LazyStackedTensorDict'
```