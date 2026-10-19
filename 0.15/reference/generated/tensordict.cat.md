# cat

tensordict.cat(*input: Sequence['T']*, *dim: [int](https://docs.python.org/3/builtins/functions.html#int) = 0*, ***, *out=None*) → T

Concatenates tensordicts along an existing batch dimension `dim`.

Same as `torch.cat(input, dim)`. Every input must have the same keys,
and the same batch size and entry shapes except along `dim`. If the
keys differ, this function raises a `KeyError`. If the shapes do not
match, it raises a `RuntimeError`. To join tensordicts along a new
dimension, use [`stack()`](tensordict.stack.html#tensordict.stack) or
[`lazy_stack()`](tensordict.lazy_stack.html#tensordict.lazy_stack) (see [Stacking and variable-size data](../../concepts/stacking.html#stacking)).

Parameters:

- **input** (*sequence**of*[*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*or**tensorclass instances*) - the
tensordicts to concatenate.
- **dim** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the batch dimension to concatenate along.
Defaults to `0`. A `dim` below `-batch_dims` warns, and is
used on the dims of the leaves. Starting with TensorDict 0.17,
the default will change to raising an `IndexError`, as for a
tensor.

Keyword Arguments:

**out** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*,**optional*) - a tensordict to write the result
into. Defaults to `None`.

Returns:

A [`TensorDict`](tensordict.TensorDict.html#tensordict.TensorDict). Tensorclass inputs give an
instance of their class. If every input is a
[`LazyStackedTensorDict`](tensordict.LazyStackedTensorDict.html#tensordict.LazyStackedTensorDict), the result is a
[`LazyStackedTensorDict`](tensordict.LazyStackedTensorDict.html#tensordict.LazyStackedTensorDict).

Examples

```
>>> import torch
>>> import tensordict
>>> from tensordict import TensorDict
>>> td = tensordict.cat(
... [TensorDict(a=torch.zeros(3), batch_size=[3]),
... TensorDict(a=torch.ones(4), batch_size=[4])]
... )
>>> td.batch_size
torch.Size([7])
```