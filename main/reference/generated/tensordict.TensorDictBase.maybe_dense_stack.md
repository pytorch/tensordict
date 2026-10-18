# TensorDictBase.maybe_dense_stack

*classmethod*TensorDictBase.maybe_dense_stack(*input*, *dim: [int](https://docs.python.org/3/builtins/functions.html#int) = 0*, ***, *out=None*, ***kwargs*)

Stacks tensordicts densely if possible, or into a lazy stack otherwise.

The result is a [`TensorDict`](tensordict.TensorDict.html#tensordict.TensorDict) if the tensordicts
have the same keys and shapes, and a
[`LazyStackedTensorDict`](tensordict.LazyStackedTensorDict.html#tensordict.LazyStackedTensorDict) otherwise. See
[Stacking and variable-size data](../../concepts/stacking.html#stacking).

Parameters:

- **input** (*sequence**of*[*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)) - the tensordicts to stack.
- **dim** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the new batch dimension. Defaults to `0`.

Keyword Arguments:

- **out** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*,**optional*) - a tensordict to write the result
into. Defaults to `None`.
- ****kwargs** - passed to
[`LazyStackedTensorDict.maybe_dense_stack`](tensordict.LazyStackedTensorDict.html#tensordict.LazyStackedTensorDict.maybe_dense_stack), for
example `strict`.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> a = TensorDict(x=torch.zeros(3), batch_size=[3])
>>> b = TensorDict(x=torch.ones(3), batch_size=[3])
>>> c = TensorDict(y=torch.ones(3), batch_size=[3])
>>> type(TensorDict.maybe_dense_stack([a, b])).__name__
'TensorDict'
>>> type(TensorDict.maybe_dense_stack([a, c])).__name__
'LazyStackedTensorDict'
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).