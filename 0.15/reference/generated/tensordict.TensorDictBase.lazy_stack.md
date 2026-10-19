# TensorDictBase.lazy_stack

*classmethod*TensorDictBase.lazy_stack(*input*, *dim: [int](https://docs.python.org/3/builtins/functions.html#int) = 0*, ***, *out=None*, ***kwargs*)

Stacks tensordicts into a lazy stack.

A lazy stack keeps the tensordicts as they are and stacks their values
only when you read them. The tensordicts can have different keys and
different feature shapes. See [Stacking and variable-size data](../../concepts/stacking.html#stacking).

Parameters:

- **input** (*sequence**of*[*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)) - the tensordicts to stack.
- **dim** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the new batch dimension. Defaults to `0`.

Keyword Arguments:

- **out** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*,**optional*) - a lazy stack to write the result
into. Defaults to `None`.
- ****kwargs** - passed to
[`LazyStackedTensorDict.lazy_stack`](tensordict.LazyStackedTensorDict.html#tensordict.LazyStackedTensorDict.lazy_stack), for example
`stack_dim_name` or `strict_shape`.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> a = TensorDict(x=torch.zeros(3), batch_size=[3])
>>> b = TensorDict(y=torch.ones(3), batch_size=[3])
>>> stack = TensorDict.lazy_stack([a, b])
>>> type(stack).__name__, stack.batch_size
('LazyStackedTensorDict', torch.Size([2, 3]))
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).