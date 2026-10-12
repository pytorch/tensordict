# TensorDictBase.lazy_stack

*classmethod*TensorDictBase.lazy_stack(*input*, *dim: [int](https://docs.python.org/3/builtins/functions.html#int) = 0*, ***, *out=None*, ***kwargs*)

Creates a lazy stack of tensordicts.

A lazy stack keeps the tensordicts as they are and stacks their values
only when you read them. The tensordicts can have different keys and
different feature shapes. See [Stacking and variable-size data](../../concepts/stacking.html#stacking), and
[`tensordict.lazy_stack()`](tensordict.lazy_stack.html#tensordict.lazy_stack) for the arguments.

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