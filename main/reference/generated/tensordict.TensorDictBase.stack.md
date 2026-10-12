# TensorDictBase.stack

*classmethod*TensorDictBase.stack(*input*, *dim: [int](https://docs.python.org/3/builtins/functions.html#int) = 0*, ***, *out=None*)

Stacks tensordicts into a single tensordict along the given dimension.

This call is equivalent to calling [`torch.stack()`](https://docs.pytorch.org/docs/stable/generated/torch.stack.html#torch.stack) but is compatible with torch.compile.
The tensordicts must have the same batch size and the same keys.
Otherwise, the call raises `RuntimeError`. To stack tensordicts with
different keys or shapes, use [`lazy_stack()`](tensordict.TensorDictBase.lazy_stack.html#tensordict.TensorDictBase.lazy_stack). See
[Stacking and variable-size data](../../concepts/stacking.html#stacking).

Parameters:

- **input** (*sequence**of*[*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)) - the tensordicts to stack.
- **dim** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the new batch dimension. Defaults to `0`.

Keyword Arguments:

**out** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*,**optional*) - a tensordict to write the result
into.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> a = TensorDict(x=torch.zeros(3), batch_size=[3])
>>> b = TensorDict(x=torch.ones(3), batch_size=[3])
>>> TensorDict.stack([a, b]).batch_size
torch.Size([2, 3])
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).