# TensorDictBase.stack_tensors

TensorDictBase.stack_tensors(**keys: [NestedKey](tensordict.NestedKey.html#tensordict.NestedKey)*, *out_key: [NestedKey](tensordict.NestedKey.html#tensordict.NestedKey)*, *dim: [int](https://docs.python.org/3/builtins/functions.html#int) = 0*, *keep_entries: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*) → Self

Stacks entries into a new entry and possibly remove the original values.

Parameters:

**keys** (*sequence**of*[*NestedKey*](tensordict.NestedKey.html#tensordict.NestedKey)) - entries to stack.

Keyword Arguments:

- **out_key** ([*NestedKey*](tensordict.NestedKey.html#tensordict.NestedKey)) - new key name for the stacked inputs.
- **keep_entries** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)*,**optional*) - if `False`, entries in `keys` will be deleted.
Defaults to `False`.
- **dim** ([*int*](tensordict.TensorDictBase.int.html#tensordict.TensorDictBase.int)*,**optional*) - the dimension along which the stack must occur.
Defaults to `0`.

Returns: self

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict(a=torch.zeros(()), b=torch.ones(()))
>>> _ = td.stack_tensors("a", "b", out_key="c")
>>> assert "a" not in td
>>> assert (td["c"] == torch.tensor([0, 1])).all()
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).