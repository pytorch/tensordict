# TensorDictBase.stack_tensors

TensorDictBase.stack_tensors(**keys: [NestedKey](tensordict.NestedKey.html#tensordict.NestedKey)*, *out_key: [NestedKey](tensordict.NestedKey.html#tensordict.NestedKey)*, *dim: [int](https://docs.python.org/3/builtins/functions.html#int) = 0*, *keep_entries: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*) → Self

Stacks entries of the tensordict into a new entry, and removes them by default.

Parameters:

***keys** ([*NestedKey*](tensordict.NestedKey.html#tensordict.NestedKey)) - the entries to stack, with [`torch.stack()`](https://docs.pytorch.org/docs/stable/generated/torch.stack.html#torch.stack).

Keyword Arguments:

- **out_key** ([*NestedKey*](tensordict.NestedKey.html#tensordict.NestedKey)) - the key of the result.
- **dim** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the new dimension, counted in the dimensions
of the result. Defaults to `0`.
- **keep_entries** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `False`, removes the entries
of `keys` once the result is written. If the stack or the
write fails, every entry is kept with its original value.
Defaults to `False`.

Returns:

`self`.

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