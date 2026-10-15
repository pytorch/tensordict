# TensorDictBase.cat_tensors

TensorDictBase.cat_tensors(**keys: [NestedKey](tensordict.NestedKey.html#tensordict.NestedKey)*, *out_key: [NestedKey](tensordict.NestedKey.html#tensordict.NestedKey)*, *dim: [int](https://docs.python.org/3/builtins/functions.html#int) = 0*, *keep_entries: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*) → Self

Concatenates entries into a new entry and possibly remove the original values.

Parameters:

**keys** (*sequence**of*[*NestedKey*](tensordict.NestedKey.html#tensordict.NestedKey)) - entries to concatenate.

Keyword Arguments:

- **out_key** ([*NestedKey*](tensordict.NestedKey.html#tensordict.NestedKey)) - new key name for the concatenated inputs.
- **keep_entries** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)*,**optional*) - if `False`, entries in `keys` will be deleted.
Defaults to `False`. Entries are deleted only after the result
has been written. If concatenation fails, or if the result cannot
be written, every input entry is kept with its original value.
- **dim** ([*int*](tensordict.TensorDictBase.int.html#tensordict.TensorDictBase.int)*,**optional*) - the dimension along which the concatenation must occur.
Defaults to `0`.

Returns: self

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict(a=torch.zeros(1), b=torch.ones(1))
>>> _ = td.cat_tensors("a", "b", out_key="c")
>>> assert "a" not in td
>>> assert (td["c"] == torch.tensor([0, 1])).all()
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).