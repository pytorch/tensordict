# TensorDictBase.fill

TensorDictBase.fill_(*key: [NestedKey](tensordict.NestedKey.html#tensordict.NestedKey)*, *value: [float](https://docs.python.org/3/builtins/functions.html#float) | [bool](https://docs.python.org/3/builtins/functions.html#bool)*) → Self

Fills the entry at `key` with `value` in place and returns `self`.

If `key` points to a nested tensordict, every leaf of it is filled.

Parameters:

- **key** ([*NestedKey*](tensordict.NestedKey.html#tensordict.NestedKey)) - the key of the entry to fill.
- **value** (*Number**or*[*bool*](https://docs.python.org/3/builtins/functions.html#bool)) - the value to write.

Raises:

[**KeyError**](https://docs.python.org/3/builtins/exceptions.html#KeyError) - if `key` is missing.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict(a=torch.zeros(2), b=TensorDict(c=torch.zeros(2)))
>>> td.fill_("a", 1.0)["a"]
tensor([1., 1.])
>>> td.fill_("b", 2.0)["b", "c"]
tensor([2., 2.])
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).