# TensorDictBase.bytes

TensorDictBase.bytes(***, *count_duplicates: [bool](https://docs.python.org/3/builtins/functions.html#bool) = True*) → [int](https://docs.python.org/3/builtins/functions.html#int)

Returns the number of bytes of the tensor leaves.

Each tensor leaf counts `numel() * dtype.itemsize` bytes, and a leaf
with a gradient counts the bytes of its gradient too. Non-tensor data
is not counted. A view counts the bytes of its own elements, not those
of the storage it shares.

Keyword Arguments:

**count_duplicates** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, a tensor stored
under several keys is counted once per key. If `False`, it
is counted once. Two different views of one tensor are counted
twice in both cases. Defaults to `True`.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> x = torch.zeros(6)
>>> td = TensorDict(a=x, b=x, c=torch.zeros(2, 3))
>>> td.bytes()
72
>>> td.bytes(count_duplicates=False)
48
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).