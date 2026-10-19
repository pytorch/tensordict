# TensorDictBase.is_empty

TensorDictBase.is_empty() → [bool](https://docs.python.org/3/builtins/functions.html#bool)

Returns `True` if the tensordict has no leaf.

Empty sub-tensordicts do not count as leaves. A lazy stack counts only
the tensors that all its members share: it returns `True` when they
share none, and warns when some member holds a leaf. Starting with
TensorDict 0.17, the default will change to returning `True` only if
every member is empty.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> TensorDict().is_empty()
True
>>> TensorDict({"nested": TensorDict()}).is_empty()
True
>>> TensorDict({"a": torch.zeros(())}).is_empty()
False
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).