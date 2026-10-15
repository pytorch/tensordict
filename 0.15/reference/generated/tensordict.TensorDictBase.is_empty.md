# TensorDictBase.is_empty

TensorDictBase.is_empty() → [bool](https://docs.python.org/3/builtins/functions.html#bool)

Returns `True` if the tensordict has no leaf.

Empty sub-tensordicts do not count as leaves.

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