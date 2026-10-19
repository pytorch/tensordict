# TensorDictBase.detach

TensorDictBase.detach_() → Self

Detaches every tensor leaf from the autograd graph in place, and returns the tensordict.

Each leaf is detached with [`torch.Tensor.detach_()`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.detach_.html#torch.Tensor.detach_): it keeps its
storage and no longer requires gradients.

Returns:

`self`.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict(a=torch.zeros(3, requires_grad=True) + 1)
>>> a = td["a"]
>>> _ = td.detach_()
>>> a.requires_grad, td["a"] is a
(False, True)
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).