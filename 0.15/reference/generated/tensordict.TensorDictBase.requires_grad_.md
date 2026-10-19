# TensorDictBase.requires_grad

TensorDictBase.requires_grad_(*requires_grad=True*) → Self

Sets `requires_grad` on every tensor leaf in place, and returns the tensordict.

Parameters:

**requires_grad** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - whether autograd records the
operations on the leaves. Defaults to `True`.

Returns:

`self`.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict(a=torch.zeros(3), b=TensorDict(c=torch.zeros(3)))
>>> td.requires_grad
False
>>> td.requires_grad_()["b", "c"].requires_grad
True
>>> td.requires_grad
True
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).