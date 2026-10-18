# TensorDictBase.abs

TensorDictBase.abs() → Self

Returns a new tensordict with [`torch.abs()`](https://docs.pytorch.org/docs/stable/generated/torch.abs.html#torch.abs) applied to every leaf.

The other unary operations ([`exp()`](tensordict.TensorDictBase.exp.html#tensordict.TensorDictBase.exp), [`log()`](tensordict.TensorDictBase.log.html#tensordict.TensorDictBase.log),
[`sqrt()`](tensordict.TensorDictBase.sqrt.html#tensordict.TensorDictBase.sqrt), [`sin()`](tensordict.TensorDictBase.sin.html#tensordict.TensorDictBase.sin)...) work the same way. Each has an
in-place twin with a trailing underscore ([`abs_()`](tensordict.TensorDictBase.abs_.html#tensordict.TensorDictBase.abs_),
[`exp_()`](tensordict.TensorDictBase.exp_.html#tensordict.TensorDictBase.exp_)...) that writes into the leaves and returns `self`.
See [In-place operations](../../concepts/operations.html#inplace-conventions).

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict(
... a=torch.tensor([-1.0, 2.0]),
... b=TensorDict(c=torch.tensor([-3.0, 0.0]), batch_size=[2]),
... batch_size=[2],
... )
>>> td.abs()["b", "c"]
tensor([3., 0.])
>>> td.exp()["a"]
tensor([0.3679, 7.3891])
>>> td.abs_() is td
True
>>> td["a"]
tensor([1., 2.])
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).