# TensorDictBase.grad

*property*TensorDictBase.grad*: [Any](https://docs.python.org/3/library/typing.html#typing.Any)*

A tensordict of the `.grad` of every leaf, or `None` if no leaf has one.

Leaves whose `.grad` is `None` are left out. Assigning `td.grad = other` sets the `.grad` of each leaf from the
matching leaf of `other`, or copies into it if it exists.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict(x=torch.ones(3, requires_grad=True), y=torch.ones(3, requires_grad=True), batch_size=[3])
>>> (td["x"] * 2).sum().backward()
>>> td.grad["x"], "y" in td.grad.keys()
(tensor([2., 2., 2.]), False)
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).