# TensorDictBase.zero_grad

TensorDictBase.zero_grad(*set_to_none: [bool](https://docs.python.org/3/builtins/functions.html#bool) = True*) → Self

Resets the `.grad` of every leaf, nested ones included, and returns `self`.

Parameters:

**set_to_none** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, sets each `.grad` to
`None`. If `False`, fills each existing `.grad` with zeros in
place. Defaults to `True`.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict(x=torch.ones(3, requires_grad=True), batch_size=[3])
>>> td["x"].sum().backward()
>>> print(td.zero_grad()["x"].grad)
None
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).