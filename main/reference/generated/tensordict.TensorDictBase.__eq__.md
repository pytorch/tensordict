# TensorDictBase.__eq__

TensorDictBase.__eq__(*other: [Any](https://docs.python.org/3/library/typing.html#typing.Any)*) → Self | [bool](https://docs.python.org/3/builtins/functions.html#bool)

Compares the tensordict with `other` leaf by leaf with `==`.

`other` can be a tensordict or a `dict` with the same keys, paired
with the tensordict key by key (a `KeyError` is raised if the keys
differ); a tensor, which is aligned with the batch dimensions of each
leaf; or a number. A batch size that differs from the tensordict's is
broadcast with it. An `other` of another type, such as `None`,
gives `False`. The other comparisons (`!=`, `<`, `<=`, `>`,
`>=`) work the same way.

Parameters:

**other** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*,*[*dict*](https://docs.python.org/3/builtins/stdtypes.html#dict)*,*[*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*or**Number*) - the value to
compare with.

Returns:

A new tensordict of boolean leaves, or `False`.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict(a=torch.tensor([1.0, 2.0]), b=torch.zeros(2), batch_size=[2])
>>> (td == 1)["a"]
tensor([ True, False])
>>> (td == td.clone()).all()
True
>>> (td >= torch.tensor([2.0, 0.0]))["b"]
tensor([False, True])
>>> td == None
False
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).