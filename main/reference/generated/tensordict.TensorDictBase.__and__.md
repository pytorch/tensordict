# TensorDictBase.__and__

TensorDictBase.__and__(*other: [TensorCollection](tensordict.TensorCollection.html#tensordict.TensorCollection) | [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | [float](https://docs.python.org/3/builtins/functions.html#float)*) → Self

Computes `self & other` leaf by leaf and returns a new tensordict.

The operation is a logical AND for boolean leaves and a bitwise AND for
integer leaves. A tensordict `other` is paired with the tensordict key
by key and must have the same keys, or a `KeyError` is raised.

Parameters:

**other** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*,*[*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*or**Number*) - the other operand.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict(a=torch.tensor([1.0, 2.0, 3.0]), batch_size=[3])
>>> ((td > 1) & (td < 3))["a"]
tensor([False, True, False])
>>> (~(td > 1))["a"]
tensor([ True, False, False])
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).