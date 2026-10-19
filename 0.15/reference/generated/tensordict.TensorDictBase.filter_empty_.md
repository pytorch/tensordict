# TensorDictBase.filter_empty

TensorDictBase.filter_empty_()

Removes the nested tensordicts that have no leaf, in place, and returns `self`.

A nested tensordict that holds only empty tensordicts is removed too.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict(a=torch.zeros(3), b={}, c={"d": {}}, batch_size=[3])
>>> list(td.filter_empty_().keys())
['a']
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).