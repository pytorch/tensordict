# TensorDictBase.unflatten

TensorDictBase.unflatten(*dim*, *unflattened_size*, ***, *inplace: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*)

Unflattens a tensordict dim expanding it to a desired shape.

Parameters:

- **dim** ([*int*](https://docs.python.org/3/builtins/functions.html#int)) - specifies the dimension of the input tensor to be
unflattened.
- **unflattened_size** (*shape*) - is the new shape of the unflattened
dimension of the tensordict.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict({
... "a": torch.arange(60).view(3, 4, 5),
... "b": torch.arange(12).view(3, 4)},
... batch_size=[3, 4])
>>> td_flat = td.flatten(0, 1)
>>> td_unflat = td_flat.unflatten(0, [3, 4])
>>> assert (td == td_unflat).all()
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).