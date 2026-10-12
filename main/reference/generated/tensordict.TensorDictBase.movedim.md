# TensorDictBase.movedim

TensorDictBase.movedim(*source: [int](https://docs.python.org/3/builtins/functions.html#int) | [tuple](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[int](https://docs.python.org/3/builtins/functions.html#int), ...]*, *destination: [int](https://docs.python.org/3/builtins/functions.html#int) | [tuple](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[int](https://docs.python.org/3/builtins/functions.html#int), ...]*)

Moves the dimension(s) of input at the position(s) in source to the position(s) in destination.

Other dimensions of input that are not explicitly moved remain in their
original order and appear at the positions not specified in destination.

Parameters:

- **source** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*or*[*tuple*](https://docs.python.org/3/builtins/stdtypes.html#tuple)*of**ints*) - Original positions of the dims to move.
These must be unique.
- **destination** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*or*[*tuple*](https://docs.python.org/3/builtins/stdtypes.html#tuple)*of**ints*) - Destination positions for each of
the original dims. These must also be unique.

Returns:

a new tensordict with the batch dimensions moved to the desired positions.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict({"a": torch.randn(3, 4, 5)}, batch_size=[3, 4])
>>> print(td.movedim(0, 1).shape)
torch.Size([4, 3])
>>> print(td.movedim((0, 1), (1, 0)).shape)
torch.Size([4, 3])
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).