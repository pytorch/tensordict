# TensorDictBase.roll

TensorDictBase.roll(*shifts: [int](https://docs.python.org/3/builtins/functions.html#int) | [tuple](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[int](https://docs.python.org/3/builtins/functions.html#int), ...]*, *dims: [int](https://docs.python.org/3/builtins/functions.html#int) | [tuple](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[int](https://docs.python.org/3/builtins/functions.html#int), ...] = None*, ***, *inplace: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*)

Roll the tensordict along the given dimensions.

Elements that are shifted beyond the last position are re-introduced at
the first position.

Parameters:

- **shifts** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*or*[*tuple*](https://docs.python.org/3/builtins/stdtypes.html#tuple)*of**ints*) - The number of places by which the elements
of the tensordict are shifted. If shifts is a tuple, dims must be a
tuple of the same size, and each dimension will be rolled by the
corresponding value.
- **dims** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*or*[*tuple*](https://docs.python.org/3/builtins/stdtypes.html#tuple)*of**ints**,**optional*) - Axis along which to roll.
By default, the tensordict is flattened before rolling.

Keyword Arguments:

**inplace** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)*,**optional*) - If `True`, this tensordict's identity
and key set are preserved; each leaf storage is replaced by
its rolled counterpart one leaf at a time, keeping peak memory
close to `1x`. The batch_size is unchanged by `roll`.
Defaults to `False`.

Returns:

a new tensordict with elements rolled. When `inplace=True` this
is `self`.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict({"a": torch.arange(6).view(2, 3)}, batch_size=[2, 3])
>>> print(td["a"])
tensor([[0, 1, 2],
 [3, 4, 5]])
>>> print(td.roll(1, 0)["a"])
tensor([[3, 4, 5],
 [0, 1, 2]])
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).