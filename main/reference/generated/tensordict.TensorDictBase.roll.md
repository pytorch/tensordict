# TensorDictBase.roll

TensorDictBase.roll(*shifts: [int](https://docs.python.org/3/builtins/functions.html#int) | [tuple](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[int](https://docs.python.org/3/builtins/functions.html#int), ...]*, *dims: [int](https://docs.python.org/3/builtins/functions.html#int) | [tuple](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[int](https://docs.python.org/3/builtins/functions.html#int), ...] = None*, ***, *inplace: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*)

Returns a tensordict with the elements rolled along batch dimensions, as [`torch.roll()`](https://docs.pytorch.org/docs/stable/generated/torch.roll.html#torch.roll) does.

Elements shifted past the last position come back at the first one.
The batch size and the names do not change, and the feature dimensions
are not rolled. The data is copied.

It can be used as a context manager, as [`unsqueeze()`](tensordict.TensorDictBase.unsqueeze.html#tensordict.TensorDictBase.unsqueeze) can: when
the `with` block exits, the entries of the block are written back to
the tensordict with the inverse operation.

Parameters:

- **shifts** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*or*[*tuple*](https://docs.python.org/3/builtins/stdtypes.html#tuple)*of*[*int*](https://docs.python.org/3/builtins/functions.html#int)) - the number of places to shift the
elements by. With a tuple, `dims` is a tuple of the same
size, and each dimension is rolled by its own shift.
- **dims** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*or*[*tuple*](https://docs.python.org/3/builtins/stdtypes.html#tuple)*of*[*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the batch dimensions to roll.
Defaults to `None`, which rolls the flattened batch and
restores its shape.

Keyword Arguments:

**inplace** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, the tensordict keeps its
identity and keys, its leaves are replaced one by one, and the
call returns `self`. See [In-place operations](../../concepts/operations.html#inplace-conventions). Defaults
to `False`.

Returns:

The rolled tensordict, or `self` with `inplace=True`.

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