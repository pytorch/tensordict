# TensorDictBase.refine_names

TensorDictBase.refine_names(**names*) → Self

Names the unnamed batch dimensions, and checks the named ones, in place.

Each entry of `names` goes with one batch dimension. An unnamed
dimension takes the new name, or stays unnamed for `None`. A named
dimension must get its own name. `names` can hold one `...`, which
keeps the names of the dimensions that it covers. This follows
`torch.Tensor.refine_names()`.

Parameters:

***names** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*or**None*) - one name per batch dimension, or fewer with
one `...`.

Returns:

`self`.

Raises:

[**RuntimeError**](https://docs.python.org/3/builtins/exceptions.html#RuntimeError) - if a named dimension gets another name, or `None`.

Examples

```
>>> from tensordict import TensorDict
>>> td = TensorDict(batch_size=[3, 4, 5], names=["a", None, None])
>>> td.refine_names("a", None, "c").names
['a', None, 'c']
>>> td.refine_names(..., "b", "c").names
['a', 'b', 'c']
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).