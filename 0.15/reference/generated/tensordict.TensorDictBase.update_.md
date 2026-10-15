# TensorDictBase.update

TensorDictBase.update_(*input_dict_or_td: [dict](https://docs.python.org/3/builtins/stdtypes.html#dict)[[str](https://docs.python.org/3/builtins/stdtypes.html#str), [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | TensorCollection] | T | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *clone: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, ***, *non_blocking: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *keys_to_update: [Sequence](https://docs.python.org/3/library/typing.html#typing.Sequence)[[NestedKey](tensordict.NestedKey.html#tensordict.NestedKey)] | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, ***kwargs*) → Self

Updates the TensorDict in-place with values from either a dictionary or another TensorDict.

Unlike [`update()`](tensordict.TensorDictBase.update.html#tensordict.TensorDictBase.update), this function will throw an error if the key is unknown to `self`.

Parameters:

- **input_dict_or_td** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*or*[*dict*](https://docs.python.org/3/builtins/stdtypes.html#dict)*,**optional*) - input data to be written
in self. If `None`, only the keyword arguments are used.
- **clone** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - whether the tensors in the input (
tensor) dict should be cloned before being set. Defaults to `False`.

Keyword Arguments:

- **keys_to_update** (*sequence**of**NestedKeys**,**optional*) - if provided, only
the list of keys in `key_to_update` will be updated.
This is aimed at avoiding calls to
`data_dest.update_(data_src.select(*keys_to_update))`.
- **non_blocking** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)*,**optional*) - if `True` and this copy is between
different devices, the copy may occur asynchronously with respect
to the host.
- ****kwargs** - additional key-value pairs to write into `self` as top-level entries. As with the
positional form, any key that does not already exist in `self` raises [`KeyError`](https://docs.python.org/3/builtins/exceptions.html#KeyError).
When both `input_dict_or_td` and `kwargs` are provided, kwargs win on key conflict.

Note

Keyword arguments are treated as top-level keys only. Nested structures still work when
the kwarg value is itself a dict or tensor collection. Keys whose names collide with reserved
parameters (`clone`, `non_blocking`, `keys_to_update`) must be passed through the positional
dict form.

Note

On a shared or memory-mapped tensordict, a non-tensor value that its shared buffer cannot
store (for instance `None`, or an int where a string is shared) is skipped, with a
[`UserWarning`](https://docs.python.org/3/builtins/exceptions.html#UserWarning) that names its key.

Returns:

self

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> a = torch.randn(3)
>>> b = torch.randn(3, 4)
>>> td = TensorDict({"a": a, "b": b}, batch_size=[3])
>>> other_td = TensorDict({"a": a*0, "b": b*0}, batch_size=[])
>>> _ = td.update_(other_td)
>>> assert td['a'] is not other_td['a']
>>> assert (td['a'] == other_td['a']).all()
>>> assert (td['a'] == 0).all()
>>> # keyword form for top-level entries (key must already exist)
>>> _ = td.update_(a=torch.ones(3))
>>> assert (td["a"] == 1).all()
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).