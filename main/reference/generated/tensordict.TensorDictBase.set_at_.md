# TensorDictBase.set_at

TensorDictBase.set_at_(*key: [NestedKey](tensordict.NestedKey.html#tensordict.NestedKey)*, *value: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | TensorCollection*, *index: [None](https://docs.python.org/3/builtins/constants.html#None) | [int](https://docs.python.org/3/builtins/functions.html#int) | [slice](https://docs.python.org/3/builtins/functions.html#slice) | [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | [List](https://docs.python.org/3/library/typing.html#typing.List)[[Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[Any](https://docs.python.org/3/library/typing.html#typing.Any), ...]*, ***, *non_blocking: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*) → Self

Writes `value` into the existing entry `key` at the index `index`, in place.

`td.set_at_(key, value, index)` is the same as `td[key][index] = value`.

Parameters:

- **key** ([*NestedKey*](tensordict.NestedKey.html#tensordict.NestedKey)) - key to be modified. The key must exist: a
missing key raises `KeyError`.
- **value** ([*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) - value to be set at the index index
- **index** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,**tensor**or*[*tuple*](https://docs.python.org/3/builtins/stdtypes.html#tuple)) - index where to write the values.

Keyword Arguments:

**non_blocking** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)*,**optional*) - if `True` and this copy is between
different devices, the copy may occur asynchronously with respect
to the host.

Returns:

self

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict({}, batch_size=[3, 4])
>>> x = torch.randn(3, 4)
>>> td = td.set("x", x)
>>> td = td.set_at_("x", value=torch.ones(1, 4), index=slice(1))
>>> assert (x[0] == 1).all()
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).