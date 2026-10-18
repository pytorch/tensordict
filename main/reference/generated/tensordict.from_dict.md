# from_dict

tensordict.from_dict(*d*, ***, *auto_batch_size: [bool](https://docs.python.org/3/builtins/functions.html#bool) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *batch_dims: [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *device: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *batch_size: [Size](https://docs.pytorch.org/docs/stable/size.html#torch.Size) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *names: [Sequence](https://docs.python.org/3/library/typing.html#typing.Sequence)[[str](https://docs.python.org/3/builtins/stdtypes.html#str)] | [None](https://docs.python.org/3/builtins/constants.html#None) = None*) → [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase)

Converts a dictionary to a tensordict, recursively.

Nested dicts become nested tensordicts, and the values are converted with
[`from_any()`](tensordict.from_any.html#tensordict.from_any). Unlike the [`TensorDict`](tensordict.TensorDict.html#tensordict.TensorDict) constructor,
it can compute the batch size. See
[`TensorDict.from_dict`](tensordict.TensorDictBase.from_dict.html#tensordict.TensorDictBase.from_dict).

Parameters:

**d** ([*dict*](https://docs.python.org/3/builtins/stdtypes.html#dict)) - the dictionary to convert.

Keyword Arguments:

- **auto_batch_size** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, sets the batch size to
the largest leading shape that all the entries share, for the
result and for each nested tensordict. Defaults to `None`,
which behaves like `False` unless `batch_dims` is given.
- **batch_dims** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - with `auto_batch_size`, the maximum
number of batch dimensions. Turns on `auto_batch_size` when that
is `None`. Exclusive with `batch_size`. Defaults to `None`.
- **device** ([*device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,*[*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*or*[*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the device of the
result. Leaves on another device are moved to it. `None` leaves
each leaf where it is, and the result has no device. Defaults to
`None`.
- **batch_size** ([*Size*](https://docs.pytorch.org/docs/stable/size.html#torch.Size)*or*[*list*](https://docs.python.org/3/builtins/stdtypes.html#list)*of*[*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the batch size of
the result: the leading dimensions that all its entries share.
Takes precedence over `auto_batch_size`. Defaults to `None`,
an empty batch size unless it is computed.
- **names** ([*list*](https://docs.python.org/3/builtins/stdtypes.html#list)*of*[*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - the names of the batch dimensions, one
per dimension. Used only with `batch_size`: when the batch size
is empty or computed, the names are dropped. Defaults to `None`.

Returns:

A [`TensorDict`](tensordict.TensorDict.html#tensordict.TensorDict).

Raises:

[**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) - if both `batch_size` and `batch_dims` are given.

Examples

```
>>> import torch
>>> from tensordict import from_dict
>>> input_dict = {"a": torch.zeros(3, 4), "b": {"c": torch.zeros(3, 4, 5)}}
>>> from_dict(input_dict).batch_size
torch.Size([])
>>> td = from_dict(input_dict, auto_batch_size=True)
>>> td.batch_size, td["b"].batch_size
(torch.Size([3, 4]), torch.Size([3, 4, 5]))
>>> from_dict(input_dict, batch_dims=1).batch_size
torch.Size([3])
>>> from_dict(input_dict, batch_size=[3], names=["n"]).names
['n']
```