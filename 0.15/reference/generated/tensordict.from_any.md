# from_any

tensordict.from_any(*obj*, ***, *auto_batch_size: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *batch_dims: [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *device: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *batch_size: [Size](https://docs.pytorch.org/docs/stable/size.html#torch.Size) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*)

Converts a dict, tuple, dataclass, array or file to a tensordict, recursively.

The conversion depends on the type of `obj`: see
[`TensorDictBase.from_any`](tensordict.TensorDictBase.from_any.html#tensordict.TensorDictBase.from_any) for
the list. A tensor or a tensor collection is returned as it is.

Parameters:

**obj** - the object to convert.

Keyword Arguments:

- **auto_batch_size** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, sets the batch size to
the largest leading shape that all the entries share. Exclusive
with `batch_size`, except for a dict, where `batch_size`
wins, and for an `h5py.File`, where `auto_batch_size`
wins. Defaults to `False`.
- **batch_dims** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - with `auto_batch_size`, the maximum
number of batch dimensions. Ignored without
`auto_batch_size`, with a `FutureWarning` when it would
change the result. Starting with TensorDict 0.17, the default
will change to turning on `auto_batch_size` when
`batch_dims` is given, as `from_dict` does. Defaults to
`None`.
- **device** ([*device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,*[*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*or*[*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the device of the
result. Leaves on another device are moved to it. `None` leaves
each leaf where it is, and the result has no device. Defaults to
`None`.
- **batch_size** ([*Size*](https://docs.pytorch.org/docs/stable/size.html#torch.Size)*or*[*list*](https://docs.python.org/3/builtins/stdtypes.html#list)*of*[*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the batch size of
the result: the leading dimensions that all its entries share.
Exclusive with `auto_batch_size`, except for a dict, where
`batch_size` wins and no batch size is computed, and for an
`h5py.File`, where `auto_batch_size` wins and
`batch_size` is ignored. Defaults to `None`.

Returns:

A tensordict (a [`PersistentTensorDict`](tensordict.PersistentTensorDict.html#tensordict.PersistentTensorDict) for an HDF5
file), a tensor or a [`NonTensorStack`](tensordict.NonTensorStack.html#tensordict.NonTensorStack) for a list,
or `obj` itself for the other types.

Examples

```
>>> import torch
>>> from tensordict import from_any
>>> td = from_any({"obs": torch.zeros(3, 4), "pair": (torch.zeros(3), 1)})
>>> sorted(td.keys(include_nested=True, leaves_only=True), key=str)
[('pair', '0'), ('pair', '1'), 'obs']
```