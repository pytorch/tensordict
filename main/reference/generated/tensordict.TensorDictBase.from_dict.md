# TensorDictBase.from_dict

*abstract classmethod*TensorDictBase.from_dict(*input_dict*, ***, *auto_batch_size: [bool](https://docs.python.org/3/builtins/functions.html#bool) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *batch_size: [Size](https://docs.pytorch.org/docs/stable/size.html#torch.Size) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *device: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *batch_dims: [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *names: [List](https://docs.python.org/3/library/typing.html#typing.List)[[str](https://docs.python.org/3/builtins/stdtypes.html#str)] | [None](https://docs.python.org/3/builtins/constants.html#None) = None*)

Returns a tensordict built from a dictionary, which can be nested.

The nested dictionaries become nested tensordicts. Without
`batch_size`, `auto_batch_size` or `batch_dims`, the batch size
is `[]`. With `auto_batch_size=True`, each nested tensordict gets
its own largest batch size, as with [`auto_batch_size_()`](tensordict.TensorDictBase.auto_batch_size_.html#tensordict.TensorDictBase.auto_batch_size_).
[`TensorDict`](tensordict.TensorDict.html#tensordict.TensorDict) returns a tensordict passed as
`input_dict` unchanged, and ignores the keyword arguments: call
[`auto_batch_size_()`](tensordict.TensorDictBase.auto_batch_size_.html#tensordict.TensorDictBase.auto_batch_size_) on it instead. See [Batch size, shapes, and indexing](../../concepts/batch_size.html#batch-size).

Parameters:

**input_dict** ([*dict*](https://docs.python.org/3/builtins/stdtypes.html#dict)) - the entries. A key can be a string or a tuple
of strings, and a value a tensor, a dictionary, or any object
that [`TensorDict`](tensordict.TensorDict.html#tensordict.TensorDict) accepts. With a non-empty
`batch_size`, [`TensorDict`](tensordict.TensorDict.html#tensordict.TensorDict) stores a
Python `int`, `float` or `bool` as
[`NonTensorData`](tensordict.NonTensorData.html#tensordict.NonTensorData), so that the output of
[`to_dict()`](tensordict.TensorDictBase.to_dict.html#tensordict.TensorDictBase.to_dict) converts back. Otherwise, it stores it as a
0-dim tensor.

Keyword Arguments:

- **auto_batch_size** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, sets the batch size
to the largest leading shape that all the entries share.
Defaults to `None`, which is `True` if `batch_dims` is
passed and `False` otherwise.
- **batch_size** ([*Size*](https://docs.pytorch.org/docs/stable/size.html#torch.Size)*or*[*list*](https://docs.python.org/3/builtins/stdtypes.html#list)*of*[*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the batch size of
the result: the leading dimensions that all its entries share.
Takes precedence over `auto_batch_size=True`: no batch size
is computed. Exclusive with `batch_dims`. Defaults to
`None`.
- **device** ([*device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,*[*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*or*[*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the device of the
result. Leaves on another device are moved to it. `None`
leaves each leaf where it is, and the result has no device.
Defaults to `None`.
- **batch_dims** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the maximum number of batch dimensions
of the computed batch size. Turns on `auto_batch_size`.
Exclusive with `batch_size`. Defaults to `None`.
- **names** ([*list*](https://docs.python.org/3/builtins/stdtypes.html#list)*of*[*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - the names of the batch dimensions,
one per dimension. Defaults to `None`.

Raises:

[**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) - if both `batch_size` and `batch_dims` are passed.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> data = {"a": torch.zeros(3, 4), "b": {"c": torch.zeros(3, 4, 5)}}
>>> TensorDict.from_dict(data).batch_size
torch.Size([])
>>> td = TensorDict.from_dict(data, auto_batch_size=True)
>>> td.batch_size, td["b"].batch_size
(torch.Size([3, 4]), torch.Size([3, 4, 5]))
>>> TensorDict.from_dict(data, batch_dims=1).batch_size
torch.Size([3])
>>> TensorDict.from_dict(data, batch_size=[3]).batch_size
torch.Size([3])
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).