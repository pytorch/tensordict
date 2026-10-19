# TensorDictBase.from_namedtuple

*classmethod*TensorDictBase.from_namedtuple(*named_tuple*, ***, *auto_batch_size: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *batch_dims: [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *device: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *batch_size: [Size](https://docs.pytorch.org/docs/stable/size.html#torch.Size) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*)

Converts a named tuple to a tensordict whose keys are its field names.

Each field value is converted with [`from_any()`](tensordict.TensorDictBase.from_any.html#tensordict.TensorDictBase.from_any), so a nested
named tuple or dict becomes a nested tensordict and a number becomes a
tensor. The `torch.return_types` results of `torch.max`,
`torch.sort` and the like are accepted too. [`to_namedtuple()`](tensordict.TensorDictBase.to_namedtuple.html#tensordict.TensorDictBase.to_namedtuple)
converts back.

Parameters:

**named_tuple** (*NamedTuple*) - the named tuple to convert.

Keyword Arguments:

- **auto_batch_size** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, sets the batch size
to the largest leading shape that all the entries share.
Exclusive with `batch_size`. Defaults to `False`.
- **batch_dims** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - with `auto_batch_size`, the maximum
number of batch dimensions. Ignored without
`auto_batch_size`, with a `FutureWarning` when it would
change the result. Starting with TensorDict 0.17, the default
will change to turning on `auto_batch_size` when
`batch_dims` is given, as `from_dict` does. Defaults to
`None`.
- **device** ([*device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,*[*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*or*[*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the device of the
result. Leaves on another device are moved to it. `None`
leaves each leaf where it is, and the result has no device.
Defaults to `None`.
- **batch_size** ([*Size*](https://docs.pytorch.org/docs/stable/size.html#torch.Size)*or*[*list*](https://docs.python.org/3/builtins/stdtypes.html#list)*of*[*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the batch size of
the result: the leading dimensions that all its entries share.
It applies to the nested tensordicts too. Exclusive with
`auto_batch_size`. Defaults to `None`, an empty batch size.

Returns:

A [`TensorDict`](tensordict.TensorDict.html#tensordict.TensorDict).

Raises:

[**TypeError**](https://docs.python.org/3/builtins/exceptions.html#TypeError) - if both `auto_batch_size=True` and `batch_size`
 are given.

Examples

```
>>> from tensordict import TensorDict
>>> import torch
>>> data = TensorDict({
... "a_tensor": torch.zeros((3)),
... "nested": {"a_tensor": torch.zeros((3)), "a_string": "zero!"}}, [3])
>>> nt = data.to_namedtuple()
>>> nt
GenericDict(a_tensor=tensor([0., 0., 0.]), nested=GenericDict(a_tensor=tensor([0., 0., 0.]), a_string='zero!'))
>>> td = TensorDict.from_namedtuple(nt, auto_batch_size=True)
>>> td.batch_size, td["nested", "a_string"]
(torch.Size([3]), 'zero!')
>>> sorted(TensorDict.from_namedtuple(torch.tensor([2, 0, 1]).max(0)).keys())
['indices', 'values']
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).