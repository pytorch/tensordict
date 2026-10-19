# TensorDictBase.from_dataclass

*classmethod*TensorDictBase.from_dataclass(*dataclass*, ***, *dest_cls: [Type](https://docs.python.org/3/library/typing.html#typing.Type) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *auto_batch_size: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *batch_dims: [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *as_tensorclass: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *device: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *batch_size: [Size](https://docs.pytorch.org/docs/stable/size.html#torch.Size) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*)

Converts a dataclass instance to a tensordict, with one entry per field.

Each field value is converted with [`from_any()`](tensordict.TensorDictBase.from_any.html#tensordict.TensorDictBase.from_any), so a nested
dataclass or dict becomes a nested tensordict and a number becomes a
tensor. With `as_tensorclass=True`, the conversion is that of the
function [`from_dataclass()`](tensordict.from_dataclass.html#tensordict.from_dataclass) instead, which returns a
tensorclass.

Parameters:

**dataclass** - the dataclass instance to convert. With
`as_tensorclass=True`, it can also be a dataclass type.

Keyword Arguments:

- **dest_cls** ([*type*](https://docs.python.org/3/builtins/functions.html#type)*,**optional*) - with `as_tensorclass=True`, the
tensorclass to convert to. Ignored otherwise. Defaults to
`None`, which creates a new tensorclass with the fields of
the dataclass.
- **auto_batch_size** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, sets the batch size
to the largest leading shape that all the entries share.
Exclusive with `batch_size`. Defaults to `False`.
- **batch_dims** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - with `auto_batch_size`, the maximum
number of batch dimensions. Ignored without
`auto_batch_size`. Defaults to `None`.
- **as_tensorclass** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, returns a
tensorclass instance (or a tensorclass type, for a dataclass
type) through [`from_dataclass()`](tensordict.from_dataclass.html#tensordict.from_dataclass). Defaults to
`False`.
- **device** ([*device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,*[*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*or*[*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the device of the
result. Leaves on another device are moved to it. `None`
leaves each leaf where it is, and the result has no device.
Defaults to `None`.
- **batch_size** ([*Size*](https://docs.pytorch.org/docs/stable/size.html#torch.Size)*or*[*list*](https://docs.python.org/3/builtins/stdtypes.html#list)*of*[*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the batch size of
the result: the leading dimensions that all its entries share.
Exclusive with `auto_batch_size`. Defaults to `None`, an
empty batch size.

Returns:

A [`TensorDict`](tensordict.TensorDict.html#tensordict.TensorDict), or a tensorclass instance or
type if `as_tensorclass=True`.

Raises:

[**TypeError**](https://docs.python.org/3/builtins/exceptions.html#TypeError) - if `dataclass` is not a dataclass and
 `as_tensorclass=False`, if it is a dataclass type with a
 field that cannot be read from the class (a field without a
 default, or a field of a tensorclass type) and
 `as_tensorclass=False`, or if both `auto_batch_size=True`
 and `batch_size` are given.

Examples

```
>>> import dataclasses
>>> import torch
>>> from tensordict import TensorDict
>>> @dataclasses.dataclass
... class Data:
... obs: torch.Tensor
... reward: torch.Tensor
>>> data = Data(obs=torch.zeros(3, 4), reward=torch.ones(3))
>>> td = TensorDict.from_dataclass(data, auto_batch_size=True)
>>> type(td).__name__, td.batch_size
('TensorDict', torch.Size([3]))
>>> tc = TensorDict.from_dataclass(data, as_tensorclass=True, batch_size=[3])
>>> type(tc).__name__, tc.batch_size
('Data_tc', torch.Size([3]))
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).