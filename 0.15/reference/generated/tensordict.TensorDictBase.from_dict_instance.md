# TensorDictBase.from_dict_instance

TensorDictBase.from_dict_instance(*input_dict*, ***, *auto_batch_size: [bool](https://docs.python.org/3/builtins/functions.html#bool) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *batch_size=None*, *device=None*, *batch_dims=None*, *names=None*)

Builds a tensordict from a dictionary, with the types of the entries of this tensordict.

Works like [`from_dict()`](tensordict.TensorDictBase.from_dict.html#tensordict.TensorDictBase.from_dict), except for the
nested dicts: a nested dict whose key is a nested tensordict or
tensorclass of this tensordict is converted to that type, recursively.
The other nested dicts become [`TensorDict`](tensordict.TensorDict.html#tensordict.TensorDict)
instances. Use it to turn the output of [`to_dict()`](tensordict.TensorDictBase.to_dict.html#tensordict.TensorDictBase.to_dict) back into
tensorclasses. The values of this tensordict are not used, and the
result is a [`TensorDict`](tensordict.TensorDict.html#tensordict.TensorDict).

Parameters:

**input_dict** ([*dict*](https://docs.python.org/3/builtins/stdtypes.html#dict)) - the dictionary to convert. Nested dicts become
nested entries.

Keyword Arguments:

- **auto_batch_size** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, sets the batch size
to the largest leading shape that all the entries share.
Defaults to `None`, which behaves like `False` unless
`batch_dims` is given.
- **batch_size** ([*Size*](https://docs.pytorch.org/docs/stable/size.html#torch.Size)*or*[*list*](https://docs.python.org/3/builtins/stdtypes.html#list)*of*[*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the batch size of
the result: the leading dimensions that all its entries
share. Defaults to `None`, which gives an empty batch size
unless the batch size is computed.
- **device** ([*device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,*[*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*or*[*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the device of the
result. Leaves on another device are moved to it. `None`
leaves each leaf where it is, and the result has no device.
Defaults to `None`.
- **batch_dims** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - with `auto_batch_size`, the maximum
number of batch dimensions. Turns on `auto_batch_size` when
that is `None`. Exclusive with `batch_size`. Defaults to
`None`.
- **names** ([*list*](https://docs.python.org/3/builtins/stdtypes.html#list)*of*[*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - the names of the batch dimensions,
one per dimension. Used only with `batch_size`: when the
batch size is empty or computed, the names are dropped.
Defaults to `None`.

Raises:

[**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) - if both `batch_size` and `batch_dims` are given.

Examples

```
>>> import torch
>>> from tensordict import TensorDict, tensorclass
>>> @tensorclass
... class MyClass:
... x: torch.Tensor
>>> td = TensorDict(a=torch.zeros(()), b=MyClass(x=torch.zeros(())))
>>> data = {"a": torch.ones(()), "b": {"x": torch.ones(())}}
>>> type(td.from_dict_instance(data)["b"]).__name__
'MyClass'
>>> type(TensorDict.from_dict(data)["b"]).__name__
'TensorDict'
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).