# TensorDictBase.from_any

*classmethod*TensorDictBase.from_any(*obj*, ***, *auto_batch_size: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *batch_dims: [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *device: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *batch_size: [Size](https://docs.pytorch.org/docs/stable/size.html#torch.Size) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*)

Converts a dict, tuple, dataclass, array or file to a tensordict, recursively.

`from_any` picks the conversion from the type of `obj`, and
applies it to the values inside `obj` too. It accepts more types
than the [`TensorDict`](tensordict.TensorDict.html#tensordict.TensorDict) constructor, at the cost of
some overhead and of opinionated choices about how each type maps to
entries:

- a dict or [`collections.UserDict`](https://docs.python.org/3/library/collections.html#collections.UserDict): [`from_dict()`](tensordict.TensorDictBase.from_dict.html#tensordict.TensorDictBase.from_dict);
- a named tuple: [`from_namedtuple()`](tensordict.TensorDictBase.from_namedtuple.html#tensordict.TensorDictBase.from_namedtuple);
- a tuple: [`from_tuple()`](tensordict.TensorDictBase.from_tuple.html#tensordict.TensorDictBase.from_tuple), with the keys `"0"`, `"1"`, ...
This includes the results of `torch.return_types` (`max`,
`sort`, `topk`, ...), which also emit a [`FutureWarning`](https://docs.python.org/3/builtins/exceptions.html#FutureWarning).
Starting with TensorDict 0.17, the default will change to their
field names (`"values"`, `"indices"`, ...), as
[`from_namedtuple()`](tensordict.TensorDictBase.from_namedtuple.html#tensordict.TensorDictBase.from_namedtuple) gives;
- a dataclass instance: [`from_dataclass()`](tensordict.TensorDictBase.from_dataclass.html#tensordict.TensorDictBase.from_dataclass) (the result is a
tensordict, not a tensorclass). A dataclass type whose fields all
have defaults gives a tensordict of the defaults too, with a
[`FutureWarning`](https://docs.python.org/3/builtins/exceptions.html#FutureWarning). Starting with TensorDict 0.17, the default
will change to returning the type unchanged, as for any other
class;
- a NumPy structured array: [`from_struct_array()`](tensordict.TensorDictBase.from_struct_array.html#tensordict.TensorDictBase.from_struct_array);
- a pandas DataFrame: [`from_pandas()`](tensordict.TensorDictBase.from_pandas.html#tensordict.TensorDictBase.from_pandas);
- an open `h5py.File`: a
[`PersistentTensorDict`](tensordict.PersistentTensorDict.html#tensordict.PersistentTensorDict) (see [`from_h5()`](tensordict.TensorDictBase.from_h5.html#tensordict.TensorDictBase.from_h5)).

A tensor or a tensor collection is returned as it is. A list becomes a
tensor if [`torch.tensor()`](https://docs.pytorch.org/docs/stable/generated/torch.tensor.html#torch.tensor) takes it, and a
[`NonTensorStack`](tensordict.NonTensorStack.html#tensordict.NonTensorStack) otherwise. Any other object is
returned unchanged.

Parameters:

**obj** - the object to convert.

Keyword Arguments:

- **auto_batch_size** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, sets the batch size
to the largest leading shape that all the entries share.
Exclusive with `batch_size`. Defaults to `False`.
- **batch_dims** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - with `auto_batch_size`, the maximum
number of batch dimensions. Ignored without
`auto_batch_size`. Defaults to `None`.
- **device** ([*device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,*[*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*or*[*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the device of the
result. Leaves on another device are moved to it. `None`
leaves each leaf where it is, and the result has no device.
Defaults to `None`.
- **batch_size** ([*Size*](https://docs.pytorch.org/docs/stable/size.html#torch.Size)*or*[*list*](https://docs.python.org/3/builtins/stdtypes.html#list)*of*[*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the batch size of
the result: the leading dimensions that all its entries share.
Exclusive with `auto_batch_size`. Defaults to `None`.

Returns:

A tensordict (a [`PersistentTensorDict`](tensordict.PersistentTensorDict.html#tensordict.PersistentTensorDict) for an
HDF5 file), a tensor or a [`NonTensorStack`](tensordict.NonTensorStack.html#tensordict.NonTensorStack) for
a list, or `obj` itself for the other types.

Examples

```
>>> import dataclasses
>>> import torch
>>> from tensordict import TensorDict
>>> @dataclasses.dataclass
... class Data:
... obs: torch.Tensor
... extra: tuple
>>> data = Data(obs=torch.zeros(3, 4), extra=(torch.ones(3), {"x": torch.ones(3)}))
>>> td = TensorDict.from_any(data, auto_batch_size=True)
>>> td.batch_size
torch.Size([3])
>>> sorted(td.keys(include_nested=True, leaves_only=True), key=str)
[('extra', '0'), ('extra', '1', 'x'), 'obs']
>>> TensorDict.from_any([1, 2, 3])
tensor([1, 2, 3])
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).