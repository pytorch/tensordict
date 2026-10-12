# TensorDictBase.from_any

*classmethod*TensorDictBase.from_any(*obj*, ***, *auto_batch_size: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *batch_dims: [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *device: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *batch_size: [Size](https://docs.pytorch.org/docs/stable/size.html#torch.Size) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*)

Recursively converts any object to a TensorDict.

Note

`from_any` is less restrictive than the regular TensorDict constructor. It can cast data structures like
dataclasses or tuples to a tensordict using custom heuristics. This approach may incur some extra overhead and
involves more opinionated choices in terms of mapping strategies.

Note

This method recursively converts the input object to a TensorDict. If the object is already a
TensorDict (or any similar tensor collection object), it will be returned as is.

Parameters:

**obj** - The object to be converted.

Keyword Arguments:

- **auto_batch_size** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)*,**optional*) - if `True`, the batch size will be computed automatically.
Defaults to `False`.
- **batch_dims** ([*int*](tensordict.TensorDictBase.int.html#tensordict.TensorDictBase.int)*,**optional*) - If auto_batch_size is `True`, defines how many dimensions the output tensordict
should have. Defaults to `None` (full batch-size at each level).
- **device** ([*device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,**optional*) - The device on which the TensorDict will be created.
- **batch_size** ([*Size*](https://docs.pytorch.org/docs/stable/size.html#torch.Size)*,**optional*) - The batch size of the TensorDict.
Exclusive with `auto_batch_size`.

Returns:

A TensorDict representation of the input object.

Supported objects:

- Dataclasses through [`from_dataclass()`](tensordict.TensorDictBase.from_dataclass.html#tensordict.TensorDictBase.from_dataclass) (dataclasses will be converted to TensorDict instances, not tensorclasses).
- Namedtuples through [`from_namedtuple()`](tensordict.TensorDictBase.from_namedtuple.html#tensordict.TensorDictBase.from_namedtuple).
- Dictionaries through [`from_dict()`](tensordict.TensorDictBase.from_dict.html#tensordict.TensorDictBase.from_dict).
- Tuples through [`from_tuple()`](tensordict.TensorDictBase.from_tuple.html#tensordict.TensorDictBase.from_tuple).
- NumPy's structured arrays through [`from_struct_array()`](tensordict.TensorDictBase.from_struct_array.html#tensordict.TensorDictBase.from_struct_array).
- HDF5 objects through [`from_h5()`](tensordict.TensorDictBase.from_h5.html#tensordict.TensorDictBase.from_h5).

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).