# TensorDictBase.from_dataclass

*classmethod*TensorDictBase.from_dataclass(*dataclass*, ***, *dest_cls: [Type](https://docs.python.org/3/library/typing.html#typing.Type) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *auto_batch_size: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *batch_dims: [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *as_tensorclass: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *device: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *batch_size: [Size](https://docs.pytorch.org/docs/stable/size.html#torch.Size) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*)

Converts a dataclass into a TensorDict instance.

Parameters:

**dataclass** - The dataclass instance to be converted.

Keyword Arguments:

- **dest_cls** ([*tensorclass*](tensordict.tensorclass.html#tensordict.tensorclass)*,**optional*) - A tensorclass type to be used to map the data. If not provided, a new
class is created. Without effect if `obj` is a type or as_tensorclass is False.
- **auto_batch_size** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)*,**optional*) - If `True`, automatically determines and applies batch size to the
resulting TensorDict. Defaults to `False`.
- **batch_dims** ([*int*](tensordict.TensorDictBase.int.html#tensordict.TensorDictBase.int)*,**optional*) - If `auto_batch_size` is `True`, defines how many dimensions the output
tensordict should have. Defaults to `None` (full batch-size at each level).
- **as_tensorclass** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)*,**optional*) - If `True`, delegates the conversion to the free function
[`from_dataclass()`](tensordict.from_dataclass.html#tensordict.from_dataclass) and returns a tensor-compatible class ([`tensorclass()`](tensordict.tensorclass.html#tensordict.tensorclass))
or instance instead of a TensorDict. Defaults to `False`.
- **device** ([*device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,**optional*) - The device on which the TensorDict will be created.
Defaults to `None`.
- **batch_size** ([*Size*](https://docs.pytorch.org/docs/stable/size.html#torch.Size)*,**optional*) - The batch size of the TensorDict.
Defaults to `None`.

Returns:

A TensorDict instance derived from the provided dataclass, unless as_tensorclass is True, in which case a tensor-compatible class or instance is returned.

Raises:

[**TypeError**](https://docs.python.org/3/builtins/exceptions.html#TypeError) - If the provided input is not a dataclass instance.

Warning

This method is distinct from the free function from_dataclass and serves a different purpose.
While the free function returns a tensor-compatible class or instance, this method returns a TensorDict instance.

Note

- This method creates a new TensorDict instance with keys corresponding to the fields of the input dataclass.
- Each key in the resulting TensorDict is initialized using the cls.from_any method.
- The auto_batch_size option allows for automatic batch size determination and application to the
resulting TensorDict.

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).