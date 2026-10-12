# from_dataclass

tensordict.from_dataclass(*obj: [Any](https://docs.python.org/3/library/typing.html#typing.Any)*, ***, *dest_cls: [Type](https://docs.python.org/3/library/typing.html#typing.Type) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *auto_batch_size: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *batch_dims: [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *batch_size: [Size](https://docs.pytorch.org/docs/stable/size.html#torch.Size) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *frozen: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *autocast: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *nocast: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *inplace: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *shadow: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *tensor_only: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *device: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*) → [Any](https://docs.python.org/3/library/typing.html#typing.Any)

Converts a dataclass instance or a type into a tensorclass instance or type, respectively.

This function takes a dataclass instance or a dataclass type and converts it into a tensor-compatible class,
optionally applying various configurations such as auto-batching, immutability, and type casting.

Parameters:

**obj** (*Any*) - The dataclass instance or type to be converted. If a type is provided, a new class is returned.

Keyword Arguments:

- **dest_cls** ([*tensorclass*](tensordict.tensorclass.html#tensordict.tensorclass)*,**optional*) - A tensorclass type to be used to map the data. If not provided, a new
class is created. Without effect if `obj` is a type.
- **auto_batch_size** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)*,**optional*) - If `True`, automatically determines and applies batch size to the resulting object. Defaults to `False`.
- **batch_dims** ([*int*](tensordict.TensorDictBase.int.html#tensordict.TensorDictBase.int)*,**optional*) - If auto_batch_size is `True`, defines how many dimensions the output tensordict should have. Defaults to `None` (full batch-size at each level).
- **batch_size** ([*Size*](https://docs.pytorch.org/docs/stable/size.html#torch.Size)*,**optional*) - The batch size of the TensorDict. Defaults to `None`.
- **frozen** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)*,**optional*) - If `True`, the resulting class or instance will be immutable. Defaults to `False`.
- **autocast** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)*,**optional*) - If `True`, enables automatic type casting for the resulting class or instance. Defaults to `False`.
- **nocast** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)*,**optional*) - If `True`, disables any type casting for the resulting class or instance. Defaults to `False`.
- **tensor_only** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)*,**optional*) - if `True`, it is expected that all items in tensorclass will be
tensor instances (tensor-compatible, since non-tensor data is converted to tensors if possible).
This can bring significant speed-ups at the cost of flexible interactions with non-tensor data.
Defaults to `False`.
- **inplace** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)*,**optional*) - If `True`, the dataclass type passed will be modified in-place. Defaults to `False`.
Without effect if an instance is provided.
- **device** ([*device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,**optional*) - The device on which the TensorDict will be created. Defaults to `None`.
- **shadow** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)*,**optional*) - Disables the validation of field names against the names of the tensorclass
members (e.g. `batch_size` or `sum`). Use with caution, as this may cause unintended consequences.
Defaults to False.

Returns:

A tensor-compatible class or instance derived from the provided dataclass.

Raises:

[**TypeError**](https://docs.python.org/3/builtins/exceptions.html#TypeError) - If the provided input is not a dataclass instance or type.

Examples

```
>>> from dataclasses import dataclass
>>> import torch
>>> from tensordict.tensorclass import from_dataclass
>>>
>>> @dataclass
... class X:
... a: int
... b: torch.Tensor
...
>>> x = X(0, 0)
>>> x2 = from_dataclass(x)
>>> print(x2)
X_tc(
 a=Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int64, is_shared=False),
 b=Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int64, is_shared=False),
 batch_size=torch.Size([]),
 device=None,
 is_shared=False)
>>> X2 = from_dataclass(X, autocast=True)
>>> print(X2(a=0, b=0))
X_tc(
 a=NonTensorData(data=0, batch_size=torch.Size([]), device=None),
 b=Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int64, is_shared=False),
 batch_size=torch.Size([]),
 device=None,
 is_shared=False)
```

Note

If a dataclass type is provided, a new class is returned with the specified configurations.
If a dataclass instance is provided, a new instance of the tensor-compatible class is returned.
The auto_batch_size, frozen, autocast, and nocast options allow for flexible configuration of the resulting class or instance.

Warning

Whereas [`from_dataclass()`](tensordict.TensorDictBase.from_dataclass.html#tensordict.TensorDictBase.from_dataclass) will return a [`TensorDict`](tensordict.TensorDict.html#tensordict.TensorDict) instance
by default, this method will return a tensorclass instance or type.