# TensorDictBase.from_tuple

*classmethod*TensorDictBase.from_tuple(*obj*, ***, *auto_batch_size: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *batch_dims: [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *device: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *batch_size: [Size](https://docs.pytorch.org/docs/stable/size.html#torch.Size) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*)

Converts a tuple to a TensorDict.

Parameters:

**obj** - The tuple instance to be converted.

Keyword Arguments:

- **auto_batch_size** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)*,**optional*) - If `True`, the batch size will be computed automatically. Defaults to `False`.
- **batch_dims** ([*int*](tensordict.TensorDictBase.int.html#tensordict.TensorDictBase.int)*,**optional*) - If auto_batch_size is `True`, defines how many dimensions the output tensordict
should have. Defaults to `None` (full batch-size at each level).
- **device** ([*device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,**optional*) - The device on which the TensorDict will be created. Defaults to `None`.
- **batch_size** ([*Size*](https://docs.pytorch.org/docs/stable/size.html#torch.Size)*,**optional*) - The batch size of the TensorDict. Defaults to `None`.

Returns:

A TensorDict representation of the input tuple.

Examples

```
>>> from tensordict import TensorDict
>>> my_tuple = (1, 2, 3)
>>> td = TensorDict.from_tuple(my_tuple)
>>> print(td)
TensorDict(
 fields={
 0: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int64, is_shared=False),
 1: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int64, is_shared=False),
 2: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int64, is_shared=False)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False)
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).