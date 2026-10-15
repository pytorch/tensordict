# TensorDictBase.from_pandas

*classmethod*TensorDictBase.from_pandas(*dataframe*, ***, *auto_batch_size: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *batch_dims: [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *device: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *batch_size: [Size](https://docs.pytorch.org/docs/stable/size.html#torch.Size) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *separator: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *dtype: [dtype](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*) → Self

Converts a pandas DataFrame to a TensorDict.

Numeric columns become tensors, string/object columns become
[`NonTensorData`](tensordict.NonTensorData.html#tensordict.NonTensorData).

Parameters:

**dataframe** (*pd.DataFrame*) - The pandas DataFrame to convert.

Keyword Arguments:

- **auto_batch_size** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)*,**optional*) - If `True`, the batch size will
be computed automatically. Defaults to `False`.
- **batch_dims** ([*int*](tensordict.TensorDictBase.int.html#tensordict.TensorDictBase.int)*,**optional*) - If `auto_batch_size` is `True`,
defines how many dimensions the output tensordict should have.
Defaults to `None`.
- **device** ([*device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,**optional*) - The device for tensor data.
Defaults to `None`.
- **batch_size** ([*Size*](https://docs.pytorch.org/docs/stable/size.html#torch.Size)*,**optional*) - The batch size. Defaults to
`[num_rows]`.
- **separator** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - If provided, column names are split on
this separator to create nested TensorDicts. For example, with
`separator="."`, a column `"obs.x"` becomes
`td["obs", "x"]`. Defaults to `None`.
- **dtype** ([*dtype*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype)*,**optional*) - If provided, all numeric columns
are cast to this dtype. Defaults to `None`.

Returns:

A TensorDict representation of the DataFrame.

Examples

```
>>> from tensordict import TensorDict
>>> import pandas as pd
>>> df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
>>> td = TensorDict.from_pandas(df)
>>> print(td)
TensorDict(
 fields={
 a: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.int64, is_shared=False),
 b: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float64, is_shared=False)},
 batch_size=torch.Size([3]),
 device=None,
 is_shared=False)
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).