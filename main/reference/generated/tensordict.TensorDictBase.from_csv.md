# TensorDictBase.from_csv

*classmethod*TensorDictBase.from_csv(*path*, ***, *auto_batch_size: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *batch_dims: [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *device: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *batch_size: [Size](https://docs.pytorch.org/docs/stable/size.html#torch.Size) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *separator: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *dtype: [dtype](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, ***kwargs*) → Self

Creates a TensorDict from a CSV file.

Requires either pandas or pyarrow to be installed.

Parameters:

**path** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*or**Path*) - Path to the CSV file.

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
this separator to create nested TensorDicts. Defaults to `None`.
- **dtype** ([*dtype*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype)*,**optional*) - If provided, all numeric columns
are cast to this dtype. Defaults to `None`.
- ****kwargs** - Additional keyword arguments forwarded to the CSV reader
(`pandas.read_csv` or `pyarrow.csv.read_csv`).

Returns:

A TensorDict representation of the CSV data.

Examples

```
>>> import os
>>> import tempfile
>>> import torch
>>> from tensordict import TensorDict
>>> data = TensorDict(obs=torch.arange(3.0), reward=torch.ones(3), batch_size=[3])
>>> with tempfile.TemporaryDirectory() as tmpdir:
... path = os.path.join(tmpdir, "data.csv")
... data.to_csv(path)
... td = TensorDict.from_csv(path)
... td32 = TensorDict.from_csv(path, dtype=torch.float32)
>>> td.batch_size, td["obs"].dtype, td32["obs"].dtype
(torch.Size([3]), torch.float64, torch.float32)
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).