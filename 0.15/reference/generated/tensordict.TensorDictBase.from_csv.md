# TensorDictBase.from_csv

*classmethod*TensorDictBase.from_csv(*path*, ***, *auto_batch_size: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *batch_dims: [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *device: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *batch_size: [Size](https://docs.pytorch.org/docs/stable/size.html#torch.Size) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *separator: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *dtype: [dtype](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, ***kwargs*) → Self

Reads a CSV file into a tensordict, with one entry per column.

Requires pandas or pyarrow: the file is read with
`pandas.read_csv()` when pandas is installed, and with
`pyarrow.csv.read_csv` otherwise. The columns are converted as in
[`from_pandas()`](tensordict.TensorDictBase.from_pandas.html#tensordict.TensorDictBase.from_pandas). [`to_csv()`](tensordict.TensorDictBase.to_csv.html#tensordict.TensorDictBase.to_csv) writes such a file. Through
pandas, a string column is one value for the whole batch (see the
note in [`from_pandas()`](tensordict.TensorDictBase.from_pandas.html#tensordict.TensorDictBase.from_pandas)); through pyarrow, it is a
[`NonTensorStack`](tensordict.NonTensorStack.html#tensordict.NonTensorStack) with one value per row.

Parameters:

**path** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*or**Path*) - the path of the CSV file.

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
Exclusive with `auto_batch_size`. Defaults to `None`,
which gives `[num_rows]`.
- **separator** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - if given, splits the column names on it
to build nested entries: with `separator="."`, the column
`"obs.x"` becomes the entry `("obs", "x")`. Defaults to
`None`, which keeps the column names as keys.
- **dtype** ([*dtype*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype)*,**optional*) - if given, casts the numeric and
boolean columns to it. Defaults to `None`, which keeps the
dtype of each column.
- ****kwargs** - passed to the CSV reader.

Returns:

A tensordict of the class this method is called on
([`TensorDict`](tensordict.TensorDict.html#tensordict.TensorDict) for `TensorDictBase`).

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