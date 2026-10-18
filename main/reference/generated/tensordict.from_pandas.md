# from_pandas

tensordict.from_pandas(*dataframe*, ***, *auto_batch_size: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *batch_dims: [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *device: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *batch_size: [Size](https://docs.pytorch.org/docs/stable/size.html#torch.Size) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *separator: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *dtype: [dtype](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*) → [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase)

Converts a pandas DataFrame to a tensordict, with one entry per column.

See [`TensorDictBase.from_pandas`](tensordict.TensorDictBase.from_pandas.html#tensordict.TensorDictBase.from_pandas) for how each column type is
converted.

Parameters:

**dataframe** (*pd.DataFrame*) - the DataFrame to convert.

Keyword Arguments:

- **auto_batch_size** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, sets the batch size to
the largest leading shape that all the entries share. Exclusive
with `batch_size`. Defaults to `False`.
- **batch_dims** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - with `auto_batch_size`, the maximum
number of batch dimensions. Ignored without `auto_batch_size`.
Defaults to `None`.
- **device** ([*device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,*[*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*or*[*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the device of the
result. Leaves on another device are moved to it. `None` leaves
each leaf where it is, and the result has no device. Defaults to
`None`.
- **batch_size** ([*Size*](https://docs.pytorch.org/docs/stable/size.html#torch.Size)*or*[*list*](https://docs.python.org/3/builtins/stdtypes.html#list)*of*[*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the batch size of
the result: the leading dimensions that all its entries share.
Exclusive with `auto_batch_size`. Defaults to `None`, which
gives `[num_rows]`.
- **separator** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - if given, splits the column names on it
to build nested entries: with `separator="."`, the column
`"obs.x"` becomes the entry `("obs", "x")`. Defaults to
`None`.
- **dtype** ([*dtype*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype)*,**optional*) - if given, casts the numeric and
boolean columns to it. Defaults to `None`.

Returns:

A [`TensorDict`](tensordict.TensorDict.html#tensordict.TensorDict).

Examples

```
>>> import pandas as pd
>>> from tensordict import from_pandas
>>> df = pd.DataFrame({"obs.x": [1.0, 2.0], "obs.y": [3.0, 4.0], "done": [False, True]})
>>> td = from_pandas(df, separator=".")
>>> td["obs", "x"], td.batch_size
(tensor([1., 2.], dtype=torch.float64), torch.Size([2]))
```