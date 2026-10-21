# TensorDictBase.from_pandas

*classmethod*TensorDictBase.from_pandas(*dataframe*, ***, *auto_batch_size: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *batch_dims: [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *device: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *batch_size: [Size](https://docs.pytorch.org/docs/stable/size.html#torch.Size) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *separator: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *dtype: [dtype](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*) → Self

Converts a pandas DataFrame to a tensordict, with one entry per column.

The batch size is `[num_rows]`. Numeric and boolean columns become
tensors, a string or object column becomes one
[`NonTensorData`](tensordict.NonTensorData.html#tensordict.NonTensorData) that holds the whole column, a
categorical column becomes the tensor of its codes, and a datetime
column becomes an `int64` tensor of nanoseconds.
[`to_pandas()`](tensordict.TensorDictBase.to_pandas.html#tensordict.TensorDictBase.to_pandas) converts back.

Note

A string or object column is one value for the whole batch, so
`td[i]["name"]` gives the values of every row, not the value of
row `i`. To get one value per row, set the column again as a
list: `td["name"] = df["name"].tolist()` stores a
[`NonTensorStack`](tensordict.NonTensorStack.html#tensordict.NonTensorStack), whose `td[i]["name"]` is
the value of row `i`. It builds one object per row, about 20
microseconds each, so it is much slower than reading the column
for large frames.

Parameters:

**dataframe** (*pd.DataFrame*) - the DataFrame to convert.

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

Returns:

A tensordict of the class this method is called on
([`TensorDict`](tensordict.TensorDict.html#tensordict.TensorDict) for `TensorDictBase`).

Examples

```
>>> from tensordict import TensorDict
>>> import pandas as pd
>>> df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
>>> td = TensorDict.from_pandas(df)
>>> print(td)
TensorDict(batch_size=[3], fields={a: i64[], b: f64[]})
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).