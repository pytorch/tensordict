# TensorDictBase.to_pandas

TensorDictBase.to_pandas(***, *separator: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*)

Converts the tensordict to a pandas DataFrame, with one column per leaf.

Each leaf must hold one value per row: a tensor of shape
`[num_rows]`, or non-tensor data. Tensors are copied to the CPU and
converted to NumPy arrays. Non-tensor data becomes a column of Python
values. Requires pandas.

Keyword Arguments:

**separator** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - if given, joins the keys of nested
entries with it to build the column names: with
`separator="."`, the entry `("obs", "x")` becomes the
column `"obs.x"`. Required when the tensordict has nested
tensordicts. Defaults to `None`.

Returns:

A `pd.DataFrame` with one row per element of the tensordict.

Raises:

[**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) - if the tensordict has a nested tensordict and no
 `separator`, or if a tensor has more than one dimension.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict({"a": torch.arange(3), "b": torch.zeros(3)}, [3])
>>> df = td.to_pandas()
>>> print(df)
 a b
0 0 0.0
1 1 0.0
2 2 0.0
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).