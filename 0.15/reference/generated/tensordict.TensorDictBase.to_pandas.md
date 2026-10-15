# TensorDictBase.to_pandas

TensorDictBase.to_pandas(***, *separator: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*)

Converts this TensorDict to a pandas DataFrame.

Each leaf key becomes a column. Tensor values are converted to numpy
arrays, [`NonTensorData`](tensordict.NonTensorData.html#tensordict.NonTensorData) values are converted to
Python lists.

Keyword Arguments:

**separator** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - If provided, nested keys are joined
with this separator to produce flat column names. For example,
`td["obs", "x"]` becomes column `"obs.x"` with
`separator="."`. Required when the TensorDict contains
nested sub-TensorDicts. Defaults to `None`.

Returns:

A pandas DataFrame.

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