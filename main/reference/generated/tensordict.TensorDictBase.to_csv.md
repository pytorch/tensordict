# TensorDictBase.to_csv

TensorDictBase.to_csv(*path*, ***, *separator: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, ***kwargs*)

Writes the tensordict to a CSV file, with one column per leaf.

The tensordict is converted with [`to_pandas()`](tensordict.TensorDictBase.to_pandas.html#tensordict.TensorDictBase.to_pandas), so the same
rules apply, and written with `pandas.DataFrame.to_csv()`
without the index. Requires pandas. [`from_csv()`](tensordict.TensorDictBase.from_csv.html#tensordict.TensorDictBase.from_csv) reads the
file back.

Parameters:

**path** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*or**Path*) - the path of the CSV file.

Keyword Arguments:

- **separator** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - if given, joins the keys of nested
entries with it to build the column names: with
`separator="."`, the entry `("obs", "x")` becomes the
column `"obs.x"`. Required when the tensordict has nested
tensordicts. Defaults to `None`.
- ****kwargs** - passed to `pandas.DataFrame.to_csv()`.

Raises:

[**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) - as [`to_pandas()`](tensordict.TensorDictBase.to_pandas.html#tensordict.TensorDictBase.to_pandas).

Examples

```
>>> import os
>>> import tempfile
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict(a=torch.arange(2), b=TensorDict(c=torch.zeros(2)), batch_size=[2])
>>> with tempfile.TemporaryDirectory() as tmpdir:
... path = os.path.join(tmpdir, "data.csv")
... td.to_csv(path, separator=".")
... with open(path) as f:
... print(f.read())
a,b.c
0,0.0
1,0.0
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).