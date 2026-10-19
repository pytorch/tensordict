# TensorDictBase.to_parquet

TensorDictBase.to_parquet(*path*, ***, *separator: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, ***kwargs*)

Writes the tensordict to a Parquet file, with one column per leaf.

The columns are built as in [`to_pandas()`](tensordict.TensorDictBase.to_pandas.html#tensordict.TensorDictBase.to_pandas), so the same rules
apply. Requires pyarrow or pandas. When pyarrow is installed, the
file is written with `pyarrow.parquet.write_table`, from a pandas
DataFrame if pandas is installed and from the columns directly
otherwise. With pandas alone, the file is written with
`pandas.DataFrame.to_parquet()`, which then needs fastparquet.
[`from_parquet()`](tensordict.TensorDictBase.from_parquet.html#tensordict.TensorDictBase.from_parquet) reads the file back.

Parameters:

**path** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*or**Path*) - the path of the Parquet file.

Keyword Arguments:

- **separator** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - if given, joins the keys of nested
entries with it to build the column names: with
`separator="."`, the entry `("obs", "x")` becomes the
column `"obs.x"`. Required when the tensordict has nested
tensordicts. Defaults to `None`.
- ****kwargs** - passed to the Parquet writer.

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
... path = os.path.join(tmpdir, "data.parquet")
... td.to_parquet(path, separator=".")
... td2 = TensorDict.from_parquet(path, separator=".")
>>> td2["b", "c"]
tensor([0., 0.])
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).