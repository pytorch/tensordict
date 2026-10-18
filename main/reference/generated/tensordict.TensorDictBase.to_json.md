# TensorDictBase.to_json

TensorDictBase.to_json(*path*, ***, *separator: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *lines: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, ***kwargs*)

Writes the tensordict to a JSON file, as an array of records.

Each element of the tensordict becomes one record, with one field per
leaf, as in [`to_pandas()`](tensordict.TensorDictBase.to_pandas.html#tensordict.TensorDictBase.to_pandas). With pandas installed, the file is
written with `pandas.DataFrame.to_json()` (`orient="records"`),
and otherwise with the [`json`](https://docs.python.org/3/library/json.html#module-json) module. [`from_json()`](tensordict.TensorDictBase.from_json.html#tensordict.TensorDictBase.from_json) reads
the file back.

Parameters:

**path** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*or**Path*) - the path of the JSON file.

Keyword Arguments:

- **separator** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - if given, joins the keys of nested
entries with it to build the column names: with
`separator="."`, the entry `("obs", "x")` becomes the
column `"obs.x"`. Required when the tensordict has nested
tensordicts. Defaults to `None`.
- **lines** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, writes JSON Lines, one record
per line. Defaults to `False`.
- ****kwargs** - passed to `pandas.DataFrame.to_json()`. Ignored
without pandas.

Examples

```
>>> import os
>>> import tempfile
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict(a=torch.arange(2), b=TensorDict(c=torch.zeros(2)), batch_size=[2])
>>> with tempfile.TemporaryDirectory() as tmpdir:
... path = os.path.join(tmpdir, "data.json")
... td.to_json(path, separator=".")
... with open(path) as f:
... print(f.read())
[{"a":0,"b.c":0.0},{"a":1,"b.c":0.0}]
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).