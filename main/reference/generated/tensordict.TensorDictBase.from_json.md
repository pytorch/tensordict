# TensorDictBase.from_json

*classmethod*TensorDictBase.from_json(*path*, ***, *auto_batch_size: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *batch_dims: [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *device: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *batch_size: [Size](https://docs.pytorch.org/docs/stable/size.html#torch.Size) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *separator: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *dtype: [dtype](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *lines: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, ***kwargs*) → Self

Reads a JSON file of records into a tensordict, with one entry per field.

The file holds an array of records (`[{"a": 1, "b": 2.0}, ...]`),
or one record per line with `lines=True` (JSON Lines). Each field
becomes an entry with one value per record. The file is read with
`pandas.read_json()` when pandas is installed, and with the
[`json`](https://docs.python.org/3/library/json.html#module-json) module otherwise. Through pandas, a string field is one
value for the whole batch (see the note in [`from_pandas()`](tensordict.TensorDictBase.from_pandas.html#tensordict.TensorDictBase.from_pandas));
through the [`json`](https://docs.python.org/3/library/json.html#module-json) module, it is a
[`NonTensorStack`](tensordict.NonTensorStack.html#tensordict.NonTensorStack) with one value per record. To
convert a single JSON object, load
it with [`json.load()`](https://docs.python.org/3/library/json.html#json.load) and pass it to [`from_dict()`](tensordict.TensorDictBase.from_dict.html#tensordict.TensorDictBase.from_dict).
[`to_json()`](tensordict.TensorDictBase.to_json.html#tensordict.TensorDictBase.to_json) writes such a file.

Parameters:

**path** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*or**Path*) - the path of the JSON file.

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
- **lines** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, reads the file as JSON Lines,
one record per line. Defaults to `False`.
- ****kwargs** - passed to `pandas.read_json()`. Ignored without
pandas.

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
... path = os.path.join(tmpdir, "data.json")
... data.to_json(path)
... td = TensorDict.from_json(path)
... path_lines = os.path.join(tmpdir, "data.jsonl")
... data.to_json(path_lines, lines=True)
... td_lines = TensorDict.from_json(path_lines, lines=True)
>>> td.batch_size, td_lines.batch_size
(torch.Size([3]), torch.Size([3]))
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).