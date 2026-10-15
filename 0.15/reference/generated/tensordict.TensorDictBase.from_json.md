# TensorDictBase.from_json

*classmethod*TensorDictBase.from_json(*path*, ***, *auto_batch_size: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *batch_dims: [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *device: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *batch_size: [Size](https://docs.pytorch.org/docs/stable/size.html#torch.Size) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *separator: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *dtype: [dtype](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *lines: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, ***kwargs*) → Self

Creates a TensorDict from a JSON file.

Supports both standard JSON (array of records) and JSON Lines format.
For nested JSON objects, use [`from_dict()`](tensordict.from_dict.html#tensordict.from_dict) instead.

Requires pandas for best results. Falls back to stdlib `json`
for simple cases.

Parameters:

**path** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*or**Path*) - Path to the JSON file.

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
- **lines** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)*,**optional*) - If `True`, reads the file as JSON Lines
(one JSON object per line). Defaults to `False`.
- ****kwargs** - Additional keyword arguments forwarded to the JSON
reader.

Returns:

A TensorDict representation of the JSON data.

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