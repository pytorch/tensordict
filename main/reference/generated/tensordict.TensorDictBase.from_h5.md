# TensorDictBase.from_h5

*classmethod*TensorDictBase.from_h5(*filename*, ***, *mode: [str](https://docs.python.org/3/builtins/stdtypes.html#str) = 'r'*, *auto_batch_size: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *batch_dims: [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *batch_size: [Size](https://docs.pytorch.org/docs/stable/size.html#torch.Size) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*)

Creates a PersistentTensorDict from a h5 file.

Parameters:

**filename** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)) - The path to the h5 file.

Keyword Arguments:

- **mode** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - Reading mode. Defaults to `"r"`.
- **auto_batch_size** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)*,**optional*) - If `True`, the batch size will be computed automatically.
Defaults to `False`.
- **batch_dims** ([*int*](tensordict.TensorDictBase.int.html#tensordict.TensorDictBase.int)*,**optional*) - If auto_batch_size is `True`, defines how many dimensions the output
tensordict should have. Defaults to `None` (full batch-size at each level).
- **batch_size** ([*Size*](https://docs.pytorch.org/docs/stable/size.html#torch.Size)*,**optional*) - The batch size of the TensorDict. Defaults to `None`.

Returns:

A PersistentTensorDict representation of the input h5 file.

Examples

```
>>> import os
>>> import tempfile
>>> import torch
>>> from tensordict import TensorDict
>>> data = TensorDict(key1=torch.zeros(3), key2=torch.ones(3), batch_size=[3])
>>> with tempfile.TemporaryDirectory() as tmpdir:
... path = os.path.join(tmpdir, "file.h5")
... _ = data.to_h5(path)
... td = TensorDict.from_h5(path)
... print(td)
PersistentTensorDict(
 fields={
 key1: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False),
 key2: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([3]),
 device=None,
 is_shared=False)
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).