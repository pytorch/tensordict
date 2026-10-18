# TensorDictBase.from_consolidated

*classmethod*TensorDictBase.from_consolidated(*filename*)

Reconstructs a tensordict from a file written by `consolidate(filename=...)`.

The result is consolidated: its leaves are views into one
copy-on-write memory map of the file, so in-place writes into them do
not reach the file. It is not locked. The batch sizes and names are as
they were when the file was written. Same as
[`from_consolidated()`](tensordict.from_consolidated.html#tensordict.from_consolidated).

Parameters:

**filename** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*or**Path*) - the file to read.

Examples

```
>>> import tempfile
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict(a=torch.zeros(3), b={"c": torch.ones(3)}, batch_size=[3])
>>> with tempfile.TemporaryDirectory() as path:
... _ = td.consolidate(filename=f"{path}/td.bin")
... loaded = TensorDict.from_consolidated(f"{path}/td.bin")
... print(loaded["b", "c"], loaded.is_consolidated())
tensor([1., 1., 1.]) True
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).

## Tutorials that use `from_consolidated`

![](../../_images/sphx_glr_serialization_speed_thumb.png)

[Benchmarking TensorDict serialization speed](../../tutorials/serialization_speed.html)

Benchmarking TensorDict serialization speed