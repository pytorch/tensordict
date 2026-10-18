# from_consolidated

tensordict.from_consolidated(*filename*)

Reconstructs a tensordict from a file written by `consolidate(filename=...)`.

A consolidated file is the single file that
[`consolidate()`](tensordict.TensorDictBase.consolidate.html#tensordict.TensorDictBase.consolidate) writes when you pass it a
`filename`. Same as [`tensordict.TensorDict.from_consolidated()`](tensordict.TensorDictBase.from_consolidated.html#tensordict.TensorDictBase.from_consolidated).

The batch sizes and names are as they were when the file was written.
To write later changes into the file, call
`consolidate(filename=filename)` again on the consolidated tensordict.

Parameters:

**filename** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*or**Path*) - the path of the file.

Returns:

A [`TensorDict`](tensordict.TensorDict.html#tensordict.TensorDict). Its entries are views into one
storage that is read from the file.

Examples

```
>>> import tempfile
>>> import torch
>>> import tensordict
>>> from tensordict import TensorDict
>>> td = TensorDict(a=torch.zeros(3), b={"c": torch.ones(3)}, batch_size=[3])
>>> with tempfile.TemporaryDirectory() as tmpdir:
... _ = td.consolidate(filename=f"{tmpdir}/td.bin")
... loaded = tensordict.from_consolidated(f"{tmpdir}/td.bin")
... print(loaded["b", "c"])
tensor([1., 1., 1.])
```