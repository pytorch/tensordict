# load_memmap

tensordict.load_memmap(*prefix: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [Path](https://docs.python.org/3/library/pathlib.html#pathlib.Path)*, *device: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *non_blocking: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, ***, *out: TensorCollection | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *robust_key: [bool](https://docs.python.org/3/builtins/functions.html#bool) | [None](https://docs.python.org/3/builtins/constants.html#None) = True*, *subpath: [NestedKey](tensordict.NestedKey.html#tensordict.NestedKey) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *mode: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *num_threads: [int](https://docs.python.org/3/builtins/functions.html#int) = 0*, *allow_pickle: [bool](https://docs.python.org/3/builtins/functions.html#bool) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*) → TensorCollection

Loads a memory-mapped tensordict from disk.

Same as [`load()`](tensordict.load.html#tensordict.load), and as
[`tensordict.TensorDict.load_memmap()`](tensordict.TensorDictBase.load_memmap.html#tensordict.TensorDictBase.load_memmap), which holds the full text of
every argument. See [Save and load tensordicts](../../saving.html#saving) for the ways to save and load a
tensordict.

Parameters:

- **prefix** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*or**Path*) - the directory, or the `.tdz` archive file,
that holds the saved tensordict.
- **device** ([*device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,**optional*) - if given, the data is cast to that
device. With `"meta"`, no file is read.
- **non_blocking** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, the cast to `device`
is not followed by a synchronization. Defaults to `False`.

Keyword Arguments:

- **out** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*,**optional*) - a tensordict to load the data into.
- **robust_key** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)*,**optional*) - must match the value used to save.
Defaults to `True`.
- **subpath** ([*NestedKey*](tensordict.NestedKey.html#tensordict.NestedKey)*or*[*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - the nested tensordict to load,
as a nested key (`("module", "0")`) or a `"/"`-separated path
(`"module/0"`). Only that subtree is read.
- **mode** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - `"r"` maps the files copy-on-write (in-place
writes stay in memory), `"r+"` maps them shared (in-place writes
reach the files). See [`tensordict.TensorDict.load_memmap()`](tensordict.TensorDictBase.load_memmap.html#tensordict.TensorDictBase.load_memmap)
for the default.
- **num_threads** ([*int*](tensordict.TensorDictBase.int.html#tensordict.TensorDictBase.int)*,**optional*) - the number of threads that decompress
the leaves of a compressed archive. It has no effect on other
data. Defaults to `0`.
- **allow_pickle** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)*,**optional*) - if `True`, non-tensor entries that
were pickled are unpickled. Only load pickles from trusted
sources.

Returns:

the saved tensordict, of the class it had when it was saved. Its
leaves are [`MemoryMappedTensor`](tensordict.MemoryMappedTensor.html#tensordict.MemoryMappedTensor) instances.

Examples

```
>>> import tempfile
>>> import torch
>>> import tensordict
>>> from tensordict import TensorDict
>>> td = TensorDict(a=torch.zeros(3), batch_size=[3])
>>> with tempfile.TemporaryDirectory() as tmpdir:
... _ = tensordict.memmap(td, tmpdir)
... loaded = tensordict.load_memmap(tmpdir)
... print(type(loaded["a"]).__name__)
MemoryMappedTensor
```