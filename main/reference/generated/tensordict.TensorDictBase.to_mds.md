# TensorDictBase.to_mds

TensorDictBase.to_mds(***, *columns: [dict](https://docs.python.org/3/builtins/stdtypes.html#dict)[[str](https://docs.python.org/3/builtins/stdtypes.html#str), [str](https://docs.python.org/3/builtins/stdtypes.html#str)] | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *out: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [tuple](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[str](https://docs.python.org/3/builtins/stdtypes.html#str), [str](https://docs.python.org/3/builtins/stdtypes.html#str)]*, *keep_local: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *compression: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *hashes: [list](https://docs.python.org/3/builtins/stdtypes.html#list)[[str](https://docs.python.org/3/builtins/stdtypes.html#str)] | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *size_limit: [int](https://docs.python.org/3/builtins/functions.html#int) | [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [None](https://docs.python.org/3/builtins/constants.html#None) = 67108864*, *writer: 'streaming.MDSWriter' | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *finish_writer: [bool](https://docs.python.org/3/builtins/functions.html#bool) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, ***kwargs*) → [None](https://docs.python.org/3/builtins/constants.html#None)

Writes the tensordict to a MosaicML Streaming (MDS) dataset.

Each element along the first batch dimension is one sample of the
dataset; a tensordict without batch dimensions is one sample. Tensors are
written as NumPy arrays, and zero-dimensional ones as Python numbers.
Read the dataset back with `streaming.StreamingDataset`. See the
[streaming documentation](https://docs.mosaicml.com/projects/streaming).

The MDS writer has limited support for nested dictionaries: call
[`flatten_keys()`](tensordict.TensorDictBase.flatten_keys.html#tensordict.TensorDictBase.flatten_keys) before writing, and
[`unflatten_keys()`](tensordict.TensorDictBase.unflatten_keys.html#tensordict.TensorDictBase.unflatten_keys) after reading.

The type of a non-tensor column is read from its first sample, which
must not be `None`; the other samples must have the same type, or be
`None`.

Keyword Arguments:

- **columns** ([*dict*](https://docs.python.org/3/builtins/stdtypes.html#dict)*of**str to str**,**optional*) - the MDS encoding of each
column, for example `{"obs": "ndarray", "name": "str"}`.
Defaults to `None`, which infers them from the first sample.
- **out** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*or*[*tuple*](https://docs.python.org/3/builtins/stdtypes.html#tuple)*of*[*str*](https://docs.python.org/3/builtins/stdtypes.html#str)) - 

where to write the shard files:

- a local directory: the shards are written there;
- a remote location: the shards are written to a local temporary
directory, uploaded, and deleted;
- a `(local_dir, remote_dir)` tuple: the shards are written to
`local_dir` and uploaded to `remote_dir`.
- **keep_local** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True` and the dataset is uploaded,
keeps the local copy. Defaults to `False`.
- **compression** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - the compression, as `"name"` or
`"name:level"`. Defaults to `None`.
- **hashes** ([*list*](https://docs.python.org/3/builtins/stdtypes.html#list)*of*[*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - the hash algorithms to apply to the
shard files. Defaults to `None`.
- **size_limit** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*or*[*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - the size after which a new shard
starts, in bytes or as a string such as `"100kb"`. `None`
writes one shard. Ignored if `writer` is passed. Defaults to
`1 << 26` (64 MiB).
- **writer** (*streaming.MDSWriter**,**optional*) - the writer to use. Defaults to
`None`, which creates one from `out` and the other keyword
arguments, and finishes it after writing.
- **finish_writer** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - whether to finish a writer passed as
`writer` after writing, which writes the dataset index. Pass
`False` to write more samples through it, and finish it
yourself afterwards, once. A writer that `to_mds` creates is
always finished. Defaults to `None`, which leaves the writer
open for a tensordict without batch dimensions, and finishes it,
with a `FutureWarning`, for a tensordict with batch dimensions.
Starting with TensorDict 0.17, the default will change to leaving
the writer open in both cases.
- ****kwargs** - passed to `streaming.MDSWriter`.

Raises:

- [**ImportError**](https://docs.python.org/3/builtins/exceptions.html#ImportError) - if `mosaicml-streaming` is not installed.
- [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) - if the tensordict is empty.

Examples

The example needs `mosaicml-streaming` (`pip install mosaicml-streaming`).

```
>>> import tempfile
>>> import torch
>>> from tensordict import LazyStackedTensorDict, TensorDict
>>> td = LazyStackedTensorDict(
... TensorDict(a=0, c=torch.zeros(2), d="a string"),
... TensorDict(a=1, c=torch.zeros(3), d="another string"),
... )
>>> from streaming import StreamingDataset 
>>> with tempfile.TemporaryDirectory() as tmpdir: 
... td.to_mds(out=tmpdir)
... dataset = StreamingDataset(local=tmpdir, remote=None, batch_size=2)
... loader = torch.utils.data.DataLoader(
... dataset, batch_size=2, collate_fn=LazyStackedTensorDict.from_list
... )
... batch = next(iter(loader)) # a lazy stack of the two samples
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).