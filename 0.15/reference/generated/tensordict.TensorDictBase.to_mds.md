# TensorDictBase.to_mds

TensorDictBase.to_mds(***, *columns: [dict](https://docs.python.org/3/builtins/stdtypes.html#dict)[[str](https://docs.python.org/3/builtins/stdtypes.html#str), [str](https://docs.python.org/3/builtins/stdtypes.html#str)] | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *out: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [tuple](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[str](https://docs.python.org/3/builtins/stdtypes.html#str), [str](https://docs.python.org/3/builtins/stdtypes.html#str)]*, *keep_local: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *compression: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *hashes: [list](https://docs.python.org/3/builtins/stdtypes.html#list)[[str](https://docs.python.org/3/builtins/stdtypes.html#str)] | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *size_limit: [int](https://docs.python.org/3/builtins/functions.html#int) | [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [None](https://docs.python.org/3/builtins/constants.html#None) = 67108864*, *writer: 'streaming.MDSWriter' | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, ***kwargs*) → [None](https://docs.python.org/3/builtins/constants.html#None)

Writes the content of a TensorCollection to a streaming dataset.

Keyword Arguments:

- **out** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*|**Tuple**[*[*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,*[*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*]*) - 

Output dataset directory to save shard files.

1. If `out` is a local directory, shard files are saved locally.
2. If `out` is a remote directory, a local temporary directory is created to

cache the shard files and then the shard files are uploaded to a remote
location. At the end, the temp directory is deleted once shards are uploaded.
3. If `out` is a tuple of `(local_dir, remote_dir)`, shard files are saved in the

local_dir and also uploaded to a remote location.
- **columns** ([*dict*](https://docs.python.org/3/builtins/stdtypes.html#dict)*[*[*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,*[*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*]*) - an optional dict of columns. Will be automatically inferred from
the tensor collection if not provided.
- **keep_local** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)) - If the dataset is uploaded, whether to keep the local dataset directory
or remove it after uploading. Defaults to `False`.
- **compression** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - Optional compression or compression:level. Defaults to
`None`.
- **hashes** (*List**[*[*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*]**,**optional*) - Optional list of hash algorithms to apply to shard files.
Defaults to `None`.
- **size_limit** (*Union**[*[*int*](tensordict.TensorDictBase.int.html#tensordict.TensorDictBase.int)*,*[*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*]**,**optional*) - Optional shard size limit, after which point to
start a new shard. If `None`, puts everything in one shard. Can specify bytes
human-readable format as well, for example `"100kb"` for 100 kilobyte
(100*1024) and so on. Defaults to `1 << 26`.
Ignored if `writer` is passed.
- **writer** (*MDSWriter**,**optional*) - the writer to use. If `None`, it is
created from `out` and the other keyword arguments.
- ****kwargs** (*Any*) - Additional settings for the Writer.

Note

The MDSWriter has limited support for nested dictionaries. The proper way to handle nested tensordicts
is to use the [`flatten_keys()`](tensordict.TensorDictBase.flatten_keys.html#tensordict.TensorDictBase.flatten_keys) method before writing, and [`unflatten_keys()`](tensordict.TensorDictBase.unflatten_keys.html#tensordict.TensorDictBase.unflatten_keys) method after reading.

Warning

This method requires mosaicml-streaming to be installed.

Warning

For non-tensor data, the type of the data must be fixed. The way tensordict recovers the data type is by
looking at the first element of the list. If it's None an error will be thrown. Otherwise all the
data in the list must have the same type (or be None if missing).

See also

See the Mosaic streaming library API at [https://docs.mosaicml.com/projects/streaming](https://docs.mosaicml.com/projects/streaming)

The following example writes a dataset and loads it in a PyTorch
dataloader. It needs `mosaicml-streaming`, so it is not run when the
documentation is tested.

Examples

```
>>> import tempfile
>>> from typing import Any
>>> from tensordict import TensorDict, LazyStackedTensorDict
>>> import torch
>>>
>>>
>>> td = LazyStackedTensorDict(
... TensorDict(a=0, b=1, c=torch.randn(2), d="a string"),
... TensorDict(a=0, b=1, c=torch.randn(3), d="another string"),
... TensorDict(a=0, b=1, c=torch.randn(3), d="yet another string"),
... )
>>>
>>> with tempfile.TemporaryDirectory() as tempdir: 
... # Create a dataset on one process / thread / node...
... td.to_mds(out=tempdir)
...
... # Create a dataloader
... from streaming import StreamingDataset
...
... # Load the dataset on another thread / process / node...
... dataset = StreamingDataset(local=tempdir, remote=None, batch_size=2)
...
... # Use the class `from_list` method as a collate_fn
... dl = torch.utils.data.DataLoader(dataset=dataset, batch_size=2, collate_fn=LazyStackedTensorDict.from_list)
... for batch in dl:
... print("batch", batch)
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).