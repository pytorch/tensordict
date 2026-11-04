Note

Go to the end
to download the full example code.

# Storing TensorDicts in zarr

**Author**: [Vincent Moens](https://github.com/vmoens)

[zarr](https://zarr.dev) splits each array into chunks that you can read,
write, and compress one at a time. The same tree of groups and arrays can go to
a local directory, a single zip file, or an S3-compatible object store. This
tutorial is for readers who store tensordicts outside the local disk, or who
share them with other zarr tools.

What you will learn:

- How to save and load tensordicts with
[`to_zarr()`](../reference/generated/tensordict.TensorDictBase.to_zarr.html#tensordict.TensorDictBase.to_zarr) and
[`from_zarr()`](../reference/generated/tensordict.TensorDictBase.from_zarr.html#tensordict.TensorDictBase.from_zarr).
- How to read slices of a store without loading it in memory.
- How to set chunks and compression.
- How to pack a store into a single zip file.
- How to pre-allocate a zarr buffer with
[`from_schema()`](../reference/generated/tensordict.TensorDictBase.from_schema.html#tensordict.TensorDictBase.from_schema).

The zarr backend needs `zarr>=3.1.1`. To install it, run
`pip install "tensordict[zarr]"`.

## Why zarr?

For local checkpoints, the memory-mapped format (see the
[saving guide](../saving.html#saving)) is faster. A leaf is a tensor at the end of a
nested key. The difference is largest for tensordicts with many small
leaves, because zarr pays one store round trip per array. Use zarr when the
data must be on object storage, when you need compression per chunk, or when
xarray or dask must read it. Both libraries read and write zarr.

```
import tempfile
from pathlib import Path

import torch
from tensordict import TensorDict

root = Path(tempfile.mkdtemp())
```

## Saving and loading

[`to_zarr()`](../reference/generated/tensordict.TensorDictBase.to_zarr.html#tensordict.TensorDictBase.to_zarr) writes the tensordict to a
store and returns a [`PersistentTensorDict`](../reference/generated/tensordict.PersistentTensorDict.html#tensordict.PersistentTensorDict) that reads
from it, as [`to_h5()`](../reference/generated/tensordict.TensorDictBase.to_h5.html#tensordict.TensorDictBase.to_h5) does for HDF5.

```
td = TensorDict(
 {
 "images": torch.randn(128, 3, 32, 32),
 "labels": torch.randint(10, (128,)),
 "meta": {"description": "a small dataset"},
 },
 batch_size=[128],
)
td_zarr = td.to_zarr(root / "dataset.zarr")
td_zarr
```

**PersistentTensorDict** batch_size=[128]

- images: f32[3,32,32]
- labels: i64[]
- meta: PersistentTensorDict 1 entry

- description: NonTensorData

The store keeps the batch size, the dimension names, and the non-tensor
entries (like the `"description"` string above).
[`from_zarr()`](../reference/generated/tensordict.TensorDictBase.from_zarr.html#tensordict.TensorDictBase.from_zarr) reads them back from the
store:

```
td_back = TensorDict.from_zarr(root / "dataset.zarr")
assert td_back.batch_size == torch.Size([128])
assert td_back["meta", "description"] == "a small dataset"
```

Stores written by other tools (xarray, or zarr itself) do not have the
tensordict metadata. For these stores, `from_zarr` infers the batch size
from the leading dimensions that all the arrays share, as `from_h5` does.

## Lazy access

`from_zarr` does not read tensor data. It reads an entry from the store
when you access it. [`get_at()`](../reference/generated/tensordict.TensorDictBase.get_at.html#tensordict.TensorDictBase.get_at) reads a part
of an entry. Use it to sample from a dataset that does not fit in memory:

```
row = td_back.get_at("images", 3)
batch = td_back.get_at("images", slice(0, 16))
picks = td_back.get_at("images", torch.tensor([1, 17, 121]))
```

## Chunking

By default, tensordict writes each tensor as one uncompressed chunk. For
checkpoints that you save and load whole, this is the fastest layout. It
also makes the timings comparable with the other formats (see the
[serialization speed tutorial](serialization_speed.html#sphx-glr-tutorials-serialization-speed-py)).
With one chunk, `get_at` must read the whole array. To read rows of a
dataset, pass `chunks=`. tensordict passes it to
[`zarr.Group.create_array()`](https://zarr.readthedocs.io/en/stable/api/zarr/group/#zarr.Group.create_array). A row read then reads only the chunks that
hold the row:

```
td.to_zarr(root / "chunked.zarr", chunks=(16, 3, 32, 32))
td_chunked = TensorDict.from_zarr(root / "chunked.zarr")
# this now reads a single 16-image chunk from disk:
row = td_chunked.get_at("images", 3)
```

## Compression

Pass any zarr codec with `compressors=`. zarr compresses each chunk on its
own. With chunks, the store is smaller and still supports partial reads.

```
from zarr.codecs import ZstdCodec

td.to_zarr(root / "compressed.zarr", chunks=(16, 3, 32, 32), compressors=ZstdCodec())
```

**PersistentTensorDict** batch_size=[128]

- images: f32[3,32,32]
- labels: i64[]
- meta: PersistentTensorDict 1 entry

- description: NonTensorData

## Alternative stores

The `filename` argument accepts any `zarr.abc.store.Store`. A
`ZipStore` packs the whole tree into one file that you can copy or send.
Write a zip store once, and read it many times: a zip file cannot replace an
entry.

Note

`to_zarr` writes the metadata file (`zarr.json`) of each group once,
with its attributes. `PersistentTensorDict.from_dict` still writes some
of them twice in two cases. With a dict and no `batch_size`, it writes
the `zarr.json` of each group twice: it finds the batch size after
writing the data. With `names=`, it writes the `zarr.json` of each
nested group twice: it sets the names after writing the data. A zip file
cannot replace an entry, so the zip gets a second entry with the same
name, and the `zipfile` module of Python warns
`UserWarning: Duplicate name: 'zarr.json'`
([#2252](https://github.com/pytorch/tensordict/issues/2252)). Reading
the store works: `zipfile` returns the last entry. To avoid it, pass a
`TensorDict` that has its batch size and names, for instance
`TensorDict(data, batch_size=[4], names=["t"])`.

```
from zarr.storage import ZipStore

store = ZipStore(root / "dataset.zarr.zip", mode="w")
td.to_zarr(store).close()

td_zip = TensorDict.from_zarr(ZipStore(root / "dataset.zarr.zip", mode="r"))
assert (td_zip["labels"] == td["labels"]).all()
td_zip.close()
```

Remote object stores work in the same way, through the fsspec support of
zarr. To write to S3-compatible storage, pass
`zarr.storage.FsspecStore.from_url("s3://bucket/dataset.zarr")` as
`filename`.

## Pre-allocation

Like the other backends, zarr supports
[`from_schema()`](../reference/generated/tensordict.TensorDictBase.from_schema.html#tensordict.TensorDictBase.from_schema). Give the keys, shapes, and
dtypes first, then fill the buffer row by row. The full dataset is never in
memory.

```
buffer = TensorDict.from_schema(
 {"obs": ([4], torch.float32), "reward": ([], torch.float32)},
 batch_size=[1000],
 storage="zarr",
 filename=root / "buffer.zarr",
)
for i in range(4):
 buffer[i] = TensorDict(
 {"obs": torch.randn(4), "reward": torch.randn(())}, batch_size=[]
 )
```

## Conclusion

You can now save a tensordict to zarr and load it again, and read parts of
its entries with `get_at` without loading the store. You can set chunks
and compression, pack a store into a zip file, and fill a pre-allocated
buffer. The zarr store is a [`PersistentTensorDict`](../reference/generated/tensordict.PersistentTensorDict.html#tensordict.PersistentTensorDict), like
an HDF5 file. For local checkpoints, use the memory-mapped format instead.

## Further reading

- The [storage guide](../storage.html#storage) compares all the backends and lists
the keyword arguments for zarr.
- The [serialization speed tutorial](serialization_speed.html#sphx-glr-tutorials-serialization-speed-py) benchmarks zarr against
the other on-disk formats.
- The [zarr documentation](https://zarr.readthedocs.io) covers stores,
codecs, and the v3 format in depth.

**Total running time of the script:** (0 minutes 0.129 seconds)

[`Download Jupyter notebook: zarr_storage.ipynb`](../_downloads/299d79c6946d13f056a08f21d56a241b/zarr_storage.ipynb)

[`Download Python source code: zarr_storage.py`](../_downloads/ef55a7451f47acbe75e5c29ec192d2a3/zarr_storage.py)

[`Download zipped: zarr_storage.zip`](../_downloads/517d21c4840328354a5d7984d2cde6e1/zarr_storage.zip)

[Gallery generated by Sphinx-Gallery](https://sphinx-gallery.github.io)