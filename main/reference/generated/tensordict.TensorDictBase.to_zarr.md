# TensorDictBase.to_zarr

TensorDictBase.to_zarr(*filename*, ***, *allow_pickle: [bool](https://docs.python.org/3/builtins/functions.html#bool) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, ***kwargs*) → [Any](https://docs.python.org/3/library/typing.html#typing.Any)

Writes the tensordict to a new zarr store and returns a tensordict backed by it.

Each leaf becomes an array of the store, and each nested tensordict a
group. An existing store at `filename` is overwritten. The batch
size and the names of the batch dimensions are saved in the store
attributes, so that [`from_zarr()`](tensordict.TensorDictBase.from_zarr.html#tensordict.TensorDictBase.from_zarr) restores them. Requires
`zarr>=3.1.1`. See [Save and load tensordicts](../../saving.html#saving).

Parameters:

- **filename** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**path**or**zarr store*) - the path of the zarr store (a
directory), or a `zarr.abc.store.Store` instance (e.g. a
`zarr.storage.ZipStore`).
- ****kwargs** - passed to [`zarr.Group.create_array()`](https://zarr.readthedocs.io/en/stable/api/zarr/group/#zarr.Group.create_array). By default
each tensor is stored as one uncompressed chunk. Pass
`chunks=...` or `compressors=...` to change that (for
example to read rows without loading the whole array, or to
compress on disk). `chunks` applies to the leading
dimensions of each leaf, and the trailing dimensions are left
whole: `chunks=(16,)` chunks every leaf along its first
dimension in blocks of 16.

Keyword Arguments:

**allow_pickle** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - the read policy of the returned
tensordict for pickled non-tensor values. `False` rejects
them. `None` and `True` read them, since this call wrote
them. Opening the store again with [`from_zarr()`](tensordict.TensorDictBase.from_zarr.html#tensordict.TensorDictBase.from_zarr) uses
the policy of that call. Defaults to `None`.

Returns:

A [`PersistentTensorDict`](tensordict.PersistentTensorDict.html#tensordict.PersistentTensorDict) backed by the new store.

Examples

```
>>> import os
>>> import tempfile
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict(a=torch.zeros(1000), b={"c": torch.zeros(1000, 3)}, batch_size=[1000])
>>> with tempfile.TemporaryDirectory() as tmpdir:
... td_zarr = td.to_zarr(os.path.join(tmpdir, "store.zarr"), chunks=(100,))
... print(type(td_zarr).__name__, td_zarr.batch_size, td_zarr["b", "c"].shape)
PersistentTensorDict torch.Size([1000]) torch.Size([1000, 3])
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).

## Tutorials that use `to_zarr`

![](../../_images/sphx_glr_serialization_speed_thumb.png)

[Benchmarking TensorDict serialization speed](../../tutorials/serialization_speed.html)

Benchmarking TensorDict serialization speed
![](../../_images/sphx_glr_zarr_storage_thumb.png)

[Storing TensorDicts in zarr](../../tutorials/zarr_storage.html)

Storing TensorDicts in zarr