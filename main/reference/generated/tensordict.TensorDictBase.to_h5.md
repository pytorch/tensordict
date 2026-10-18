# TensorDictBase.to_h5

TensorDictBase.to_h5(*filename*, ***kwargs*) → [Any](https://docs.python.org/3/library/typing.html#typing.Any)

Writes the tensordict to a new HDF5 file and returns a tensordict backed by it.

Each leaf becomes a dataset of the file, and each nested tensordict a
group. An existing file at `filename` is overwritten. The batch size
and the names of the batch dimensions are kept. Requires `h5py`.
[`from_h5()`](tensordict.TensorDictBase.from_h5.html#tensordict.TensorDictBase.from_h5) opens the file again. See [Save and load tensordicts](../../saving.html#saving).

Parameters:

- **filename** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*or**path*) - the path of the HDF5 file.
- ****kwargs** - passed to `h5py.Group.create_dataset()`, for example
`compression="gzip"`.

Returns:

A [`PersistentTensorDict`](tensordict.PersistentTensorDict.html#tensordict.PersistentTensorDict) backed by the new file.

Examples

```
>>> import os
>>> import tempfile
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict(a=torch.zeros(1000), b={"c": torch.zeros(1000, 3)}, batch_size=[1000])
>>> with tempfile.TemporaryDirectory() as tmpdir:
... td_h5 = td.to_h5(os.path.join(tmpdir, "data.h5"), compression="gzip")
... print(type(td_h5).__name__, td_h5.batch_size, td_h5["b", "c"].shape)
... td_h5.close()
PersistentTensorDict torch.Size([1000]) torch.Size([1000, 3])
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).

## Tutorials that use `to_h5`

![](../../_images/sphx_glr_zarr_storage_thumb.png)

[Storing TensorDicts in zarr](../../tutorials/zarr_storage.html)

Storing TensorDicts in zarr