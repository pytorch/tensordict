# TensorDictBase.to_h5

TensorDictBase.to_h5(*filename*, ***kwargs*) → [Any](https://docs.python.org/3/library/typing.html#typing.Any)

Converts a tensordict to a PersistentTensorDict with the h5 backend.

Parameters:

- **filename** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*or**path*) - path to the h5 file.
- ****kwargs** - kwargs to be passed to `h5py.File.create_dataset()`.

Returns:

A [`PersistentTensorDict`](tensordict.PersistentTensorDict.html#tensordict.PersistentTensorDict) instance linked to the newly created file.

Examples

```
>>> import torch
>>> import tempfile
>>> import timeit
>>>
>>> from tensordict import TensorDict, MemoryMappedTensor
>>> td = TensorDict({
... "a": MemoryMappedTensor.from_tensor(torch.zeros(()).expand(1_000_000)),
... "b": {"c": MemoryMappedTensor.from_tensor(torch.zeros(()).expand(1_000_000, 3))},
... }, [1_000_000])
>>>
>>> file = tempfile.NamedTemporaryFile()
>>> td_h5 = td.to_h5(file.name, compression="gzip", compression_opts=9)
>>> print(td_h5)
PersistentTensorDict(
 fields={
 a: Tensor(shape=torch.Size([1000000]), device=cpu, dtype=torch.float32, is_shared=False),
 b: PersistentTensorDict(
 fields={
 c: Tensor(shape=torch.Size([1000000, 3]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([1000000]),
 device=None,
 is_shared=False)},
 batch_size=torch.Size([1000000]),
 device=None,
 is_shared=False)
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).

## Tutorials that use `to_h5`

![](../../_images/sphx_glr_zarr_storage_thumb.png)

[Storing TensorDicts in zarr](../../tutorials/zarr_storage.html)

Storing TensorDicts in zarr