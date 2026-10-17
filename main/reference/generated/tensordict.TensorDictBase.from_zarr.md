# TensorDictBase.from_zarr

*classmethod*TensorDictBase.from_zarr(*filename*, ***, *mode: [str](https://docs.python.org/3/builtins/stdtypes.html#str) = 'r'*, *auto_batch_size: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *batch_dims: [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *batch_size: [Size](https://docs.pytorch.org/docs/stable/size.html#torch.Size) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *non_tensor_as_stack: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *allow_pickle: [bool](https://docs.python.org/3/builtins/functions.html#bool) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*)

Creates a PersistentTensorDict from a zarr store.

Requires `zarr>=3.0` to be installed.

Parameters:

**filename** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**path**or**zarr store*) - The path to the zarr store (a
directory), or a `zarr.abc.store.Store` instance (e.g. a
`zarr.storage.ZipStore`).

Keyword Arguments:

- **mode** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - Reading mode. Defaults to `"r"`.
- **auto_batch_size** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)*,**optional*) - If `True`, the batch size will be computed automatically.
Defaults to `False`.
- **batch_dims** ([*int*](tensordict.TensorDictBase.int.html#tensordict.TensorDictBase.int)*,**optional*) - If auto_batch_size is `True`, defines how many dimensions the output
tensordict should have. Defaults to `None` (full batch-size at each level).
- **batch_size** ([*Size*](https://docs.pytorch.org/docs/stable/size.html#torch.Size)*,**optional*) - The batch size of the TensorDict. Defaults to `None`
(read from the metadata written by [`to_zarr()`](tensordict.TensorDictBase.to_zarr.html#tensordict.TensorDictBase.to_zarr), or automatically determined
for stores written by other tools).
- **non_tensor_as_stack** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)*,**optional*) - if `True`, a non-tensor
array that tensordict did not write, such as a string column
with one string per row, reads as a
[`NonTensorStack`](tensordict.NonTensorStack.html#tensordict.NonTensorStack) of its rows when its
leading dims equal the batch size. Otherwise it reads as one
[`NonTensorData`](tensordict.NonTensorData.html#tensordict.NonTensorData) that holds the whole
array. Defaults to `False`.
- **allow_pickle** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)*or*[*None*](https://docs.python.org/3/builtins/constants.html#None)*,**optional*) - whether to decode pickled
non-tensor values. `False` rejects them before decoding;
`True` permits them only for data you trust. The default,
`None`, permits them with a `FutureWarning` when read.
Starting with TensorDict 0.17, the default will change to
`allow_pickle=False`. Tensor and JSON entries are unaffected.

Returns:

A PersistentTensorDict representation of the input zarr store.

Examples

```
>>> import os
>>> import tempfile
>>> import torch
>>> from tensordict import TensorDict
>>> data = TensorDict(key1=torch.zeros(3), key2=torch.ones(3), batch_size=[3])
>>> with tempfile.TemporaryDirectory() as tmpdir:
... path = os.path.join(tmpdir, "store.zarr")
... _ = data.to_zarr(path)
... td = TensorDict.from_zarr(path)
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

## Tutorials that use `from_zarr`

![](../../_images/sphx_glr_serialization_speed_thumb.png)

[Benchmarking TensorDict serialization speed](../../tutorials/serialization_speed.html)

Benchmarking TensorDict serialization speed
![](../../_images/sphx_glr_zarr_storage_thumb.png)

[Storing TensorDicts in zarr](../../tutorials/zarr_storage.html)

Storing TensorDicts in zarr