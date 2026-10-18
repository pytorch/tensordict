# TensorDictBase.from_zarr

*classmethod*TensorDictBase.from_zarr(*filename*, ***, *mode: [str](https://docs.python.org/3/builtins/stdtypes.html#str) = 'r'*, *auto_batch_size: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *batch_dims: [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *batch_size: [Size](https://docs.pytorch.org/docs/stable/size.html#torch.Size) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *non_tensor_as_stack: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *allow_pickle: [bool](https://docs.python.org/3/builtins/functions.html#bool) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*)

Opens a zarr store as a [`PersistentTensorDict`](tensordict.PersistentTensorDict.html#tensordict.PersistentTensorDict).

The data stays in the store: entries are read when they are accessed.
Requires `zarr>=3.0`. See [Save and load tensordicts](../../saving.html#saving) and
[`PersistentTensorDict`](tensordict.PersistentTensorDict.html#tensordict.PersistentTensorDict).

Parameters:

**filename** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**path**or**zarr store*) - the path to the zarr store (a
directory), or a `zarr.abc.store.Store` instance (e.g. a
`zarr.storage.ZipStore`).

Keyword Arguments:

- **mode** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - how the store is opened: `"r"` (read
only), `"r+"` (read and write), and so on, as in
[`zarr.open_group()`](https://zarr.readthedocs.io/en/stable/api/zarr/functions/open_group/#zarr.open_group). Defaults to `"r"`.
- **auto_batch_size** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, sets the batch size
to the largest leading shape that all the entries share.
Exclusive with `batch_size`. Defaults to `False`.
- **batch_dims** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - with `auto_batch_size`, the maximum
number of batch dimensions. Ignored without
`auto_batch_size`. Defaults to `None`.
- **batch_size** ([*Size*](https://docs.pytorch.org/docs/stable/size.html#torch.Size)*or*[*list*](https://docs.python.org/3/builtins/stdtypes.html#list)*of*[*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the batch size of
the result: the leading dimensions that all its entries share.
Exclusive with `auto_batch_size`. Defaults to `None`, which
reads the batch size and the names of the batch dimensions that [`to_zarr()`](tensordict.TensorDictBase.to_zarr.html#tensordict.TensorDictBase.to_zarr) stored, or, for
a store written by another tool, gives each nested tensordict
the largest batch size that its entries share.
- **non_tensor_as_stack** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, a non-tensor
array that tensordict did not write, such as a string column
with one string per row, reads as a
[`NonTensorStack`](tensordict.NonTensorStack.html#tensordict.NonTensorStack) of its rows when its
leading dimensions equal the batch size. Otherwise it reads as
one [`NonTensorData`](tensordict.NonTensorData.html#tensordict.NonTensorData) that holds the whole
array. Defaults to `False`.
- **allow_pickle** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - whether to decode pickled
non-tensor values. `False` rejects them before decoding.
`True` permits them: use it only for data you trust.
`None` permits them with a `FutureWarning` when read.
Starting with TensorDict 0.17, the default will change to
`allow_pickle=False`. Tensor and JSON entries are unaffected.
Defaults to `None`.

Returns:

A [`PersistentTensorDict`](tensordict.PersistentTensorDict.html#tensordict.PersistentTensorDict) backed by the store.

Raises:

[**TypeError**](https://docs.python.org/3/builtins/exceptions.html#TypeError) - if both `auto_batch_size=True` and `batch_size`
 are given.

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