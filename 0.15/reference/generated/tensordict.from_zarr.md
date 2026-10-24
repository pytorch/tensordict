# from_zarr

tensordict.from_zarr(*zarr_file*, ***, *mode: [str](https://docs.python.org/3/builtins/stdtypes.html#str) = 'r'*, *auto_batch_size: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *batch_dims: [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *batch_size: [Size](https://docs.pytorch.org/docs/stable/size.html#torch.Size) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *non_tensor_as_stack: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *allow_pickle: [bool](https://docs.python.org/3/builtins/functions.html#bool) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*) → [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase)

Opens a zarr store as a [`PersistentTensorDict`](tensordict.PersistentTensorDict.html#tensordict.PersistentTensorDict).

Requires `zarr>=3.1.1`. See
[`TensorDictBase.from_zarr`](tensordict.TensorDictBase.from_zarr.html#tensordict.TensorDictBase.from_zarr).

Parameters:

**zarr_file** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**path**or**zarr store*) - the path to the zarr store (a
directory), or a `zarr.abc.store.Store` instance.

Keyword Arguments:

- **mode** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - how the store is opened: `"r"` (read only),
`"r+"` (read and write), and so on. Defaults to `"r"`.
- **auto_batch_size** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, sets the batch size to
the largest leading shape that all the entries share. Exclusive
with `batch_size`. Defaults to `False`.
- **batch_dims** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - with `auto_batch_size`, the maximum
number of batch dimensions. Ignored without
`auto_batch_size`, with a `FutureWarning` when it would
change the result. Starting with TensorDict 0.17, the default
will change to turning on `auto_batch_size` when
`batch_dims` is given, as `from_dict` does. Defaults to
`None`.
- **batch_size** ([*Size*](https://docs.pytorch.org/docs/stable/size.html#torch.Size)*or*[*list*](https://docs.python.org/3/builtins/stdtypes.html#list)*of*[*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the batch size of
the result: the leading dimensions that all its entries share.
Exclusive with `auto_batch_size`. Defaults to `None`, which
reads the batch size that [`to_zarr()`](tensordict.TensorDictBase.to_zarr.html#tensordict.TensorDictBase.to_zarr)
stored, or computes it for a store written by another tool.
- **non_tensor_as_stack** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, a non-tensor
array that tensordict did not write reads as a
[`NonTensorStack`](tensordict.NonTensorStack.html#tensordict.NonTensorStack) of its rows when its leading
dimensions equal the batch size. Defaults to `False`.
- **allow_pickle** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - whether to decode pickled non-tensor
values. `False` rejects them. `True` permits them, only for
data you trust. `None` permits them with a `FutureWarning`.
Starting with TensorDict 0.17, the default will change to
`allow_pickle=False`. Defaults to `None`.

Returns:

A [`PersistentTensorDict`](tensordict.PersistentTensorDict.html#tensordict.PersistentTensorDict) backed by the store.