# from_h5

tensordict.from_h5(*h5_file*, ***, *mode: [str](https://docs.python.org/3/builtins/stdtypes.html#str) = 'r'*, *auto_batch_size: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *batch_dims: [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *batch_size: [Size](https://docs.pytorch.org/docs/stable/size.html#torch.Size) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *non_tensor_as_stack: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*) → [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase)

Opens an HDF5 file as a [`PersistentTensorDict`](tensordict.PersistentTensorDict.html#tensordict.PersistentTensorDict).

See [`TensorDictBase.from_h5`](tensordict.TensorDictBase.from_h5.html#tensordict.TensorDictBase.from_h5).

Parameters:

**h5_file** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*or**path*) - the path to the HDF5 file.

Keyword Arguments:

- **mode** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - how the file is opened, as in
`h5py.File`: `"r"` (read only), `"r+"` (read and
write), and so on. Defaults to `"r"`.
- **auto_batch_size** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, sets the batch size to
the largest leading shape that all the entries share. Exclusive
with `batch_size`. Defaults to `False`.
- **batch_dims** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - with `auto_batch_size`, the maximum
number of batch dimensions. Ignored without `auto_batch_size`.
Defaults to `None`.
- **batch_size** ([*Size*](https://docs.pytorch.org/docs/stable/size.html#torch.Size)*or*[*list*](https://docs.python.org/3/builtins/stdtypes.html#list)*of*[*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the batch size of
the result: the leading dimensions that all its entries share.
Exclusive with `auto_batch_size`. Defaults to `None`, which
gives each nested tensordict the largest batch size that its
entries share.
- **non_tensor_as_stack** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, a non-tensor
dataset that tensordict did not write reads as a
[`NonTensorStack`](tensordict.NonTensorStack.html#tensordict.NonTensorStack) of its rows when its leading
dimensions equal the batch size. Defaults to `False`.

Returns:

A [`PersistentTensorDict`](tensordict.PersistentTensorDict.html#tensordict.PersistentTensorDict) backed by the file.