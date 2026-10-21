# TensorDictBase.from_h5

*classmethod*TensorDictBase.from_h5(*filename*, ***, *mode: [str](https://docs.python.org/3/builtins/stdtypes.html#str) = 'r'*, *auto_batch_size: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *batch_dims: [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *batch_size: [Size](https://docs.pytorch.org/docs/stable/size.html#torch.Size) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *non_tensor_as_stack: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*)

Opens an HDF5 file as a [`PersistentTensorDict`](tensordict.PersistentTensorDict.html#tensordict.PersistentTensorDict).

The data stays in the file: entries are read when they are accessed.
Requires `h5py`. See [Save and load tensordicts](../../saving.html#saving) and
[`PersistentTensorDict`](tensordict.PersistentTensorDict.html#tensordict.PersistentTensorDict).

Parameters:

**filename** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*or**path*) - the path to the HDF5 file.

Keyword Arguments:

- **mode** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - how the file is opened, as in
`h5py.File`: `"r"` (read only), `"r+"` (read and
write), and so on. Defaults to `"r"`.
- **auto_batch_size** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, sets the batch size
to the largest leading shape that all the entries share.
Exclusive with `batch_size`. Defaults to `False`.
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
gives each nested tensordict the largest batch size that its
entries share.
- **non_tensor_as_stack** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, a non-tensor
dataset that tensordict did not write, such as a string column
with one string per row, reads as a
[`NonTensorStack`](tensordict.NonTensorStack.html#tensordict.NonTensorStack) of its rows when its
leading dimensions equal the batch size. Otherwise it reads as
one [`NonTensorData`](tensordict.NonTensorData.html#tensordict.NonTensorData) that holds the whole
dataset. Defaults to `False`.

Returns:

A [`PersistentTensorDict`](tensordict.PersistentTensorDict.html#tensordict.PersistentTensorDict) backed by the file.

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
... path = os.path.join(tmpdir, "file.h5")
... _ = data.to_h5(path)
... td = TensorDict.from_h5(path)
... print(td)
PersistentTensorDict(batch_size=[3], fields={key1: f32[], key2: f32[]})
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).