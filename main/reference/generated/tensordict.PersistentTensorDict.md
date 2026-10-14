# PersistentTensorDict

`PersistentTensorDict` has the methods of [`TensorDictBase`](tensordict.TensorDictBase.html#tensordict.TensorDictBase),
listed by task in [Methods by task](../td.html#api-tensordict-methods). This page lists only the
methods that `PersistentTensorDict` adds, and the overrides that have their own
documentation.

*class*tensordict.PersistentTensorDict(***, *batch_size=None*, *filename=None*, *group=None*, *mode='r'*, *backend='h5'*, *device=None*, *validate_batch_size: [bool](https://docs.python.org/3/builtins/functions.html#bool) = True*, *non_tensor_as_stack: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, ***kwargs*)

Bases: `_NoShapeOps`, `_NoVmap`, `_NoSetNonTensor`, [`TensorDictBase`](tensordict.TensorDictBase.html#tensordict.TensorDictBase)

A tensordict whose entries live in an HDF5 file or a Zarr store.

The HDF5 backend (`backend="h5"`) needs `h5py`. The Zarr backend
(`backend="zarr"`) needs `zarr>=3.0`. Each read of an entry loads it
from the file into a new tensor, so `ptd["a"] is ptd["a"]` is
`False`, and in-place changes to the tensor that you read do not reach
the file. Writes (`ptd["a"] = value`) go to the file, and so do the
in-place methods of the tensordict, such as `ptd.add_(1)` and
`ptd.update_(other)`.

Use it to read or write existing HDF5 or Zarr data. To keep tensordict
data on disk for fast reloads, use [`memmap()`](tensordict.TensorDictBase.memmap.html#tensordict.TensorDictBase.memmap)
instead. [Large datasets and storage backends](../../storage.html#storage) compares the storage backends, and
[Which container should I use?](../../choosing_a_container.html#choosing-a-container) compares the containers.

Open a file with [`from_h5()`](tensordict.from_h5.html#tensordict.from_h5) or [`from_zarr()`](tensordict.from_zarr.html#tensordict.from_zarr). Create one from
data with [`from_dict()`](tensordict.from_dict.html#tensordict.from_dict). Call `close()` when you are done: the
file stays open when the object goes out of scope. To load every entry
into memory, call [`to_tensordict()`](tensordict.TensorDictBase.to_tensordict.html#tensordict.TensorDictBase.to_tensordict).

Like other [`TensorDictBase`](tensordict.TensorDictBase.html#tensordict.TensorDictBase) subclasses,
`PersistentTensorDict` has a `device` attribute. This does not
mean that the data is stored on that device: the entries are cast to it
when they are read.

With the h5 backend, a string comes back as `bytes`. Starting with
TensorDict 0.17, the default will change to `str` for the strings that
tensordict wrote, which it marks with an attribute; reading one as
`bytes` warns until then. Strings that other tools wrote stay `bytes`.

Keyword Arguments:

- **batch_size** ([*Size*](https://docs.pytorch.org/docs/stable/size.html#torch.Size)*or**compatible*) - the tensordict batch size.
Defaults to `torch.Size(())`.
- **filename** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - the path to the h5 file or zarr store (for the
zarr backend, a `zarr.abc.store.Store` instance is also accepted).
Exclusive with `group`.
- **group** (*h5py.Group**or*[*Group*](https://zarr.readthedocs.io/en/stable/api/zarr/group/#zarr.Group)*,**optional*) - a file or a group that
contains data. Exclusive with `filename`.
- **mode** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - how the file is opened, as in `h5py.File`:
`"r"` (read only), `"r+"` (read and write, the file must
exist), `"w"` (create, or truncate an existing file), `"w-"`
or `"x"` (create, fail if the file exists), `"a"` (read and
write, create the file if needed). Defaults to `"r"`.
- **backend** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - storage backend, `"h5"` (requires `h5py`) or
`"zarr"` (requires `zarr>=3.0`). Defaults to `"h5"`.
- **device** ([*device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*or**compatible**,**optional*) - the device that the
entries are cast to when they are read. Defaults to `None`: the
entries are read on CPU and `ptd.device` is `None`.
- **validate_batch_size** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)*,**optional*) - if `True`, the batch size is
checked against the stored array shapes at construction time (this
requires iterating over the store metadata). Defaults to `True`.
- **non_tensor_as_stack** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)*,**optional*) - if `True`, a non-tensor array
that tensordict did not write, such as a string column written by
another tool, reads as a
[`NonTensorStack`](tensordict.NonTensorStack.html#tensordict.NonTensorStack) of its rows when its leading
dims equal the batch size, so that indexing the tensordict selects
its rows. Otherwise it reads as one
[`NonTensorData`](tensordict.NonTensorData.html#tensordict.NonTensorData) that holds the whole array.
Defaults to `False`.
- ****kwargs** - kwargs to be passed to `h5py.File.create_dataset()` (h5
backend) or [`zarr.Group.create_array()`](https://zarr.readthedocs.io/en/stable/api/zarr/group/#zarr.Group.create_array) (zarr backend). With the
zarr backend, arrays are stored as a single chunk without compression
by default; pass `chunks=...` and/or `compressors=...` to override.

Note

PersistentTensorDict instances are not closed when they go out of
scope. Call `close()` to close the file.

Examples

```
>>> import tempfile
>>> import torch
>>> from tensordict import PersistentTensorDict
>>> with tempfile.NamedTemporaryFile() as f:
... data = PersistentTensorDict(filename=f.name, batch_size=[3], mode="w")
... data["a", "b"] = torch.randn(3, 4)
... print(data)
PersistentTensorDict(
 fields={
 a: PersistentTensorDict(
 fields={
 b: Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([3]),
 device=None,
 is_shared=False)},
 batch_size=torch.Size([3]),
 device=None,
 is_shared=False)
```

close()

Closes the persistent tensordict.

contiguous(***, *canonical: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*)

Materializes a PersistentTensorDict on a regular TensorDict.

flatten_keys(*separator: [str](https://docs.python.org/3/builtins/stdtypes.html#str) = '.'*, *inplace: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*) → T

Returns an in-memory copy where the nested keys are joined by `separator`.

The content is first loaded with [`to_tensordict()`](tensordict.TensorDictBase.to_tensordict.html#tensordict.TensorDictBase.to_tensordict).

Parameters:

- **separator** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - the separator between the nested items. Defaults to `"."`.
- **inplace** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - must be `False`; `True` raises a `ValueError`.
Defaults to `False`.

See [`flatten_keys()`](tensordict.TensorDictBase.flatten_keys.html#tensordict.TensorDictBase.flatten_keys).

*classmethod*from_dict(*input_dict*, *filename*, ***, *auto_batch_size: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *batch_size=None*, *device=None*, *backend: [str](https://docs.python.org/3/builtins/stdtypes.html#str) = 'h5'*, ***kwargs*)

Converts a dictionary or a TensorDict to a h5 file or zarr store.

Parameters:

- **input_dict** ([*dict*](https://docs.python.org/3/builtins/stdtypes.html#dict)*,*[*TensorDict*](tensordict.TensorDict.html#tensordict.TensorDict)*or**compatible*) - data to be stored.
- **filename** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*or**path*) - path to the h5 file or zarr store.

Keyword Arguments:

- **auto_batch_size** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)*,**optional*) - not used. The batch size comes
from `batch_size`, or from the input as described there,
whatever the value of this argument
([#2247](https://github.com/pytorch/tensordict/issues/2247)). Defaults
to `False`.
- **batch_size** (*tensordict batch-size**,**optional*) - if provided, batch size
of the tensordict. If not, a tensordict input keeps its batch
size, and a dict input gets the largest leading shape that all
its entries share.
- **device** ([*device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*or**compatible**,**optional*) - the device where to
expect the tensor once they are returned. Defaults to `None`
(on cpu by default).
- **backend** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - storage backend, `"h5"` or `"zarr"`.
Defaults to `"h5"`.
- ****kwargs** - kwargs to be passed to `h5py.File.create_dataset()`
(h5 backend) or [`zarr.Group.create_array()`](https://zarr.readthedocs.io/en/stable/api/zarr/group/#zarr.Group.create_array) (zarr backend).

Returns:

A `PersistentTensorDict` instance linked to the newly created file.

Examples

```
>>> import os
>>> import tempfile
>>> import torch
>>> from tensordict import PersistentTensorDict
>>> with tempfile.TemporaryDirectory() as tmpdir:
... ptd = PersistentTensorDict.from_dict(
... {"a": torch.zeros(3, 4), "b": {"c": torch.ones(3)}},
... os.path.join(tmpdir, "data.h5"),
... )
... print(ptd.batch_size, ptd["b", "c"])
... ptd.close()
torch.Size([3]) tensor([1., 1., 1.])
```

*classmethod*from_h5(*filename*, ***, *mode='r'*, *batch_size: torch.size | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *non_tensor_as_stack: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*)

Creates a PersistentTensorDict from a h5 file.

This function will automatically determine the batch-size for each nested tensordict (unless `batch_size`
is provided).

Parameters:

**filename** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)) - The path to the h5 file.

Keyword Arguments:

- **mode** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - how the file is opened: `"r"` (read only),
`"r+"` (read and write), or another mode of `h5py.File`
(see `PersistentTensorDict`). Defaults to `"r"`.
- **batch_size** ([*Size*](https://docs.pytorch.org/docs/stable/size.html#torch.Size)*,**optional*) - The batch size of the TensorDict. Defaults to None (batch-size automatically
determined).
- **non_tensor_as_stack** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)*,**optional*) - if `True`, a non-tensor
dataset that tensordict did not write, such as a string column
with one string per row, reads as a
[`NonTensorStack`](tensordict.NonTensorStack.html#tensordict.NonTensorStack) of its rows when its
leading dims equal the batch size. Otherwise it reads as one
[`NonTensorData`](tensordict.NonTensorData.html#tensordict.NonTensorData) that holds the whole
dataset. Defaults to `False`.

Returns:

A PersistentTensorDict representation of the input h5 file.

Examples

```
>>> import os
>>> import tempfile
>>> import torch
>>> from tensordict import PersistentTensorDict
>>> with tempfile.TemporaryDirectory() as tmpdir:
... filename = os.path.join(tmpdir, "file.h5")
... PersistentTensorDict.from_dict(
... {"key1": torch.zeros(3), "key2": torch.ones(3)}, filename
... ).close()
... ptd = PersistentTensorDict.from_h5(filename)
... print(ptd)
... ptd.close()
PersistentTensorDict(
 fields={
 key1: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False),
 key2: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([3]),
 device=None,
 is_shared=False)
```

*classmethod*from_zarr(*filename*, ***, *mode='r'*, *batch_size: [Size](https://docs.pytorch.org/docs/stable/size.html#torch.Size) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *non_tensor_as_stack: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*)

Creates a PersistentTensorDict from a zarr store.

The batch size (and dimension names) are read from the
`"__tensordict__"` attribute written by `from_dict()` /
[`to_zarr()`](tensordict.TensorDictBase.to_zarr.html#tensordict.TensorDictBase.to_zarr) when present; otherwise the
batch size of each nested tensordict is determined automatically
(unless `batch_size` is provided).

Parameters:

**filename** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**path**or**zarr store*) - the path to the zarr store
(a directory), or a `zarr.abc.store.Store` instance.

Keyword Arguments:

- **mode** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - Reading mode. Defaults to `"r"`.
- **batch_size** ([*Size*](https://docs.pytorch.org/docs/stable/size.html#torch.Size)*,**optional*) - The batch size of the TensorDict.
Defaults to `None` (batch-size read from metadata or
automatically determined).
- **non_tensor_as_stack** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)*,**optional*) - if `True`, a non-tensor
array that tensordict did not write, such as a string column
with one string per row, reads as a
[`NonTensorStack`](tensordict.NonTensorStack.html#tensordict.NonTensorStack) of its rows when its
leading dims equal the batch size. Otherwise it reads as one
[`NonTensorData`](tensordict.NonTensorData.html#tensordict.NonTensorData) that holds the whole
array. Defaults to `False`.

Returns:

A PersistentTensorDict representation of the input zarr store.

Examples

```
>>> import os
>>> import tempfile
>>> import torch
>>> from tensordict import PersistentTensorDict
>>> with tempfile.TemporaryDirectory() as tmpdir:
... store = os.path.join(tmpdir, "store.zarr")
... PersistentTensorDict.from_dict(
... {"key1": torch.zeros(3), "key2": torch.ones(3)},
... store,
... batch_size=[3],
... backend="zarr",
... ).close()
... ptd = PersistentTensorDict.from_zarr(store)
... print(ptd)
... ptd.close()
PersistentTensorDict(
 fields={
 key1: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False),
 key2: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([3]),
 device=None,
 is_shared=False)
```

get_at(*key: [NestedKey](tensordict.NestedKey.html#tensordict.NestedKey)*, *idx: [None](https://docs.python.org/3/builtins/constants.html#None) | [int](https://docs.python.org/3/builtins/functions.html#int) | [slice](https://docs.python.org/3/builtins/functions.html#slice) | [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | [List](https://docs.python.org/3/library/typing.html#typing.List)[[Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[Any](https://docs.python.org/3/library/typing.html#typing.Any), ...]*, *default: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | TensorCollection = NO_DEFAULT*) → [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | TensorCollection

Gets the value of the entry `key` at the index `idx`.

Parameters:

- **key** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,*[*tuple*](https://docs.python.org/3/builtins/stdtypes.html#tuple)*of*[*str*](https://docs.python.org/3/builtins/stdtypes.html#str)) - key to be retrieved.
- **idx** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,*[*slice*](https://docs.python.org/3/builtins/functions.html#slice)*,*[*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*,**iterable*) - index of the entry.
- **default** ([*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*,**optional*) - default value to return if the key is
not present in the tensordict. If not provided, a missing key
raises a `KeyError`.

See [`get_at()`](tensordict.TensorDictBase.get_at.html#tensordict.TensorDictBase.get_at).

*property*kwargs*: [dict](https://docs.python.org/3/builtins/stdtypes.html#dict)[[str](https://docs.python.org/3/builtins/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)]*

The keyword arguments that the tensordict passes to the backend when it creates a dataset.

Deprecated since version 0.15: This is internal and has no replacement.

## Tutorials that use `PersistentTensorDict`

![](../../_images/sphx_glr_zarr_storage_thumb.png)

[Storing TensorDicts in zarr](../../tutorials/zarr_storage.html)

Storing TensorDicts in zarr