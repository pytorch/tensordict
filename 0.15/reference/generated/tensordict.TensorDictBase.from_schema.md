# TensorDictBase.from_schema

*classmethod*TensorDictBase.from_schema(*schema: [dict](https://docs.python.org/3/builtins/stdtypes.html#dict)[[str](https://docs.python.org/3/builtins/stdtypes.html#str), [tuple](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[list](https://docs.python.org/3/builtins/stdtypes.html#list)[[int](https://docs.python.org/3/builtins/functions.html#int)] | [Size](https://docs.pytorch.org/docs/stable/size.html#torch.Size), [dtype](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype)]]*, ***, *batch_size: [Sequence](https://docs.python.org/3/library/typing.html#typing.Sequence)[[int](https://docs.python.org/3/builtins/functions.html#int)] | [Size](https://docs.pytorch.org/docs/stable/size.html#torch.Size) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *storage: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *device=None*, ***kwargs*) → [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase)

Allocates a zero-filled tensordict from a schema of shapes and dtypes.

Each entry of `schema` maps a key to an `(element_shape, dtype)`
pair, and the entry gets the shape `[*batch_size, *element_shape]`.
`storage` selects where the zeros are stored, so that a buffer on
disk or on a server can be created without building it in memory
first.

Parameters:

**schema** ([*dict*](https://docs.python.org/3/builtins/stdtypes.html#dict)) - maps each key (a `str`) to a pair
`(element_shape, dtype)`, where `element_shape` (a list of
int or a `torch.Size`) is the shape of the entry after the
batch dimensions.

Keyword Arguments:

- **batch_size** ([*Size*](https://docs.pytorch.org/docs/stable/size.html#torch.Size)*or*[*list*](https://docs.python.org/3/builtins/stdtypes.html#list)*of*[*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the batch size of
the result, prepended to every element shape. Defaults to
`None`, an empty batch size.
- **storage** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - 

where the entries are stored:

- `None`: a [`TensorDict`](tensordict.TensorDict.html#tensordict.TensorDict) of regular
tensors.
- `"memmap"`: memory-mapped tensors, in the directory given
by `prefix=<dir>` (see [`memmap_like()`](tensordict.TensorDictBase.memmap_like.html#tensordict.TensorDictBase.memmap_like)).
- `"h5"`: a [`PersistentTensorDict`](tensordict.PersistentTensorDict.html#tensordict.PersistentTensorDict) in the
HDF5 file given by `filename=<path>`.
- `"zarr"`: a [`PersistentTensorDict`](tensordict.PersistentTensorDict.html#tensordict.PersistentTensorDict) in
the zarr store given by `filename=<path or store>`
(requires `zarr>=3.0`).
- `"shared"`: tensors in shared memory (see
[`share_memory_()`](tensordict.TensorDictBase.share_memory_.html#tensordict.TensorDictBase.share_memory_)).
- `"redis"` or `"dragonfly"`: a
[`TensorDictStore`](tensordict.store.TensorDictStore.html#tensordict.store.TensorDictStore) on a server,
through [`TensorDictStore.from_schema`](tensordict.store.TensorDictStore.html#tensordict.store.TensorDictStore.from_schema).

Defaults to `None`.
- **device** ([*device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,*[*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*or*[*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the device of the
result. The `"memmap"` storage ignores it. Defaults to
`None`.
- ****kwargs** - passed to the backend: `prefix` and the other
arguments of [`memmap_like()`](tensordict.TensorDictBase.memmap_like.html#tensordict.TensorDictBase.memmap_like) for `"memmap"`,
`filename` for `"h5"` and `"zarr"`, `host`, `port`
and the like for `"redis"` and `"dragonfly"`.

Returns:

A new tensordict of the type that `storage` selects, filled with
zeros.

Raises:

[**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) - if `storage` is not one of the values above.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict.from_schema(
... {"obs": ([84, 84, 3], torch.uint8),
... "reward": ([], torch.float32)},
... batch_size=[1000],
... )
>>> td["obs"].shape
torch.Size([1000, 84, 84, 3])
```

```
>>> import tempfile
>>> with tempfile.TemporaryDirectory() as d:
... td_mm = TensorDict.from_schema(
... {"obs": ([4], torch.float32)},
... batch_size=[8],
... storage="memmap",
... prefix=d,
... )
... print(td_mm.is_memmap())
True
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).

## Tutorials that use `from_schema`

![](../../_images/sphx_glr_tensordict_preallocation_thumb.png)

[Filling a TensorDict row by row](../../tutorials/tensordict_preallocation.html)

Filling a TensorDict row by row
![](../../_images/sphx_glr_zarr_storage_thumb.png)

[Storing TensorDicts in zarr](../../tutorials/zarr_storage.html)

Storing TensorDicts in zarr