# TensorDictBase.from_schema

*classmethod*TensorDictBase.from_schema(*schema: [dict](https://docs.python.org/3/builtins/stdtypes.html#dict)[[str](https://docs.python.org/3/builtins/stdtypes.html#str), [tuple](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[list](https://docs.python.org/3/builtins/stdtypes.html#list)[[int](https://docs.python.org/3/builtins/functions.html#int)] | [Size](https://docs.pytorch.org/docs/stable/size.html#torch.Size), [dtype](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype)]]*, ***, *batch_size: [Sequence](https://docs.python.org/3/library/typing.html#typing.Sequence)[[int](https://docs.python.org/3/builtins/functions.html#int)] | [Size](https://docs.pytorch.org/docs/stable/size.html#torch.Size) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *storage: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *device=None*, ***kwargs*) → [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase)

Pre-allocate a zero-filled TensorDict from a schema.

Creates a [`TensorDictBase`](tensordict.TensorDictBase.html#tensordict.TensorDictBase) whose storage backend is selected
by `storage`. Each entry in `schema` maps a field name to an
`(element_shape, dtype)` pair; the full stored shape is
`[*batch_size, *element_shape]`.

Parameters:

**schema** - Mapping from field name to `(element_shape, dtype)`.
`element_shape` is the per-element shape (excluding
`batch_size`).

Keyword Arguments:

- **batch_size** - Overall batch dimensions prepended to every element
shape. Defaults to `()`.
- **storage** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*or*[*None*](https://docs.python.org/3/builtins/constants.html#None)) - 

Backend selector:

- `None` - plain [`TensorDict`](tensordict.TensorDict.html#tensordict.TensorDict) with regular tensors.
- `"memmap"` - memory-mapped tensors on disk.
Pass `prefix=<dir>` in *kwargs*.
- `"h5"` - HDF5 via [`PersistentTensorDict`](tensordict.PersistentTensorDict.html#tensordict.PersistentTensorDict).
Pass `filename=<path>` in *kwargs*.
- `"zarr"` - zarr (requires `zarr>=3.0`) via
[`PersistentTensorDict`](tensordict.PersistentTensorDict.html#tensordict.PersistentTensorDict). Pass `filename=<path or store>`
in *kwargs*.
- `"shared"` - CPU shared-memory tensors.
- `"redis"` / `"dragonfly"` - delegates to
`TensorDictStore.from_schema()`.
- **device** - Device for the resulting tensors (ignored by some
backends).
- ****kwargs** - Backend-specific arguments forwarded to the
underlying constructor (e.g. `prefix` for memmap,
`filename` for h5, `host`/`port` for redis).

Returns:

A new [`TensorDictBase`](tensordict.TensorDictBase.html#tensordict.TensorDictBase) subclass instance with
pre-allocated (zero-filled) keys.

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
... assert td_mm.is_memmap()
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