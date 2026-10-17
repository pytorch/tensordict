# TensorDictBase.to_zarr

TensorDictBase.to_zarr(*filename*, ***, *allow_pickle: [bool](https://docs.python.org/3/builtins/functions.html#bool) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, ***kwargs*) → [Any](https://docs.python.org/3/library/typing.html#typing.Any)

Converts a tensordict to a PersistentTensorDict with the zarr backend.

Requires `zarr>=3.0` to be installed. The batch size (and dimension
names) are persisted in the store attributes such that
[`from_zarr()`](tensordict.TensorDictBase.from_zarr.html#tensordict.TensorDictBase.from_zarr) restores them without inference.

Parameters:

- **filename** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**path**or**zarr store*) - path to the zarr store (a
directory), or a `zarr.abc.store.Store` instance (e.g. a
`zarr.storage.ZipStore`).
- ****kwargs** - kwargs to be passed to [`zarr.Group.create_array()`](https://zarr.readthedocs.io/en/stable/api/zarr/group/#zarr.Group.create_array).
By default each tensor is stored as a single uncompressed chunk;
pass `chunks=...` and/or `compressors=...` to override (e.g.
for out-of-core row access or on-disk compression). Since leaves
have different ranks, `chunks` constrains the leading
dimensions of each leaf and trailing dimensions are left whole
(e.g. `chunks=(16,)` chunks every leaf along its first
dimension in blocks of 16).

Keyword Arguments:

**allow_pickle** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)*or*[*None*](https://docs.python.org/3/builtins/constants.html#None)*,**optional*) - read policy for the returned
handle. `None` (default) trusts the values just written, as
does `True`. `False` rejects reads of pickled non-tensor
values. Reopening the store with [`from_zarr()`](tensordict.TensorDictBase.from_zarr.html#tensordict.TensorDictBase.from_zarr) uses that
call's policy, not the writer's.

Returns:

A [`PersistentTensorDict`](tensordict.PersistentTensorDict.html#tensordict.PersistentTensorDict) instance linked to the newly created store.

Examples

```
>>> import tempfile
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict({
... "a": torch.zeros(1000),
... "b": {"c": torch.zeros(1000, 3)},
... }, [1000])
>>> td_zarr = td.to_zarr(tempfile.mkdtemp() + "/store.zarr")
>>> print(td_zarr)
PersistentTensorDict(
 fields={
 a: Tensor(shape=torch.Size([1000]), device=cpu, dtype=torch.float32, is_shared=False),
 b: PersistentTensorDict(
 fields={
 c: Tensor(shape=torch.Size([1000, 3]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([1000]),
 device=None,
 is_shared=False)},
 batch_size=torch.Size([1000]),
 device=None,
 is_shared=False)
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).

## Tutorials that use `to_zarr`

![](../../_images/sphx_glr_serialization_speed_thumb.png)

[Benchmarking TensorDict serialization speed](../../tutorials/serialization_speed.html)

Benchmarking TensorDict serialization speed
![](../../_images/sphx_glr_zarr_storage_thumb.png)

[Storing TensorDicts in zarr](../../tutorials/zarr_storage.html)

Storing TensorDicts in zarr