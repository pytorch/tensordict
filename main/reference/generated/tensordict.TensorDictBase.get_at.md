# TensorDictBase.get_at

TensorDictBase.get_at(*key*, *index*)

TensorDictBase.get_at(*key*, *index*, *default*)

Returns the entry at `key`, indexed with `index`.

`td.get_at(key, index)` is the same as `td.get(key)[index]`, but a
tensordict stored on disk or in a store reads only the indexed part.

Parameters:

- **key** ([*NestedKey*](tensordict.NestedKey.html#tensordict.NestedKey)) - the key to read.
- **index** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,*[*slice*](https://docs.python.org/3/builtins/functions.html#slice)*,*[*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*or*[*tuple*](https://docs.python.org/3/builtins/stdtypes.html#tuple)) - the index to apply to
the entry.
- **default** (*Any**,**optional*) - the value returned if the key is missing.
Defaults to `None`.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict({"x": torch.arange(3)}, batch_size=[3])
>>> td.get_at("x", 1)
tensor(1)
>>> td.get_at("x", torch.tensor([True, False, True]))
tensor([0, 2])
>>> print(td.get_at("y", 1))
None
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).

## Tutorials that use `get_at`

![](../../_images/sphx_glr_zarr_storage_thumb.png)

[Storing TensorDicts in zarr](../../tutorials/zarr_storage.html)

Storing TensorDicts in zarr