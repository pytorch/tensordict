# TensorDictBase.get_at

TensorDictBase.get_at(*key*, *index*)

TensorDictBase.get_at(*key*, *index*, *default*)

Get the value of a tensordict from the key key at the index idx.

Parameters:

- **key** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,*[*tuple*](https://docs.python.org/3/builtins/stdtypes.html#tuple)*of*[*str*](https://docs.python.org/3/builtins/stdtypes.html#str)) - key to be retrieved.
- **index** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,*[*slice*](https://docs.python.org/3/builtins/functions.html#slice)*,*[*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*,**iterable*) - index of the tensor.
- **default** ([*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) - default value to return if the key is
not present in the tensordict.

Returns:

indexed tensor.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict({"x": torch.arange(3)}, batch_size=[])
>>> td.get_at("x", index=1)
tensor(1)
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).

## Tutorials that use `get_at`

![](../../_images/sphx_glr_zarr_storage_thumb.png)

[Storing TensorDicts in zarr](../../tutorials/zarr_storage.html)

Storing TensorDicts in zarr