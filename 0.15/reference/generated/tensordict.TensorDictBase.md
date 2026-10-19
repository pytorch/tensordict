# TensorDictBase

The methods of `TensorDictBase` are shared by every tensordict class and
by tensorclasses. They are listed by task in [Methods by task](../td.html#api-tensordict-methods),
with one page for each method.

*class*tensordict.TensorDictBase(**args*, ***kwargs*)

The abstract base class of every tensordict.

[`TensorDict`](tensordict.TensorDict.html#tensordict.TensorDict), [`LazyStackedTensorDict`](tensordict.LazyStackedTensorDict.html#tensordict.LazyStackedTensorDict),
[`PersistentTensorDict`](tensordict.PersistentTensorDict.html#tensordict.PersistentTensorDict) and the other tensordict classes
subclass it, and tensorclasses have the same methods. You do not create a
`TensorDictBase`: use it in type hints and in `isinstance` checks. A
tensorclass is not an instance of it: use
[`is_tensor_collection()`](tensordict.is_tensor_collection.html#tensordict.is_tensor_collection) to accept both. The methods are
listed by task in [Methods by task](../td.html#api-tensordict-methods).

Examples

```
>>> import torch
>>> from tensordict import TensorDict, TensorDictBase, lazy_stack
>>> td = TensorDict(a=torch.zeros(3), batch_size=[3])
>>> isinstance(td, TensorDictBase), isinstance(lazy_stack([td, td]), TensorDictBase)
(True, True)
```

## Tutorials that use `TensorDictBase`

![](../../_images/sphx_glr_data_fashion_thumb.png)

[Using TensorDict for datasets](../../tutorials/data_fashion.html)

Using TensorDict for datasets
![](../../_images/sphx_glr_functional_thumb.png)

[Functional calls and ensembles with TensorDict](../../tutorials/functional.html)

Functional calls and ensembles with TensorDict
![](../../_images/sphx_glr_serialization_speed_thumb.png)

[Benchmarking TensorDict serialization speed](../../tutorials/serialization_speed.html)

Benchmarking TensorDict serialization speed
![](../../_images/sphx_glr_streamed_tensordict_thumb.png)

[Building tensordicts from streams](../../tutorials/streamed_tensordict.html)

Building tensordicts from streams
![](../../_images/sphx_glr_tensordict_keys_thumb.png)

[Getting, setting, and nesting keys](../../tutorials/tensordict_keys.html)

Getting, setting, and nesting keys
![](../../_images/sphx_glr_tensordict_memory_thumb.png)

[Devices and memory-mapped tensors](../../tutorials/tensordict_memory.html)

Devices and memory-mapped tensors
![](../../_images/sphx_glr_tensordict_module_thumb.png)

[Building models with TensorDictModule](../../tutorials/tensordict_module.html)

Building models with TensorDictModule
![](../../_images/sphx_glr_tensordict_preallocation_thumb.png)

[Filling a TensorDict row by row](../../tutorials/tensordict_preallocation.html)

Filling a TensorDict row by row
![](../../_images/sphx_glr_tensordict_shapes_thumb.png)

[Changing the shape of a TensorDict](../../tutorials/tensordict_shapes.html)

Changing the shape of a TensorDict
![](../../_images/sphx_glr_tensordict_slicing_thumb.png)

[Slicing, indexing, and masking](../../tutorials/tensordict_slicing.html)

Slicing, indexing, and masking
![](../../_images/sphx_glr_zarr_storage_thumb.png)

[Storing TensorDicts in zarr](../../tutorials/zarr_storage.html)

Storing TensorDicts in zarr