# TensorCollection

*class*tensordict.TensorCollection(**args*, ***kwargs*)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

The abstract base class of tensordicts and tensorclasses.

[`TensorDictBase`](tensordict.TensorDictBase.html#tensordict.TensorDictBase) and [`TensorClass`](tensordict.TensorClass.html#tensordict.TensorClass)
inherit from it. It cannot be instantiated.

Use it in type hints and `isinstance` checks for "a tensordict or a
tensorclass". Classes decorated with `@tensorclass` do not inherit from
it, only subclasses of [`TensorClass`](tensordict.TensorClass.html#tensordict.TensorClass) do; to check both,
use [`is_tensor_collection()`](tensordict.is_tensor_collection.html#tensordict.is_tensor_collection).

Examples

```
>>> import torch
>>> from tensordict import TensorClass, TensorCollection, TensorDict
>>> class Data(TensorClass):
... x: torch.Tensor
>>> td = TensorDict(x=torch.zeros(3), batch_size=[3])
>>> isinstance(td, TensorCollection)
True
>>> isinstance(Data(x=torch.zeros(3), batch_size=[3]), TensorCollection)
True
>>> isinstance(torch.zeros(3), TensorCollection)
False
```

## Tutorials that use `TensorCollection`

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
![](../../_images/sphx_glr_tensorclass_fashion_thumb.png)

[Using tensorclasses for datasets](../../tutorials/tensorclass_fashion.html)

Using tensorclasses for datasets
![](../../_images/sphx_glr_tensorclass_imagenet_thumb.png)

[Batched data loading with tensorclasses](../../tutorials/tensorclass_imagenet.html)

Batched data loading with tensorclasses
![](../../_images/sphx_glr_tensorclass_particles_thumb.png)

[Simulating particles with a tensorclass](../../tutorials/tensorclass_particles.html)

Simulating particles with a tensorclass
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