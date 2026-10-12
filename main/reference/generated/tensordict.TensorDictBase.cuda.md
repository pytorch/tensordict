# TensorDictBase.cuda

TensorDictBase.cuda(*device: [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, ***kwargs*) → Self

Casts a tensordict to a cuda device (if not already on it).

Parameters:

**device** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - if provided, the cuda device on which the
tensor should be cast.

This function also supports all the keyword arguments of [`to()`](tensordict.TensorDictBase.to.html#tensordict.TensorDictBase.to).

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).

## Tutorials that use `cuda`

![](../../_images/sphx_glr_tensordict_memory_thumb.png)

[Devices and memory-mapped tensors](../../tutorials/tensordict_memory.html)

Devices and memory-mapped tensors