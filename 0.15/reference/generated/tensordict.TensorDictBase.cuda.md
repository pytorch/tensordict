# TensorDictBase.cuda

TensorDictBase.cuda(*device: [int](https://docs.python.org/3/builtins/functions.html#int) | [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, ***kwargs*) → Self

Returns the tensordict on a CUDA device, as [`to()`](tensordict.TensorDictBase.to.html#tensordict.TensorDictBase.to) does.

A tensordict that is already on that device is returned as it is.

Parameters:

- **device** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,*[*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*or*[*device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,**optional*) - the CUDA device. An
integer (or a string of digits) is a device index. Defaults to
`None`, which uses the current CUDA device.
- ****kwargs** - passed to [`to()`](tensordict.TensorDictBase.to.html#tensordict.TensorDictBase.to), for example `non_blocking`.

Raises:

[**RuntimeError**](https://docs.python.org/3/builtins/exceptions.html#RuntimeError) - if `device` is not a CUDA device.

Examples

The example needs a CUDA device, so it is not run when the
documentation is tested.

```
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict(a=torch.zeros(3), batch_size=[3])
>>> td.cuda().device 
device(type='cuda', index=0)
>>> td.cuda(1).device 
device(type='cuda', index=1)
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).

## Tutorials that use `cuda`

![](../../_images/sphx_glr_tensordict_memory_thumb.png)

[Devices and memory-mapped tensors](../../tutorials/tensordict_memory.html)

Devices and memory-mapped tensors