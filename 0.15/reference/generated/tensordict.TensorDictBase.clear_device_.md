# TensorDictBase.clear_device

TensorDictBase.clear_device_() → Self

Sets the device of the tensordict and its sub-tensordicts to `None`, and returns it.

The leaves are not moved. After this call, you can set leaves on any
device. See [Devices, dtypes, and memory](../../concepts/devices.html#devices).

Returns:

`self`.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict(a=torch.zeros(3), batch_size=[3], device="cpu")
>>> td = td.clear_device_()
>>> print(td.device)
None
>>> td["a"].device
device(type='cpu')
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).

## Tutorials that use `clear_device_`

![](../../_images/sphx_glr_tensordict_memory_thumb.png)

[Devices and memory-mapped tensors](../../tutorials/tensordict_memory.html)

Devices and memory-mapped tensors