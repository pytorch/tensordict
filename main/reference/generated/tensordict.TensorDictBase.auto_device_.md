# TensorDictBase.auto_device

TensorDictBase.auto_device_() → Self

Sets the device of the tensordict to the device of its leaves, if they all share one.

If the leaves are on several devices, the device is set to `None`.
The nested tensordicts get the same device. The leaves are not moved.
See [Devices, dtypes, and memory](../../concepts/devices.html#devices).

Returns:

`self`.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict(a=torch.zeros(3), batch_size=[3])
>>> print(td.device)
None
>>> td.auto_device_().device
device(type='cpu')
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).