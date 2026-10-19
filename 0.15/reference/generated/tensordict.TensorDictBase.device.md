# TensorDictBase.device

*property*TensorDictBase.device*: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | [None](https://docs.python.org/3/builtins/constants.html#None)*

The device of the tensordict, or `None` if it has none.

If the tensordict has a device, all its leaves, nested ones included,
are on that device: the tensors that you pass to the constructor or
set later are moved to it. If the device is `None`, the leaves can be
on different devices. The device is not inferred from the tensors.
Assigning it raises a `RuntimeError`: use [`to()`](tensordict.TensorDictBase.to.html#tensordict.TensorDictBase.to) to get a
tensordict on another device. See [Devices, dtypes, and memory](../../concepts/devices.html#devices).

Examples

The examples use the `"meta"` device so that they run on CPU.
A CUDA device works the same way.

```
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict({
... "cpu": torch.randn(3, device='cpu'),
... "meta": torch.randn(3, device='meta'),
... }, batch_size=[], device=None)
>>> td['cpu'].device
device(type='cpu')
>>> td['meta'].device
device(type='meta')
>>> print(td.device)
None
>>> td = TensorDict({
... "x": torch.randn(3, device='cpu'),
... "y": TensorDict({'z': torch.randn(3, device='cpu')}, batch_size=[], device=None),
... }, batch_size=[], device='meta')
>>> td['x'].device
device(type='meta')
>>> td['y'].device # nested tensordicts are also mapped onto the appropriate device.
device(type='meta')
>>> td['y', 'z'].device
device(type='meta')
>>> td['w'] = torch.zeros(3) # a new entry is moved to the device too
>>> td['w'].device
device(type='meta')
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).