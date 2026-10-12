# TensorDictBase.device

*abstract property*TensorDictBase.device*: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | [None](https://docs.python.org/3/builtins/constants.html#None)*

Device of a TensorDict.

If the TensorDict has a specified device, all
its tensors (incl. nested ones) must live on the same device: the
tensors that you pass to the constructor or set later are moved to
it. If the TensorDict device is `None`, different values can be
located on different devices. The device is not inferred from the
tensors. See [Devices, dtypes and memory](../../concepts/devices.html#devices).

Returns:

torch.device object indicating the device where the tensors
are placed, or None if TensorDict does not have a device.

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