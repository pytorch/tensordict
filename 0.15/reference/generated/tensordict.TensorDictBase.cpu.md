# TensorDictBase.cpu

TensorDictBase.cpu(***kwargs*) → Self

Returns the tensordict on the CPU, as `to("cpu", **kwargs)` does.

A tensordict whose device is already the CPU is returned as it is.

Parameters:

****kwargs** - passed to [`to()`](tensordict.TensorDictBase.to.html#tensordict.TensorDictBase.to), for example `non_blocking`.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict(a=torch.zeros(3), batch_size=[3])
>>> td.cpu().device
device(type='cpu')
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).