# TensorDictBase.copy

TensorDictBase.copy_(*tensordict: T*, *non_blocking: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*) → Self

Copies the entries of `tensordict` into the existing entries, and returns `self`.

The same as [`update_()`](tensordict.TensorDictBase.update_.html#tensordict.TensorDictBase.update_), under the name of
[`torch.Tensor.copy_()`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.copy_.html#torch.Tensor.copy_): the existing tensors keep their storage,
and every key of `tensordict` must exist in the tensordict.

Parameters:

- **tensordict** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*or*[*dict*](https://docs.python.org/3/builtins/stdtypes.html#dict)) - the entries to copy.
- **non_blocking** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, copies between devices
can run asynchronously with respect to the host. Defaults to
`False`.

Raises:

- [**KeyError**](https://docs.python.org/3/builtins/exceptions.html#KeyError) - if a key of `tensordict` is missing from the
 tensordict.
- [**RuntimeError**](https://docs.python.org/3/builtins/exceptions.html#RuntimeError) - if a value does not fit the shape of its entry.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> x = torch.zeros(3)
>>> td = TensorDict(x=x, batch_size=[3])
>>> _ = td.copy_(TensorDict(x=torch.ones(3), batch_size=[3]))
>>> x
tensor([1., 1., 1.])
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).