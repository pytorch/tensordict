# TensorDictBase.attrs

TensorDictBase.attrs(***, *fields: [Sequence](https://docs.python.org/3/library/typing.html#typing.Sequence)[[str](https://docs.python.org/3/builtins/stdtypes.html#str)] = ('device', 'dtype', 'shape')*, *num_threads: [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*) → Self

Returns a tensordict that describes each tensor leaf with a [`TensorAttrs`](tensordict.TensorAttrs.html#tensordict.TensorAttrs).

Each tensor leaf is replaced by a [`TensorAttrs`](tensordict.TensorAttrs.html#tensordict.TensorAttrs) that
records its device, dtype and shape. The result has an empty batch size
and no device. Pass it to [`to()`](tensordict.TensorDictBase.to.html#tensordict.TensorDictBase.to) to cast another tensordict leaf
by leaf, to the device and dtype of the matching leaf, when the leaves
do not all share one device or dtype.

Keyword Arguments:

- **fields** (*sequence**of*[*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - the attributes to record, any
subset of `("device", "dtype", "shape")`. The others are
`None`. Defaults to `("device", "dtype", "shape")`.
- **num_threads** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the number of threads that build the
result. Defaults to `None`, which uses the calling thread
only.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict(
... {"a": torch.zeros(3, device="cpu"),
... "b": torch.zeros(3, dtype=torch.int32)},
... batch_size=[3],
... )
>>> attrs = td.attrs(fields=("device", "dtype"))
>>> target = TensorDict(a=torch.zeros(3, device="cpu"), b=torch.zeros(3), batch_size=[3])
>>> out = target.to(attrs) # casts "b" to int32
>>> out["b"].dtype
torch.int32
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).