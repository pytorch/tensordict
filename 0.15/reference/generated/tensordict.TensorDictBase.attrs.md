# TensorDictBase.attrs

TensorDictBase.attrs(***, *fields: [Sequence](https://docs.python.org/3/library/typing.html#typing.Sequence)[[str](https://docs.python.org/3/builtins/stdtypes.html#str)] = ('device', 'dtype', 'shape')*, *num_threads: [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*) → Self

Return a deviceless tensordict whose leaves are [`TensorAttrs`](tensordict.TensorAttrs.html#tensordict.TensorAttrs).

Each tensor leaf of `self` is replaced by a [`TensorAttrs`](tensordict.TensorAttrs.html#tensordict.TensorAttrs)
describing the requested tensor attributes. The result is intended to be passed to
[`to()`](tensordict.TensorDictBase.to.html#tensordict.TensorDictBase.to) to drive per-leaf device/dtype casting when the source tensordict
is heterogeneous.

Keyword Arguments:

- **fields** (*sequence**of*[*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - which attributes to record on each
[`TensorAttrs`](tensordict.TensorAttrs.html#tensordict.TensorAttrs). Accepts any subset of
`("device", "dtype", "shape")`. Attributes not listed remain `None`.
Defaults to `("device", "dtype", "shape")`.
- **num_threads** ([*int*](tensordict.TensorDictBase.int.html#tensordict.TensorDictBase.int)*or*[*None*](https://docs.python.org/3/builtins/constants.html#None)*,**optional*) - number of threads to use when
iterating leaves. Defaults to `None` (single-threaded). Construction
of [`TensorAttrs`](tensordict.TensorAttrs.html#tensordict.TensorAttrs) is Python-bound, so threading typically yields
little; exposed for symmetry with [`to()`](tensordict.TensorDictBase.to.html#tensordict.TensorDictBase.to).

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
>>> out = target.to(attrs) # casts `b` to int32 per-leaf
>>> out["b"].dtype
torch.int32
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).