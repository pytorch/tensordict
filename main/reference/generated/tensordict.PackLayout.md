# PackLayout

*class*tensordict.PackLayout(*keys: [tuple](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[NestedKey](tensordict.NestedKey.html#tensordict.NestedKey), ...]*, *shapes: [tuple](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[Size](https://docs.pytorch.org/docs/stable/size.html#torch.Size), ...]*, *dim: [int](https://docs.python.org/3/builtins/functions.html#int) = -1*, *nested: [tuple](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[tuple](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[NestedKey](tensordict.NestedKey.html#tensordict.NestedKey), [Size](https://docs.pytorch.org/docs/stable/size.html#torch.Size), [type](https://docs.python.org/3/builtins/functions.html#type) | [None](https://docs.python.org/3/builtins/constants.html#None)], ...] = ()*, *device: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *cls: [type](https://docs.python.org/3/builtins/functions.html#type) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

The layout of a tensordict packed by [`pack()`](tensordict.TensorDictBase.pack.html#tensordict.TensorDictBase.pack).

It records the column of each leaf in the packed tensor, and the nesting,
batch dimensions and classes needed to rebuild the tensordict with
`unpack()`. It is frozen, hashable and compares by value, so a module
can cache it and `torch.compile` can treat it as a constant.

Only tensorclasses are rebuilt with their class. Other subclasses of
[`TensorDictBase`](tensordict.TensorDictBase.html#tensordict.TensorDictBase), such as
[`TypedTensorDict`](tensordict.TypedTensorDict.html#tensordict.TypedTensorDict) or
`TensorDictParams`, come back as a
[`TensorDict`](tensordict.TensorDict.html#tensordict.TensorDict).

Variables:

- **keys** ([*tuple*](https://docs.python.org/3/builtins/stdtypes.html#tuple)*of*[*NestedKey*](tensordict.NestedKey.html#tensordict.NestedKey)) - the leaf keys in column order.
- **shapes** ([*tuple*](https://docs.python.org/3/builtins/stdtypes.html#tuple)*of*[*Size*](https://docs.pytorch.org/docs/stable/size.html#torch.Size)) - the feature shape of each leaf, that is
its shape after the batch dimensions of the packed tensordict.
Lists and tuples of ints are converted to `torch.Size`.
- **dim** ([*int*](https://docs.python.org/3/builtins/functions.html#int)) - the position of the packed dimension in the packed tensor.
[`pack()`](tensordict.TensorDictBase.pack.html#tensordict.TensorDictBase.pack) stores it counted from the
end (a negative int), so `td.pack(1)` and `td.pack(-1)` give
equal layouts when `td` has one batch dimension, and `flat`
may have extra leading batch dimensions.
- **nested** ([*tuple*](https://docs.python.org/3/builtins/stdtypes.html#tuple)) - one `(key, batch_shape, cls)` entry per nested tensor
collection, where `batch_shape` is its batch shape after the
batch dimensions of the root and `cls` is its tensorclass, or
`None` for a tensordict. Batch shapes given as lists or tuples
of ints are converted to `torch.Size`.
- **device** ([*device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*or**None*) - the device of the packed tensordict.
- **cls** ([*type*](https://docs.python.org/3/builtins/functions.html#type)*or**None*) - the tensorclass of the packed tensordict, or `None`.
- **offsets** ([*tuple*](https://docs.python.org/3/builtins/stdtypes.html#tuple)*of*[*int*](https://docs.python.org/3/builtins/functions.html#int)) - the first column of each leaf.
- **numel** ([*int*](https://docs.python.org/3/builtins/functions.html#int)) - the number of columns of the packed tensor.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict(p=torch.zeros(5), v=torch.ones(5, 3), batch_size=[5])
>>> flat, layout = td.pack()
>>> layout.keys, layout.offsets, layout.numel
(('p', 'v'), (0, 1), 4)
>>> out = layout.unpack(flat[:2])
>>> out["v"].shape
torch.Size([2, 3])
```

unpack(*flat: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*) → [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase)

Rebuilds a tensordict from a packed tensor.

The leaves of the result are views of `flat`. Its batch size is the
shape of `flat` without the packed dimension, so `flat` may have
other batch dimensions than the packed tensordict, for instance after
an `index_select`.

Parameters:

**flat** ([*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) - a tensor with `numel` entries along
dimension `dim`.

Returns:

A [`TensorDict`](tensordict.TensorDict.html#tensordict.TensorDict), or an instance of `cls`,
on the device of `flat` if the packed tensordict had a device.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict(p=torch.zeros(5), v=torch.ones(5, 3), batch_size=[5])
>>> flat, layout = td.pack()
>>> out = layout.unpack(flat)
>>> assert (out == td).all()
>>> out["p"].data_ptr() == flat.data_ptr()
True
```