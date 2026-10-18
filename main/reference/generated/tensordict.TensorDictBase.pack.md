# TensorDictBase.pack

TensorDictBase.pack(*dim: [int](https://docs.python.org/3/builtins/functions.html#int) = -1*, ***, *layout: [PackLayout](tensordict.PackLayout.html#tensordict.PackLayout) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*) → [tuple](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor), [PackLayout](tensordict.PackLayout.html#tensordict.PackLayout)]

Packs all the leaves of a tensordict into one tensor of features.

Each leaf is flattened after the batch dimensions, and the leaves are
concatenated in the order of `keys(True, True, sort=True)`, so the
columns depend on the set of keys only. Leaves can have different
ranks. The result has shape `(*batch_size, F)` for `dim=-1`, where
`F` is the sum of the numbers of features of the leaves, and it is
contiguous for any `dim`. [`PackLayout.unpack`](tensordict.PackLayout.html#tensordict.PackLayout.unpack) inverts it.

Packing along the last dim copies the data once, with one
[`torch.cat()`](https://docs.pytorch.org/docs/stable/generated/torch.cat.html#torch.cat). Along another dim the result can take a second
copy: [`torch.cat()`](https://docs.pytorch.org/docs/stable/generated/torch.cat.html#torch.cat) keeps the channels-last strides that moving
the feature dim gives a 4-d or 5-d result, and the result is then made
contiguous.

All the leaves must share one dtype and one device. Non-tensor data
cannot be packed: drop it with [`filter_non_tensor_data()`](tensordict.TensorDictBase.filter_non_tensor_data.html#tensordict.TensorDictBase.filter_non_tensor_data) first.
A lazy stack packs like its dense equivalent if all its members have
the same keys and feature shapes.

Parameters:

**dim** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the position of the packed dimension in the
result, in `[-(batch_dims + 1), batch_dims]` as for
[`torch.stack()`](https://docs.pytorch.org/docs/stable/generated/torch.stack.html#torch.stack). Defaults to `-1`.

Keyword Arguments:

**layout** ([*PackLayout*](tensordict.PackLayout.html#tensordict.PackLayout)*,**optional*) - a layout to pack into. The columns
then follow its key order, and a missing, extra or reshaped
leaf raises. `dim` must match the layout's dim. Defaults to
`None`, which builds a new layout in sorted key order.

Returns:

A tuple of the packed tensor and its
[`PackLayout`](tensordict.PackLayout.html#tensordict.PackLayout).

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict(
... p=torch.zeros(5),
... v=torch.ones(5, 3),
... nested={"stress": torch.ones(5, 3, 3)},
... batch_size=[5],
... )
>>> flat, layout = td.pack()
>>> flat.shape
torch.Size([5, 13])
>>> layout.keys
(('nested', 'stress'), 'p', 'v')
>>> out = layout.unpack(flat.index_select(0, torch.tensor([0, 2])))
>>> out.batch_size
torch.Size([2])
>>> flat2, _ = td.pack(layout=layout)
>>> assert (flat2 == flat).all()
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).