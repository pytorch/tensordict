Note

Go to the end
to download the full example code.

# Slicing, indexing, and masking

**Author**: [Tom Begley](https://github.com/tcbegley)

One index on a [`TensorDict`](../reference/generated/tensordict.TensorDict.html#tensordict.TensorDict) indexes every tensor in it in
the same way. If you now index the images, the labels, and the weights of a
batch one by one, one index replaces these lines. This tutorial is for
readers who know how to index a [`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor).

What you will learn:

- How an index applies to the batch dimensions of every entry.
- How to write a tensordict into another one with an index.
- How a boolean mask changes the batch size.

When you create a [`TensorDict`](../reference/generated/tensordict.TensorDict.html#tensordict.TensorDict), you give it a
`batch_size`: the leading dimensions that all its entries share (see
[Batch size, shapes, and indexing](../concepts/batch_size.html#batch-size)). You can index these batch dimensions as you index a
[`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor). The index applies to all the entries of the
[`TensorDict`](../reference/generated/tensordict.TensorDict.html#tensordict.TensorDict).

For example, the tensordict below has two batch dimensions.
`tensordict[0]` returns a new [`TensorDict`](../reference/generated/tensordict.TensorDict.html#tensordict.TensorDict) with the same
keys. Its values are the first "row" of each entry of the original
[`TensorDict`](../reference/generated/tensordict.TensorDict.html#tensordict.TensorDict).

```
import torch
from tensordict import TensorDict

tensordict = TensorDict(
 {"a": torch.zeros(3, 4, 5), "b": torch.zeros(3, 4)}, batch_size=[3, 4]
)

print(tensordict[0])
```

```
TensorDict(batch_size=[4], fields={a: f32[5], b: f32[]})
```

The syntax is the same as for tensors. This index drops the first row of
each entry:

```
print(tensordict[1:])
```

```
TensorDict(batch_size=[2,4], fields={a: f32[5], b: f32[]})
```

You can index several dimensions at once:

```
print(tensordict[:, 2:])
```

```
TensorDict(batch_size=[3,2], fields={a: f32[5], b: f32[]})
```

`Ellipsis` (`...`) stands for as many `:` as needed to index all the
batch dimensions (`tensordict.batch_dims`).

```
print(tensordict[..., 2:])
```

```
TensorDict(batch_size=[3,2], fields={a: f32[5], b: f32[]})
```

Note

An index on a tensordict applies to the batch dimensions only. The
tensordict above has `batch_size=[3, 4]`, and `"a"` has shape
`[3, 4, 5]`:

- `tensordict["a"][..., 2:]` reads `"a"` and then indexes its last
dimension. The result has shape `[3, 4, 3]`.
- `tensordict[..., 2:]["a"]` indexes the last batch dimension
(dimension 1) and then reads `"a"`. The result has shape `[3, 2, 5]`.

```
assert tensordict["a"][..., 2:].shape == torch.Size([3, 4, 3])
assert tensordict[..., 2:]["a"].shape == torch.Size([3, 2, 5])
```

## Setting values with an index

`tensordict[index] = other` writes the entries of the tensordict
`other` into `tensordict` at the index. The batch size of `other`
must be the batch size of `tensordict[index]`, or broadcast to it. If it
does not, the assignment raises a `RuntimeError`.

```
tensordict = TensorDict(
 {"a": torch.zeros(3, 4, 5), "b": torch.zeros(3, 4)}, batch_size=[3, 4]
)

td2 = TensorDict({"a": torch.ones(2, 4, 5), "b": torch.ones(2, 4)}, batch_size=[2, 4])
tensordict[:-1] = td2
print(tensordict["a"], tensordict["b"])
```

```
tensor([[[1., 1., 1., 1., 1.],
 [1., 1., 1., 1., 1.],
 [1., 1., 1., 1., 1.],
 [1., 1., 1., 1., 1.]],

 [[1., 1., 1., 1., 1.],
 [1., 1., 1., 1., 1.],
 [1., 1., 1., 1., 1.],
 [1., 1., 1., 1., 1.]],

 [[0., 0., 0., 0., 0.],
 [0., 0., 0., 0., 0.],
 [0., 0., 0., 0., 0.],
 [0., 0., 0., 0., 0.]]]) tensor([[1., 1., 1., 1.],
 [1., 1., 1., 1.],
 [0., 0., 0., 0.]])
```

## Masking

A boolean mask works as it does for tensors. The mask has the shape of the
batch dimensions that it masks.

```
mask = torch.tensor(
 [[1, 0, 1, 0], [1, 0, 1, 0], [1, 0, 1, 0]],
 dtype=torch.bool,
)
masked_tensordict = tensordict[mask]
print(masked_tensordict)
```

```
TensorDict(batch_size=[6], fields={a: f32[5], b: f32[]})
```

The mask selects 6 of the 12 elements of the batch. As with tensors, the
dimensions that the mask covers become one dimension, with one element per
`True` value of the mask. The result has `batch_size=[6]`, and `"a"`
has shape `[6, 5]`:

```
assert masked_tensordict.batch_size == torch.Size([6])
assert masked_tensordict["a"].shape == torch.Size([6, 5])
```

## Conclusion

You can now index, slice, and mask a [`TensorDict`](../reference/generated/tensordict.TensorDict.html#tensordict.TensorDict) as you
do a tensor, and write a tensordict into another one with an index. An index
or a mask applies to the batch dimensions of every entry, and keeps the
other dimensions. The same indexing works on a
[`TensorClass`](../reference/generated/tensordict.TensorClass.html#tensordict.TensorClass), the class form of a tensordict with typed
fields (see [TensorClass: a dataclass of tensors](../tensorclass.html#tensorclass-guide)).

## Further reading

- [Batch size, shapes, and indexing](../concepts/batch_size.html#batch-size) explains batch sizes, shapes, and indexing.
- [Changing the shape of a TensorDict](tensordict_shapes.html#sphx-glr-tutorials-tensordict-shapes-py) shows the other shape
operations: reshape, split, stack, permute.
- [Filling a TensorDict row by row](tensordict_preallocation.html#sphx-glr-tutorials-tensordict-preallocation-py) fills an empty
tensordict row by row with an index.
- [Indexing, masking, and splitting](../reference/td.html#api-methods-indexing) lists the indexing methods.

**Total running time of the script:** (0 minutes 0.004 seconds)

[`Download Jupyter notebook: tensordict_slicing.ipynb`](../_downloads/245c7aad3c5052fda391900ac51962d9/tensordict_slicing.ipynb)

[`Download Python source code: tensordict_slicing.py`](../_downloads/2ee47d503f16d1bb0c7d67b4eecde9cc/tensordict_slicing.py)

[`Download zipped: tensordict_slicing.zip`](../_downloads/c1fc5aa7548e6b1437d36fb47bc36ce5/tensordict_slicing.zip)

[Gallery generated by Sphinx-Gallery](https://sphinx-gallery.github.io)