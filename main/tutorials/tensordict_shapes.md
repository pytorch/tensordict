Note

Go to the end
to download the full example code.

# Changing the shape of a TensorDict

**Author**: [Tom Begley](https://github.com/tcbegley)

A [`TensorDict`](../reference/generated/tensordict.TensorDict.html#tensordict.TensorDict) has the shape operations of a tensor, such
as `reshape`, `split`, `stack`, and `permute`. Each call changes every
tensor in it in the same way, so you write the call once, and the tensors stay
aligned. This tutorial is for readers who know these operations on a
[`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor).

What you will learn:

- How a shape operation on a tensordict applies to the batch dimensions of
every entry.
- How to split, unbind, stack, and concatenate tensordicts.
- The difference between a lazy stack and a dense stack.
- How to change the shape of a tensordict for the length of a `with` block.

When you create a [`TensorDict`](../reference/generated/tensordict.TensorDict.html#tensordict.TensorDict), you give it a
`batch_size`: the leading dimensions that all the entries share. An entry
can have more dimensions after them (see [Batch size, shapes, and indexing](../concepts/batch_size.html#batch-size)). The shape
operations below change the batch dimensions of every entry, and keep the
other dimensions.

```
import torch
from tensordict import TensorDict

torch.manual_seed(0)
```

```
<torch._C.Generator object at 0x7f4c31fecf90>
```

## Indexing

An index applies to the batch dimensions. Each entry of the
[`TensorDict`](../reference/generated/tensordict.TensorDict.html#tensordict.TensorDict) is indexed in the same way. The
[Slicing, indexing, and masking](tensordict_slicing.html#sphx-glr-tutorials-tensordict-slicing-py) tutorial covers indexing
in detail.

```
a = torch.rand(3, 4)
b = torch.rand(3, 4, 5)
tensordict = TensorDict({"a": a, "b": b}, batch_size=[3, 4])

indexed_tensordict = tensordict[:2, 1]
assert indexed_tensordict["a"].shape == torch.Size([2])
assert indexed_tensordict["b"].shape == torch.Size([2, 5])
```

## Reshaping

[`TensorDict.reshape`](../reference/generated/tensordict.TensorDictBase.reshape.html#tensordict.TensorDictBase.reshape) works like
[`torch.Tensor.reshape()`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.reshape.html#torch.Tensor.reshape). It reshapes the batch dimensions of every
entry, and keeps the other dimensions, as the shape of `b` below shows. It
also updates the `batch_size` attribute.

```
reshaped_tensordict = tensordict.reshape(-1)
assert reshaped_tensordict.batch_size == torch.Size([12])
assert reshaped_tensordict["a"].shape == torch.Size([12])
assert reshaped_tensordict["b"].shape == torch.Size([12, 5])
```

## Splitting

[`TensorDict.split`](../reference/generated/tensordict.TensorDictBase.split.html#tensordict.TensorDictBase.split) works like
[`torch.Tensor.split()`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.split.html#torch.Tensor.split). It splits the [`TensorDict`](../reference/generated/tensordict.TensorDict.html#tensordict.TensorDict) into
chunks. Each chunk is a [`TensorDict`](../reference/generated/tensordict.TensorDict.html#tensordict.TensorDict) with the same keys as
the original one. Its entries are views of the entries of the original
[`TensorDict`](../reference/generated/tensordict.TensorDict.html#tensordict.TensorDict): they share memory with them.

```
chunks = tensordict.split([3, 1], dim=1)
assert chunks[0].batch_size == torch.Size([3, 3])
assert chunks[1].batch_size == torch.Size([3, 1])
torch.testing.assert_close(chunks[0]["a"], tensordict["a"][:, :-1])
```

Note

If a function or method takes a `dim` argument, a negative `dim` counts
from the end of the `batch_size` of the [`TensorDict`](../reference/generated/tensordict.TensorDict.html#tensordict.TensorDict)
that you call it on. If the tensordict contains nested tensordicts with
more batch dimensions, a negative `dim` still counts from the end of
the batch dimensions of the root tensordict.

```
>>> tensordict = TensorDict(
... {
... "a": torch.rand(3, 4),
... "nested": TensorDict({"b": torch.rand(3, 4, 5)}, [3, 4, 5])
... },
... [3, 4],
... )
>>> # dim=-2 is the first dimension for all entries: the root
>>> # TensorDict has 2 batch dimensions, the nested TensorDict has 3
>>> chunks = tensordict.split([2, 1], dim=-2)
>>> assert chunks[0].batch_size == torch.Size([2, 4])
>>> assert chunks[0]["nested"].batch_size == torch.Size([2, 4, 5])
```

In this example, [`TensorDict.split`](../reference/generated/tensordict.TensorDictBase.split.html#tensordict.TensorDictBase.split)
gives the same result as with `dim=tensordict.batch_dims - 2`.

## Unbinding

[`TensorDict.unbind`](../reference/generated/tensordict.TensorDictBase.unbind.html#tensordict.TensorDictBase.unbind) works like
[`torch.Tensor.unbind()`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.unbind.html#torch.Tensor.unbind). It removes a dimension and returns a `tuple`
of all the slices along that dimension.

```
slices = tensordict.unbind(dim=1)
assert len(slices) == 4
assert all(s.batch_size == torch.Size([3]) for s in slices)
torch.testing.assert_close(slices[0]["a"], tensordict["a"][:, 0])
```

## Stacking and concatenating

[`torch.cat()`](https://docs.pytorch.org/docs/stable/generated/torch.cat.html#torch.cat) and [`torch.stack()`](https://docs.pytorch.org/docs/stable/generated/torch.stack.html#torch.stack) accept lists of
[`TensorDict`](../reference/generated/tensordict.TensorDict.html#tensordict.TensorDict) instances.

### Stacking

You can stack tensordicts in two ways. [`torch.stack()`](https://docs.pytorch.org/docs/stable/generated/torch.stack.html#torch.stack) copies the values
into new tensors and returns a [`TensorDict`](../reference/generated/tensordict.TensorDict.html#tensordict.TensorDict).
[`lazy_stack()`](../reference/generated/tensordict.lazy_stack.html#tensordict.lazy_stack) keeps the original tensordicts in a list and
returns a [`LazyStackedTensorDict`](../reference/generated/tensordict.LazyStackedTensorDict.html#tensordict.LazyStackedTensorDict), which stacks a value
only when you read it. Use a lazy stack when the tensordicts have different
shapes, devices, or keys, or when you read only a few keys of the stack. The
[Stacking and variable-size data](../concepts/stacking.html#stacking) page explains both kinds of stacks.

```
from tensordict import lazy_stack

cloned_tensordict = tensordict.clone()
stacked_tensordict = lazy_stack([tensordict, cloned_tensordict], dim=0)
print(stacked_tensordict)
```

```
LazyStackedTensorDict(
 fields={
 a: Tensor(shape=torch.Size([2, 3, 4]), device=cpu, dtype=torch.float32, is_shared=False),
 b: Tensor(shape=torch.Size([2, 3, 4, 5]), device=cpu, dtype=torch.float32, is_shared=False)},
 exclusive_fields={
 },
 batch_size=torch.Size([2, 3, 4]),
 device=None,
 is_shared=False,
 stack_dim=0)
```

Like [`torch.stack()`](https://docs.pytorch.org/docs/stable/generated/torch.stack.html#torch.stack) on tensors, [`torch.stack()`](https://docs.pytorch.org/docs/stable/generated/torch.stack.html#torch.stack) on tensordicts
returns a dense [`TensorDict`](../reference/generated/tensordict.TensorDict.html#tensordict.TensorDict):

```
dense_stack = torch.stack([tensordict, cloned_tensordict])
assert isinstance(dense_stack, TensorDict)
```

If you index a [`LazyStackedTensorDict`](../reference/generated/tensordict.LazyStackedTensorDict.html#tensordict.LazyStackedTensorDict) along the stack
dimension, you get back the original [`TensorDict`](../reference/generated/tensordict.TensorDict.html#tensordict.TensorDict).

```
assert stacked_tensordict[0] is tensordict
assert stacked_tensordict[1] is cloned_tensordict
```

If you read a key of the [`LazyStackedTensorDict`](../reference/generated/tensordict.LazyStackedTensorDict.html#tensordict.LazyStackedTensorDict), it
stacks the values of this key. If the value is a nested
[`TensorDict`](../reference/generated/tensordict.TensorDict.html#tensordict.TensorDict), you get another
[`LazyStackedTensorDict`](../reference/generated/tensordict.LazyStackedTensorDict.html#tensordict.LazyStackedTensorDict).

```
assert stacked_tensordict["a"].shape == torch.Size([2, 3, 4])
```

Note

A lazy stack stacks a value each time you read it. If you read a value of
the stack more than once, convert the
[`LazyStackedTensorDict`](../reference/generated/tensordict.LazyStackedTensorDict.html#tensordict.LazyStackedTensorDict) to a dense
[`TensorDict`](../reference/generated/tensordict.TensorDict.html#tensordict.TensorDict) first, with
[`LazyStackedTensorDict.to_tensordict`](../reference/generated/tensordict.TensorDictBase.to_tensordict.html#tensordict.TensorDictBase.to_tensordict)
or [`LazyStackedTensorDict.contiguous`](../reference/generated/tensordict.TensorDictBase.contiguous.html#tensordict.TensorDictBase.contiguous).

```
>>> assert isinstance(stacked_tensordict.to_tensordict(), TensorDict)
>>> assert isinstance(stacked_tensordict.contiguous(), TensorDict)
```

The result holds the stacked values, and reading a value does not stack
it again.

### Concatenating

Concatenation is never lazy. [`torch.cat()`](https://docs.pytorch.org/docs/stable/generated/torch.cat.html#torch.cat) on a list of
[`TensorDict`](../reference/generated/tensordict.TensorDict.html#tensordict.TensorDict) instances returns a
[`TensorDict`](../reference/generated/tensordict.TensorDict.html#tensordict.TensorDict). Each of its entries concatenates the
entries with the same key in the tensordicts of the list.

```
concatenated_tensordict = torch.cat([tensordict, cloned_tensordict], dim=0)
assert isinstance(concatenated_tensordict, TensorDict)
assert concatenated_tensordict.batch_size == torch.Size([6, 4])
assert concatenated_tensordict["b"].shape == torch.Size([6, 4, 5])
```

## Expanding

[`TensorDict.expand`](../reference/generated/tensordict.TensorDictBase.expand.html#tensordict.TensorDictBase.expand) expands all the
entries of a [`TensorDict`](../reference/generated/tensordict.TensorDict.html#tensordict.TensorDict). As with
[`torch.Tensor.expand()`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.expand.html#torch.Tensor.expand), the expanded entries are views: they share
memory with the original entries. Call `clone()` on the result before
you write into it.

```
exp_tensordict = tensordict.expand(2, *tensordict.batch_size)
assert exp_tensordict.batch_size == torch.Size([2, 3, 4])
torch.testing.assert_close(exp_tensordict["a"][0], exp_tensordict["a"][1])
```

## Squeezing and unsqueezing

The [`squeeze()`](../reference/generated/tensordict.TensorDictBase.squeeze.html#tensordict.TensorDictBase.squeeze) and
[`unsqueeze()`](../reference/generated/tensordict.TensorDictBase.unsqueeze.html#tensordict.TensorDictBase.unsqueeze) methods remove or add a batch
dimension of size one.

```
tensordict = TensorDict({"a": torch.rand(3, 1, 4)}, [3, 1, 4])
squeezed_tensordict = tensordict.squeeze()
assert squeezed_tensordict["a"].shape == torch.Size([3, 4])
print(squeezed_tensordict, end="\n\n")

unsqueezed_tensordict = tensordict.unsqueeze(-1)
assert unsqueezed_tensordict["a"].shape == torch.Size([3, 1, 4, 1])
print(unsqueezed_tensordict)
```

```
TensorDict(
 fields={
 a: Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([3, 4]),
 device=None,
 is_shared=False)

TensorDict(
 fields={
 a: Tensor(shape=torch.Size([3, 1, 4, 1]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([3, 1, 4, 1]),
 device=None,
 is_shared=False)
```

Note

[`unsqueeze()`](../reference/generated/tensordict.TensorDictBase.unsqueeze.html#tensordict.TensorDictBase.unsqueeze),
[`squeeze()`](../reference/generated/tensordict.TensorDictBase.squeeze.html#tensordict.TensorDictBase.squeeze), [`view()`](../reference/generated/tensordict.TensorDictBase.view.html#tensordict.TensorDictBase.view),
[`permute()`](../reference/generated/tensordict.TensorDictBase.permute.html#tensordict.TensorDictBase.permute), and
[`transpose()`](../reference/generated/tensordict.TensorDictBase.transpose.html#tensordict.TensorDictBase.transpose) apply the operation to the
entries when you call them, and return a [`TensorDict`](../reference/generated/tensordict.TensorDict.html#tensordict.TensorDict).

These methods apply only to the batch dimensions. The other dimensions of
the entries do not change:

```
tensordict = TensorDict({"a": torch.rand(3, 1, 1, 4)}, [3, 1])
squeezed_tensordict = tensordict.squeeze()
# only one of the two dimensions of size one is removed: the other
# one is not a batch dimension
assert squeezed_tensordict["a"].shape == torch.Size([3, 1, 4])
```

## Viewing

[`view()`](../reference/generated/tensordict.TensorDictBase.view.html#tensordict.TensorDictBase.view) returns a
[`TensorDict`](../reference/generated/tensordict.TensorDict.html#tensordict.TensorDict) of views of the original entries.

```
tensordict = TensorDict({"a": torch.arange(12)}, [12])
# the entries are viewed at this step
viewed_tensordict = tensordict.view((2, 3, 2))

# "a" is a view of the original tensor
assert viewed_tensordict["a"].shape == torch.Size([2, 3, 2])
```

## Permuting the batch dimensions

[`TensorDict.permute`](../reference/generated/tensordict.TensorDictBase.permute.html#tensordict.TensorDictBase.permute) reorders the batch
dimensions, like [`torch.permute()`](https://docs.pytorch.org/docs/stable/generated/torch.permute.html#torch.permute). The other dimensions do not change.

Like the other shape operations above, `permute` applies to the entries
when you call it.

```
tensordict = TensorDict({"a": torch.rand(3, 4), "b": torch.rand(3, 4, 5)}, [3, 4])
# swap the batch dimensions
permuted_tensordict = tensordict.permute([1, 0])

assert permuted_tensordict["a"].shape == torch.Size([4, 3])
assert permuted_tensordict["b"].shape == torch.Size([4, 3, 5])
```

## Reversible operations as context managers

You can use the result of some reversible operations as a context manager,
in a `with` block. At the end of the block, tensordict applies the inverse
operation and writes the changes back into the original tensordict. The
shape operations [`view()`](../reference/generated/tensordict.TensorDictBase.view.html#tensordict.TensorDictBase.view),
[`permute()`](../reference/generated/tensordict.TensorDictBase.permute.html#tensordict.TensorDictBase.permute),
[`transpose()`](../reference/generated/tensordict.TensorDictBase.transpose.html#tensordict.TensorDictBase.transpose),
[`squeeze()`](../reference/generated/tensordict.TensorDictBase.squeeze.html#tensordict.TensorDictBase.squeeze), and
[`unsqueeze()`](../reference/generated/tensordict.TensorDictBase.unsqueeze.html#tensordict.TensorDictBase.unsqueeze) work in this way. So do
[`to_module()`](../reference/generated/tensordict.TensorDictBase.to_module.html#tensordict.TensorDictBase.to_module) for functional calls, and
[`lock_()`](../reference/generated/tensordict.TensorDictBase.lock_.html#tensordict.TensorDictBase.lock_) and
[`unlock_()`](../reference/generated/tensordict.TensorDictBase.unlock_.html#tensordict.TensorDictBase.unlock_). This example uses `transpose`:

```
tensordict = TensorDict({"a": torch.rand(3, 4), "b": torch.rand(3, 4, 5)}, [3, 4])

with tensordict.transpose(1, 0) as tdt:
 tdt.set("c", torch.ones(4, 3)) # the batch size of tdt is [4, 3]
```

After the block, the original tensordict has the `"c"` entry, with the
batch dimensions in the original order:

```
assert (tensordict.get("c") == 1).all()
assert tensordict["c"].shape == torch.Size([3, 4])
```

## Gathering values

[`TensorDict.gather`](../reference/generated/tensordict.TensorDictBase.gather.html#tensordict.TensorDictBase.gather) gathers values along
a batch dimension with an index tensor, like [`torch.gather()`](https://docs.pytorch.org/docs/stable/generated/torch.gather.html#torch.gather).

```
index = torch.randint(4, (3, 4))
gathered_tensordict = tensordict.gather(dim=1, index=index)
print("index:\n", index, end="\n\n")
print("tensordict['a']:\n", tensordict["a"], end="\n\n")
print("gathered_tensordict['a']:\n", gathered_tensordict["a"], end="\n\n")
```

```
index:
 tensor([[0, 1, 2, 1],
 [1, 3, 1, 0],
 [0, 0, 3, 3]])

tensordict['a']:
 tensor([[0.0500, 0.4663, 0.9397, 0.2961],
 [0.9515, 0.6811, 0.0488, 0.8163],
 [0.4423, 0.2768, 0.8998, 0.0960]])

gathered_tensordict['a']:
 tensor([[0.0500, 0.4663, 0.9397, 0.4663],
 [0.6811, 0.8163, 0.6811, 0.9515],
 [0.4423, 0.4423, 0.0960, 0.0960]])
```

## Conclusion

You can now index, `reshape`, `split`, `unbind`, `expand`,
`squeeze`, `unsqueeze`, `view`, `permute`, and `gather` a
[`TensorDict`](../reference/generated/tensordict.TensorDict.html#tensordict.TensorDict) with one call each. Each call changes the
batch dimensions of every entry and keeps the other dimensions. You can
stack tensordicts densely with [`torch.stack()`](https://docs.pytorch.org/docs/stable/generated/torch.stack.html#torch.stack), or lazily with
[`lazy_stack()`](../reference/generated/tensordict.lazy_stack.html#tensordict.lazy_stack) when their shapes differ. You can also change
the shape for the length of a `with` block, and get the changes back in the
original tensordict.

The same operations work on a [`TensorClass`](../reference/generated/tensordict.TensorClass.html#tensordict.TensorClass), the class
form of a tensordict with typed fields (see [TensorClass](../tensorclass.html#tensorclass-guide)).

## Further reading

- [Batch size, shapes, and indexing](../concepts/batch_size.html#batch-size) explains batch sizes, shapes, and indexing.
- [Stacking and variable-size data](../concepts/stacking.html#stacking) explains lazy stacks and data of different sizes.
- [Slicing, indexing, and masking](tensordict_slicing.html#sphx-glr-tutorials-tensordict-slicing-py) shows indexing, masking,
and writing into a tensordict with an index.
- [Shape, batch dimensions, and stacking](../reference/td.html#api-methods-shape) lists all the shape methods.

**Total running time of the script:** (0 minutes 0.009 seconds)

[`Download Jupyter notebook: tensordict_shapes.ipynb`](../_downloads/11a23e1bafc489334eaf020bf977a5bb/tensordict_shapes.ipynb)

[`Download Python source code: tensordict_shapes.py`](../_downloads/55cd8c52e91d60c34a488d01e9d82ff3/tensordict_shapes.py)

[`Download zipped: tensordict_shapes.zip`](../_downloads/71264da051c82d595130ca8e60350024/tensordict_shapes.zip)

[Gallery generated by Sphinx-Gallery](https://sphinx-gallery.github.io)