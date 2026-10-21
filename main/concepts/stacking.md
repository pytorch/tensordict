# Stacking and variable-size data

[`torch.stack()`](https://docs.pytorch.org/docs/stable/generated/torch.stack.html#torch.stack) and [`torch.cat()`](https://docs.pytorch.org/docs/stable/generated/torch.cat.html#torch.cat) join tensordicts as they join
tensors, so a list of samples becomes one batch in one call. A lazy stack also
joins tensordicts whose keys or shapes differ, such as graphs with different
numbers of nodes or sequences of different lengths.

## Dense stacking: torch.stack and torch.cat

[`torch.stack()`](https://docs.pytorch.org/docs/stable/generated/torch.stack.html#torch.stack) puts tensordicts along a new batch dimension.
[`torch.cat()`](https://docs.pytorch.org/docs/stable/generated/torch.cat.html#torch.cat) joins them along an existing batch dimension. Here, each
tensordict holds the 3-D positions of 2 particles:

```
>>> import torch
>>> from tensordict import TensorDict
>>> a = TensorDict({"pos": torch.zeros(2, 3)}, batch_size=[2])
>>> b = TensorDict({"pos": torch.ones(2, 3)}, batch_size=[2])
>>> stacked = torch.stack([a, b])
>>> type(stacked).__name__, stacked.batch_size
('TensorDict', torch.Size([2, 2]))
>>> torch.stack([a, b], dim=1)["pos"].shape
torch.Size([2, 2, 3])
>>> torch.cat([a, b]).batch_size
torch.Size([4])
```

Both return a regular [`TensorDict`](../reference/generated/tensordict.TensorDict.html#tensordict.TensorDict), with one new tensor for
each key. This is a dense stack: it copies the data, so a later change to the
result does not change the inputs. [`tensordict.stack()`](../reference/generated/tensordict.stack.html#tensordict.stack) and
[`tensordict.cat()`](../reference/generated/tensordict.cat.html#tensordict.cat) are the same functions.

`torch.stack` raises a `RuntimeError` if the tensordicts do not have the
same keys, the same batch size, and the same shape for each entry.
`torch.cat` accepts different sizes along the joined dimension, but it
raises a `KeyError` if the keys differ:

```
>>> c = TensorDict({"pos": torch.zeros(2, 2)}, batch_size=[2]) # 2-D positions
>>> torch.stack([a, c])
Traceback (most recent call last):
...
RuntimeError: The shapes of the tensors to stack is incompatible: torch.Size([2, 2]) vs torch.Size([2, 3]) for key pos.
>>> torch.stack([a, TensorDict({"vel": torch.zeros(2, 3)}, batch_size=[2])])
Traceback (most recent call last):
...
RuntimeError: The sets of keys in the tensordicts to stack are exclusive. Consider using `LazyStackedTensorDict.maybe_dense_stack` instead.
```

For tensordicts that differ in these ways, use a lazy stack.

## Lazy stacks

A lazy stack is a [`LazyStackedTensorDict`](../reference/generated/tensordict.LazyStackedTensorDict.html#tensordict.LazyStackedTensorDict). It keeps the list
of the original tensordicts, and it stacks their entries only when you read
them. [`lazy_stack()`](../reference/generated/tensordict.lazy_stack.html#tensordict.lazy_stack) always returns one. In this example, each
tensordict holds 10 time steps of one robot:

```
>>> from tensordict import lazy_stack
>>> tds = [
... TensorDict({"battery": torch.zeros(10), "joints": torch.zeros(10, 4)}, batch_size=[10])
... for _ in range(3)
... ]
>>> stacked = lazy_stack(tds) # no data is stacked here
>>> type(stacked).__name__, stacked.batch_size
('LazyStackedTensorDict', torch.Size([3, 10]))
>>> stacked["battery"].shape # stacks the "battery" entries only
torch.Size([3, 10])
```

You also get a lazy stack from [`torch.stack()`](https://docs.pytorch.org/docs/stable/generated/torch.stack.html#torch.stack) or [`torch.cat()`](https://docs.pytorch.org/docs/stable/generated/torch.cat.html#torch.cat) of lazy
stacks, and from [`maybe_dense_stack()`](../reference/generated/tensordict.maybe_dense_stack.html#tensordict.maybe_dense_stack) if the inputs cannot
make a dense stack. The table in Which function to use lists each
case.

### Read and write a lazy stack

A lazy stack reads and writes the original tensordicts:

- An index along the stack dimension returns the original tensordict.
- A write of an entry, or an in-place change, goes to the original
tensordicts. A new key goes to each of them.
- Any other index returns a new lazy stack, for example `stacked[:, 0]`.

For example:

```
>>> stacked[0] is tds[0]
True
>>> stacked["battery"] = torch.ones(3, 10)
>>> tds[0]["battery"].sum()
tensor(10.)
```

Each read of an entry stacks the values again, and allocates a new tensor. If
you read the same keys many times, call
[`contiguous()`](../reference/generated/tensordict.TensorDictBase.contiguous.html#tensordict.TensorDictBase.contiguous) once and read from its
result. `contiguous()` returns a regular TensorDict that owns its data, so a
write to it does not change the original tensordicts:

```
>>> dense = stacked.contiguous()
>>> type(dense).__name__
'TensorDict'
>>> dense["battery"] += 1
>>> tds[0]["battery"].sum() # not changed
tensor(10.)
```

[`to_tensordict()`](../reference/generated/tensordict.TensorDictBase.to_tensordict.html#tensordict.TensorDictBase.to_tensordict) does the same.
`contiguous()` raises a `RuntimeError` if the stacked entries have
different shapes. See Variable-size data for that case.

A lazy stack can name its stack dimension, with `stack_dim_name` in
[`lazy_stack()`](../reference/generated/tensordict.lazy_stack.html#tensordict.lazy_stack), or by setting its `names`. If the stacked
tensordicts have no names, `contiguous()`, `to_tensordict()` and
[`pad()`](../reference/generated/tensordict.pad.html#tensordict.pad) give a TensorDict without that name. `clone()`,
which returns a lazy stack, and
[`densify()`](../reference/generated/tensordict.TensorDictBase.densify.html#tensordict.TensorDictBase.densify) keep it. To keep it in a
TensorDict, call `densify()`, or name the dimension again with
[`refine_names()`](../reference/generated/tensordict.TensorDictBase.refine_names.html#tensordict.TensorDictBase.refine_names):

```
>>> named = lazy_stack(tds, stack_dim_name="robot")
>>> named.names
['robot', None]
>>> named.contiguous().names
[None, None]
>>> named.contiguous().refine_names("robot", ...).names
['robot', None]
>>> named.densify().names
['robot', None]
>>> named.clone().names
['robot', None]
```

If the stacked tensordicts have names, all of these methods keep every name.

Note

Only stacking is lazy. Shape operations such as `squeeze`, `unsqueeze`,
`permute`, and `view` return a regular TensorDict. The legacy lazy
versions of these operations exist only with
[`set_lazy_legacy()`](../reference/generated/tensordict.set_lazy_legacy.html#tensordict.set_lazy_legacy) set to `True`. This mode is deprecated
and will be removed in tensordict 0.17.

### Different keys and batch sizes

A lazy stack accepts tensordicts with different keys. The keys that are not in
every tensordict are exclusive keys. `keys()` does not list them, and you
cannot read them from the stack. Read them from one tensordict of the stack
instead:

```
>>> mixed = lazy_stack([
... TensorDict({"pos": torch.zeros(3)}, batch_size=[3]),
... TensorDict({"pos": torch.zeros(3), "charge": torch.zeros(3)}, batch_size=[3]),
... ])
>>> list(mixed.keys())
['pos']
>>> mixed[1]["charge"].shape
torch.Size([3])
```

A lazy stack also accepts tensordicts with different batch sizes, if they have
the same number of batch dimensions. The batch size then shows `-1` for each
dimension that differs. Here, two systems have 2 and 4 particles:

```
>>> ragged = lazy_stack([
... TensorDict({"pos": torch.zeros(2, 3)}, batch_size=[2]),
... TensorDict({"pos": torch.zeros(4, 3)}, batch_size=[4]),
... ])
>>> ragged.batch_size
torch.Size([2, -1])
>>> ragged[1].batch_size
torch.Size([4])
```

## Which function to use

The table gives the type that each function returns, for a list of two
tensordicts of each kind. "Lazy stack" means a
[`LazyStackedTensorDict`](../reference/generated/tensordict.LazyStackedTensorDict.html#tensordict.LazyStackedTensorDict), and "raises" means that the
function raises an error.

| Inputs | `torch.stack` | `lazy_stack` | `maybe_dense_stack` | `torch.cat` | `pad_sequence` |
| --- | --- | --- | --- | --- | --- |
| Same keys and shapes | `TensorDict` | lazy stack | `TensorDict` | `TensorDict` | `TensorDict` |
| Entry shapes differ | raises | lazy stack | lazy stack | raises | raises, unless only the `pad_dim` dimension differs |
| Keys differ | raises | lazy stack | lazy stack | raises `KeyError` | raises `TypeError` |
| Batch sizes differ (same number of dimensions) | raises | lazy stack, `-1` in the batch size | lazy stack, `-1` in the batch size | `TensorDict` | `TensorDict` (padded) |
| Tensorclass instances | the tensorclass | the tensorclass | the tensorclass | the tensorclass | the tensorclass |
| Lazy stacks | lazy stack | lazy stack | lazy stack | lazy stack | `TensorDict` |

Use these rules to choose:

- If the inputs always have the same keys and shapes, use `torch.stack`. It
gives a regular TensorDict: a read returns the stored tensor, without a new
stack.
- If you must write back into the inputs, or if the keys or shapes can differ,
use `lazy_stack`.
- If the inputs usually match but sometimes do not, use
`maybe_dense_stack`. It gives a dense stack when it can, and a lazy stack
otherwise.

## Variable-size data

Some data does not have the same shape for each element: sequences of
different lengths, graphs with different numbers of nodes, strings. A dense
TensorDict cannot hold it, because each entry is one tensor.

### Lazy stacks

A lazy stack holds tensordicts whose entries have different shapes. The shape
of such an entry shows `-1` for each dimension that differs. `stacked[key]`
then raises a `RuntimeError`. To read the entry, pass one of these keyword
arguments to [`get()`](../reference/generated/tensordict.TensorDictBase.get.html#tensordict.TensorDictBase.get):

- `as_list=True` returns a list with one tensor for each element.
- `as_padded_tensor=True` returns one tensor, padded to the longest element.
Set the padding with `padding_value` (default `0.0`) and
`padding_side` (`"right"` or `"left"`).
- `as_nested_tensor=True` returns a nested tensor ([`torch.nested`](https://docs.pytorch.org/docs/stable/nested.html#module-torch.nested)). The
default layout is `torch.jagged`. Set another one with `layout`.

A dense tensordict returns the stored tensor for `as_list=True` and
`as_nested_tensor=True`, and warns. Starting with TensorDict 0.17, the
default will change to returning `list(tensor.unbind(0))` and a nested
tensor, as a lazy stack does.

For example, with three graphs of 2, 3, and 1 nodes, and 2 features for each
node:

```
>>> _ = torch.manual_seed(0)
>>> graphs = lazy_stack([TensorDict({"nodes": torch.randn(n, 2)}) for n in [2, 3, 1]])
>>> graphs
LazyStackedTensorDict(batch_size=[3], stack_dim=0, fields={nodes: f32[-1,2]})
>>> [t.shape for t in graphs.get("nodes", as_list=True)]
[torch.Size([2, 2]), torch.Size([3, 2]), torch.Size([1, 2])]
>>> graphs.get("nodes", as_padded_tensor=True).shape
torch.Size([3, 3, 2])
>>> graphs.get("nodes", as_nested_tensor=True).is_nested
True
>>> graphs[1]["nodes"].shape
torch.Size([3, 2])
```

[`densify()`](../reference/generated/tensordict.TensorDictBase.densify.html#tensordict.TensorDictBase.densify) turns the whole lazy stack
into a regular TensorDict. The entries with different shapes become nested
tensors. A key that only some members have is filled with an empty entry for
the members that lack it (zero rows when the members that have it agree on its
feature dims), so it appears in the result although `keys()` of the lazy
stack leaves it out. Pass `layout=torch.jagged`: the
default, `torch.strided`, gives a warning from PyTorch, which recommends the
jagged layout.

```
>>> dense = graphs.densify(layout=torch.jagged)
>>> type(dense).__name__, dense["nodes"].is_nested
('TensorDict', True)
```

A TensorDict can also hold a nested tensor with the `torch.jagged` layout
directly. Its first dimension is a batch dimension:

```
>>> nt = torch.nested.nested_tensor([torch.ones(2, 2), torch.ones(3, 2)], layout=torch.jagged)
>>> data = TensorDict({"nodes": nt}, batch_size=[2])
>>> data[1]["nodes"].shape
torch.Size([3, 2])
```

Indexing (with an int, a slice, a mask or a tensor of indices), `split`,
`chunk` and [`torch.cat()`](https://docs.pytorch.org/docs/stable/generated/torch.cat.html#torch.cat) work along that dimension, although torch does
not do them on the nested tensor itself. Leaves that share their offsets, such
as token ids and their attention mask, still share them in the result.
[`torch.stack()`](https://docs.pytorch.org/docs/stable/generated/torch.stack.html#torch.stack) raises, because the result would need a second ragged
dimension: use [`lazy_stack()`](../reference/generated/tensordict.lazy_stack.html#tensordict.lazy_stack) instead.

```
>>> data = torch.cat([data, data[torch.tensor([1])]])
>>> data.batch_size, [t.shape for t in data[1:]["nodes"].unbind()]
(torch.Size([3]), [torch.Size([3, 2]), torch.Size([3, 2])])
```

### Padding with pad_sequence

[`pad_sequence()`](../reference/generated/tensordict.pad_sequence.html#tensordict.pad_sequence) pads the entries of a list of tensordicts to
the longest one along `pad_dim`, and stacks them. With `return_mask=True`,
it adds a `"masks"` entry that marks the values that are not padding. Use it
to get one dense TensorDict, for example for a model that takes a padded
batch. Here, three recordings of a 2-channel sensor have 2, 3, and 1 samples:

```
>>> from tensordict import pad_sequence
>>> tds = [TensorDict({"signal": torch.ones(n, 2)}) for n in [2, 3, 1]]
>>> padded = pad_sequence(tds, return_mask=True)
>>> padded["signal"].shape
torch.Size([3, 3, 2])
>>> padded["masks", "signal"]
tensor([[ True, True, False],
 [ True, True, True],
 [ True, False, False]])
```

Pass `padding_side="left"` to put the padding before the data instead.
[`unpad_sequence()`](../reference/generated/tensordict.unpad_sequence.html#tensordict.unpad_sequence) goes back: it reads the masks and returns
a lazy stack of the samples, each with its own length:

```
>>> from tensordict import unpad_sequence
>>> samples = unpad_sequence(padded)
>>> [sample["signal"].shape for sample in samples.tensordicts]
[torch.Size([2, 2]), torch.Size([3, 2]), torch.Size([1, 2])]
```

`pad_sequence` raises a `RuntimeError` for 0-dim entries ("Cannot pad
scalars"). Give such entries one dimension, or stack them separately.

### Strings and other Python objects

A list of strings, one for each element of the batch, becomes a
[`NonTensorStack`](../reference/generated/tensordict.NonTensorStack.html#tensordict.NonTensorStack). The strings can have any length:

```
>>> TensorDict({"text": ["a", "bb", "ccc"]}, batch_size=[3])[1]["text"]
'bb'
```

[Non-tensor data and unbatched entries](non_tensor.html#non-tensor-data) explains non-tensor data and what reads return.

### Choose an option

- To keep each element as it is, and to write into the elements, use a lazy
stack.
- To give a model one tensor with a mask, use `pad_sequence`, or
`get(key, as_padded_tensor=True)` on a lazy stack.
- To give a model a nested tensor, use `get(key, as_nested_tensor=True)` or
`densify(layout=torch.jagged)`.
- To give a variable-length attention kernel a batch without padding, pack
the sequences, as in [Pack variable-length sequences](../cookbook.html#cookbook-packing).
- For strings and other Python objects, use a `NonTensorStack`.

## Lazy pre-allocation

To fill a tensordict one element at a time, create an empty tensordict with
the full batch size, and assign each element to an index. The first assignment
creates each entry, filled with zeros, with the batch size of the tensordict.
The next assignments write into these entries in place. In this example, a
simulation writes one state for each of its `N` steps:

```
>>> N = 4
>>> def simulate(i):
... return TensorDict({"pos": torch.full((3,), float(i)), "info": {"step": torch.tensor(i)}})
>>> data = TensorDict(batch_size=[N])
>>> for i in range(N):
... data[i] = simulate(i)
>>> data["pos"]
tensor([[0., 0., 0.],
 [1., 1., 1.],
 [2., 2., 2.],
 [3., 3., 3.]])
>>> data["info", "step"]
tensor([0, 1, 2, 3])
```

A later assignment can add a new key: the other elements of that entry are
zeros. A value with another shape raises a `RuntimeError`. The
[Filling a TensorDict row by row](../tutorials/tensordict_preallocation.html) tutorial steps through this loop.

## See also

- [`LazyStackedTensorDict`](../reference/generated/tensordict.LazyStackedTensorDict.html#tensordict.LazyStackedTensorDict), [`stack()`](../reference/generated/tensordict.stack.html#tensordict.stack),
[`lazy_stack()`](../reference/generated/tensordict.lazy_stack.html#tensordict.lazy_stack), [`maybe_dense_stack()`](../reference/generated/tensordict.maybe_dense_stack.html#tensordict.maybe_dense_stack),
[`cat()`](../reference/generated/tensordict.cat.html#tensordict.cat), and [`pad_sequence()`](../reference/generated/tensordict.pad_sequence.html#tensordict.pad_sequence) in the
[API reference](../reference/index.html#api-reference).
- The tutorials [Changing the shape of a TensorDict](../tutorials/tensordict_shapes.html) (stacking and
concatenating) and [Building tensordicts from streams](../tutorials/streamed_tensordict.html) (building lazy
stacks from a stream of data).