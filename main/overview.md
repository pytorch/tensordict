# What is TensorDict?

A [`TensorDict`](reference/generated/tensordict.TensorDict.html#tensordict.TensorDict) is a dictionary of tensors that behaves
like one tensor. You index, reshape, move, and save the whole group with one
call. TensorDict also makes sure that every tensor keeps the dimensions that
the group shares. A [`TensorClass`](reference/generated/tensordict.TensorClass.html#tensordict.TensorClass) gives the same behavior
to a class with typed fields and methods.

## TensorDict: the dict form

All the entries of a TensorDict share their first dimensions, called the
batch size. This TensorDict holds the state of 4 robots: 7 joint angles for
each robot, and the camera images in a nested TensorDict:

```
>>> import torch
>>> from tensordict import TensorDict
>>> robots = TensorDict(
... {
... "joints": torch.zeros(4, 7),
... "camera": {
... "rgb": torch.zeros(4, 3, 32, 32, dtype=torch.uint8),
... "depth": torch.zeros(4, 32, 32),
... },
... },
... batch_size=[4],
... )
```

A string, or a tuple of strings for a nested entry, reads one entry. Any
other index applies to the batch dimensions of every entry, nested entries
too. Device moves and dtype casts also apply to every entry:

```
>>> robots["camera", "depth"].shape
torch.Size([4, 32, 32])
>>> robots[:2]["camera", "rgb"].shape
torch.Size([2, 3, 32, 32])
>>> robots.to(torch.float64)["camera", "rgb"].dtype
torch.float64
```

You can also save a TensorDict to disk, send it to other processes or nodes,
and pass it to [`torch.nn.Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module) objects through [`tensordict.nn`](reference/nn.html#module-tensordict.nn).
With [`torch.func`](https://docs.pytorch.org/docs/stable/func.api.html#module-torch.func), a TensorDict of parameters makes model ensembles and
parameter edits shorter to write (see [torch.compile, torch.func, torch.export, and CUDA graphs](compile.html#compile)). TensorDict started as
part of TorchRL and is now a separate library.

## TensorClass: the class form

A [`TensorClass`](reference/generated/tensordict.TensorClass.html#tensordict.TensorClass) declares its fields as class annotations,
like a dataclass, and it has the tensor behavior of a TensorDict. A method
that you write once runs on one item or on a batch of any shape. This class
holds bounding boxes for object detection, with a batch of 8 images and 5
boxes for each image:

```
>>> from tensordict import TensorClass
>>> class Box(TensorClass):
... low: torch.Tensor # [*batch, 2], the lower corner
... high: torch.Tensor # [*batch, 2], the upper corner
...
... def area(self) -> torch.Tensor:
... return (self.high - self.low).prod(-1)
>>> boxes = Box(
... low=torch.zeros(8, 5, 2), high=torch.full((8, 5, 2), 2.0), batch_size=[8, 5]
... )
>>> boxes.area().shape
torch.Size([8, 5])
>>> boxes[0, 1].area()
tensor(4.)
>>> type(boxes.reshape(40)).__name__
'Box'
```

Use a TensorDict when your code chooses the keys at run time. Use a
TensorClass when the fields are fixed and you want methods on them.
[Which container should I use?](choosing_a_container.html#choosing-a-container) covers the other containers.

## What it replaces: a dictionary and tree_map

`torch.utils._pytree.tree_map()` applies a function to each tensor of a
nested dictionary. It does some of the work of a TensorDict, but it does not
know about a batch dimension. The examples below use a batch of 3 graphs,
each with 5 nodes and 10 edges:

```
>>> from torch.utils._pytree import tree_map
>>> graphs = {
... "nodes": torch.randn(3, 5, 16),
... "edges": {"index": torch.randint(5, (3, 2, 10)), "weight": torch.rand(3, 10)},
... }
>>> td = TensorDict(graphs, batch_size=[3])
```

To split the graphs with tree_map, you get one dictionary of tuples. You
must then build the three dictionaries by hand, and that code grows with
each level of nesting:

```
>>> unbound = tree_map(lambda x: x.unbind(0), graphs)
>>> type(unbound["edges"]["weight"])
<class 'tuple'>
>>> samples = [
... {
... "nodes": unbound["nodes"][i],
... "edges": {
... "index": unbound["edges"]["index"][i],
... "weight": unbound["edges"]["weight"][i],
... },
... }
... for i in range(3)
... ]
```

[`unbind()`](reference/generated/tensordict.TensorDictBase.unbind.html#tensordict.TensorDictBase.unbind) returns the three tensordicts
directly, at any nesting depth. [`split()`](reference/generated/tensordict.TensorDictBase.split.html#tensordict.TensorDictBase.split) and
[`chunk()`](reference/generated/tensordict.TensorDictBase.chunk.html#tensordict.TensorDictBase.chunk) work the same way:

```
>>> g0, g1, g2 = td.unbind(0)
>>> g0["edges", "weight"].shape
torch.Size([10])
```

tree_map applies an index to each tensor as it is, so the result depends on
the shape of each tensor. A TensorDict applies the index to the batch
dimensions only:

```
>>> mapped = tree_map(lambda x: x[..., :2], graphs)
>>> mapped["nodes"].shape, mapped["edges"]["weight"].shape
(torch.Size([3, 5, 2]), torch.Size([3, 2]))
>>> sliced = td[..., :2]
>>> sliced["nodes"].shape, sliced["edges", "weight"].shape
(torch.Size([2, 5, 16]), torch.Size([2, 10]))
```

A dictionary accepts a tensor of any shape. A TensorDict raises an error
when an entry does not match the batch size:

```
>>> graphs["edges"]["type"] = torch.zeros(5) # no error
>>> td["edges", "type"] = torch.zeros(5)
Traceback (most recent call last):
...
RuntimeError: batch dimension mismatch, got self.batch_size=torch.Size([3]) and value.shape=torch.Size([5]) for key 'type'. The leading dimensions of a value must match the batch size.
```

TensorDict is registered as a pytree node, so tree_map and [`torch.func`](https://docs.pytorch.org/docs/stable/func.api.html#module-torch.func)
also work on a TensorDict. The result is a TensorDict:

```
>>> type(tree_map(lambda x: x + 1, td)).__name__
'TensorDict'
```

## What it is not

A TensorDict is not *exactly* a [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict). A dict is a good model: a
TensorDict has the methods of a mapping, `keys`, `values`, `items`,
`get`, `pop`, `update`, `in`, and `**data`. But `for x in data`,
`len(data)`, `bool(data)`, and `data == other` follow the rules of a
tensor:

```
>>> isinstance(robots, dict)
False
>>> len(robots) # the size of the first batch dimension, not the number of keys
4
>>> sorted(robots.keys())
['camera', 'joints']
```

[A TensorDict is not exactly a dict](concepts/batch_size.html#batch-size-not-a-dict) lists these differences.

A TensorDict is not a [`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) either. Each entry keeps its own
dtype, device, and the dimensions after the batch dimensions.
`robots.shape` is the batch size, not the shape of an entry:

```
>>> isinstance(robots, torch.Tensor)
False
>>> robots.shape
torch.Size([4])
```

A TensorDict is not a data frame. For tables that you filter, group, and
join, use pandas or Polars. [`to_pandas()`](reference/generated/tensordict.TensorDictBase.to_pandas.html#tensordict.TensorDictBase.to_pandas) and
[`from_pandas()`](reference/generated/tensordict.TensorDictBase.from_pandas.html#tensordict.TensorDictBase.from_pandas) convert between the two.

## Where to go next

The user guide explains one concept on each page:

- [Batch size, shapes, and indexing](concepts/batch_size.html#batch-size): the batch size, indexing, shape operations, and dimension
names.
- [Keys, nested keys, and locking](concepts/keys.html#keys): string and nested keys, reading and writing entries, nested
tensordicts, and locking.
- [Stacking and variable-size data](concepts/stacking.html#stacking): dense and lazy stacks, and data with different shapes.
- [Arithmetic, apply, reductions, and gradients](concepts/operations.html#operations): arithmetic, [`apply()`](reference/generated/tensordict.TensorDictBase.apply.html#tensordict.TensorDictBase.apply),
reductions, gradients, and the in-place conventions.
- [Devices, dtypes, and memory](concepts/devices.html#devices): devices, dtypes, and memory.
- [Non-tensor data and unbatched entries](concepts/non_tensor.html#non-tensor-data): strings and other Python objects, and entries outside
the batch size.
- [TensorClass](tensorclass.html#tensorclass-guide) and [TypedTensorDict](typed_tensordict.html#typed-tensordict-guide): containers with
typed fields. [Which container should I use?](choosing_a_container.html#choosing-a-container) compares them.
- [Modules: TensorDictModule and tensordict.nn](modules.html#modules-guide): [`TensorDictModule`](reference/generated/tensordict.nn.TensorDictModule.html#tensordict.nn.TensorDictModule) and the other
modules that read and write tensordicts.
- [Common surprises](pitfalls.html#pitfalls): the behaviors that surprise new users most often.

For saving, storage, data loading, distributed training, and
[`torch.compile()`](https://docs.pytorch.org/docs/stable/generated/torch.compile.html#torch.compile), see [Save and load tensordicts](saving.html#saving), [Large datasets and storage backends](storage.html#storage), [Load data and share it between processes](data_loading.html#data-loading),
[Distributed tensordicts](distributed.html#distributed), and [torch.compile, torch.func, torch.export, and CUDA graphs](compile.html#compile).

## See also

- [`TensorDict`](reference/generated/tensordict.TensorDict.html#tensordict.TensorDict) and [`TensorClass`](reference/generated/tensordict.TensorClass.html#tensordict.TensorClass) in the
[API reference](reference/index.html#api-reference).
- The tutorials [Getting, setting, and nesting keys](tutorials/tensordict_keys.html),
[Slicing, indexing, and masking](tutorials/tensordict_slicing.html), and [Changing the shape of a TensorDict](tutorials/tensordict_shapes.html).