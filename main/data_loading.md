# Load data and share it between processes

A tensordict with a batch dimension is a dataset. A
[`DataLoader`](https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) reads each batch from it with one index
operation, and the batch is a tensordict. Processes on one node can also share
one tensordict, in shared memory or in memory-mapped files, and see the writes
of the other processes.
For data from Hugging Face `datasets` and for training with Lightning,
see [Use tensordict with other libraries](integrations.html#integrations).

## Use a tensordict as a dataset

A tensordict is a map-style dataset: `len(td)` is the size of its first
batch dimension, and `td[i]` is one sample (see [Batch size, shapes, and indexing](concepts/batch_size.html#batch-size)). It also
has `__getitems__`, which reads a whole batch with one index operation.
Pass `collate_fn=lambda x: x` to keep that batch as it is:

```
>>> import torch
>>> from torch.utils.data import DataLoader
>>> from tensordict import TensorDict
>>> _ = torch.manual_seed(0)
>>> data = TensorDict(
... {"image": torch.randn(10, 1, 8, 8), "label": torch.arange(10)}, batch_size=[10]
... )
>>> loader = DataLoader(data, batch_size=4, shuffle=True, collate_fn=lambda x: x)
>>> batches = list(loader)
>>> [batch["label"].tolist() for batch in batches]
[[5, 1, 2, 7], [9, 4, 6, 3], [0, 8]]
>>> type(batches[0]).__name__, batches[0].batch_size
('TensorDict', torch.Size([4]))
```

The default `collate_fn` returns that batch as it is too:

```
>>> next(iter(DataLoader(data, batch_size=4))).batch_size
torch.Size([4])
```

The same works for a memory-mapped tensordict (see [Save and load tensordicts](saving.html#saving)). Each batch
then reads only its rows from disk.

Without a `DataLoader`, index the tensordict with a permutation. With no
worker processes, this gives the same batches:

```
>>> for idx in torch.randperm(len(data)).split(4):
... batch = data[idx]
>>> batch.batch_size
torch.Size([2])
```

## A dataset that returns tensordicts

A dataset can return one tensordict for each sample, for example a list of
tensordicts, or a [`Dataset`](https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.Dataset) whose `__getitem__`
builds a tensordict. The `DataLoader` then gets a list of tensordicts for
each batch. The default `collate_fn` stacks them into one tensordict, as
[`collate()`](reference/generated/tensordict.collate.html#tensordict.collate) does (see below). A batch of tensorclasses keeps
its class:

```
>>> samples = [TensorDict({"features": torch.randn(3), "label": i}) for i in range(10)]
>>> batch = next(iter(DataLoader(samples, batch_size=4)))
>>> batch.batch_size, batch["label"]
(torch.Size([4]), tensor([0, 1, 2, 3]))
>>> from tensordict import TensorClass
>>> class Sample(TensorClass):
... features: torch.Tensor
>>> batch = next(iter(DataLoader([Sample(features=torch.randn(3)) for _ in range(10)], batch_size=4)))
>>> type(batch).__name__, batch.features.shape
('Sample', torch.Size([4, 3]))
```

`collate_fn=torch.stack` gives the same batches:

```
>>> batch = next(iter(DataLoader(samples, batch_size=4, collate_fn=torch.stack)))
>>> batch.batch_size, batch["label"]
(torch.Size([4]), tensor([0, 1, 2, 3]))
```

If the samples have different shapes, `torch.stack` raises an error. Pass
[`lazy_stack()`](reference/generated/tensordict.lazy_stack.html#tensordict.lazy_stack) instead. It returns a
[`LazyStackedTensorDict`](reference/generated/tensordict.LazyStackedTensorDict.html#tensordict.LazyStackedTensorDict), which keeps each sample as it is
(see [Stacking and variable-size data](concepts/stacking.html#stacking)). This example loads point clouds with different numbers
of points:

```
>>> from tensordict import lazy_stack
>>> clouds = [TensorDict({"points": torch.randn(n, 3)}) for n in (5, 2, 7, 3)]
>>> batch = next(iter(DataLoader(clouds, batch_size=4, collate_fn=lazy_stack)))
>>> type(batch).__name__, batch.batch_size
('LazyStackedTensorDict', torch.Size([4]))
>>> [cloud["points"].shape for cloud in batch.unbind(0)]
[torch.Size([5, 3]), torch.Size([2, 3]), torch.Size([7, 3]), torch.Size([3, 3])]
```

[`collate()`](reference/generated/tensordict.collate.html#tensordict.collate) does both: it stacks the samples densely if
their shapes match, and lazily otherwise. It also reads samples that hold
tensordicts or tensorclasses in a tuple or a dict, such as `(td, label)`
pairs, which `torch.stack` cannot stack:

```
>>> from tensordict import collate
>>> pairs = [(TensorDict({"features": torch.randn(3)}), i) for i in range(10)]
>>> features, labels = next(iter(DataLoader(pairs, batch_size=4, collate_fn=collate)))
>>> type(features).__name__, features.batch_size, labels
('TensorDict', torch.Size([4]), tensor([0, 1, 2, 3]))
>>> batch = next(iter(DataLoader(clouds, batch_size=4, collate_fn=collate)))
>>> type(batch).__name__
'LazyStackedTensorDict'
```

The default `collate_fn` gives the same batches, so you need to pass
[`collate()`](reference/generated/tensordict.collate.html#tensordict.collate) only for its keyword arguments, shown below:

```
>>> features, labels = next(iter(DataLoader(pairs, batch_size=4)))
>>> type(features).__name__, labels
('TensorDict', tensor([0, 1, 2, 3]))
>>> type(next(iter(DataLoader(clouds, batch_size=4)))).__name__
'LazyStackedTensorDict'
```

To pad some entries instead, each with its own value, or to stack them as
jagged nested tensors, pass the keyword arguments of
[`collate()`](reference/generated/tensordict.collate.html#tensordict.collate) with [`functools.partial()`](https://docs.python.org/3/library/functools.html#functools.partial). The other
entries are collated as above:

```
>>> from functools import partial
>>> tokens = [TensorDict({"ids": torch.ones(n, dtype=torch.long), "points": torch.randn(n, 3)}) for n in (2, 4, 3)]
>>> collate_fn = partial(collate, pad_keys={"ids": 7}, nested_keys=["points"], mask_key="mask")
>>> batch = next(iter(DataLoader(tokens, batch_size=3, collate_fn=collate_fn)))
>>> batch["ids"].shape, batch["mask", "ids"].sum(-1), batch["points"].is_nested
(torch.Size([3, 4]), tensor([2, 4, 3]), True)
```

## Batch samplers

A batch sampler gives the indices of a whole batch. Pass it as
`batch_sampler`. The `DataLoader` then calls `data.__getitems__` with
the list of indices of each batch, and `collate_fn` gets the tensordict:

```
>>> from torch.utils.data import BatchSampler, SequentialSampler
>>> sampler = BatchSampler(SequentialSampler(data), batch_size=4, drop_last=True)
>>> loader = DataLoader(data, batch_sampler=sampler, collate_fn=lambda x: x)
>>> [batch["label"].tolist() for batch in loader]
[[0, 1, 2, 3], [4, 5, 6, 7]]
```

Passing it as `sampler`, with `batch_size=None`, gives the same batches
through `data[indices]`, but the workers of the `DataLoader` then send
an HDF5 or zarr [`PersistentTensorDict`](reference/generated/tensordict.PersistentTensorDict.html#tensordict.PersistentTensorDict) batch as a lazy
view, which the main process reads.

## Worker processes and pinned memory

With `num_workers` greater than 0, the `DataLoader` reads the batches in
worker processes. Each worker gets the dataset when it starts:

- With the `fork` start method (the default on Linux up to Python 3.13),
the worker inherits a copy-on-write view of the memory of the main process.
- With `spawn` (the default on macOS and Windows) and `forkserver` (the
default on Linux from Python 3.14), the `DataLoader` pickles the dataset
to each worker. A memory-mapped tensordict pickles as its file names, so
the workers open the same files and do not copy the data. Pickle cannot
send a `lambda`, so define `collate_fn` as a function at the top level
of a module.

In a worker, `__getitems__` of an HDF5 or zarr
[`PersistentTensorDict`](reference/generated/tensordict.PersistentTensorDict.html#tensordict.PersistentTensorDict) reads the rows into a
[`TensorDict`](reference/generated/tensordict.TensorDict.html#tensordict.TensorDict) in memory, so the batch is not a view that
writes to the file, as it is without workers.

The worker examples start processes, so they are complete scripts. Save one
to a file and run it with `python`:

```
import tempfile

import torch
from torch.utils.data import DataLoader

from tensordict import TensorDict

def keep_batch(batch):
 return batch

if __name__ == "__main__":
 with tempfile.TemporaryDirectory() as path:
 data = TensorDict(
 {
 "image": torch.randn(1_000, 1, 8, 8),
 "label": torch.arange(1_000),
 },
 batch_size=[1_000],
 ).memmap_(path)
 loader = DataLoader(
 data, batch_size=100, shuffle=True, num_workers=2, collate_fn=keep_batch
 )
 total = sum(batch["label"].sum().item() for batch in loader)
 print(total) # 499500
```

With `pin_memory=True`, the `DataLoader` calls
[`pin_memory()`](reference/generated/tensordict.TensorDictBase.pin_memory.html#tensordict.TensorDictBase.pin_memory) on each batch, which pins every
leaf. From pinned memory, `to(device, non_blocking=True)` copies the batch
to the GPU while the CPU goes on. These lines need a GPU, so they are not run
when this page is tested:

```
>>> loader = DataLoader(data, batch_size=4, collate_fn=lambda x: x, pin_memory=True) 
>>> batch = next(iter(loader)).to("cuda", non_blocking=True)
```

Without a `DataLoader`, `td.to("cuda", non_blocking_pin=True)` pins the
leaves with several threads (`num_threads`), and copies them to the GPU
(see [Devices, dtypes, and memory](concepts/devices.html#devices)).

## Share a tensordict between processes

[`share_memory_()`](reference/generated/tensordict.TensorDictBase.share_memory_.html#tensordict.TensorDictBase.share_memory_) moves each tensor of a
tensordict to shared memory. When you then pass the tensordict to another
process, as an argument or through a `torch.multiprocessing` queue, that
process uses the same memory. Each process sees the writes of the others,
without a copy.

`share_memory_` locks the tensordict, so write into its tensors in place,
for example with `td[i] = ...` or `td["x"].add_(1)` (see
[Locking and temporary changes](concepts/keys.html#keys-locking)). In this script, four processes each simulate one
system, and write its state into one row:

```
import torch
import torch.multiprocessing as mp

from tensordict import TensorDict

def simulate(systems, i):
 systems[i] = TensorDict({"state": torch.full((3,), float(i))})

if __name__ == "__main__":
 systems = TensorDict({"state": torch.zeros(4, 3)}, batch_size=[4]).share_memory_()
 processes = [mp.Process(target=simulate, args=(systems, i)) for i in range(4)]
 for p in processes:
 p.start()
 for p in processes:
 p.join()
 print(systems["state"][:, 0]) # tensor([0., 1., 2., 3.])
```

Call `share_memory_` before you start the processes. Without it, the
result depends on the start method. With `fork`, the child writes into a
copy, and the parent does not see the writes. With `spawn`, PyTorch moves
the tensors to shared memory when it pickles them, so the parent sees the
writes.

`memmap_()` works in the same way as `share_memory_()`, with the tensors
in files: memory-mapped files in a directory if you give a path, and
anonymous temporary files if you do not. Use files if the data is larger than
RAM, or if processes that you start later must open the data. The size of
`/dev/shm` limits shared memory.

A queue works the same way. The process that gets the tensordict uses the
memory of the process that sent it:

```
import torch
import torch.multiprocessing as mp

from tensordict import TensorDict

def consume(queue):
 systems = queue.get()
 systems["state"].add_(100)

if __name__ == "__main__":
 queue = mp.Queue()
 systems = TensorDict({"state": torch.zeros(4, 3)}, batch_size=[4]).share_memory_()
 p = mp.Process(target=consume, args=(queue,))
 p.start()
 queue.put(systems)
 p.join()
 print(systems["state"][0]) # tensor([100., 100., 100.])
```

## Several processes on one dataset on disk

Processes can also open a memory-mapped dataset by its path, with
[`load_memmap()`](reference/generated/tensordict.TensorDictBase.load_memmap.html#tensordict.TensorDictBase.load_memmap). Readers that do not change the
data can pass `mode="r"` (see [Save and load tensordicts](saving.html#saving)). tensordict does not lock the
files, so give each writer its own rows (see [Share data between processes](storage.html#storage-sharing)). This
script preallocates a dataset on disk, fills it from two processes, and reads
it back:

```
import tempfile

import torch
import torch.multiprocessing as mp

from tensordict import TensorDict

def fill(path, start, stop):
 data = TensorDict.load_memmap(path)
 for i in range(start, stop):
 data[i] = TensorDict({"image": torch.full((3, 8, 8), i, dtype=torch.uint8)})

if __name__ == "__main__":
 with tempfile.TemporaryDirectory() as path:
 TensorDict.from_schema(
 {"image": ([3, 8, 8], torch.uint8)},
 batch_size=[100],
 storage="memmap",
 prefix=path,
 )
 writers = [
 mp.Process(target=fill, args=(path, 0, 50)),
 mp.Process(target=fill, args=(path, 50, 100)),
 ]
 for p in writers:
 p.start()
 for p in writers:
 p.join()
 data = TensorDict.load_memmap(path, mode="r")
 # tensor([ 0, 49, 50, 99], dtype=torch.uint8)
 print(data["image"][[0, 49, 50, 99], 0, 0, 0])
```

## Process a dataset in parallel with map

[`map()`](reference/generated/tensordict.TensorDictBase.map.html#tensordict.TensorDictBase.map) splits a tensordict into chunks along
one dimension, calls a function on each chunk in a pool of processes, and
concatenates the results. If the tensordict is memory-mapped or in shared
memory, the chunks go to the workers without a copy. Return a new
tensordict from the function, and do not add keys to its input: the chunk of
a memory-mapped tensordict is locked. If you pass a preallocated
memory-mapped tensordict as `out`, the workers write the results into it.
This script computes the kinetic energy of 8 particles in two processes:

```
import tempfile

import torch

from tensordict import TensorDict

def kinetic_energy(chunk):
 energy = 0.5 * chunk["mass"] * (chunk["vel"] ** 2).sum(-1)
 return TensorDict({"energy": energy}, batch_size=chunk.batch_size)

if __name__ == "__main__":
 with tempfile.TemporaryDirectory() as path:
 particles = TensorDict(
 {
 "mass": torch.ones(8),
 "vel": torch.arange(8.0).unsqueeze(-1).repeat(1, 3),
 },
 batch_size=[8],
 ).memmap_()
 out = TensorDict({"energy": torch.zeros(8)}, batch_size=[8]).memmap_like(path)
 particles.map(
 kinetic_energy,
 num_workers=2,
 chunksize=2,
 out=out,
 generator=torch.default_generator,
 )
 # MemoryMappedTensor([ 0.0000, 1.5000, 6.0000, 13.5000])
 print(out["energy"][:4])
```

`map` also takes `num_chunks`, a `pool` that you create, and a
`generator` that seeds each worker. Starting with TensorDict 0.17, the
default will change to drawing the seed from the global RNG;
`generator=torch.default_generator` does that now.
[`map_iter()`](reference/generated/tensordict.TensorDictBase.map_iter.html#tensordict.TensorDictBase.map_iter) returns the results one chunk at
a time.

## See also

- [Large datasets and storage backends](storage.html#storage): memory-mapped, HDF5, zarr, shared-memory, and Redis
backends.
- [Save and load tensordicts](saving.html#saving): memory-mapped directories and the `mode` of
`load_memmap`.
- [Distributed tensordicts](distributed.html#distributed): send tensordicts between nodes.
- [Train at scale](training_at_scale.html#training-at-scale): prefetch batches to the GPU, and resume a data
loader.
- The reference of [`share_memory_()`](reference/generated/tensordict.TensorDictBase.share_memory_.html#tensordict.TensorDictBase.share_memory_),
[`pin_memory()`](reference/generated/tensordict.TensorDictBase.pin_memory.html#tensordict.TensorDictBase.pin_memory),
[`map()`](reference/generated/tensordict.TensorDictBase.map.html#tensordict.TensorDictBase.map), and [`lazy_stack()`](reference/generated/tensordict.lazy_stack.html#tensordict.lazy_stack).
- Tutorials: [Using TensorDict for datasets](tutorials/data_fashion.html#sphx-glr-tutorials-data-fashion-py),
[Using tensorclasses for datasets](tutorials/tensorclass_fashion.html#sphx-glr-tutorials-tensorclass-fashion-py), and
[Batched data loading with tensorclasses](tutorials/tensorclass_imagenet.html#sphx-glr-tutorials-tensorclass-imagenet-py).