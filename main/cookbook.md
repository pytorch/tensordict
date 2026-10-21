# How-to recipes

Short recipes for common training and data tasks. Each one answers a "how
do I ... with tensordict?" question, and links to the page that explains the
details.

The examples run on a CPU, in order, in one Python session. They start with
these imports and a temporary directory, `root`:

```
>>> import tempfile
>>> from pathlib import Path
>>> import torch
>>> from torch import nn
>>> from torch.utils.data import DataLoader
>>> from tensordict import TensorDict
>>> tmp = tempfile.TemporaryDirectory()
>>> root = Path(tmp.name)
>>> _ = torch.manual_seed(0)
```

## Batch samples of different lengths

Sequences, point clouds, and graphs often have a different length for each
sample. [`pad_sequence()`](reference/generated/tensordict.pad_sequence.html#tensordict.pad_sequence) pads the samples to the longest one
and stacks them. With `return_mask=True`, it adds a `"masks"` entry that
marks the values that are not padding.
[`unpad_sequence()`](reference/generated/tensordict.unpad_sequence.html#tensordict.unpad_sequence) gives back the samples:

```
>>> from tensordict import pad_sequence, unpad_sequence
>>> samples = [
... TensorDict({"tokens": torch.arange(n), "embedding": torch.ones(n, 4)}, batch_size=[n])
... for n in (3, 1, 2)
... ]
>>> padded = pad_sequence(samples, return_mask=True)
>>> padded["tokens"]
tensor([[0, 1, 2],
 [0, 0, 0],
 [0, 1, 0]])
>>> padded["masks", "tokens"]
tensor([[ True, True, True],
 [ True, False, False],
 [ True, True, False]])
>>> [s.batch_size for s in unpad_sequence(padded).tensordicts]
[torch.Size([3]), torch.Size([1]), torch.Size([2])]
```

To skip the padding, stack the samples with [`lazy_stack()`](reference/generated/tensordict.lazy_stack.html#tensordict.lazy_stack),
and read an entry as a nested tensor with the jagged layout. Its
`values()` holds the samples one after the other, and its `offsets()`
gives the index where each sample starts, the layout of the `cu_seqlens`
argument of variable-length attention kernels:

```
>>> from tensordict import lazy_stack
>>> ragged = lazy_stack(samples)
>>> nt = ragged.get("tokens", as_nested_tensor=True)
>>> nt.values()
tensor([0, 1, 2, 0, 0, 1])
>>> nt.offsets()
tensor([0, 3, 4, 6])
>>> nt.offsets().diff() # the length of each sample
tensor([3, 1, 2])
```

[Variable-size data](concepts/stacking.html#stacking-variable-size) compares these options.

## Load a memory-mapped dataset with a DataLoader

Save the dataset once with [`memmap()`](reference/generated/tensordict.TensorDictBase.memmap.html#tensordict.TensorDictBase.memmap). Each
leaf goes to its own file. [`load_memmap()`](reference/generated/tensordict.TensorDictBase.load_memmap.html#tensordict.TensorDictBase.load_memmap)
opens the files, and a batch reads only its rows from disk. A tensordict
reads a whole batch with one index operation, and
[`collate()`](reference/generated/tensordict.collate.html#tensordict.collate) keeps that batch as it is. Unlike a `lambda`,
`collate` also works with worker processes that pickle it:

```
>>> from tensordict import collate
>>> _ = TensorDict(
... {"image": torch.randn(100, 1, 8, 8), "label": torch.arange(100)}, batch_size=[100]
... ).memmap(root / "dataset")
>>> dataset = TensorDict.load_memmap(root / "dataset")
>>> device = "cuda" if torch.cuda.is_available() else "cpu"
>>> loader = DataLoader(
... dataset,
... batch_size=32,
... shuffle=True,
... collate_fn=collate,
... pin_memory=torch.cuda.is_available(),
... )
>>> for batch in loader:
... batch = batch.to(device, non_blocking=True)
>>> batch.batch_size, batch["image"].shape
(torch.Size([4]), torch.Size([4, 1, 8, 8]))
```

With `pin_memory=True`, the `DataLoader` pins each batch, so that the
copy to the GPU with `non_blocking=True` does not block the CPU.
[Load data and share it between processes](data_loading.html#data-loading) covers worker processes, batch samplers, and datasets
that return one tensordict per sample.

## Split a dataset and loop over shuffled minibatches

Index the dataset with a random permutation, and split it with
[`split()`](reference/generated/tensordict.TensorDictBase.split.html#tensordict.TensorDictBase.split). Every entry gets the same split.
For the minibatches, split a new permutation in each epoch:

```
>>> data = TensorDict(
... {"x": torch.randn(100, 3), "y": torch.randn(100, 1)}, batch_size=[100]
... )
>>> train, val = data[torch.randperm(len(data))].split([80, 20])
>>> train.batch_size, val.batch_size
(torch.Size([80]), torch.Size([20]))
>>> model = nn.Linear(3, 1)
>>> optim = torch.optim.SGD(model.parameters(), lr=0.1)
>>> for epoch in range(3):
... for idx in torch.randperm(len(train)).split(16):
... batch = train[idx]
... loss = nn.functional.mse_loss(model(batch["x"]), batch["y"])
... optim.zero_grad()
... loss.backward()
... optim.step()
... with torch.no_grad():
... val_loss = nn.functional.mse_loss(model(val["x"]), val["y"])
```

To split into `k` parts of nearly equal size, use
[`chunk()`](reference/generated/tensordict.TensorDictBase.chunk.html#tensordict.TensorDictBase.chunk).

## Set optimizer options for groups of parameters

[`from_module()`](reference/generated/tensordict.TensorDict.html#tensordict.TensorDict.from_module) collects the parameters of a
module in a tensordict, with one nested tensordict per sub-module. Its
leaves are the parameters of the module, not copies.
[`split_keys()`](reference/generated/tensordict.TensorDictBase.split_keys.html#tensordict.TensorDictBase.split_keys) with a function
`fn(key, value)` splits the leaves into groups, for example to apply weight
decay to the weight matrices only, and not to the biases and the norm
layers:

```
>>> model = nn.Sequential(nn.Linear(3, 8), nn.LayerNorm(8), nn.Linear(8, 2))
>>> params = TensorDict.from_module(model)
>>> params["0", "weight"] is model[0].weight
True
>>> matrices, others = params.split_keys(lambda key, value: value.ndim >= 2)
>>> sorted(matrices.keys(True, True))
[('0', 'weight'), ('2', 'weight')]
>>> optim = torch.optim.AdamW(
... [
... {"params": list(matrices.values(True, True)), "weight_decay": 0.01},
... {"params": list(others.values(True, True)), "weight_decay": 0.0},
... ],
... lr=1e-3,
... )
```

`keys(True, True)` and `values(True, True)` list the leaves, nested
ones included. The function gets the key as a tuple of strings, so it can
also test a name, for example `key[-1] == "bias"`. A list of keys selects
whole sub-modules: `params.split_keys(["0"])` puts the first layer in its
own group.

## Keep a moving average of the parameters

An exponential moving average (EMA) keeps a smoothed copy of the parameters.
Make the copy with `detach().clone()`. After each optimizer step, move it
towards the parameters with [`lerp_()`](reference/generated/tensordict.TensorDictBase.lerp_.html#tensordict.TensorDictBase.lerp_), under
`torch.no_grad()`, as for any update that the optimizer does not make.
To evaluate the model with the average,
[`to_module()`](reference/generated/tensordict.TensorDictBase.to_module.html#tensordict.TensorDictBase.to_module) as a context manager puts it in
the model for some lines only:

```
>>> ema = params.detach().clone()
>>> decay = 0.99
>>> x = torch.randn(16, 3)
>>> for step in range(10):
... loss = model(x).pow(2).mean()
... optim.zero_grad()
... loss.backward()
... optim.step()
... with torch.no_grad():
... _ = ema.lerp_(params, 1 - decay)
>>> with ema.to_module(model):
... y_ema = model(x)
>>> model[0].weight is params["0", "weight"] # the model has its own parameters again
True
```

`ema` is a tensordict, so `ema.to(device)` and `ema.save(path)` work
as for any other one. [Parameters as tensordicts](modules.html#modules-parameters) explains `to_module`.

## Save and resume training

Put the parameters, the moving average, and the step counter in one
tensordict, and save it with [`save()`](reference/generated/tensordict.TensorDictBase.save.html#tensordict.TensorDictBase.save).
A tensordict cannot hold the state dict of an optimizer: its keys are
integers, and its `param_groups` hold lists of Python values. Save it with
[`torch.save()`](https://docs.pytorch.org/docs/stable/generated/torch.save.html#torch.save):

```
>>> state = TensorDict({"model": params, "ema": ema, "step": step})
>>> _ = state.save(root / "checkpoint")
>>> torch.save(optim.state_dict(), root / "optim.pt")
```

To resume, load the files. `to_module(..., inplace=True)` copies the values
into the parameters of the model, so the optimizer keeps the same
parameters. [`clone()`](reference/generated/tensordict.TensorDictBase.clone.html#tensordict.TensorDictBase.clone) copies the moving
average from the files into memory:

```
>>> state = TensorDict.load_memmap(root / "checkpoint")
>>> _ = state["model"].to_module(model, inplace=True)
>>> ema = state["ema"].clone()
>>> optim.load_state_dict(torch.load(root / "optim.pt", weights_only=True))
>>> start = int(state["step"]) + 1
>>> start
10
```

[Save and load tensordicts](saving.html#saving) explains the files, and how to load a part of a checkpoint.

## Train an ensemble with torch.vmap

[`from_modules()`](reference/generated/tensordict.TensorDictBase.from_modules.html#tensordict.TensorDictBase.from_modules) stacks the parameters of several
models with the same structure, along a new first dimension. With
`as_module=True`, the result is a
[`TensorDictParams`](reference/generated/tensordict.TensorDictParams.html#tensordict.TensorDictParams), a module whose parameters an
optimizer can take. [`torch.vmap()`](https://docs.pytorch.org/docs/stable/generated/torch.vmap.html#torch.vmap) then calls one model with each set of
parameters, and one backward pass gives the gradients of all the models:

```
>>> members = [nn.Linear(3, 1) for _ in range(5)]
>>> ensemble = TensorDict.from_modules(*members, as_module=True)
>>> ensemble["weight"].shape
torch.Size([5, 1, 3])
>>> base = nn.Linear(3, 1).to("meta") # holds the structure, not the values
>>> def call(params, x):
... with params.to_module(base):
... return base(x)
>>> optim = torch.optim.Adam(ensemble.parameters(), lr=1e-2)
>>> x, y = torch.randn(32, 3), torch.randn(32, 1)
>>> for step in range(10):
... pred = torch.vmap(call, (0, None))(ensemble, x) # one prediction per member
... loss = (pred - y).pow(2).mean()
... optim.zero_grad()
... loss.backward()
... optim.step()
>>> pred.shape
torch.Size([5, 32, 1])
```

`ensemble[2]` holds the parameters of the third model. To copy them into
its own module, use `ensemble.data[2].to_module(members[2], inplace=True)`.
[Modules: TensorDictModule and tensordict.nn](modules.html#modules-guide) shows the same pattern with
[`EnsembleModule`](reference/generated/tensordict.nn.EnsembleModule.html#tensordict.nn.EnsembleModule), and [torch.compile, torch.func, torch.export, and CUDA graphs](compile.html#compile) with
[`torch.compile()`](https://docs.pytorch.org/docs/stable/generated/torch.compile.html#torch.compile).

## Process a dataset in parallel

[`map()`](reference/generated/tensordict.TensorDictBase.map.html#tensordict.TensorDictBase.map) splits a tensordict into chunks,
calls a function on each chunk in a pool of worker processes, and writes the
results into `out`. On a memory-mapped dataset, the workers read the
chunks from the files, and write the results into the files of `out`,
without a copy. The function must return a new tensordict, because the
chunks are locked.

The example starts processes, so it is a complete script. Save it to a file
and run it with `python`. It normalizes each image of a dataset on disk:

```
import tempfile

import torch

from tensordict import TensorDict

def normalize(chunk):
 image = chunk["image"].float()
 mean = image.mean(dim=(-2, -1), keepdim=True)
 std = image.std(dim=(-2, -1), keepdim=True)
 return TensorDict({"image": (image - mean) / std}, batch_size=chunk.batch_size)

if __name__ == "__main__":
 with tempfile.TemporaryDirectory() as path:
 images = torch.randint(256, (1_000, 1, 8, 8), dtype=torch.uint8)
 dataset = TensorDict({"image": images}, batch_size=[1_000]).memmap(f"{path}/raw")
 out = TensorDict.from_schema(
 {"image": ([1, 8, 8], torch.float32)},
 batch_size=[1_000],
 storage="memmap",
 prefix=f"{path}/normalized",
 )
 dataset.map(
 normalize,
 num_workers=2,
 chunksize=100,
 out=out,
 generator=torch.default_generator,
 )
 print(out["image"][0].mean().abs() < 1e-5) # tensor(True)
```

[Load data and share it between processes](data_loading.html#data-loading) lists the options of `map`, and shows how processes
share one tensordict in memory.

## Cast the floating-point leaves only

`td.to(torch.bfloat16)` casts every leaf, the integer and boolean ones
too. Pass `cast_filter=is_floating` to cast only the floating-point leaves,
as [`torch.nn.Module.to()`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.to) does. Labels and masks then keep their dtype:

```
>>> from tensordict import is_floating
>>> batch = TensorDict(
... {
... "image": torch.randn(4, 1, 8, 8),
... "label": torch.arange(4),
... "mask": torch.ones(4, 8, 8, dtype=torch.bool),
... },
... batch_size=[4],
... )
>>> bf16 = batch.to(torch.bfloat16, cast_filter=is_floating)
>>> bf16["image"].dtype, bf16["label"].dtype, bf16["mask"].dtype
(torch.bfloat16, torch.int64, torch.bool)
>>> batch.half(cast_filter=is_floating)["label"].dtype
torch.int64
```

[`is_inexact()`](reference/generated/tensordict.is_inexact.html#tensordict.is_inexact) also casts complex leaves. A tensorclass can
set a default `cast_filter` for all its casts, with
`@tensorclass(cast_filter=is_floating)` (see [`tensorclass()`](reference/generated/tensordict.tensorclass.html#tensordict.tensorclass)).

```
>>> tmp.cleanup()
```