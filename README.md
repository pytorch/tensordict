[![Docs](https://img.shields.io/static/v1?logo=github&style=flat&color=pink&label=docs&message=tensordict)][#docs-package]
[![Discord](https://img.shields.io/badge/Discord-blue?logo=discord&logoColor=white)](https://discord.com/invite/tz3TgTAe3D)
[![Python version](https://img.shields.io/pypi/pyversions/tensordict.svg)](https://www.python.org/downloads/)
[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)][#github-license]
<a href="https://pypi.org/project/tensordict"><img src="https://img.shields.io/pypi/v/tensordict" alt="pypi version"></a>
[![Downloads](https://static.pepy.tech/personalized-badge/tensordict?period=total&units=international_system&left_color=blue&right_color=orange&left_text=Downloads)][#pepy-package]
[![Conda (channel only)](https://img.shields.io/conda/vn/conda-forge/tensordict?logo=anaconda&style=flat&color=orange)][#conda-forge-package]

[#docs-package]: https://docs.pytorch.org/tensordict/stable/
[#github-license]: https://github.com/pytorch/tensordict/blob/main/LICENSE
[#pepy-package]: https://pepy.tech/projects/tensordict
[#conda-forge-package]: https://anaconda.org/conda-forge/tensordict

# TensorDict

**TensorDict is a batched, nested `dict[str, Tensor]` that behaves like a tensor.**

All tensors in a TensorDict share their leading dimensions, the `batch_size`.
TensorDict makes sure that each tensor you write has these dimensions. When you
index, reshape, stack, move, save or add a TensorDict, the operation applies to
every tensor in it, the nested ones too. These tensors are the *leaves* of the
TensorDict. A TensorDict also works under `torch.compile`.

```text
TensorDict(batch_size=[32])
|-- obs:      Tensor[32, 128]
|-- action:   Tensor[32]
|-- reward:   Tensor[32]
`-- next:
    `-- obs:  Tensor[32, 128]
```

[**Getting started**](https://docs.pytorch.org/tensordict/stable/getting_started.html) |
[**Documentation**][#docs-package] |
[**30-second demo**](#30-second-demo) |
[**Why TensorDict**](#why-tensordict) |
[**Patterns**](#patterns) |
[**Installation**](#installation) |
[**Ecosystem**](#ecosystem) |
[**Citation**](#citation)

## 30-second demo

```python
import torch
from tensordict import TensorDict

batch = TensorDict(
    {
        "obs": torch.randn(32, 128),
        "action": torch.randint(0, 4, (32,)),
        "reward": torch.randn(32),
        "next": {"obs": torch.randn(32, 128)},
    },
    batch_size=[32],
)

mini = batch[:8]                 # slices every leaf
device = "cuda" if torch.cuda.is_available() else "cpu"
on_device = batch.to(device)     # moves every leaf
scaled = batch * 0.5             # arithmetic on every leaf
merged = batch + batch           # adds the leaves with the same key
stacked = torch.stack([batch, batch], 0)

print(mini.shape)                # torch.Size([8])
print(stacked.shape)             # torch.Size([2, 32])
```

The object is still a mapping, but the batch acts like a tensor. Write the
operation once, and it applies to every tensor of the same sample, rollout,
batch, parameter set or dataset shard.

## Why TensorDict

A TensorDict uses string keys like a `dict`, and it can also hold non-tensor
values. The table shows the code that it replaces.

| With a plain `dict` | With `TensorDict` |
|---------------------|-------------------|
| Keep the leading dimensions aligned by hand | One `batch_size`, checked on each write |
| Call `.to(device)` for every tensor | `td.to(device)` moves every leaf |
| Write slicing, stacking and reshaping code | `td[:32]`, `torch.stack`, `td.reshape` |
| Recurse through nested dicts by hand | Tuple keys: `td["agents", "policy"]` |
| Repeat arithmetic over the leaves | `td + td`, `td * scalar`, `td.abs()` |
| Invent a checkpoint format | `td.save`, `td.memmap`, `TensorDict.load_memmap` |
| Write new code for each new structure | `torch.stack`, `torch.cat` and `torch.compile` accept a TensorDict |

Use TensorDict when one sample of your data has more than one tensor, and you
want to index, move or save all of them with one call.

## Speed and memory

- Arithmetic such as `td + td`, `td * 0.5` and `td.abs()`, and the in-place
  versions, call one PyTorch `_foreach` kernel for all leaves when that kernel
  exists.
- `td.to(device)` copies every leaf with `non_blocking=True`. If the target is
  not a CUDA device (for example a copy from GPU to CPU), TensorDict
  synchronizes once, after the last copy. If you need blocking copies, pass
  `non_blocking=False`.
- A lazy stack keeps a list of TensorDicts, and does not copy them into one
  tensor for each key.
- Memory-mapped leaves stay on disk until you read them.
- Shape operations with `inplace=True`, such as `td.repeat(2, inplace=True)`,
  release each old leaf when its replacement is written. Peak memory stays
  close to one copy of the data.
- The test suite runs the core operations under `torch.compile`.

The [benchmark dashboard](https://docs.pytorch.org/tensordict/dev/bench/)
shows the results of the benchmarks in `benchmarks/` over time.

## Patterns

### One batch through the whole training step

A dataset, a model and a loss can share one TensorDict instead of a long list
of arguments. This loop trains a linear model on 4 mini-batches:

```python
import torch
from torch import nn
from tensordict import TensorDict
from tensordict.nn import TensorDictModule

model = TensorDictModule(nn.Linear(3, 1), in_keys=["x"], out_keys=["pred"])
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
data = TensorDict(x=torch.randn(64, 3), y=torch.randn(64, 1), batch_size=[64])

for batch in data.split(16):
    batch = model(batch)  # reads batch["x"], writes batch["pred"]
    loss = (batch["pred"] - batch["y"]).pow(2).mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

The loop does not change when the data gets new keys, for example for
classification, segmentation, RL rollouts or LLM post-training batches.

### Nested data

```python
import torch
from tensordict import TensorDict

td = TensorDict(
    {
        "agents": {
            "policy": torch.randn(64, 8),
            "value": torch.randn(64, 1),
        },
        "env": {
            "reward": torch.randn(64),
            "done": torch.zeros(64, dtype=torch.bool),
        },
    },
    batch_size=[64],
)

policy = td["agents", "policy"]
td["env", "reward"] = td["env", "reward"].clip(-1, 1)
print(td[:4]["agents"].batch_size)  # torch.Size([4])
```

A tuple key reads or writes a nested entry. Indexing and the other operations
apply to the nested TensorDicts too.

### Module parameters

A TensorDict can hold the parameters of a module. Use it to call the module
with other parameters, to stack parameter sets for `torch.vmap`, or to save
them.

```python
import torch
from torch import nn
from tensordict import TensorDict

module = nn.Linear(3, 4)
params = TensorDict.from_module(module)
zeros = params.detach().clone().zero_()

with zeros.to_module(module):  # the module uses the zero parameters here
    out = module(torch.ones(3))
print(out)  # all zeros; the original parameters are back after the block
```

TorchRL modules and functional training code use the same mechanism.

### Save and memory-map a batch

```python
import tempfile

import torch
from tensordict import TensorDict

td = TensorDict(
    {"tokens": torch.randint(0, 1000, (128, 32)), "scores": torch.randn(128)},
    batch_size=[128],
)
with tempfile.TemporaryDirectory() as path:
    td.memmap(path)  # one file per leaf
    reloaded = TensorDict.load_memmap(path)
    print(reloaded[:2]["tokens"].shape)  # torch.Size([2, 32])
```

`load_memmap` maps the files into memory, so it reads data only when you use
it. Memory-mapped TensorDicts hold offline datasets, replay buffers and
checkpoints, and several processes can read the same files.

## Key features

- Tensor operations on the whole structure: indexing, slicing, device and
  dtype casts, reshaping, stacking and concatenation.
  [[guide]](https://docs.pytorch.org/tensordict/stable/concepts/batch_size.html)
  [[tutorial]](https://docs.pytorch.org/tensordict/stable/tutorials/tensordict_shapes.html)
- Nested structures with tuple keys.
  [[guide]](https://docs.pytorch.org/tensordict/stable/concepts/keys.html)
  [[tutorial]](https://docs.pytorch.org/tensordict/stable/tutorials/tensordict_keys.html)
- Memory: non-blocking device copies
  [[guide]](https://docs.pytorch.org/tensordict/stable/concepts/devices.html),
  lazy stacks [[guide]](https://docs.pytorch.org/tensordict/stable/concepts/stacking.html),
  preallocation [[tutorial]](https://docs.pytorch.org/tensordict/stable/tutorials/tensordict_preallocation.html),
  memory mapping and consolidation [[guide]](https://docs.pytorch.org/tensordict/stable/saving.html).
- Functional programming with parameter TensorDicts, `to_module`,
  `torch.vmap` and `torch.compile`.
  [[tutorial]](https://docs.pytorch.org/tensordict/stable/tutorials/functional.html)
  [[guide]](https://docs.pytorch.org/tensordict/stable/compile.html)
- `TensorClass`: a dataclass with typed fields and the TensorDict operations.
  [[guide]](https://docs.pytorch.org/tensordict/stable/tensorclass.html)
  [[tutorial]](https://docs.pytorch.org/tensordict/stable/tutorials/tensorclass_fashion.html)
- Distributed and multi-process pipelines: send, receive and share
  TensorDicts between workers, devices and machines.
  [[guide]](https://docs.pytorch.org/tensordict/stable/distributed.html)
- Serialization and memory mapping for checkpoints and datasets.
  [[guide]](https://docs.pytorch.org/tensordict/stable/saving.html)

For a longer tour, read [Getting started](https://docs.pytorch.org/tensordict/stable/getting_started.html), then the
[documentation][#docs-package]. The changes of each release are in the
[release notes](https://github.com/pytorch/tensordict/releases).

## Installation

TensorDict requires Python 3.10 or later and PyTorch 2.13 or later.

```bash
pip install tensordict
```

or

```bash
conda install -c conda-forge tensordict
```

The [installation page](https://docs.pytorch.org/tensordict/stable/installation.html) covers nightly builds, the
optional dependencies (HDF5, Zarr, pandas, Parquet, ONNX), source installs and
the TorchRL versions.

## Ecosystem

TensorDict started in reinforcement learning, where a batch is often a nested
trajectory. These projects use it for RL rollouts, LLM post-training samples,
robotics trajectories, simulation state, model parameters, checkpointed datasets
and scientific pipelines.

| Domain | Projects |
|--------|----------|
| **Reinforcement Learning** | [TorchRL](https://github.com/pytorch/rl) (PyTorch), [DreamerV3-torch](https://github.com/NM512/dreamerv3-torch), [Dreamer4](https://github.com/nicklashansen/dreamer4), [SkyRL](https://github.com/NovaSky-AI/SkyRL) |
| **LLM Post-Training** | [verl](https://github.com/verl-project/verl), [ROLL](https://github.com/alibaba/ROLL) (Alibaba), [LMFlow](https://github.com/OptimalScale/LMFlow), [LoongFlow](https://github.com/baidu-baige/LoongFlow) (Baidu) |
| **Robotics and Simulation** | [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground) (Google DeepMind), [ProtoMotions](https://github.com/NVlabs/ProtoMotions) (NVIDIA), [holosoma](https://github.com/amazon-far/holosoma) (Amazon) |
| **Physics and Scientific ML** | [PhysicsNeMo](https://github.com/NVIDIA/physicsnemo) (NVIDIA) |
| **Genomics** | [Medaka](https://github.com/nanoporetech/medaka) (Oxford Nanopore) |

## Citation

If you use TensorDict, please cite the TorchRL paper:

```bibtex
@misc{bou2023torchrl,
      title={TorchRL: A data-driven decision-making library for PyTorch},
      author={Albert Bou and Matteo Bettini and Sebastian Dittert and Vikash Kumar and Shagun Sodhani and Xiaomeng Yang and Gianni De Fabritiis and Vincent Moens},
      year={2023},
      eprint={2306.00577},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## License

TensorDict is licensed under the MIT License. See [LICENSE][#github-license] for details.
