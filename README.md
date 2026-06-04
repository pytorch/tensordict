[![Docs](https://img.shields.io/static/v1?logo=github&style=flat&color=pink&label=docs&message=tensordict)][#docs-package]
[![Discord](https://img.shields.io/badge/Discord-blue?logo=discord&logoColor=white)](https://discord.gg/tz3TgTAe3D)
[![Python version](https://img.shields.io/pypi/pyversions/tensordict.svg)](https://www.python.org/downloads/)
[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)][#github-license]
<a href="https://pypi.org/project/tensordict"><img src="https://img.shields.io/pypi/v/tensordict" alt="pypi version"></a>
[![Downloads](https://static.pepy.tech/personalized-badge/tensordict?period=total&units=international_system&left_color=blue&right_color=orange&left_text=Downloads)][#pepy-package]
[![Conda (channel only)](https://img.shields.io/conda/vn/conda-forge/tensordict?logo=anaconda&style=flat&color=orange)][#conda-forge-package]

[#docs-package]: https://docs.pytorch.org/tensordict/stable/
[#docs-package-benchmark]: https://docs.pytorch.org/tensordict/stable/dev/bench/
[#github-license]: https://github.com/pytorch/tensordict/blob/main/LICENSE
[#pepy-package]: https://pepy.tech/project/tensordict
[#conda-forge-package]: https://anaconda.org/conda-forge/tensordict

# TensorDict

**TensorDict is a batched, nested `dict[str, Tensor]` that behaves like a tensor.**

Move it, slice it, reshape it, stack it, save it, compile it, or do arithmetic on
it: every tensor leaf follows the same operation, and one shared `batch_size`
keeps the structure honest.

```text
TensorDict(batch_size=[32])
|-- obs:      Tensor[32, 128]
|-- action:   Tensor[32]
|-- reward:   Tensor[32]
`-- next:
    `-- obs:  Tensor[32, 128]
```

[**30-second demo**](#30-second-demo) |
[**Why TensorDict**](#why-tensordict) |
[**What is new in 0.13**](#what-is-new-in-013) |
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
on_device = batch.to(device)       # moves every leaf; non-blocking internally
scaled = batch * 0.5             # arithmetic on the whole structure
merged = batch + batch           # leaf-wise TensorDict arithmetic
stacked = torch.stack([batch, batch], 0)

print(mini.shape)                # torch.Size([8])
print(stacked.shape)             # torch.Size([2, 32])
```

The object remains a mapping, but the batch acts like a tensor. That is the
point: write the operation once, apply it to every tensor that belongs to the
same example, rollout, batch, parameter set, or dataset shard.

## Why TensorDict

Plain dictionaries are flexible. TensorDict keeps that flexibility and adds the
parts tensor programs need once the code gets serious.

| With a plain `dict` | With `TensorDict` |
|---------------------|-------------------|
| Manually keep leading dimensions aligned | One `batch_size` validates the structure |
| Repeat `.to(device)` for every tensor | `td.to(device)` moves the full batch |
| Hand-roll slicing, stacking, reshaping | `td[:32]`, `torch.stack`, `td.reshape` |
| Manually recurse through nested state | Nested keys are first-class |
| Duplicate arithmetic over leaves | `td + td`, `td * scalar`, `td.abs()` |
| Invent checkpoint formats | `td.save`, `td.memmap`, `load_memmap` |
| Hope generic code keeps working | PyTorch-native APIs, `torch.compile` coverage |

Use TensorDict when the unit of data is not one tensor anymore, but it should
still move through your program like one tensor.

## Performance is part of the API

TensorDict is not just syntax for recursive Python loops. Core paths are built
for high-throughput PyTorch workloads:

- **Arithmetic dispatch**: operations such as `td + td`, `td * 0.5`, `td.abs()`
  and in-place variants apply directly to leaves and use PyTorch foreach kernels
  where available.
- **Device and host transfers**: D2H and H2D copies are dispatched across the
  full structure. TensorDict uses non-blocking leaf transfers internally when
  possible, so the common path is just `td.to(device)`; pass
  `non_blocking=False` only when you need an explicitly synchronous transfer.
- **Shape operations without boilerplate**: indexing, `view`, `reshape`,
  `permute`, `unsqueeze`, `squeeze`, `flatten`, `unflatten`, `stack` and `cat`
  operate on the batch structure rather than on hand-maintained lists of leaves.
- **Low-allocation workflows**: lazy stacks, preallocation, memory mapping and
  `inplace=True` shape-changing operations help reduce peak memory in data-heavy
  pipelines.
- **Compile-aware internals**: TensorDict is used in compiled training and RL
  loops, and the codebase carries dedicated `torch.compile` coverage for hot
  paths.

For deeper numbers, see the [benchmark notes][#docs-package-benchmark].

## What is new in 0.13

TensorDict 0.13 focuses on making structured tensor programs more practical in
large training systems:

- **Tabular import/export** for pandas, CSV, Parquet and JSON workflows.
- **More `inplace=True` shape operations**, including `gather`, `repeat`,
  `repeat_interleave`, `roll`, `reshape`, `flatten`, `unflatten` and
  `contiguous`.
- **Improved `torch.compile` behavior** for TensorClass initialization,
  dynamic-shape export, locking paths and shallow clones.
- **Safer memmap filenames by default** through robust key encoding.
- **A migration path for module state preservation** with
  `to_module(..., preserve_module_state=...)`.
- **CPU-only release wheels** for TensorDict, avoiding duplicate GPU wheel
  artifacts for a package whose compiled extension is device-independent.

## Patterns

### One batch through the whole training step

TensorDict lets datasets, models and losses agree on one container instead of a
long argument list.

```python
for batch in dataloader:
    batch = batch.to(device)
    batch = model(batch)
    loss = loss_module(batch)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

That loop can stay stable while the schema changes from classification to
segmentation, RL rollouts, model-based prediction or LLM post-training batches.

### Nested data without custom plumbing

```python
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
```

Nested keys are part of the API, not an afterthought.

### Functional modules and parameter sets

TensorDict can hold module parameters, swap them into modules, vectorize over
ensembles and make model state explicit.

```python
from tensordict import TensorDict

params = TensorDict.from_module(module)

with params.to_module(module, preserve_module_state=True):
    out = module(inputs)
```

This is the same foundation used by TorchRL modules and functional training
utilities.

### Checkpoint and share large tensor batches

```python
td = TensorDict({"tokens": tokens, "scores": scores}, batch_size=[n])
td.memmap("/tmp/batch")          # memory-map every leaf
reloaded = TensorDict.load_memmap("/tmp/batch")
```

Memory-mapped TensorDicts are useful for large offline datasets, replay buffers,
inter-process handoff and checkpointed intermediate state.

## Key features

- **Tensor-like collection ops**: indexing, slicing, device casting, dtype
  casting, reshaping, stacking and concatenation.
  [[tutorial]](https://docs.pytorch.org/tensordict/stable/tutorials/tensordict_shapes.html)
- **Nested structures** with tuple keys and predictable batch semantics.
  [[tutorial]](https://docs.pytorch.org/tensordict/stable/tutorials/tensordict_keys.html)
- **Fast memory workflows**: asynchronous transfers, memmap, consolidated
  tensors, lazy stacks and preallocation.
  [[tutorial]](https://docs.pytorch.org/tensordict/stable/tutorials/tensordict_memory.html)
- **Functional programming** with parameter TensorDicts, `to_module` and
  compatibility with `torch.vmap`.
  [[tutorial]](https://docs.pytorch.org/tensordict/stable/tutorials/functional.html)
- **`@tensorclass`**: a tensor-aware dataclass for structured tensor objects.
  [[tutorial]](https://docs.pytorch.org/tensordict/stable/tutorials/tensorclass_fashion.html)
- **Distributed and multiprocessed pipelines** across workers, devices and
  machines. [[doc]](https://docs.pytorch.org/tensordict/stable/distributed.html)
- **Serialization and memory mapping** for efficient checkpointing and dataset
  storage. [[doc]](https://docs.pytorch.org/tensordict/stable/saving.html)

For a longer tour, start with [GETTING_STARTED.md](GETTING_STARTED.md) or the
[online documentation][#docs-package].

## Installation

**With pip**:

```bash
pip install tensordict
```

**With conda**:

```bash
conda install -c conda-forge tensordict
```

**Nightly builds**:

```bash
pip install tensordict-nightly
```

**From source with an existing PyTorch install**:

```bash
pip install -e . --no-deps
```

If you use `uv` with PyTorch nightlies, keep `torch` pinned to the PyTorch wheel
index or install TensorDict with `--no-deps` so the resolver does not replace
your existing PyTorch build:

```bash
uv pip install -e . --no-deps
uv pip install -e . --prerelease=allow -f "https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html"
```

## Ecosystem

TensorDict started in reinforcement learning, where batches quickly become
nested trajectories. It is now used anywhere tensor batches are structured data:
RL rollouts, LLM post-training samples, robotics trajectories, simulation state,
model parameters, checkpointed datasets and scientific pipelines.

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

TensorDict is licensed under the MIT License. See [LICENSE](LICENSE) for details.
