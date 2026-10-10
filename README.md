# TensorDict

**A dictionary of PyTorch tensors that behaves like one tensor.**

[![PyPI](https://img.shields.io/pypi/v/tensordict)](https://pypi.org/project/tensordict/)
[![conda-forge](https://img.shields.io/conda/vn/conda-forge/tensordict)](https://anaconda.org/conda-forge/tensordict)
[![Python](https://img.shields.io/pypi/pyversions/tensordict)](https://pypi.org/project/tensordict/)
[![Downloads per month](https://static.pepy.tech/badge/tensordict/month)](https://pepy.tech/projects/tensordict)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](https://github.com/pytorch/tensordict/blob/main/LICENSE)

In PyTorch code, a batch is rarely one tensor. A reinforcement-learning step
holds observations, actions, rewards and the next observations. An LLM
training sample holds tokens, masks, log-probabilities and rewards. In a plain
`dict`, every operation on such a batch is a loop over its tensors, and nested
dicts need recursion. A `TensorDict` keeps the tensors and their shared batch
dimensions together, so each operation is one call:

```diff
- batch = {key: value.to("cuda") for key, value in batch.items()}
- first8 = {key: value[:8] for key, value in batch.items()}
- stacked = {key: torch.stack([b[key] for b in batches]) for key in batches[0]}
+ batch = batch.to("cuda")
+ first8 = batch[:8]
+ stacked = torch.stack(batches)
```

[Getting started](https://docs.pytorch.org/tensordict/stable/getting_started.html) ·
[Documentation](https://docs.pytorch.org/tensordict/stable/) ·
[Tutorials](https://docs.pytorch.org/tensordict/stable/tutorials/index.html) ·
[API reference](https://docs.pytorch.org/tensordict/stable/reference/index.html) ·
[Questions](https://github.com/pytorch/tensordict/discussions)

## Install

```bash
pip install tensordict
```

Or use `conda install -c conda-forge tensordict`. TensorDict needs Python 3.10
or later and PyTorch 2.13 or later. For nightly builds, optional extras and
installs from source, see
[Installation](https://docs.pytorch.org/tensordict/stable/installation.html).

## Example

```python
import torch
from tensordict import TensorDict

batch = TensorDict(
    obs=torch.randn(32, 3, 64, 64),
    action=torch.randint(4, (32,)),
    reward=torch.randn(32, 1),
    next={"obs": torch.randn(32, 3, 64, 64)},  # a nested dict becomes a nested TensorDict
    batch_size=[32],                            # the leading dimension that all tensors share
)

print(batch[:8].batch_size)                    # torch.Size([8])
print(batch["next", "obs"].shape)              # torch.Size([32, 3, 64, 64])
print(torch.stack([batch, batch]).batch_size)  # torch.Size([2, 32])
print(batch.reshape(4, 8)["reward"].shape)     # torch.Size([4, 8, 1])

device = "cuda" if torch.cuda.is_available() else "cpu"
batch = batch.to(device)                       # moves every tensor, nested ones too
```

The batch size is a rule, not a label. TensorDict checks it each time you add a
tensor:

```python
batch["mask"] = torch.ones(31)
# RuntimeError: batch dimension mismatch, got self.batch_size=torch.Size([32]) and
# value.shape=torch.Size([31]) for key 'mask'. The leading dimensions of a value
# must match the batch size.
```

## What people use it for

| Work | What TensorDict does there | Used by |
|---|---|---|
| Reinforcement learning | Holds rollouts and replay-buffer samples, and lets policies read and write named entries. | [TorchRL](https://github.com/pytorch/rl), [SkyRL](https://github.com/NovaSky-AI/SkyRL) |
| LLM post-training | Keeps prompts, responses, masks, log-probabilities and rewards in one batch object that you split, pad and move together. | [verl](https://github.com/verl-project/verl), [ROLL](https://github.com/alibaba/ROLL), [LMFlow](https://github.com/OptimalScale/LMFlow) |
| Robotics and simulation | Stores the state of many parallel environments as one batched structure. | [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground), [ProtoMotions](https://github.com/NVlabs/ProtoMotions), [holosoma](https://github.com/amazon-far/holosoma) |
| Scientific ML and genomics | Groups the fields of a sample (a mesh, a read) with their batch dimensions. | [PhysicsNeMo](https://github.com/NVIDIA/physicsnemo), [Medaka](https://github.com/nanoporetech/medaka) |
| Datasets larger than memory | Writes each tensor to a memory-mapped file, and reads samples back with the same indexing as in RAM. | the TorchRL replay buffers on disk |
| Model parameters as data | Turns the parameters of a module into a TensorDict for functional calls, ensembles with `torch.vmap` and parameter averaging. | the TorchRL loss modules |

GitHub lists [more than 2,000 repositories](https://github.com/pytorch/tensordict/network/dependents)
that depend on TensorDict.

## Before you build on it

| Question | Answer |
|---|---|
| Who maintains it? | It is part of the [PyTorch organization](https://github.com/pytorch) on GitHub, and has had more than 40 releases since November 2022 ([release notes](https://github.com/pytorch/tensordict/releases)). |
| How is it tested? | Every pull request runs the test suite on Linux (CPU and GPU) and macOS, with Python 3.10 to 3.14 and with PyTorch 2.13 (the oldest supported), the latest release and nightly builds. It also runs the TorchRL test suite against the change. |
| What does it depend on? | PyTorch, NumPy, cloudpickle, packaging, pyvers, and orjson on Python before 3.13. TensorDict itself is pure Python. |
| Does it work with `torch.compile`? | Yes. A TensorDict is a pytree, so `torch.compile`, `torch.func` (`vmap`, `grad`) and `torch.export` see the tensors inside it. The [compile guide](https://docs.pytorch.org/tensordict/stable/compile.html) lists what recompiles and the known limits. |
| Can I get plain tensors back? | Yes. Each entry is a regular `torch.Tensor` (`batch["obs"]`). `to_dict()` returns a nested `dict` of tensors, and `numpy()`, `to_pandas()`, `to_h5()` and `to_zarr()` export to other formats. |
| How stable is the API? | TensorDict is not at version 1.0 yet. Before a breaking change, the affected code warns for two minor releases, and the warning names the version that makes the change. |
| License | [MIT](https://github.com/pytorch/tensordict/blob/main/LICENSE). |

Use something else if your data is a table that you filter, group and join
(pandas and Polars are built for that), or if you do not use PyTorch.

## More examples

<details>
<summary>Save a dataset to disk, then read single samples without loading it</summary>

```python
import tempfile

import torch
from tensordict import TensorDict

data = TensorDict(
    images=torch.zeros(10_000, 3, 32, 32, dtype=torch.uint8),
    labels=torch.arange(10_000),
    batch_size=[10_000],
)
path = tempfile.mkdtemp()
data.memmap(path)                      # one file per tensor, plus JSON metadata
loaded = TensorDict.load_memmap(path)  # maps the files; reads no data yet
print(loaded[1234]["labels"])          # MemoryMappedTensor(1234)
```

See [Save and load tensordicts](https://docs.pytorch.org/tensordict/stable/saving.html)
and [Large datasets and storage backends](https://docs.pytorch.org/tensordict/stable/storage.html).

</details>

<details>
<summary>Build models that read and write named entries</summary>

```python
import torch
from torch import nn
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, TensorDictSequential

policy = TensorDictSequential(
    TensorDictModule(nn.Linear(4, 32), in_keys=["obs"], out_keys=["hidden"]),
    TensorDictModule(nn.Linear(32, 2), in_keys=["hidden"], out_keys=["action"]),
)
batch = policy(TensorDict(obs=torch.randn(10, 4), batch_size=[10]))
print(batch["action"].shape)  # torch.Size([10, 2])
```

See [Modules: TensorDictModule and tensordict.nn](https://docs.pytorch.org/tensordict/stable/modules.html).

</details>

<details>
<summary>Run an ensemble of models with <code>torch.vmap</code></summary>

```python
import torch
from torch import nn
from tensordict import TensorDict

models = [nn.Linear(4, 2) for _ in range(5)]
params = TensorDict.from_modules(*models)  # the parameters of the 5 models, batch_size [5]
base = models[0]

def call(p, x):
    with p.to_module(base):
        return base(x)

out = torch.vmap(call, (0, None))(params, torch.randn(3, 4))
print(out.shape)  # torch.Size([5, 3, 2])
```

See the [functional programming tutorial](https://docs.pytorch.org/tensordict/stable/tutorials/functional.html).

</details>

<details>
<summary>Declare a typed batch with <code>TensorClass</code></summary>

```python
import torch
from tensordict import TensorClass

class Transition(TensorClass):
    obs: torch.Tensor
    reward: torch.Tensor

data = Transition(obs=torch.randn(8, 3), reward=torch.zeros(8), batch_size=[8])
print(data[:2].obs.shape)                    # torch.Size([2, 3])
print(torch.stack([data, data]).batch_size)  # torch.Size([2, 8])
```

See [TensorClass](https://docs.pytorch.org/tensordict/stable/tensorclass.html).

</details>

<details>
<summary>Compile a function that takes a TensorDict</summary>

```python
import torch
from tensordict import TensorDict

@torch.compile(fullgraph=True)
def normalize(td):
    return td.apply(lambda x: (x - x.mean()) / (x.std() + 1e-6))

td = TensorDict(a=torch.randn(16, 3), b={"c": torch.randn(16)}, batch_size=[16])
print(normalize(td).batch_size)  # torch.Size([16])
```

See [torch.compile, torch.func, torch.export and CUDA graphs](https://docs.pytorch.org/tensordict/stable/compile.html).

</details>

## Documentation

- [Getting started](https://docs.pytorch.org/tensordict/stable/getting_started.html):
  the first hour, from a first TensorDict to saving one.
- [User guide](https://docs.pytorch.org/tensordict/stable/overview.html): batch
  size, keys, stacking, operations, devices, non-tensor data, TensorClass and
  modules.
- How-to guides: [save and load](https://docs.pytorch.org/tensordict/stable/saving.html),
  [large datasets](https://docs.pytorch.org/tensordict/stable/storage.html),
  [data loading](https://docs.pytorch.org/tensordict/stable/data_loading.html),
  [distributed](https://docs.pytorch.org/tensordict/stable/distributed.html) and
  [torch.compile](https://docs.pytorch.org/tensordict/stable/compile.html).
- [Tutorials](https://docs.pytorch.org/tensordict/stable/tutorials/index.html)
  and the [API reference](https://docs.pytorch.org/tensordict/stable/reference/index.html).

## Get help and contribute

Ask questions in [GitHub Discussions](https://github.com/pytorch/tensordict/discussions)
or on [Discord](https://discord.com/invite/tz3TgTAe3D), and report bugs in
[issues](https://github.com/pytorch/tensordict/issues). To contribute, read
[CONTRIBUTING.md](https://github.com/pytorch/tensordict/blob/main/CONTRIBUTING.md).

## Citation

If you use TensorDict in your research, cite the TorchRL paper, which
introduced it:

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

TensorDict is released under the
[MIT License](https://github.com/pytorch/tensordict/blob/main/LICENSE).
