# TensorDict

**A `dict[str, torch.Tensor]` that behaves like one tensor.**

[![Docs](https://img.shields.io/badge/docs-docs.pytorch.org-blue)](https://docs.pytorch.org/tensordict/stable/)
[![PyPI](https://img.shields.io/pypi/v/tensordict)](https://pypi.org/project/tensordict/)
[![Downloads](https://static.pepy.tech/badge/tensordict)](https://pepy.tech/projects/tensordict)

PyTorch code that handles groups of tensors, such as a training batch, a
particle system, or a batch of graphs, usually repeats every operation for
every tensor, and nested dicts need recursion. A TensorDict holds the group as
one object:

```python
batch = TensorDict(
    {"image": images, "label": labels, "text": {"tokens": tokens, "mask": mask}},
    batch_size=[32],  # Optionally, TensorDicts can be batched
)

batch = batch.to("cuda")                      # moves every tensor, nested ones too
train, val = batch[:24], batch[24:]           # slices every tensor
shuffled = batch[torch.randperm(len(batch))]  # one permutation for all of them
pairs = torch.stack([batch, batch])           # batch_size [2, 32]
batch["text", "mask"]                         # nested keys are tuples
flat = batch.flatten_keys(".")                # keys: image, label, text.tokens, text.mask
```

The tensors share their leading dimensions, the batch size, so indexing,
reshaping, stacking, `.to(device)`, saving, and `torch.compile` apply to all of
them at once.

TensorClass does the same for a class with typed fields and methods, like a
dataclass. A method written once runs on one item or on a batch of any shape,
and compiles:

```python
import torch
from tensordict import TensorClass

class Particles(TensorClass):
    pos: torch.Tensor   # [*batch, 3]
    vel: torch.Tensor   # [*batch, 3]
    mass: torch.Tensor  # [*batch]

    def step(self, force: torch.Tensor, dt: float) -> "Particles":
        vel = self.vel + dt * force / self.mass.unsqueeze(-1)
        return Particles(pos=self.pos + dt * vel, vel=vel, mass=self.mass, batch_size=self.batch_size)

# 64 systems of 1,000 particles each
systems = Particles(pos=torch.randn(64, 1000, 3), vel=torch.zeros(64, 1000, 3),
                    mass=torch.ones(64, 1000), batch_size=[64, 1000])

systems = systems.step(torch.randn(64, 1000, 3), dt=0.01)  # all 64 systems
one = systems[0].step(torch.randn(1000, 3), dt=0.01)        # one system
print(type(one).__name__, one.batch_size)                    # Particles torch.Size([1000])
pairs = torch.stack([systems, systems])                      # batch_size [2, 64, 1000]

step = torch.compile(Particles.step, fullgraph=True)
systems = step(systems, torch.randn(64, 1000, 3), 0.01)
systems = systems.to("cuda" if torch.cuda.is_available() else "cpu")  # moves every field
```

Used in physics ([PhysicsNeMo](https://github.com/NVIDIA/physicsnemo)),
weather ([Earth2Studio](https://github.com/NVIDIA/earth2studio),
[geoarches](https://github.com/INRIA/geoarches)), materials
([ALCHEMI Toolkit](https://github.com/NVIDIA/nvalchemi-toolkit)), genomics
([Medaka](https://github.com/nanoporetech/medaka)), microscopy
([VisCy](https://github.com/mehta-lab/VisCy)), causal inference
([causica](https://github.com/microsoft/causica)), robotics
([MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground),
[ProtoMotions](https://github.com/NVlabs/ProtoMotions)), reinforcement learning
([TorchRL](https://github.com/pytorch/rl)), and LLM post-training
([verl](https://github.com/verl-project/verl), [ROLL](https://github.com/alibaba/ROLL)).
More than
[2,000 other repositories](https://github.com/pytorch/tensordict/network/dependents)
depend on it.

[Getting started](https://docs.pytorch.org/tensordict/stable/getting_started.html) ·
[Documentation](https://docs.pytorch.org/tensordict/stable/) ·
[Tutorials](https://docs.pytorch.org/tensordict/stable/tutorials/index.html) ·
[API reference](https://docs.pytorch.org/tensordict/stable/reference/index.html) ·
[Questions](https://github.com/pytorch/tensordict/discussions)

## Install

```bash
pip install tensordict
```

Pure Python, for Python 3.10+ and PyTorch 2.11+. Nightly builds and extras:
[Installation](https://docs.pytorch.org/tensordict/stable/installation.html).

## TensorDict: keys at run time, like a dict

```python
import torch
from tensordict import TensorDict

batch = TensorDict(
    {"images": torch.randn(32, 3, 64, 64), "labels": torch.randint(10, (32,))},
    batch_size=[32],  # the leading dimension that all tensors share
)

print(batch[:8].batch_size)                 # torch.Size([8])
print(batch.reshape(4, 8)["images"].shape)  # torch.Size([4, 8, 3, 64, 64])
```

Every tensor you add must have the batch dimensions:

```python
batch["scores"] = torch.ones(31)
# RuntimeError: batch dimension mismatch, got self.batch_size=torch.Size([32]) and
# value.shape=torch.Size([31]) for key 'scores'. The leading dimensions of a value
# must match the batch size.
```

## More examples

<details>
<summary>Stack samples of different lengths, with no padding</summary>

```python
import torch
from tensordict import TensorDict, lazy_stack

episodes = [TensorDict({"obs": torch.randn(n, 4)}, batch_size=[n]) for n in (3, 5, 2)]
batch = lazy_stack(episodes)      # keeps each sample as it is
print(batch.batch_size)           # torch.Size([3, -1]): -1 marks the dimension that varies
print(batch[1]["obs"].shape)      # torch.Size([5, 4])
print(batch[:, :2]["obs"].shape)  # torch.Size([3, 2, 4]): the first 2 steps of each
```

See [Stacking and variable-size data](https://docs.pytorch.org/tensordict/stable/concepts/stacking.html).

</details>

<details>
<summary>Save a dataset to disk, then read single samples without loading it</summary>

```python
import tempfile

import torch
from tensordict import TensorDict

data = TensorDict(
    {"images": torch.zeros(10_000, 3, 32, 32, dtype=torch.uint8), "labels": torch.arange(10_000)},
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
batch = policy(TensorDict({"obs": torch.randn(10, 4)}, batch_size=[10]))
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
<summary>Compile a function that takes a TensorDict</summary>

```python
import torch
from tensordict import TensorDict

@torch.compile(fullgraph=True)
def normalize(td):
    return td.apply(lambda x: (x - x.mean()) / (x.std() + 1e-6))

td = TensorDict({"a": torch.randn(16, 3), "b": {"c": torch.randn(16)}}, batch_size=[16])
print(normalize(td).batch_size)  # torch.Size([16])
```

See [torch.compile, torch.func, torch.export, and CUDA graphs](https://docs.pytorch.org/tensordict/stable/compile.html).

</details>

## FAQ

| Question | Answer |
|---|---|
| Does it work with `torch.compile`? | Yes, and with `torch.func` (`vmap`, `grad`) and `torch.export`, because TensorDicts are pytrees. See the [compile guide](https://docs.pytorch.org/tensordict/stable/compile.html). |
| What does the container cost? | Python work per operation and per tensor, not per element. Arithmetic uses PyTorch's foreach kernels, `.to(device)` copies without blocking, and a compiled graph holds only the tensor operations. |
| Can I get plain tensors back? | Yes. `batch["images"]` is a `torch.Tensor`, `to_dict()` returns a nested `dict`, and `numpy()`, `to_pandas()`, `to_h5()`, and `to_zarr()` export. |
| What does it depend on? | PyTorch, NumPy, cloudpickle, and pyvers, plus typing_extensions on Python 3.10. |
| What is tested? | Python 3.10 to 3.15 (with free-threaded 3.14t and 3.15t) and PyTorch 2.11 to nightly, on Linux (CPU and GPU) and macOS. Each pull request also runs the TorchRL test suite. |
| Is the API stable? | It is pre-1.0. A breaking change comes after two minor releases of warnings that name the version of the change. APIs that the docs mark as experimental (TypedTensorDict, UnbatchedTensor, the UCXX pipes, and LazyStackedTensorDictStore) can change in a minor release. |
| Who maintains it? | The [PyTorch organization](https://github.com/pytorch), with 40+ releases since 2022 ([release notes](https://github.com/pytorch/tensordict/releases)). |
| When should I use something else? | For tables that you filter, group, and join (use pandas or Polars), or outside PyTorch. |

## Documentation

- [Getting started](https://docs.pytorch.org/tensordict/stable/getting_started.html): the first hour.
- The [user guide](https://docs.pytorch.org/tensordict/stable/overview.html), and how-to guides for
  [saving](https://docs.pytorch.org/tensordict/stable/saving.html),
  [large datasets](https://docs.pytorch.org/tensordict/stable/storage.html),
  [data loading](https://docs.pytorch.org/tensordict/stable/data_loading.html),
  [distributed training](https://docs.pytorch.org/tensordict/stable/distributed.html), and
  [torch.compile](https://docs.pytorch.org/tensordict/stable/compile.html).
- [Tutorials](https://docs.pytorch.org/tensordict/stable/tutorials/index.html) and the
  [API reference](https://docs.pytorch.org/tensordict/stable/reference/index.html).

## Get help and contribute

Ask in [GitHub Discussions](https://github.com/pytorch/tensordict/discussions) or on
[Discord](https://discord.com/invite/tz3TgTAe3D), report bugs in
[issues](https://github.com/pytorch/tensordict/issues), and read
[CONTRIBUTING.md](https://github.com/pytorch/tensordict/blob/main/CONTRIBUTING.md) to contribute.

## Citation

If you use TensorDict in research, cite the TorchRL paper, which introduced it.

<details>
<summary>BibTeX</summary>

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

</details>

## License

MIT. See [LICENSE](https://github.com/pytorch/tensordict/blob/main/LICENSE).
