[![Docs](https://img.shields.io/static/v1?logo=github&style=flat&color=pink&label=docs&message=tensordict)][#docs-package]
[![Discord](https://dcbadge.vercel.app/api/server/tz3TgTAe3D)](https://discord.gg/tz3TgTAe3D)
[![Python version](https://img.shields.io/pypi/pyversions/tensordict.svg)](https://www.python.org/downloads/)
[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)][#github-license]
<a href="https://pypi.org/project/tensordict"><img src="https://img.shields.io/pypi/v/tensordict" alt="pypi version"></a>
[![Downloads](https://static.pepy.tech/personalized-badge/tensordict?period=total&units=international_system&left_color=blue&right_color=orange&left_text=Downloads)][#pepy-package]
[![Conda (channel only)](https://img.shields.io/conda/vn/conda-forge/tensordict?logo=anaconda&style=flat&color=orange)][#conda-forge-package]

[#docs-package]: https://pytorch.github.io/tensordict/
[#docs-package-benchmark]: https://pytorch.github.io/tensordict/dev/bench/
[#github-license]: https://github.com/pytorch/tensordict/blob/main/LICENSE
[#pepy-package]: https://pepy.tech/project/tensordict
[#conda-forge-package]: https://anaconda.org/conda-forge/tensordict

# TensorDict

**Tensors. Organized. Fast.**

TensorDict is a composable data structure for PyTorch that lets you manipulate
collections of tensors as if they were a single tensor -- indexing, reshaping,
moving to device, stacking, and more, all applied across every leaf at once.

```python
from tensordict import TensorDict

data = TensorDict(
    obs=torch.randn(128, 84),
    action=torch.randn(128, 4),
    reward=torch.randn(128, 1),
    batch_size=[128],
)

data_gpu = data.to("cuda")      # all tensors move together
sub = data_gpu[:64]              # all tensors are sliced
stacked = torch.stack([data, data])  # works like a tensor
```

[**Key Features**](#key-features) |
[**Examples**](#examples) |
[**Installation**](#installation) |
[**Ecosystem**](#ecosystem) |
[**Citation**](#citation) |
[**License**](#license)

## Key Features

TensorDict makes your code-bases more _readable_, _compact_, _modular_ and _fast_.
It abstracts away tailored operations, dispatching them on the leaves for you.

- **Composability**: `TensorDict` generalizes `torch.Tensor` operations to collections of tensors.
- **Speed**: asynchronous transfer to device, fast node-to-node communication through `consolidate`, compatible with `torch.compile`.
- **Shape operations**: indexing, slicing, concatenation, reshaping -- everything you can do with a tensor.
- **Distributed / multiprocessed**: distribute TensorDict instances across workers, devices and machines.
- **Serialization** and memory-mapping for efficient checkpointing.
- **Functional programming** and compatibility with `torch.vmap`.
- **Nesting**: nest TensorDict instances to create hierarchical structures.
- **Lazy preallocation**: preallocate memory without initializing tensors.
- **`@tensorclass`**: a specialized dataclass for `torch.Tensor`.

## Examples

Check our [**Getting Started**](GETTING_STARTED.md) guide for a full overview of TensorDict's features.

### Before / after

Working with groups of tensors is common in ML. Without a shared structure,
every operation must be repeated for each tensor:

```python
# Without TensorDict
obs = obs.to("cuda")
action = action.to("cuda")
reward = reward.to("cuda")
next_obs = next_obs.to("cuda")

obs_batch = obs[:32]
action_batch = action[:32]
reward_batch = reward[:32]
next_obs_batch = next_obs[:32]
```

With TensorDict, all of that collapses to:

```python
# With TensorDict
data = data.to("cuda")
data_batch = data[:32]
```

This holds for any operation: `reshape`, `unsqueeze`, `permute`, `to`, indexing,
`torch.stack`, `torch.cat`, and many more.

### Generic training loops

Using TensorDict primitives, most supervised training loops can be rewritten
in a generic way:

```python
for i, data in enumerate(dataset):
    data = model(data)
    loss = loss_module(data)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

Each step of the training loop -- data loading, model prediction, loss
computation -- can be swapped independently without touching the rest.
The same loop works across classification, segmentation, RL, and more.

### Fast copy on device

By default, device transfers are asynchronous and synchronized only when needed:

```python
td_cuda = TensorDict(**dict_of_tensors, device="cuda")
td_cpu = td_cuda.to("cpu")
td_cpu = td_cuda.to("cpu", non_blocking=False)  # force synchronous
```

### Coding an optimizer

Using TensorDict you can code the Adam optimizer as you would for a single
tensor and apply it to a collection of parameters. On CUDA, these operations
use fused kernels:

```python
class Adam:
    def __init__(self, weights: TensorDict, alpha: float=1e-3,
                 beta1: float=0.9, beta2: float=0.999,
                 eps: float = 1e-6):
        weights = weights.lock_()
        self.weights = weights
        self.t = 0

        self._mu = weights.data.clone()
        self._sigma = weights.data.mul(0.0)
        self.beta1 = beta1
        self.beta2 = beta2
        self.alpha = alpha
        self.eps = eps

    def step(self):
        self._mu.mul_(self.beta1).add_(self.weights.grad, 1 - self.beta1)
        self._sigma.mul_(self.beta2).add_(self.weights.grad.pow(2), 1 - self.beta2)
        self.t += 1
        mu = self._mu.div_(1-self.beta1**self.t)
        sigma = self._sigma.div_(1 - self.beta2 ** self.t)
        self.weights.data.add_(mu.div_(sigma.sqrt_().add_(self.eps)).mul_(-self.alpha))
```

## Ecosystem

TensorDict is used across a range of domains:

| Domain | Projects |
|--------|----------|
| **Reinforcement Learning** | [TorchRL](https://github.com/pytorch/rl) (PyTorch), [DreamerV3-torch](https://github.com/NM512/dreamerv3-torch), [SkyRL](https://github.com/NovaSky-AI/SkyRL) |
| **LLM Post-Training** | [verl](https://github.com/verl-project/verl), [ROLL](https://github.com/alibaba/ROLL) (Alibaba), [LMFlow](https://github.com/OptimalScale/LMFlow) |
| **Robotics & Simulation** | [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground) (Google DeepMind), [ProtoMotions](https://github.com/NVlabs/ProtoMotions) (NVIDIA), [holosoma](https://github.com/amazon-far/holosoma) (Amazon) |
| **Physics & Scientific ML** | [PhysicsNeMo](https://github.com/NVIDIA/physicsnemo) (NVIDIA) |
| **Drug Discovery** | [Lobster](https://github.com/prescient-design/lobster) (Genentech / Prescient Design) |
| **Agents** | [cua](https://github.com/trycua/cua), [OpenManus-RL](https://github.com/OpenManus/OpenManus-RL), [Simia](https://github.com/microsoft/Simia-Agent-Training) (Microsoft) |
| **Genomics** | [Medaka](https://github.com/nanoporetech/medaka) (Oxford Nanopore) |

See the full list of [1,600+ dependent repositories](https://github.com/pytorch/tensordict/network/dependents).

## Installation

**With pip**:

```bash
pip install tensordict
```

For the latest features:

```bash
pip install tensordict-nightly
```

**With conda**:

```bash
conda install -c conda-forge tensordict
```

**With uv + PyTorch nightlies**:

If you're using a PyTorch nightly, install tensordict with `--no-deps` to prevent
uv from re-resolving `torch` from PyPI:

```bash
uv pip install -e . --no-deps
```

Or explicitly point uv at the PyTorch nightly wheel index:

```bash
uv pip install -e . --prerelease=allow -f "https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html"
```

## Citation

If you're using TensorDict, please refer to this BibTeX entry to cite this work:
```
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
