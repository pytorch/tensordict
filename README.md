<!--- BADGES: START --->
<!---
[![Documentation](https://img.shields.io/badge/Documentation-blue.svg?style=flat)](https://pytorch.github.io/tensordict/)
--->
[![Docs - GitHub.io](https://img.shields.io/static/v1?logo=github&style=flat&color=pink&label=docs&message=tensordict)][#docs-package]
[![Discord Shield](https://dcbadge.vercel.app/api/server/tz3TgTAe3D)](https://discord.gg/tz3TgTAe3D)
[![Benchmarks](https://img.shields.io/badge/Benchmarks-blue.svg)][#docs-package-benchmark]
[![Python version](https://img.shields.io/pypi/pyversions/tensordict.svg)](https://www.python.org/downloads/)
[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)][#github-license]
<a href="https://pypi.org/project/tensordict"><img src="https://img.shields.io/pypi/v/tensordict" alt="pypi version"></a>
<a href="https://pypi.org/project/tensordict-nightly"><img src="https://img.shields.io/pypi/v/tensordict-nightly?label=nightly" alt="pypi nightly version"></a>
[![Downloads](https://static.pepy.tech/personalized-badge/tensordict?period=total&units=international_system&left_color=blue&right_color=orange&left_text=Downloads)][#pepy-package]
[![Downloads](https://static.pepy.tech/personalized-badge/tensordict-nightly?period=total&units=international_system&left_color=blue&right_color=orange&left_text=Downloads%20(nightly))][#pepy-package-nightly]
[![codecov](https://codecov.io/gh/pytorch/tensordict/branch/main/graph/badge.svg?token=9QTUG6NAGQ)][#codecov-package]
[![circleci](https://circleci.com/gh/pytorch/tensordict.svg?style=shield)][#circleci-package]
[![Conda - Platform](https://img.shields.io/conda/pn/conda-forge/tensordict?logo=anaconda&style=flat)][#conda-forge-package]
[![Conda (channel only)](https://img.shields.io/conda/vn/conda-forge/tensordict?logo=anaconda&style=flat&color=orange)][#conda-forge-package]

[#docs-package]: https://pytorch.github.io/tensordict/
[#docs-package-benchmark]: https://pytorch.github.io/tensordict/dev/bench/
[#github-license]: https://github.com/pytorch/tensordict/blob/main/LICENSE
[#pepy-package]: https://pepy.tech/project/tensordict
[#pepy-package-nightly]: https://pepy.tech/project/tensordict-nightly
[#codecov-package]: https://codecov.io/gh/pytorch/tensordict
[#circleci-package]: https://circleci.com/gh/pytorch/tensordict
[#conda-forge-package]: https://anaconda.org/conda-forge/tensordict

<!--- BADGES: END --->

# 📖 TensorDict

TensorDict is a dictionary-like class that inherits properties from tensors, making it easy to work with collections of
tensors in PyTorch. It provides a simple and intuitive way to manipulate and process tensors, allowing you to focus on
building and training your models.

[**Key Features**](#key-features) |
[**Examples**](#examples) |
[**Installation**](#installation) |
[**Citation**](#citation) |
[**License**](#license)

## Key Features
TensorDict makes your code-bases more _readable_, _compact_, _modular_ and _fast_.
It abstracts away tailored operations, making your code less error-prone as it takes care of
dispatching the operation on the leaves for you.

The key features are:

- 🧮 **Composability**: `TensorDict` generalizes `torch.Tensor` operations to collection of tensors.
- ⚡️ **Speed**: asynchronous transfer to device, fast node-to-node communication through `consolidate`, compatible with `torch.compile`.
- ✂️ **Shape operations**: Perform tensor-like operations on TensorDict instances, such as indexing, slicing or
  concatenation.
- 🌐 **Distributed / multiprocessed capabilities**: Easily distribute TensorDict instances across multiple workers,
  devices and machines.
- 💾 **Serialization** and memory-mapping
- λ **Functional programming** and compatibility with `torch.vmap`
- 📦 **Nesting**: Nest TensorDict instances to create hierarchical structures.
- ⏰ **Lazy preallocation**: Preallocate memory for TensorDict instances without initializing the tensors.
- 📝 **Specialized dataclass** for torch.Tensor ([`@tensorclass`](#tensorclass))

![tensordict.png](docs%2Ftensordict.png)

## Examples

This section presents a couple of stand-out applications of the library.
Check our [**Getting Started**](GETTING_STARTED.md) guide for an overview of TensorDict's features!

### Fast copy on device
`TensorDict` optimizes transfers from/to device to make them safe and fast.
By default, data transfers will be made asynchronously and synchronizations will be called whenever needed.
```python
# Fast and safe asynchronous copy to 'cuda'
td_cuda = TensorDict(**dict_of_tensor, device="cuda")
# Fast and safe asynchronous copy to 'cpu'
td_cpu = td_cuda.to("cpu")
# Force synchronous copy
td_cpu = td_cuda.to("cpu", non_blocking=False)
```

### Coding an optimizer
For instance, using `TensorDict` you can code the Adam optimizer as you would for a single `torch.Tensor` and apply
that to a `TensorDict` input as well. On `cuda`, these operations will rely on fused kernels, making it very fast to
execute:
```python
class Adam:
    def __init__(self, weights: TensorDict, alpha: float=1e-3,
                 beta1: float=0.9, beta2: float=0.999,
                 eps: float = 1e-6):
        # Lock for efficiency
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

### Training a model

Using tensordict primitives, most supervised training loops can be rewritten in a generic way:
```python
for i, data in enumerate(dataset):
    # the model reads and writes tensordicts
    data = model(data)
    loss = loss_module(data)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

With this level of abstraction, one can recycle a training loop for highly heterogeneous task.
Each individual step of the training loop (data collection and transform, model prediction, loss computation etc.)
can be tailored to the use case at hand without impacting the others.
For instance, the above example can be easily used across classification and segmentation tasks, among many others.

## Installation

**With Pip**:

To install the latest stable version of tensordict, simply run

```bash
pip install tensordict
```

This will work with Python 3.7 and upward as well as PyTorch 1.12 and upward.

To enjoy the latest features, one can use

```bash
pip install tensordict-nightly
```

**With Conda**:

Install `tensordict` from `conda-forge` channel.

```sh
conda install -c conda-forge tensordict
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

## Disclaimer

TensorDict is at the *beta*-stage, meaning that there may be bc-breaking changes introduced, but
they should come with a warranty.
Hopefully these should not happen too often, as the current roadmap mostly
involves adding new features and building compatibility with the broader
PyTorch ecosystem.

## License

TensorDict is licensed under the MIT License. See [LICENSE](LICENSE) for details.
