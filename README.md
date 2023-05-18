<!--- BADGES: START --->
<!---
[![Documentation](https://img.shields.io/badge/Documentation-blue.svg?style=flat)](https://pytorch-labs.github.io/tensordict/)
--->
[![Docs - GitHub.io](https://img.shields.io/static/v1?logo=github&style=flat&color=pink&label=docs&message=tensordict)][#docs-package]
[![Benchmarks](https://img.shields.io/badge/Benchmarks-blue.svg)][#docs-package-benchmark]
[![Python version](https://img.shields.io/pypi/pyversions/tensordict.svg)](https://www.python.org/downloads/)
[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)][#github-license]
<a href="https://pypi.org/project/tensordict"><img src="https://img.shields.io/pypi/v/tensordict" alt="pypi version"></a>
<a href="https://pypi.org/project/tensordict-nightly"><img src="https://img.shields.io/pypi/v/tensordict-nightly?label=nightly" alt="pypi nightly version"></a>
[![Downloads](https://static.pepy.tech/personalized-badge/tensordict?period=total&units=international_system&left_color=blue&right_color=orange&left_text=Downloads)][#pepy-package]
[![Downloads](https://static.pepy.tech/personalized-badge/tensordict-nightly?period=total&units=international_system&left_color=blue&right_color=orange&left_text=Downloads%20(nightly))][#pepy-package-nightly]
[![codecov](https://codecov.io/gh/pytorch-labs/tensordict/branch/main/graph/badge.svg?token=9QTUG6NAGQ)][#codecov-package]
[![circleci](https://circleci.com/gh/pytorch-labs/tensordict.svg?style=shield)][#circleci-package]
[![Conda - Platform](https://img.shields.io/conda/pn/conda-forge/tensordict?logo=anaconda&style=flat)][#conda-forge-package]
[![Conda (channel only)](https://img.shields.io/conda/vn/conda-forge/tensordict?logo=anaconda&style=flat&color=orange)][#conda-forge-package]

[#docs-package]: https://pytorch-labs.github.io/tensordict/
[#docs-package-benchmark]: https://pytorch-labs.github.io/tensordict/dev/bench/
[#github-license]: https://github.com/pytorch-labs/tensordict/blob/main/LICENSE
[#pepy-package]: https://pepy.tech/project/tensordict
[#pepy-package-nightly]: https://pepy.tech/project/tensordict-nightly
[#codecov-package]: https://codecov.io/gh/pytorch-labs/tensordict
[#circleci-package]: https://circleci.com/gh/pytorch-labs/tensordict
[#conda-forge-package]: https://anaconda.org/conda-forge/tensordict

<!--- BADGES: END --->

# TensorDict

[**Installation**](#installation) | [**General features**](#general) |
[**Tensor-like features**](#tensor-like-features) |  [**Distributed capabilities**](#distributed-capabilities) |
[**TensorDict for functional programming using FuncTorch**](#tensordict-for-functional-programming-using-functorch) |
[**Lazy preallocation**](#lazy-preallocation) | [**Nesting TensorDicts**](#nesting-tensordicts) | [**TensorClass**](#tensorclass)

`TensorDict` is a dictionary-like class that inherits properties from tensors,
such as indexing, shape operations, casting to device or point-to-point communication
in distributed settings.

The main purpose of TensorDict is to make code-bases more _readable_ and _modular_ by abstracting away tailored operations:
```python
for i, tensordict in enumerate(dataset):
    # the model reads and writes tensordicts
    tensordict = model(tensordict)
    loss = loss_module(tensordict)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```
With this level of abstraction, one can recycle a training loop for highly heterogeneous task.
Each individual step of the training loop (data collection and transform, model prediction, loss computation etc.)
can be tailored to the use case at hand without impacting the others.
For instance, the above example can be easily used across classification and segmentation tasks, among many others.

## Features

### General

A tensordict is primarily defined by its `batch_size` (or `shape`) and its key-value pairs:
```python
>>> from tensordict import TensorDict
>>> import torch
>>> tensordict = TensorDict({
...     "key 1": torch.ones(3, 4, 5),
...     "key 2": torch.zeros(3, 4, 5, dtype=torch.bool),
... }, batch_size=[3, 4])
```
The `batch_size` and the first dimensions of each of the tensors must be compliant.
The tensors can be of any dtype and device. Optionally, one can restrict a tensordict to
live on a dedicated device, which will send each tensor that is written there:
```python
>>> tensordict = TensorDict({
...     "key 1": torch.ones(3, 4, 5),
...     "key 2": torch.zeros(3, 4, 5, dtype=torch.bool),
... }, batch_size=[3, 4], device="cuda:0")
>>> tensordict["key 3"] = torch.randn(3, 4, device="cpu")
>>> assert tensordict["key 3"].device is torch.device("cuda:0")
```

### Tensor-like features

TensorDict objects can be indexed exactly like tensors. The resulting of indexing
a TensorDict is another TensorDict containing tensors indexed along the required dimension:
```python
>>> tensordict = TensorDict({
...     "key 1": torch.ones(3, 4, 5),
...     "key 2": torch.zeros(3, 4, 5, dtype=torch.bool),
... }, batch_size=[3, 4])
>>> sub_tensordict = tensordict[..., :2]
>>> assert sub_tensordict.shape == torch.Size([3, 2])
>>> assert sub_tensordict["key 1"].shape == torch.Size([3, 2, 5])
```

Similarly, one can build tensordicts by stacking or concatenating single tensordicts:
```python
>>> tensordicts = [TensorDict({
...     "key 1": torch.ones(3, 4, 5),
...     "key 2": torch.zeros(3, 4, 5, dtype=torch.bool),
... }, batch_size=[3, 4]) for _ in range(2)]
>>> stack_tensordict = torch.stack(tensordicts, 1)
>>> assert stack_tensordict.shape == torch.Size([3, 2, 4])
>>> assert stack_tensordict["key 1"].shape == torch.Size([3, 2, 4, 5])
>>> cat_tensordict = torch.cat(tensordicts, 0)
>>> assert cat_tensordict.shape == torch.Size([6, 4])
>>> assert cat_tensordict["key 1"].shape == torch.Size([6, 4, 5])
```

TensorDict instances can also be reshaped, viewed, squeezed and unsqueezed:
```python
>>> tensordict = TensorDict({
...     "key 1": torch.ones(3, 4, 5),
...     "key 2": torch.zeros(3, 4, 5, dtype=torch.bool),
... }, batch_size=[3, 4])
>>> print(tensordict.view(-1))
torch.Size([12])
>>> print(tensordict.reshape(-1))
torch.Size([12])
>>> print(tensordict.unsqueeze(-1))
torch.Size([3, 4, 1])
```

One can also send tensordict from device to device, place them in shared memory,
clone them, update them in-place or not, split them, unbind them, expand them etc.

If a functionality is missing, it is easy to call it using `apply()` or `apply_()`:
```python
tensordict_uniform = tensordict.apply(lambda tensor: tensor.uniform_())
```
### Distributed capabilities

Complex data structures can be cumbersome to synchronize in distributed settings.
`tensordict` solves that problem with synchronous and asynchronous helper methods
such as `recv`, `irecv`, `send` and `isend` that behave like their `torch.distributed`
counterparts:
```python
>>> # on all workers
>>> data = TensorDict({"a": torch.zeros(()), ("b", "c"): torch.ones(())}, [])
>>> # on worker 1
>>> data.isend(dst=0)
>>> # on worker 0
>>> data.irecv(src=1)
```

When nodes share a common scratch space, the
[`MemmapTensor` backend](https://pytorch-labs.github.io/tensordict/tutorials/tensordict_memory.html)
can be used
to seamlessly send, receive and read a huge amount of data.

### TensorDict for functional programming using FuncTorch

We also provide an API to use TensorDict in conjunction with [FuncTorch](https://pytorch.org/functorch).
For instance, TensorDict makes it easy to concatenate model weights to do model ensembling:
```python
>>> from torch import nn
>>> from tensordict import TensorDict
>>> from tensordict.nn import make_functional
>>> import torch
>>> from torch import vmap
>>> layer1 = nn.Linear(3, 4)
>>> layer2 = nn.Linear(4, 4)
>>> model = nn.Sequential(layer1, layer2)
>>> # we represent the weights hierarchically
>>> weights1 = TensorDict(layer1.state_dict(), []).unflatten_keys(".")
>>> weights2 = TensorDict(layer2.state_dict(), []).unflatten_keys(".")
>>> params = make_functional(model)
>>> assert (params == TensorDict({"0": weights1, "1": weights2}, [])).all()
>>> # Let's use our functional module
>>> x = torch.randn(10, 3)
>>> out = model(x, params=params)  # params is the last arg (or kwarg)
>>> # an ensemble of models: we stack params along the first dimension...
>>> params_stack = torch.stack([params, params], 0)
>>> # ... and use it as an input we'd like to pass through the model
>>> y = vmap(model, (None, 0))(x, params_stack)
>>> print(y.shape)
torch.Size([2, 10, 4])
```

Moreover, tensordict modules are compatible with `torch.fx` and `torch.compile`,
which means that you can get the best of both worlds: a codebase that is
both readable and future-proof as well as efficient and portable!


### Lazy preallocation

Pre-allocating tensors can be cumbersome and hard to scale if the list of preallocated
items varies according to the script configuration. TensorDict solves this in an elegant way.
Assume you are working with a function `foo() -> TensorDict`, e.g.
```python
def foo():
    tensordict = TensorDict({}, batch_size=[])
    tensordict["a"] = torch.randn(3)
    tensordict["b"] = TensorDict({"c": torch.zeros(2)}, batch_size=[])
    return tensordict
```
and you would like to call this function repeatedly. You could do this in two ways.
The first would simply be to stack the calls to the function:
```python
tensordict = torch.stack([foo() for _ in range(N)])
```
However, you could also choose to preallocate the tensordict:
```python
tensordict = TensorDict({}, batch_size=[N])
for i in range(N):
    tensordict[i] = foo()
```
which also results in a tensordict (when `N = 10`)
```
TensorDict(
    fields={
        a: Tensor(torch.Size([10, 3]), dtype=torch.float32),
        b: TensorDict(
            fields={
                c: Tensor(torch.Size([10, 2]), dtype=torch.float32)},
            batch_size=torch.Size([10]),
            device=None,
            is_shared=False)},
    batch_size=torch.Size([10]),
    device=None,
    is_shared=False)
```
When `i==0`, your empty tensordict will automatically be populated with empty tensors
of batch-size `N`. After that, updates will be written in-place.
Note that this would also work with a shuffled series of indices (pre-allocation does
not require you to go through the tensordict in an ordered fashion).


### Nesting TensorDicts

It is possible to nest tensordict. The only requirement is that the sub-tensordict should be indexable
under the parent tensordict, i.e. its batch size should match (but could be longer than) the parent
batch size.

We can switch easily between hierarchical and flat representations.
For instance, the following code will result in a single-level tensordict with keys `"key 1"` and `"key 2.sub-key"`:
```python
>>> tensordict = TensorDict({
...     "key 1": torch.ones(3, 4, 5),
...     "key 2": TensorDict({"sub-key": torch.randn(3, 4, 5, 6)}, batch_size=[3, 4, 5])
... }, batch_size=[3, 4])
>>> tensordict_flatten = tensordict.flatten_keys(separator=".")
```

Accessing nested tensordicts can be achieved with a single index:
```python
>>> sub_value = tensordict["key 2", "sub-key"]
```

## TensorClass

Content flexibility comes at the cost of predictability.
In some cases, developers may be looking for data structure with a more explicit behavior.
`tensordict` provides a `dataclass`-like decorator that allows for the creation of custom dataclasses that support
the tensordict operations:
```python
>>> from tensordict.prototype import tensorclass
>>> import torch
>>>
>>> @tensorclass
... class MyData:
...    image: torch.Tensor
...    mask: torch.Tensor
...    label: torch.Tensor
...
...    def mask_image(self):
...        return self.image[self.mask.expand_as(self.image)].view(*self.batch_size, -1)
...
...    def select_label(self, label):
...        return self[self.label == label]
...
>>> images = torch.randn(100, 3, 64, 64)
>>> label = torch.randint(10, (100,))
>>> mask = torch.zeros(1, 64, 64, dtype=torch.bool).bernoulli_().expand(100, 1, 64, 64)
>>>
>>> data = MyData(images, mask, label=label, batch_size=[100])
>>>
>>> print(data.select_label(1))
MyData(
    image=Tensor(torch.Size([11, 3, 64, 64]), dtype=torch.float32),
    label=Tensor(torch.Size([11]), dtype=torch.int64),
    mask=Tensor(torch.Size([11, 1, 64, 64]), dtype=torch.bool),
    batch_size=torch.Size([11]),
    device=None,
    is_shared=False)
>>> print(data.mask_image().shape)
torch.Size([100, 6117])
>>> print(data.reshape(10, 10))
MyData(
    image=Tensor(torch.Size([10, 10, 3, 64, 64]), dtype=torch.float32),
    label=Tensor(torch.Size([10, 10]), dtype=torch.int64),
    mask=Tensor(torch.Size([10, 10, 1, 64, 64]), dtype=torch.bool),
    batch_size=torch.Size([10, 10]),
    device=None,
    is_shared=False)
```
As this example shows, one can write a specific data structures with dedicated methods while still enjoying the TensorDict
artifacts such as shape operations (e.g. reshape or permutations), data manipulation (indexing, `cat` and `stack`) or calling
arbitrary functions through the `apply` method (and many more).

Tensorclasses support nesting and, in fact, all the TensorDict features.


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
@software{TensorDict,
  author = {Moens, Vincent},
  title = {{TensorDict: your PyTorch universal data carrier}},
  url = {https://github.com/pytorch-labs/tensordict},
  version = {0.1.2},
  year = {2023}
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
