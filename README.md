<!--- BADGES: START --->
<!---
[![Documentation](https://img.shields.io/badge/Documentation-blue.svg?style=flat)](https://pytorch.github.io/tensordict/)
--->
[![Docs - GitHub.io](https://img.shields.io/static/v1?logo=github&style=flat&color=pink&label=docs&message=tensordict)][#docs-package]
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

# TensorDict

[**Installation**](#installation) | [**General features**](#general) |
[**Tensor-like features**](#tensor-like-features) |  [**Distributed capabilities**](#distributed-capabilities) |
[**TensorDict for functional programming**](#tensordict-for-functional-programming) |
[**TensorDict for parameter serialization](#tensordict-for-parameter-serialization) |
[**Lazy preallocation**](#lazy-preallocation) | [**Nesting TensorDicts**](#nesting-tensordicts) | [**TensorClass**](#tensorclass)

`TensorDict` is a dictionary-like class that inherits properties from tensors,
such as indexing, shape operations, casting to device or point-to-point communication
in distributed settings.

The main purpose of TensorDict is to make code-bases more _readable_ and _modular_ by abstracting away tailored operations:
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

## Features

### General

A tensordict is primarily defined by its `batch_size` (or `shape`) and its key-value pairs:
```python
>>> from tensordict import TensorDict
>>> import torch
>>> data = TensorDict({
...     "key 1": torch.ones(3, 4, 5),
...     "key 2": torch.zeros(3, 4, 5, dtype=torch.bool),
... }, batch_size=[3, 4])
```
The `batch_size` and the first dimensions of each of the tensors must be compliant.
The tensors can be of any dtype and device. Optionally, one can restrict a tensordict to
live on a dedicated device, which will send each tensor that is written there:
```python
>>> data = TensorDict({
...     "key 1": torch.ones(3, 4, 5),
...     "key 2": torch.zeros(3, 4, 5, dtype=torch.bool),
... }, batch_size=[3, 4], device="cuda:0")
>>> data["key 3"] = torch.randn(3, 4, device="cpu")
>>> assert data["key 3"].device is torch.device("cuda:0")
```

But that is not all, you can also store nested values in a tensordict:
```python
>>> data["nested", "key"] = torch.zeros(3, 4) # the batch-size must match
```
and any nested tuple structure will be unravelled to make it easy to read code and
write ops programmatically:
```python
>>> data["nested", ("supernested", ("key",))] = torch.zeros(3, 4) # the batch-size must match
>>> assert (data["nested", "supernested", "key"] == 0).all()
>>> assert (("nested",), "supernested", (("key",),)) in data.keys(include_nested=True)  # this works too!
```

You can also store non-tensor data in tensordicts:

```python
>>> data = TensorDict({"a-tensor": torch.randn(1, 2)}, batch_size=[1, 2])
>>> data["non-tensor"] = "a string!"
>>> assert data["non-tensor"] == "a string!"
```

### Tensor-like features

TensorDict objects can be indexed exactly like tensors. The resulting of indexing
a TensorDict is another TensorDict containing tensors indexed along the required dimension:
```python
>>> data = TensorDict({
...     "key 1": torch.ones(3, 4, 5),
...     "key 2": torch.zeros(3, 4, 5, dtype=torch.bool),
... }, batch_size=[3, 4])
>>> sub_tensordict = data[..., :2]
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
>>> data = TensorDict({
...     "key 1": torch.ones(3, 4, 5),
...     "key 2": torch.zeros(3, 4, 5, dtype=torch.bool),
... }, batch_size=[3, 4])
>>> print(data.view(-1))
torch.Size([12])
>>> print(data.reshape(-1))
torch.Size([12])
>>> print(data.unsqueeze(-1))
torch.Size([3, 4, 1])
```

One can also send tensordict from device to device, place them in shared memory,
clone them, update them in-place or not, split them, unbind them, expand them etc.

If a functionality is missing, it is easy to call it using `apply()` or `apply_()`:
```python
tensordict_uniform = data.apply(lambda tensor: tensor.uniform_())
```

``apply()`` can also be great to filter a tensordict, for instance:
```python
data = TensorDict({"a": torch.tensor(1.0, dtype=torch.float), "b": torch.tensor(1, dtype=torch.int64)}, [])
data_float = data.apply(lambda x: x if x.dtype == torch.float else None) # contains only the "a" key
assert "b" not in data_float
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
[`MemmapTensor` backend](https://pytorch.github.io/tensordict/tutorials/tensordict_memory.html)
can be used
to seamlessly send, receive and read a huge amount of data.

### TensorDict for functional programming

We also provide an API to use TensorDict in conjunction with [FuncTorch](https://pytorch.org/functorch).
For instance, TensorDict makes it easy to concatenate model weights to do model ensembling:
```python
>>> from torch import nn
>>> from tensordict import TensorDict
>>> import torch
>>> from torch import vmap
>>> layer1 = nn.Linear(3, 4)
>>> layer2 = nn.Linear(4, 4)
>>> model = nn.Sequential(layer1, layer2)
>>> params = TensorDict.from_module(model)
>>> # we represent the weights hierarchically
>>> weights1 = TensorDict(layer1.state_dict(), []).unflatten_keys(".")
>>> weights2 = TensorDict(layer2.state_dict(), []).unflatten_keys(".")
>>> assert (params == TensorDict({"0": weights1, "1": weights2}, [])).all()
>>> # Let's use our functional module
>>> x = torch.randn(10, 3)
>>> with params.to_module(model):
...     out = model(x)
>>> # an ensemble of models: we stack params along the first dimension...
>>> params_stack = torch.stack([params, params], 0)
>>> # ... and use it as an input we'd like to pass through the model
>>> def func(x, params):
...     with params.to_module(model):
...         return model(x)
>>> y = vmap(func, (None, 0))(x, params_stack)
>>> print(y.shape)
torch.Size([2, 10, 4])
```

Moreover, tensordict modules are compatible with `torch.fx` and (soon) `torch.compile`,
which means that you can get the best of both worlds: a codebase that is
both readable and future-proof as well as efficient and portable!

### TensorDict for parameter serialization and building datasets

TensorDict offers an API for parameter serialization that can be >3x faster than
regular calls to `torch.save(state_dict)`. Moreover, because tensors will be saved
independently on disk, you can deserialize your checkpoint on an arbitrary slice
of the model.

```python
>>> model = nn.Sequential(nn.Linear(3, 4), nn.Linear(4, 3))
>>> params = TensorDict.from_module(model)
>>> params.memmap("/path/to/saved/folder/", num_threads=16)  # adjust num_threads for speed
>>> # load params
>>> params = TensorDict.load_memmap("/path/to/saved/folder/", num_threads=16)
>>> params.to_module(model)  # load onto model
>>> params["0"].to_module(model[0])  # load on a slice of the model
>>> # in the latter case we could also have loaded only the slice we needed
>>> params0 = TensorDict.load_memmap("/path/to/saved/folder/0", num_threads=16)
>>> params0.to_module(model[0])  # load on a slice of the model
```

The same functionality can be used to access data in a dataset stored on disk.
Soring a single contiguous tensor on disk accessed through the `tensordict.MemoryMappedTensor`
primitive and reading slices of it is not only **much** faster than loading
single files one at a time but it's also easier and safer (because there is no pickling
or third-party library involved):

```python
# allocate memory of the dataset on disk
data = TensorDict({
    "images": torch.zeros((128, 128, 3), dtype=torch.uint8),
    "labels": torch.zeros((), dtype=torch.int)}, batch_size=[])
data = data.expand(1000000)
data = data.memmap_like("/path/to/dataset")
# ==> Fill your dataset here
# Let's get 3 items of our dataset:
data[torch.tensor([1, 10000, 500000])]  # This is much faster than loading the 3 images independently
```

### Preprocessing with TensorDict.map

Preprocessing huge contiguous (or not!) datasets can be done via `TensorDict.map`
which will dispatch a task to various workers:

```python
import torch
from tensordict import TensorDict, MemoryMappedTensor
import tempfile

def process_data(data):
    images = data.get("images").flip(-2).clone()
    labels = data.get("labels") // 10
    # we update the td inplace
    data.set_("images", images)  # flip image
    data.set_("labels", labels)  # cluster labels

if __name__ == "__main__":
    # create data_preproc here
    data_preproc = data.map(process_data, num_workers=4, chunksize=0, pbar=True)  # process 1 images at a time
```

### Lazy preallocation

Pre-allocating tensors can be cumbersome and hard to scale if the list of preallocated
items varies according to the script configuration. TensorDict solves this in an elegant way.
Assume you are working with a function `foo() -> TensorDict`, e.g.
```python
def foo():
    data = TensorDict({}, batch_size=[])
    data["a"] = torch.randn(3)
    data["b"] = TensorDict({"c": torch.zeros(2)}, batch_size=[])
    return data
```
and you would like to call this function repeatedly. You could do this in two ways.
The first would simply be to stack the calls to the function:
```python
data = torch.stack([foo() for _ in range(N)])
```
However, you could also choose to preallocate the tensordict:
```python
data = TensorDict({}, batch_size=[N])
for i in range(N):
    data[i] = foo()
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
>>> data = TensorDict({
...     "key 1": torch.ones(3, 4, 5),
...     "key 2": TensorDict({"sub-key": torch.randn(3, 4, 5, 6)}, batch_size=[3, 4, 5])
... }, batch_size=[3, 4])
>>> tensordict_flatten = data.flatten_keys(separator=".")
```

Accessing nested tensordicts can be achieved with a single index:
```python
>>> sub_value = data["key 2", "sub-key"]
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
