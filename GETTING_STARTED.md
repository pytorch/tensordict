# 📖 Getting started with TensorDict

- [**Basic Usage**](#basic-usage)
- [**Prerequisite: TensorDict's Metadata**](#tensordicts-metadata)
- [**Specialized Dictionary**](#tensordict-as-a-specialized-dictionary)
- [**Nesting TensorDicts**](#nesting-tensordicts)
- [**Tensor-like features**](#tensor-like-features)
- [**TensorDicts as context managers**](#tensordicts-as-context-managers)
- [**Distributed capabilities**](#distributed-capabilities)
- [**TensorDict to represent state-dicts**](#tensordict-to-represent-state-dicts)
- [**TensorDict for functional programming**](#tensordict-for-functional-programming)
- [**TensorDict for parameter serialization and building datasets**](#tensordict-for-parameter-serialization-and-building-datasets)
- [**Preprocessing with TensorDict.map**](#preprocessing-with-tensordictmap)
- [**Lazy preallocation**](#lazy-preallocation)
- [**tensorclass**](#tensorclass)

![tensordict.png](docs%2Ftensordict.png)

## Basic usage

``TensorDict`` can be used as a drop-in replacement for python ``dict``, provided that the keys
are strings or tuples of strings.

Numerical data such as `np.ndarray`, `int`, `float` and `bool` will be cast to `torch.Tensor`
instances:
```python
>>> from tensordict import TensorDict
>>> td = TensorDict(a=0, b=1)
>>> print(td)
TensorDict(
    fields={
        a: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int64, is_shared=False),
        b: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int64, is_shared=False)},
    batch_size=torch.Size([]),
    device=None,
    is_shared=False)
```

Alternatively, a dictionary can be passed as input. The following code will create the exact same
content as above:
```python
>>> td = TensorDict({"a": 0, "b": 1})
```

`TensorDict` also supports non-tensor data:
```python
>>> td = TensorDict(a=0, non_tensor="a string!")
>>> assert td["non_tensor"] == "a string!"
```

`TensorDict` supports assignment of new entries, unless it is locked:
```python
>>> td = TensorDict()
>>> td["a"] = 0
>>> print(td)
TensorDict(
    fields={
        a: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int64, is_shared=False)},
    batch_size=torch.Size([]),
    device=None,
    is_shared=False)
>>> td = TensorDict(lock=True)
>>> td["a"] = 0
RuntimeError: Cannot modify locked TensorDict. For in-place modification, consider using the `set_()` method and make sure the key is present.
```

## TensorDict's Metadata

Unlike other [pytrees](https://github.com/pytorch/pytorch/blob/main/torch/utils/_pytree.py), TensorDict
carries metadata that make it easy to query the state of the container. The main metadata
are:
- the [``batch_size``](https://pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDict.html#tensordict.TensorDict.batch_size)
(also referred as ``shape``),

- the [``device``](https://pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDict.html#tensordict.TensorDict.device),

- the shared status
([``is_memmap``](https://pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase.is_memmap) or
[``is_shared``](https://pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase.is_shared)),

- the dimension [``names``](https://pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDict.html#tensordict.TensorDict.names),

- the [``lock``](https://pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDict.html#tensordict.TensorDict.lock_) status.

A tensordict is primarily defined by its `batch_size` (or `shape`) and its key-value pairs:
```python
>>> from tensordict import TensorDict
>>> import torch
>>> data = TensorDict({
...     "key 1": torch.ones(3, 4, 5),
...     "key 2": torch.zeros(3, 4, 5, dtype=torch.bool),
... }, batch_size=[3, 4])
```
If provided, the `batch_size` and the first dimensions of each of the tensors must be compliant.
The tensors can be of any dtype and device.

Optionally, one can restrict a tensordict to
live on a dedicated ``device``, which will send each tensor that is written there:
```python
>>> data = TensorDict({
...     "key 1": torch.ones(3, 4, 5),
...     "key 2": torch.zeros(3, 4, 5, dtype=torch.bool),
... }, batch_size=[3, 4], device="cuda:0")
```
When a tensordict has a device, all write operations will cast the tensor to the
TensorDict device:
```python
>>> data["key 3"] = torch.randn(3, 4, device="cpu")
>>> assert data["key 3"].device is torch.device("cuda:0")
```
Once the device is set, it can be cleared with the
[``clear_device_``](https://pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDict.html#tensordict.TensorDict.clear_device_)
method.

## TensorDict as a specialized dictionary
`TensorDict` possesses all the basic features of a dictionary such as
[``clear``](https://pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDict.html#tensordict.TensorDict.clear),
[``copy``](https://pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDict.html#tensordict.TensorDict.copy),
[``fromkeys``](https://pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDict.html#tensordict.TensorDict.fromkeys),
[``get``](https://pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDict.html#tensordict.TensorDict.get),
[``items``](https://pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDict.html#tensordict.TensorDict.items),
[``keys``](https://pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDict.html#tensordict.TensorDict.keys),
[``pop``](https://pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDict.html#tensordict.TensorDict.pop),
[``popitem``](https://pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDict.html#tensordict.TensorDict.popitem),
[``setdefault``](https://pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDict.html#tensordict.TensorDict.setdefault),
[``update``](https://pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDict.html#tensordict.TensorDict.update) and
[``values``](https://pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDict.html#tensordict.TensorDict.values).

You can select some of the entries of a `TensorDict` with `TensorDict.select`, or exclude them
with `TensorDict.exclude`:
```python
>>> td = TensorDict(a=0, b=1).exclude("b")
>>> print(td)
TensorDict(
    fields={
        a: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int64, is_shared=False)},
    batch_size=torch.Size([]),
    device=None,
    is_shared=False)
```

You can also split the tensordict in two different key-sets:
```python
>>> td = TensorDict(a=0, b=0, c=0)
>>> td_ab, td_c = td.split_keys(["a", "b"])
>>> td_ab
TensorDict(
    fields={
        a: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int64, is_shared=False),
        b: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int64, is_shared=False)},
    batch_size=torch.Size([]),
    device=None,
    is_shared=False)
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

## Nesting TensorDicts

It is possible to nest tensordict. The only requirement is that the sub-tensordict should be indexable
under the parent tensordict, i.e. its batch size should match (but could be longer than) the parent
batch size.

We can switch easily between hierarchical and flat representations thansk to `flatten_keys` and `unflatten_keys`.
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

## Tensor-like features

TensorDict supports many common point-wise arithmetic operations such as `==` or `+`, `+=`
and similar (provided that the underlying tensors support the said operation):
```python
>>> td = TensorDict.fromkeys(["a", "b", "c"], 0)
>>> td += 1
>>> assert (td==1).all()
```

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

## TensorDicts as context managers

Many tensordict operations can be used as context manager. In such cases, the modified version of the `TensorDict`
will update its parent tensordict as well:
```python
>>> td = TensorDict({"a": {"b": 0}})
>>> with td.flatten_keys() as tdflat:
...     tdflat["a.b"] += 1
...     tdflat["a.c"] = 0
>>> assert td["a", "b"] == 1
>>> assert td["a", "c"] == 0
```

The operations that can be used as context manager include: `flatten_keys`, `unflatten_keys`, `flatten`, `unflatten`,
`lock_`, `unlock_`, `to_module`, `permute`, `transpose`, `view`, `squeeze` and `unsqueeze`.

## Distributed capabilities

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


## TensorDict to represent state-dicts

As it can store tensors in a nested fashion, `TensorDict` is the ideal tool to represent state-dicts in a nested
manner. This representation is sometimes much clearer than the flat representation of `torch.nn.Module.state_dict`:

```python
>>> import torch.nn
>>> module = torch.nn.Sequential(torch.nn.Linear(3, 4))
>>> state_dict = TensorDict.from_module(module)
>>> print(state_dict)
TensorDict(
    fields={
        0: TensorDict(
            fields={
                bias: Parameter(shape=torch.Size([4]), device=cpu, dtype=torch.float32, is_shared=False),
                weight: Parameter(shape=torch.Size([4, 3]), device=cpu, dtype=torch.float32, is_shared=False)},
            batch_size=torch.Size([]),
            device=None,
            is_shared=False)},
    batch_size=torch.Size([]),
    device=None,
    is_shared=False)
>>> state_dict.data.zero_() # Zeros the parameters
>>> state_dict.to_module(module, inplace=True) # Update the parameters
```

## TensorDict for functional programming

Because it can store parameters and pass them to a module, we can also use `TensorDict` to make functional calls (i.e.,
calls where a modified version of the parameters is used) to a module. For instance, the following script
uses a modified version of the parameters to call the module, then replaces the original parameters back in place after
the call:
```python
>>> module = torch.nn.Sequential(torch.nn.Linear(3, 4))
>>> td = TensorDict.from_module(module)
>>> td_zero = td.detach().clone().zero_()
>>> with td_zero.to_module(module):
...     y = module(torch.randn(3))
>>> assert (y == 0).all()
```

Because `TensorDict` also supports batches of data, using `torch.vmap` to execute the model across parameter configurations
is also simple:
```python
>>> module = torch.nn.Sequential(torch.nn.Linear(3, 4))
>>> td = TensorDict.from_module(module)
>>> td_zero = td.detach().clone().zero_()
>>> td_one = td.detach().clone() * 0 + 1
>>> td_stack = torch.stack([td_zero, td_one])
>>> def call_module(x, td):
...     with td.to_module(module):
...         return module(x)
>>> x = torch.ones(2, 10, 3)
>>> y = torch.vmap(call_module)(x, td_stack)
>>> assert (y[0] == 0).all()
>>> assert (y[1] == 4).all()
```

## TensorDict for parameter serialization and building datasets

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

## Preprocessing with TensorDict.map

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

The `TensorDict.map_iter` function can also be used to iterate (optinally randomly) over a large tensordict
in a dataloader-like fashion.

## Lazy preallocation

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
