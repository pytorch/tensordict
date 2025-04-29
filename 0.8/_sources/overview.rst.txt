Overview
========

TensorDict makes it easy to organise data and write reusable, generic PyTorch code. Originally developed for TorchRL,
we've spun it out into a separate library.

TensorDict is primarily a dictionary but also a tensor-like class: it supports multiple tensor operations that are
mostly shape and storage-related. It is designed to be efficiently serialised or transmitted from node to node or
process to process. Finally, it is shipped with its own :mod:`~tensordict.nn` module which is compatible with ``torch.func``
and aims at making model ensembling and parameter manipulation easier.

On this page we will motivate :class:`~tensordict.TensorDict` and give some examples of what it can do.

Motivation
----------

TensorDict allows you to write generic code modules that are re-usable across paradigms. For instance, the following
loop can be re-used across most SL, SSL, UL and RL tasks.

>>> for i, tensordict in enumerate(dataset):
...     # the model reads and writes tensordicts
...     tensordict = model(tensordict)
...     loss = loss_module(tensordict)
...     loss.backward()
...     optimizer.step()
...     optimizer.zero_grad()

With its :mod:`~tensordict.nn` module, the package provides many tools to use :class:`~tensordict.TensorDict` in a code
base with little or no effort.

In multiprocessing or distributed settings, :class:`~tensordict.TensorDict` allows you to seamlessly dispatch data to
each worker:

>>> # creates batches of 10 datapoints
>>> splits = torch.arange(tensordict.shape[0]).split(10)
>>> for worker in range(workers):
...     idx = splits[worker]
...     pipe[worker].send(tensordict[idx])

Some operations offered by TensorDict can be done via tree_map too, but with a greater degree of complexity:

>>> td = TensorDict(
...     {"a": torch.randn(3, 11), "b": torch.randn(3, 3)}, batch_size=3
... )
>>> regular_dict = {"a": td["a"], "b": td["b"]}
>>> td0, td1, td2 = td.unbind(0)
>>> # similar structure with pytree
>>> regular_dicts = tree_map(lambda x: x.unbind(0))
>>> regular_dict1, regular_dict2, regular_dict3 = [
...     {"a": regular_dicts["a"][i], "b": regular_dicts["b"][i]}
...     for i in range(3)]

The nested case is even more compelling:

>>> td = TensorDict(
...     {"a": {"c": torch.randn(3, 11)}, "b": torch.randn(3, 3)}, batch_size=3
... )
>>> regular_dict = {"a": {"c": td["a", "c"]}, "b": td["b"]}
>>> td0, td1, td2 = td.unbind(0)
>>> # similar structure with pytree
>>> regular_dicts = tree_map(lambda x: x.unbind(0))
>>> regular_dict1, regular_dict2, regular_dict3 = [
...     {"a": {"c": regular_dicts["a"]["c"][i]}, "b": regular_dicts["b"][i]}
...     for i in range(3)

Decomposing the output dictionary in three similarly structured dictionaries after applying the unbind operation quickly
becomes significantly cumbersome when working naively with pytree. With tensordict, we provide a simple API for users
that want to unbind or split nested structures, rather than computing a nested split / unbound nested structure.

Features
--------

A :class:`~tensordict.TensorDict` is a dict-like container for tensors. To instantiate a :class:`~tensordict.TensorDict`,
you can specify key-value pairs
as well as the batch size (an empty tensordict can be created via `TensorDict()`).
The leading dimensions of any values in the :class:`~tensordict.TensorDict` must be compatible with the batch size.

    >>> import torch
    >>> from tensordict import TensorDict
    >>> tensordict = TensorDict(
    ...     {"zeros": torch.zeros(2, 3, 4), "ones": torch.ones(2, 3, 4, 5)},
    ...     batch_size=[2, 3],
    ... )

The syntax for setting or retrieving values is much like that for a regular dictionary.

    >>> zeros = tensordict["zeros"]
    >>> tensordict["twos"] = 2 * torch.ones(2, 3)

One can also index a tensordict along its batch_size which makes it possible to obtain congruent slices of data in just
a few characters (notice that indexing the nth leading dimensions with tree_map using an ellipsis would require a bit more coding):

    >>> sub_tensordict = tensordict[..., :2]

One can also use the set method with ``inplace=True`` or the :meth:`~tensordict.TensorDict.set_` method to do inplace updates of the contents.
The former is a fault-tolerant version of the latter: if no matching key is found, it will write a new one.

The contents of the TensorDict can now be manipulated collectively.
For example, to place all of the contents onto a particular device one can simply do

    >>> tensordict = tensordict.to("cuda:0")

You can then assert that the device of the tensordict is `"cuda:0"`:

    >>> assert tensordict.device == torch.device("cuda:0")

To reshape the batch dimensions one can do

    >>> tensordict = tensordict.reshape(6)

The class supports many other operations, including :func:`~torch.squeeze`, :func:`~torch.unsqueeze`,
:meth:`~tensordict.TensorDict.view`, :func:`~torch.permute`, :meth:`~tensordict.TensorDict.unbind`,
:func:`~torch.stack`, :func:`~torch.cat` and many more.

If an operation is not present, the :meth:`~tensordict.TensorDict.apply` method will usually provide the solution
that was needed.

Escaping shape operations
~~~~~~~~~~~~~~~~~~~~~~~~~

In some cases, it may be desirable to store tensors in a TensorDict without enforcing batch size consistency during
shape operations.

This can be achieved by wrapping the tensor in an :class:`~tensordict.UnbatchedTensor` instance.

An :class:`~tensordict.UnbatchedTensor` ignores its shape during shape operations on the TensorDict, allowing for
flexible storage and manipulation of tensors with arbitrary shapes.

    >>> from tensordict import UnbatchedTensor
    >>> tensordict = TensorDict({"zeros": UnbatchedTensor(torch.zeros(10))}, batch_size=[2, 3])
    >>> reshaped_td = tensordict.reshape(6)
    >>> reshaped_td["zeros"] is tensordict["zeros"]
    True

Non-tensor data
---------------

Tensordict is a powerful library for working with tensor data, but it also supports non-tensor data. This guide will
show you how to use tensordict with non-tensor data.

Creating a TensorDict with Non-Tensor Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can create a TensorDict with non-tensor data using the :class:`~tensordict.NonTensorData` class.

    >>> from tensordict import TensorDict, NonTensorData
    >>> import torch
    >>> td = TensorDict(
    ...     a=NonTensorData("a string!"),
    ...     b=torch.zeros(()),
    ... )
    >>> print(td)
    TensorDict(
        fields={
            a: NonTensorData(data=a string!, batch_size=torch.Size([]), device=None),
            b: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False)},
        batch_size=torch.Size([]),
        device=None,
        is_shared=False)

As you can see, the :class:`~tensordict.NonTensorData` object is stored in the TensorDict just like a regular tensor.

Accessing Non-Tensor Data
~~~~~~~~~~~~~~~~~~~~~~~~~

You can access the non-tensor data using the key or the get method. Regular `getattr` calls will return the content of
the :class:`~tensordict.NonTensorData` object whereas :meth:`~tensordict.TensorDict.get` will return the
:class:`~tensordict.NonTensorData` object itself.

    >>> print(td["a"])  # prints: a string!
    >>> print(td.get("a"))  # prints: NonTensorData(data=a string!, batch_size=torch.Size([]), device=None)


Batched Non-Tensor Data
~~~~~~~~~~~~~~~~~~~~~~~

If you have a batch of non-tensor data, you can store it in a TensorDict with a specified batch size.

    >>> td = TensorDict(
    ...     a=NonTensorData("a string!"),
    ...     b=torch.zeros(3),
    ...     batch_size=[3]
    ... )
    >>> print(td)
    TensorDict(
        fields={
            a: NonTensorData(data=a string!, batch_size=torch.Size([3]), device=None),
            b: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False)},
        batch_size=torch.Size([3]),
        device=None,
        is_shared=False)

In this case, we assume that all elements of the tensordict have the same non-tensor data.

    >>> print(td[0])
    TensorDict(
        fields={
            a: NonTensorData(data=a string!, batch_size=torch.Size([]), device=None),
            b: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False)},
        batch_size=torch.Size([]),
        device=None,
        is_shared=False)

To assign a different non-tensor data object to each element in a shaped tensordict, you can use stacks of non-tensor
data.

Stacked Non-Tensor Data
~~~~~~~~~~~~~~~~~~~~~~~

If you have a list of non-tensor data that you want to store in a :class:`~tensordict.TensorDict`, you can use the
:class:`~tensordict.NonTensorStack` class.

    >>> td = TensorDict(
    ...     a=NonTensorStack("a string!", "another string!", "a third string!"),
    ...     b=torch.zeros(3),
    ...     batch_size=[3]
    ... )
    >>> print(td)
    TensorDict(
        fields={
            a: NonTensorStack(
                ['a string!', 'another string!', 'a third string!'...,
                batch_size=torch.Size([3]),
                device=None),
            b: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False)},
        batch_size=torch.Size([3]),
        device=None,
        is_shared=False)

You can access the first element and you will get the first of the strings:

    >>> print(td[0])
    TensorDict(
        fields={
            a: NonTensorData(data=a string!, batch_size=torch.Size([]), device=None),
            b: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False)},
        batch_size=torch.Size([]),
        device=None,
        is_shared=False)

In contrast, using :class:`~tensordict.NonTensorData` with a list will not lead to the same result, as there is no
way to tell what to do in general with a non-tensor data that happens to be a list:

    >>> td = TensorDict(
    ...     a=NonTensorData(["a string!", "another string!", "a third string!"]),
    ...     b=torch.zeros(3),
    ...     batch_size=[3]
    ... )
    >>> print(td[0])
    TensorDict(
        fields={
            a: NonTensorData(data=['a string!', 'another string!', 'a third string!'], batch_size=torch.Size([]), device=None),
            b: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False)},
        batch_size=torch.Size([]),
        device=None,
        is_shared=False)

Stacking TensorDicts with Non-Tensor Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To stack non-tensor data, :func:`~torch.stack` will check the identity of the non-tensor objects and produce a single
:class:`~tensordict.NonTensorData` if they match, or a :class:`~tensordict.NonTensorStack` otherwise:

    >>> td = TensorDict(
    ...     a=NonTensorData("a string!"),
    ... b = torch.zeros(()),
    ... )
    >>> print(torch.stack([td, td]))
    TensorDict(
        fields={
            a: NonTensorData(data=a string!, batch_size=torch.Size([2]), device=None),
            b: Tensor(shape=torch.Size([2]), device=cpu, dtype=torch.float32, is_shared=False)},
        batch_size=torch.Size([2]),
        device=None,
        is_shared=False)

If you want to make sure the result is a stack, use :meth:`~tensordict.TensorDict.lazy_stack` instead.

    >>> print(TensorDict.lazy_stack([td, td]))
    LazyStackedTensorDict(
        fields={
            a: NonTensorStack(
                ['a string!', 'a string!'],
                batch_size=torch.Size([2]),
                device=None),
            b: Tensor(shape=torch.Size([2]), device=cpu, dtype=torch.float32, is_shared=False)},
        exclusive_fields={
        },
        batch_size=torch.Size([2]),
        device=None,
        is_shared=False,
        stack_dim=0)

Named dimensions
----------------

TensorDict and related classes also support dimension names.
The names can be given at construction time or refined later. The semantic is
similar to the torch.Tensor dimension name feature:

>>> tensordict = TensorDict({}, batch_size=[3, 4], names=["a", None])
>>> tensordict.refine_names(..., "b")
>>> tensordict.names = ["z", "y"]
>>> tensordict.rename("m", "n")
>>> tensordict.rename(m="h")

Nested TensorDicts
------------------

The values in a :class:`~tensordict.TensorDict` can themselves be TensorDicts (the nested dictionaries in the example
below will be converted to nested TensorDicts).

>>> tensordict = TensorDict(
...     {
...         "inputs": {
...             "image": torch.rand(100, 28, 28),
...             "mask": torch.randint(2, (100, 28, 28), dtype=torch.uint8)
...         },
...         "outputs": {"logits": torch.randn(100, 10)},
...     },
...     batch_size=[100],
... )

Accessing or setting nested keys can be done with tuples of strings

>>> image = tensordict["inputs", "image"]
>>> logits = tensordict.get(("outputs", "logits"))  # alternative way to access
>>> tensordict["outputs", "probabilities"] = torch.sigmoid(logits)

Lazy evaluation
---------------

Some operations on :class:`~tensordict.TensorDict` defer execution until items are accessed. For example stacking,
squeezing, unsqueezing, permuting batch dimensions and creating a view are not executed immediately on all the contents
of the :class:`~tensordict.TensorDict`. Instead they are performed lazily when values in the :class:`~tensordict.TensorDict`
are accessed. This can save a lot of unnecessary calculation should the :class:`~tensordict.TensorDict` contain many values.

>>> tensordicts = [TensorDict({
...     "a": torch.rand(10),
...     "b": torch.rand(10, 1000, 1000)}, [10])
...     for _ in range(3)]
>>> stacked = torch.stack(tensordicts, 0)  # no stacking happens here
>>> stacked_a = stacked["a"]  # we stack the a values, b values are not stacked

It also has the advantage that we can manipulate the original tensordicts in a stack:

>>> stacked["a"] = torch.zeros_like(stacked["a"])
>>> assert (tensordicts[0]["a"] == 0).all()

The caveat is that the get method has now become an expensive operation and, if repeated many times, may cause some
overhead. One can avoid this by simply calling tensordict.contiguous() after the execution of stack. To further mitigate
this, TensorDict comes with its own meta-data class (MetaTensor) that keeps track of the type, shape, dtype and device
of each entry of the dict, without performing the expensive operation.

Lazy pre-allocation
-------------------

Suppose we have some function foo() -> TensorDict and that we do something like the following:

>>> tensordict = TensorDict({}, batch_size=[N])
>>> for i in range(N):
...     tensordict[i] = foo()

When ``i == 0`` the empty :class:`~tensordict.TensorDict` will automatically be populated with empty tensors with batch
size N. In subsequent iterations of the loop the updates will all be written in-place.

TensorDictModule
----------------

To make it easy to integrate :class:`~tensordict.TensorDict` in one's code base, we provide a tensordict.nn package that allows users to
pass :class:`~tensordict.TensorDict` instances to :class:`~torch.nn.Module` objects (or any callable).

:class:`~tensordict.nn.TensorDictModule` wraps :class:`~torch.nn.Module` and accepts a single :class:`~tensordict.TensorDict` as an input. You can specify where the underlying module should take its input from, and where it should write its output. This is a key reason we can write reusable, generic high-level code such as the training loop in the motivation section.

>>> from tensordict.nn import TensorDictModule
>>> class Net(nn.Module):
...     def __init__(self):
...         super().__init__()
...         self.linear = nn.LazyLinear(1)
...
...     def forward(self, x):
...         logits = self.linear(x)
...         return logits, torch.sigmoid(logits)
>>> module = TensorDictModule(
...     Net(),
...     in_keys=["input"],
...     out_keys=[("outputs", "logits"), ("outputs", "probabilities")],
... )
>>> tensordict = TensorDict({"input": torch.randn(32, 100)}, [32])
>>> tensordict = module(tensordict)
>>> # outputs can now be retrieved from the tensordict
>>> logits = tensordict["outputs", "logits"]
>>> probabilities = tensordict.get(("outputs", "probabilities"))

To facilitate the adoption of this class, one can also pass the tensors as kwargs:

>>> tensordict = module(input=torch.randn(32, 100))

which will return a :class:`~tensordict.TensorDict` identical to the one in the previous code box. See :ref:`the export tutorial` for
more context on this feature.

A key pain-point of multiple PyTorch users is the inability of nn.Sequential to handle modules with multiple inputs.
Working with key-based graphs can easily solve that problem as each node in the sequence knows what data needs to be
read and where to write it.

For this purpose, we provide the :class:`~tensordict.nn.TensorDictSequential` class which passes data through a
sequence of ``TensorDictModules``. Each module in the sequence takes its input from, and writes its output to the
original :class:`~tensordict.TensorDict`, meaning it's possible for modules in the sequence to ignore output from their
predecessors, or take additional input from the tensordict as necessary. Here's an example:

>>> class Net(nn.Module):
...     def __init__(self, input_size=100, hidden_size=50, output_size=10):
...         super().__init__()
...         self.fc1 = nn.Linear(input_size, hidden_size)
...         self.fc2 = nn.Linear(hidden_size, output_size)
...
...     def forward(self, x):
...         x = torch.relu(self.fc1(x))
...         return self.fc2(x)
...
... class Masker(nn.Module):
...     def forward(self, x, mask):
...         return torch.softmax(x * mask, dim=1)
>>> net = TensorDictModule(
...     Net(), in_keys=[("input", "x")], out_keys=[("intermediate", "x")]
... )
>>> masker = TensorDictModule(
...     Masker(),
...     in_keys=[("intermediate", "x"), ("input", "mask")],
...     out_keys=[("output", "probabilities")],
... )
>>> module = TensorDictSequential(net, masker)
>>> tensordict = TensorDict(
...     {
...         "input": TensorDict(
...             {"x": torch.rand(32, 100), "mask": torch.randint(2, size=(32, 10))},
...             batch_size=[32],
...         )
...     },
...     batch_size=[32],
... )
>>> tensordict = module(tensordict)
>>> intermediate_x = tensordict["intermediate", "x"]
>>> probabilities = tensordict["output", "probabilities"]

In this example, the second module combines the output of the first with the mask stored under ("inputs", "mask") in the
:class:`~tensordict.TensorDict`.

:class:`~tensordict.nn.TensorDictSequential` offers a bunch of other features: one can access the list of input and
output keys by querying the in_keys and out_keys attributes. It is also possible to ask for a sub-graph by querying
:meth:`~tensordict.nn.TensorDictSequential.select_subsequence` with the desired sets of input and output keys that are desired. This will return another
:class:`~tensordict.nn.TensorDictSequential` with only the modules that are indispensable to satisfy those requirements.
The :class:`~tensordict.nn.TensorDictModule` is also compatible with :func:`~torch.vmap` and other ``torch.func``
capabilities.
