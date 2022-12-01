Overview
========

TensorDict makes it easy to organise data and write reusable, generic PyTorch code. Originally developed for TorchRL, we've spun it out into a separate library.

TensorDict is primarily a dictionary but also a tensor-like class: it supports multiple tensor operations that are mostly shape and storage-related. It is designed to be efficiently serialised or transmitted from node to node or process to process. Finally, it is shipped with its own ``tensordict.nn`` module which is compatible with ``functorch`` and aims at making model ensembling and parameter manipulation easier.

On this page we will motivate ``TensorDict`` and give some examples of what it can do.

Motivation
----------

TensorDict allows you to write generic code modules that are re-usable across paradigms. For instance, the following loop can be re-used across most SL, SSL, UL and RL tasks.

>>> for i, tensordict in enumerate(dataset):
...     # the model reads and writes tensordicts
...     tensordict = model(tensordict)
...     loss = loss_module(tensordict)
...     loss.backward()
...     optimizer.step()
...     optimizer.zero_grad()

With its ``tensordict.nn`` module, the package provides many tools to use ``TensorDict`` in a code base with little or no effort.

In multiprocessing or distributed settings, ``tensordict`` allows you to seamlessly dispatch data to each worker:

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
>>> regular_dict1, regular_dict2 regular_dict3 = [
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
>>> regular_dict1, regular_dict2 regular_dict3 = [
...     {"a": {"c": regular_dicts["a"]["c"][i]}, "b": regular_dicts["b"][i]}
...     for i in range(3)

Decomposing the output dictionary in three similarly structured dictionaries after applying the unbind operation quickly becomes significantly cumbersome when working naively with pytree. With tensordict, we provide a simple API for users that want to unbind or split nested structures, rather than computing a nested split / unbound nested structure.

Features
--------

A ``TensorDict`` is a dict-like container for tensors. To instantiate a ``TensorDict``, you must specify key-value pairs as well as the batch size. The leading dimensions of any values in the ``TensorDict`` must be compatible with the batch size.

>>> import torch
>>> from tensordict import TensorDict

>>> tensordict = TensorDict(
...     {"zeros": torch.zeros(2, 3, 4), "ones": torch.ones(2, 3, 4, 5)},
...     batch_size=[2, 3],
... )

The syntax for setting or retrieving values is much like that for a regular dictionary.

>>> zeros = tensordict["zeros"]
>>> tensordict["twos"] = 2 * torch.ones(2, 3)

One can also index a tensordict along its batch_size which makes it possible to obtain congruent slices of data in just a few characters (notice that indexing the nth leading dimensions with tree_map using an ellipsis would require a bit more coding):

>>> sub_tensordict = tensordict[..., :2]

One can also use the set method with inplace=True or the set_ method to do inplace updates of the contents. The former is a fault-tolerant version of the latter: if no matching key is found, it will write a new one.

The contents of the TensorDict can now be manipulated collectively. For example, to place all of the contents onto a particular device one can simply do

>>> tensordict = tensordict.to("cuda:0")

To reshape the batch dimensions one can do

>>> tensordict = tensordict.reshape(6)

The class supports many other operations, including squeeze, unsqueeze, view, permute, unbind, stack, cat and many more. If an operation is not present, the TensorDict.apply method will usually provide the solution that was needed.

Nested TensorDicts
------------------

The values in a ``TensorDict`` can themselves be TensorDicts (the nested dictionaries in the example below will be converted to nested TensorDicts).

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

Some operations on ``TensorDict`` defer execution until items are accessed. For example stacking, squeezing, unsqueezing, permuting batch dimensions and creating a view are not executed immediately on all the contents of the ``TensorDict``. Instead they are performed lazily when values in the ``TensorDict`` are accessed. This can save a lot of unnecessary calculation should the ``TensorDict`` contain many values.

>>> tensordicts = [TensorDict({
...     "a": torch.rand(10),
...     "b": torch.rand(10, 1000, 1000)}, [10])
...     for _ in range(3)]
>>> stacked = torch.stack(tensordicts, 0)  # no stacking happens here
>>> stacked_a = stacked["a"]  # we stack the a values, b values are not stacked

It also has the advantage that we can manipulate the original tensordicts in a stack:

>>> stacked["a"] = torch.zeros_like(stacked["a"])
>>> assert (tensordicts[0]["a"] == 0).all()

The caveat is that the get method has now become an expensive operation and, if repeated many times, may cause some overhead. One can avoid this by simply calling tensordict.contiguous() after the execution of stack. To further mitigate this, TensorDict comes with its own meta-data class (MetaTensor) that keeps track of the type, shape, dtype and device of each entry of the dict, without performing the expensive operation.

Lazy pre-allocation
-------------------

Suppose we have some function foo() -> TensorDict and that we do something like the following:

>>> tensordict = TensorDict({}, batch_size=[N])
>>> for i in range(N):
...     tensordict[i] = foo()

When ``i == 0`` the empty ``TensorDict`` will automatically be populated with empty tensors with batch size N. In subsequent iterations of the loop the updates will all be written in-place.

TensorDictModule
----------------

To make it easy to integrate ``TensorDict`` in one's code base, we provide a tensordict.nn package that allows users to pass ``TensorDict`` instances to ``nn.Module`` objects.

``TensorDictModule`` wraps ``nn.Module`` and accepts a single ``TensorDict`` as an input. You can specify where the underlying module should take its input from, and where it should write its output. This is a key reason we can write reusable, generic high-level code such as the training loop in the motivation section.

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

which will return a ``TensorDict`` identical to the one in the previous code box.

A key pain-point of multiple PyTorch users is the inability of nn.Sequential to handle modules with multiple inputs. Working with key-based graphs can easily solve that problem as each node in the sequence knows what data needs to be read and where to write it.

For this purpose, we provide the ``TensorDictSequential`` class which passes data through a sequence of ``TensorDictModules``. Each module in the sequence takes its input from, and writes its output to the original ``TensorDict``, meaning it's possible for modules in the sequence to ignore output from their predecessors, or take additional input from the tensordict as necessary. Here's an example.

>>> class Net(nn.Module):
...    def __init__(self, input_size=100, hidden_size=50, output_size=10):
...        super().__init__()
...        self.fc1 = nn.Linear(input_size, hidden_size)
...        self.fc2 = nn.Linear(hidden_size, output_size)
...
...    def forward(self, x):
...        x = torch.relu(self.fc1(x))
...        return self.fc2(x)
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
>>> probabilities = tensordict["outputs", "probabilities"]

In this example, the second module combines the output of the first with the mask stored under ("inputs", "mask") in the ``TensorDict``.

``TensorDictSequential`` offers a bunch of other features: one can access the list of input and output keys by querying the in_keys and out_keys attributes. It is also possible to ask for a sub-graph by querying ``select_subsequence()`` with the desired sets of input and output keys that are desired. This will return another ``TensorDictSequential`` with only the modules that are indispensable to satisfy those requirements. The ``TensorDictModule`` is also compatible with ``vmap`` and other ``functorch`` capabilities.

Functional Programming
----------------------

We provide and API to use ``TensorDict`` in conjunction with ``functorch``. For instance, ``TensorDict`` makes it easy to concatenate model weights to do model ensembling:

>>> from torch import nn
>>> from tensordict import TensorDict
>>> from tensordict.nn import make_functional
>>> import torch
>>> from functorch import vmap
>>> layer1 = nn.Linear(3, 4)
>>> layer2 = nn.Linear(4, 4)
>>> model = nn.Sequential(layer1, layer2)
>>> # we represent the weights hierarchically
>>> weights1 = TensorDict(layer1.state_dict(), []).unflatten_keys(separator=".")
>>> weights2 = TensorDict(layer2.state_dict(), []).unflatten_keys(separator=".")
>>> params = make_functional(model)
>>> # params provided by make_functional match state_dict:
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


The functional API is comparable if not faster than the current ``FunctionalModule`` implemented in ``functorch``.
