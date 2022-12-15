.. currentmodule:: tensordict.nn

tensordict.nn package
=====================

The tensordict.nn package makes it possible to flexibly use TensorDict within
ML pipelines.

Since TensorDict turns parts of one's code to a key-based structure, it is now
possible to build complex graph structures using these keys as hooks.
The basic building block is :obj:`TensorDictModule`, which wraps an :obj:`nn.Module`
instance with a list of input and output keys:

.. code-block::

  >>> from torch.nn import Transformer
  >>> from tensordict import TensorDict
  >>> from tensordict.nn import TensorDictModule
  >>> import torch
  >>> module = TensorDictModule(Transformer(), in_keys=["feature", "target"], out_keys=["prediction"])
  >>> data = TensorDict({"feature": torch.randn(10, 11, 512), "target": torch.randn(10, 11, 512)}, [10, 11])
  >>> data = module(data)
  >>> print(data)
  TensorDict(
      fields={
          feature: Tensor(torch.Size([10, 11, 512]), dtype=torch.float32),
          prediction: Tensor(torch.Size([10, 11, 512]), dtype=torch.float32),
          target: Tensor(torch.Size([10, 11, 512]), dtype=torch.float32)},
      batch_size=torch.Size([10, 11]),
      device=None,
      is_shared=False)

One does not necessarily need to use :obj:`TensorDictModule`, a custom :obj:`nn.Module`
with an ordered list of input and output keys (named :obj:`module.in_keys` and
:obj:`module.out_keys`) will suffice.

A key pain-point of multiple PyTorch users is the inability of nn.Sequential to
handle modules with multiple inputs. Working with key-based graphs can easily
solve that problem as each node in the sequence knows what data needs to be
read and where to write it.

For this purpose, we provide the TensorDictSequential class which passes data
through a sequence of TensorDictModules. Each module in the sequence takes its
input from, and writes its output to the original TensorDict, meaning it's possible
for modules in the sequence to ignore output from their predecessors, or take
additional input from the tensordict as necessary. Here's an example:

.. code-block::

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
  >>> class Masker(nn.Module):
  ...     def forward(self, x, mask):
  ...         return torch.softmax(x * mask, dim=1)
  ...
  >>> net = TensorDictModule(
  ...     Net(), in_keys=[("input", "x")], out_keys=[("intermediate", "x")]
  ... )
  >>> masker = TensorDictModule(
  ...     Masker(),
  ...     in_keys=[("intermediate", "x"), ("input", "mask")],
  ...     out_keys=[("output", "probabilities")],
  ... )
  >>> module = TensorDictSequential(net, masker)
  >>>
  >>> td = TensorDict(
  ...     {
  ...         "input": TensorDict(
  ...             {"x": torch.rand(32, 100), "mask": torch.randint(2, size=(32, 10))},
  ...             batch_size=[32],
  ...         )
  ...     },
  ...     batch_size=[32],
  ... )
  >>> td = module(td)
  >>> print(td)
  TensorDict(
      fields={
          input: TensorDict(
              fields={
                  mask: Tensor(torch.Size([32, 10]), dtype=torch.int64),
                  x: Tensor(torch.Size([32, 100]), dtype=torch.float32)},
              batch_size=torch.Size([32]),
              device=None,
              is_shared=False),
          intermediate: TensorDict(
              fields={
                  x: Tensor(torch.Size([32, 10]), dtype=torch.float32)},
              batch_size=torch.Size([32]),
              device=None,
              is_shared=False),
          output: TensorDict(
              fields={
                  probabilities: Tensor(torch.Size([32, 10]), dtype=torch.float32)},
              batch_size=torch.Size([32]),
              device=None,
              is_shared=False)},
      batch_size=torch.Size([32]),
      device=None,
      is_shared=False)

We can also select sub-graphs easily throught the :obj:`TensorDictSequential.select_subsequence(in_keys, out_keys)` method:

.. code-block::

  >>> sub_module = module.select_subsequence(out_keys=[("intermediate", "x")])
  >>> td = TensorDict(
  ...     {
  ...         "input": TensorDict(
  ...             {"x": torch.rand(32, 100), "mask": torch.randint(2, size=(32, 10))},
  ...             batch_size=[32],
  ...         )
  ...     },
  ...     batch_size=[32],
  ... )
  >>> sub_module(td)
  >>> print(td)  # the "output" has not been computed
  TensorDict(
      fields={
          input: TensorDict(
              fields={
                  mask: Tensor(torch.Size([32, 10]), dtype=torch.int64),
                  x: Tensor(torch.Size([32, 100]), dtype=torch.float32)},
              batch_size=torch.Size([32]),
              device=None,
              is_shared=False),
          intermediate: TensorDict(
              fields={
                  x: Tensor(torch.Size([32, 10]), dtype=torch.float32)},
              batch_size=torch.Size([32]),
              device=None,
              is_shared=False)},
      batch_size=torch.Size([32]),
      device=None,
      is_shared=False)

Finally, tensordict.nn comes with a :obj:`ProbabilisticTensorDictModule` that allows
to build distributions from network outputs and get summary statistics or samples from it
(along with the distribution parameters):

.. code-block::

  >>> import torch
  >>> from tensordict import TensorDict
  >>> from tensordict.nn import TensorDictModule
  >>> from tensordict.nn.distributions import NormalParamExtractor
  >>> from tensordict.nn.functional_modules import make_functional
  >>> from tensordict.nn.prototype import (
  ...     ProbabilisticTensorDictModule,
  ...     ProbabilisticTensorDictSequential,
  ... )
  >>> from torch.distributions import Normal
  >>> td = TensorDict(
  ...     {"input": torch.randn(3, 4), "hidden": torch.randn(3, 8)}, [3]
  ... )
  >>> net = torch.nn.GRUCell(4, 8)
  >>> module = TensorDictModule(
  ...     net, in_keys=["input", "hidden"], out_keys=["embed"]
  ... )
  >>> extractor = TensorDictModule(
  ...     NormalParamExtractor(), in_keys=["embed"], out_keys=["loc", "scale"]
  ... )
  >>> prob_module = ProbabilisticTensorDictModule(
  ...     in_keys=["loc", "scale"],
  ...     out_keys=["sample"],
  ...     distribution_class=Normal,
  ...     return_log_prob=True,
  ... )
  >>> td_module = ProbabilisticTensorDictSequential(module, extractor, prob_module)
  >>> td_module(td)
  >>> print(td)
  TensorDict(
      fields={
          action: Tensor(torch.Size([3, 4]), dtype=torch.float32),
          hidden: Tensor(torch.Size([3, 8]), dtype=torch.float32),
          input: Tensor(torch.Size([3, 4]), dtype=torch.float32),
          loc: Tensor(torch.Size([3, 4]), dtype=torch.float32),
          sample_log_prob: Tensor(torch.Size([3, 4]), dtype=torch.float32),
          scale: Tensor(torch.Size([3, 4]), dtype=torch.float32)},
      batch_size=torch.Size([3]),
      device=None,
      is_shared=False)


.. autosummary::
    :toctree: generated/
    :template: td_template_noinherit.rst

    TensorDictModule
    ProbabilisticTensorDictModule
    TensorDictSequential
    TensorDictModuleWrapper

Functional
----------

.. autosummary::
    :toctree: generated/
    :template: rl_template_noinherit.rst

    get_functional
    make_functional
    repopulate_module

Utils
-----

.. autosummary::
    :toctree: generated/
    :template: rl_template_noinherit.rst

    make_tensordict
    dispatch_kwargs
    set_interaction_mode
    mappings
    inv_softplus
    biased_softplus
