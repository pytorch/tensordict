.. currentmodule:: tensordict.nn

tensordict.nn package
=====================

The tensordict.nn package makes it possible to flexibly use TensorDict within
ML pipelines.

Since TensorDict turns parts of one's code to a key-based structure, it is now
possible to build complex graph structures using these keys as hooks.
The basic building block is :class:`~.TensorDictModule`, which wraps an :class:`torch.nn.Module`
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

One does not necessarily need to use :class:`~.TensorDictModule`, a custom :class:`torch.nn.Module`
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

  >>> from tensordict.nn import TensorDictSequential
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

We can also select sub-graphs easily through the :meth:`~.TensorDictSequential.select_subsequence` method:

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

Finally, :mod:`tensordict.nn` comes with a :class:`~.ProbabilisticTensorDictModule` that allows
to build distributions from network outputs and get summary statistics or samples from it
(along with the distribution parameters):

.. code-block::

  >>> import torch
  >>> from tensordict import TensorDict
  >>> from tensordict.nn import TensorDictModule
  >>> from tensordict.nn.distributions import NormalParamExtractor
  >>> from tensordict.nn.prototype import (
  ...     ProbabilisticTensorDictModule,
  ...     ProbabilisticTensorDictSequential,
  ... )
  >>> from torch.distributions import Normal
  >>> td = TensorDict(
  ...     {"input": torch.randn(3, 4), "hidden": torch.randn(3, 8)}, [3]
  ... )
  >>> net = torch.nn.Sequential(torch.nn.GRUCell(4, 8), NormalParamExtractor())
  >>> module = TensorDictModule(
  ...     net, in_keys=["input", "hidden"], out_keys=["loc", "scale"]
  ... )
  >>> prob_module = ProbabilisticTensorDictModule(
  ...     in_keys=["loc", "scale"],
  ...     out_keys=["sample"],
  ...     distribution_class=Normal,
  ...     return_log_prob=True,
  ... )
  >>> td_module = ProbabilisticTensorDictSequential(module, prob_module)
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

    TensorDictModuleBase
    TensorDictModule
    ProbabilisticTensorDictModule
    ProbabilisticTensorDictSequential
    TensorDictSequential
    TensorDictModuleWrapper
    CudaGraphModule
    WrapModule
    set_composite_lp_aggregate
    composite_lp_aggregate
    as_tensordict_module

Ensembles
---------
The functional approach enables a straightforward ensemble implementation.
We can duplicate and reinitialize model copies using the :class:`tensordict.nn.EnsembleModule`

.. code-block::

    >>> import torch
    >>> from torch import nn
    >>> from tensordict.nn import TensorDictModule
    >>> from torchrl.modules import EnsembleModule
    >>> from tensordict import TensorDict
    >>> net = nn.Sequential(nn.Linear(4, 32), nn.ReLU(), nn.Linear(32, 2))
    >>> mod = TensorDictModule(net, in_keys=['a'], out_keys=['b'])
    >>> ensemble = EnsembleModule(mod, num_copies=3)
    >>> data = TensorDict({'a': torch.randn(10, 4)}, batch_size=[10])
    >>> ensemble(data)
    TensorDict(
        fields={
            a: Tensor(shape=torch.Size([3, 10, 4]), device=cpu, dtype=torch.float32, is_shared=False),
            b: Tensor(shape=torch.Size([3, 10, 2]), device=cpu, dtype=torch.float32, is_shared=False)},
        batch_size=torch.Size([3, 10]),
        device=None,
        is_shared=False)

.. autosummary::
    :toctree: generated/
    :template: rl_template_noinherit.rst

    EnsembleModule

Compiling TensorDictModules
---------------------------

.. currentmodule:: tensordict.nn

Since v0.5, TensorDict components are compatible with :func:`~torch.compile`.
For instance, a :class:`~tensordict.TensorDictSequential` module can be compiled with
``torch.compile`` and reach a runtime similar to a regular PyTorch module wrapped in
a :class:`~tensordict.nn.TensorDictModule`.

Distributions
-------------

.. currentmodule:: tensordict.nn.distributions

.. autosummary::
    :toctree: generated/
    :template: rl_template_noinherit.rst

    AddStateIndependentNormalScale
    CompositeDistribution
    Delta
    NormalParamExtractor
    OneHotCategorical
    TruncatedNormal
    InteractionType
    set_interaction_type
    add_custom_mapping
    mappings


Utils
-----

.. currentmodule:: tensordict.nn

.. autosummary::
    :toctree: generated/
    :template: rl_template_noinherit.rst

    make_tensordict
    dispatch
    inv_softplus
    biased_softplus
    set_skip_existing
    skip_existing
    rand_one_hot
