.. module:: tensordict.nn

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
          feature: Tensor(shape=torch.Size([10, 11, 512]), device=cpu, dtype=torch.float32, is_shared=False),
          prediction: Tensor(shape=torch.Size([10, 11, 512]), device=cpu, dtype=torch.float32, is_shared=False),
          target: Tensor(shape=torch.Size([10, 11, 512]), device=cpu, dtype=torch.float32, is_shared=False)},
      batch_size=torch.Size([10, 11]),
      device=None,
      is_shared=False)

One does not necessarily need to use :class:`~.TensorDictModule`, a custom :class:`torch.nn.Module`
with an ordered list of input and output keys (named ``module.in_keys`` and
``module.out_keys``) will suffice.

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

  >>> from torch import nn
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
                  mask: Tensor(shape=torch.Size([32, 10]), device=cpu, dtype=torch.int64, is_shared=False),
                  x: Tensor(shape=torch.Size([32, 100]), device=cpu, dtype=torch.float32, is_shared=False)},
              batch_size=torch.Size([32]),
              device=None,
              is_shared=False),
          intermediate: TensorDict(
              fields={
                  x: Tensor(shape=torch.Size([32, 10]), device=cpu, dtype=torch.float32, is_shared=False)},
              batch_size=torch.Size([32]),
              device=None,
              is_shared=False),
          output: TensorDict(
              fields={
                  probabilities: Tensor(shape=torch.Size([32, 10]), device=cpu, dtype=torch.float32, is_shared=False)},
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
  >>> td = sub_module(td)
  >>> print(td)  # the "output" has not been computed
  TensorDict(
      fields={
          input: TensorDict(
              fields={
                  mask: Tensor(shape=torch.Size([32, 10]), device=cpu, dtype=torch.int64, is_shared=False),
                  x: Tensor(shape=torch.Size([32, 100]), device=cpu, dtype=torch.float32, is_shared=False)},
              batch_size=torch.Size([32]),
              device=None,
              is_shared=False),
          intermediate: TensorDict(
              fields={
                  x: Tensor(shape=torch.Size([32, 10]), device=cpu, dtype=torch.float32, is_shared=False)},
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
  >>> from tensordict.nn import (
  ...     ProbabilisticTensorDictModule,
  ...     ProbabilisticTensorDictSequential,
  ... )
  >>> from torch.distributions import Normal
  >>> td = TensorDict(
  ...     {"input": torch.randn(3, 4), "hidden": torch.randn(3, 8)}, [3]
  ... )
  >>> gru = TensorDictModule(
  ...     torch.nn.GRUCell(4, 8), in_keys=["input", "hidden"], out_keys=["hidden"]
  ... )
  >>> param_extractor = TensorDictModule(
  ...     NormalParamExtractor(), in_keys=["hidden"], out_keys=["loc", "scale"]
  ... )
  >>> prob_module = ProbabilisticTensorDictModule(
  ...     in_keys=["loc", "scale"],
  ...     out_keys=["sample"],
  ...     distribution_class=Normal,
  ...     return_log_prob=True,
  ... )
  >>> td_module = ProbabilisticTensorDictSequential(gru, param_extractor, prob_module)
  >>> td = td_module(td)
  >>> print(td)
  TensorDict(
      fields={
          hidden: Tensor(shape=torch.Size([3, 8]), device=cpu, dtype=torch.float32, is_shared=False),
          input: Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.float32, is_shared=False),
          loc: Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.float32, is_shared=False),
          sample: Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.float32, is_shared=False),
          sample_log_prob: Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.float32, is_shared=False),
          scale: Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.float32, is_shared=False)},
      batch_size=torch.Size([3]),
      device=None,
      is_shared=False)

Type-Safe TensorClass Modules
-----------------------------

The :class:`~.TensorClassModuleBase` provides a type-safe way to define modules that work with
:class:`~tensordict.TensorClass` inputs and outputs. This enables static type
checking and improved code clarity compared to working with string-based keys.

A :class:`~.TensorClassModuleBase` subclass specifies its input and output types through generic
type parameters. The module can be converted to work with :class:`~.TensorDict` objects using the
:meth:`~.TensorClassModuleBase.as_td_module` method, which returns a :class:`~.TensorClassModuleWrapper`:

.. code-block::

  >>> import torch
  >>> from tensordict.tensorclass import TensorClass
  >>> from tensordict.nn import TensorClassModuleBase
  >>> from tensordict import TensorDict
  >>>
  >>> # Define input and output TensorClass types
  >>> class InputTC(TensorClass):
  ...     a: torch.Tensor
  ...     b: torch.Tensor
  ...
  >>> class OutputTC(TensorClass):
  ...     total: torch.Tensor
  ...     difference: torch.Tensor
  ...
  >>> # Create a type-safe module
  >>> class MyModule(TensorClassModuleBase[InputTC, OutputTC]):
  ...     def forward(self, x: InputTC) -> OutputTC:
  ...         return OutputTC(
  ...             total=x.a + x.b,
  ...             difference=x.a - x.b,
  ...             batch_size=x.batch_size
  ...         )
  ...
  >>> # Use with TensorClass
  >>> module = MyModule()
  >>> input_tc = InputTC(a=torch.tensor([1.0, 2.0]), b=torch.tensor([3.0, 4.0]), batch_size=[2])
  >>> output = module(input_tc)
  >>> print(output.total)
  tensor([4., 6.])
  >>> print(output.difference)
  tensor([-2., -2.])
  >>>
  >>> # Convert to TensorDictModule for use in TensorDict workflows
  >>> td_module = module.as_td_module()
  >>> td = TensorDict({"a": torch.tensor([1.0, 2.0]), "b": torch.tensor([3.0, 4.0])}, batch_size=[2])
  >>> result = td_module(td)  # a new TensorDict holding the output fields only
  >>> print(result)
  TensorDict(
      fields={
          difference: Tensor(shape=torch.Size([2]), device=cpu, dtype=torch.float32, is_shared=False),
          total: Tensor(shape=torch.Size([2]), device=cpu, dtype=torch.float32, is_shared=False)},
      batch_size=torch.Size([2]),
      device=None,
      is_shared=False)

The type-safe approach offers several benefits:

* **Type checking**: IDEs and type checkers can verify correct usage at development time
* **Self-documenting**: The input and output structure is clear from the type signature
* **Refactoring**: Renaming fields in TensorClass definitions is caught by type checkers
* **Nested structures**: Support for nested TensorClass types with automatic key extraction

:class:`~.TensorClassModuleBase` modules can be composed and used in :class:`~.TensorDictSequential`
after conversion via :meth:`~.TensorClassModuleBase.as_td_module`. Since the converted module returns
a new TensorDict that contains only its output fields, the modules that follow it in the sequence (and,
by default, the result of the sequence) only see those outputs, not the converted module's input keys.


.. autosummary::
    :toctree: generated/
    :template: td_template_noinherit.rst

    TensorDictModuleBase
    TensorDictModule
    TensorClassModuleBase
    TensorClassModuleWrapper
    ProbabilisticTensorDictModule
    ProbabilisticTensorDictSequential
    TensorDictSequential
    TensorDictModuleWrapper
    CudaGraphModule
    WrapModule
    InteractionType
    set_interaction_type
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
    >>> from tensordict.nn import EnsembleModule
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
    :template: td_template_noinherit.rst

    EnsembleModule

Compiling TensorDictModules
---------------------------

.. currentmodule:: tensordict.nn

Since v0.5, TensorDict components are compatible with :func:`~torch.compile`.
For instance, a :class:`~tensordict.nn.TensorDictSequential` module can be compiled with
``torch.compile``.

Distributions
-------------

.. currentmodule:: tensordict.nn.distributions

.. autosummary::
    :toctree: generated/
    :template: td_template_noinherit.rst

    AddStateIndependentNormalScale
    CompositeDistribution
    Delta
    NormalParamExtractor
    OneHotCategorical
    TruncatedNormal


Utils
-----

.. currentmodule:: tensordict.nn

.. autosummary::
    :toctree: generated/
    :template: td_template_noinherit.rst

    make_tensordict
    dispatch
    inv_softplus
    biased_softplus
    set_skip_existing
    skip_existing
    add_custom_mapping
    mappings
    rand_one_hot
