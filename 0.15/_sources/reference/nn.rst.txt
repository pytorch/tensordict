.. _api-nn:

tensordict.nn
=============

.. module:: tensordict.nn

This page lists the modules of ``tensordict.nn``, and the probabilistic
modules, distributions, and helpers that go with them. A module reads the
entries named in its ``in_keys`` from a tensordict, and writes the entries
named in its ``out_keys`` into it. For a guide with examples, see
:ref:`modules-guide`.

The page of a module class lists its methods and the methods of its
tensordict base classes. For the methods of :class:`torch.nn.Module`, see
the PyTorch documentation.

Modules
-------

- :class:`TensorDictModuleBase` is the base class of the tensordict
  modules.
- :class:`TensorDictModule` wraps a :class:`torch.nn.Module` that takes and
  returns tensors, and gives it ``in_keys`` and ``out_keys``.
- :class:`TensorDictSequential` runs modules one after the other on the
  same tensordict.
- :class:`WrapModule` wraps a function that takes and returns a
  tensordict.
- :class:`TensorDictModuleWrapper` wraps a tensordict module and behaves
  like it, except for the methods that a subclass overrides.
- :class:`as_tensordict_module` is a decorator that turns a function into
  a :class:`TensorDictModule`.
- :class:`CudaGraphModule` captures a module or a function in a CUDA graph
  and replays it. See :ref:`compile`.

.. autosummary::
    :toctree: generated/
    :template: nn_module.rst

    TensorDictModuleBase
    TensorDictModule
    TensorDictSequential
    WrapModule
    TensorDictModuleWrapper

.. autosummary::
    :toctree: generated/
    :template: class.rst

    as_tensordict_module
    CudaGraphModule

Modules with tensorclass inputs and outputs
-------------------------------------------

A :class:`TensorClassModuleBase` subclass declares the tensorclass types
of its input and output, so that a type checker can report a wrong field
name or type.
:meth:`TensorClassModuleBase.as_td_module` returns a
:class:`TensorClassModuleWrapper`, which reads and writes a tensordict, so
that you can put the module in a :class:`TensorDictSequential`.

.. autosummary::
    :toctree: generated/
    :template: nn_module.rst

    TensorClassModuleBase
    TensorClassModuleWrapper

Probabilistic modules
---------------------

A :class:`ProbabilisticTensorDictModule` builds a distribution from
entries of the tensordict, and writes a sample, and optionally its
log-probability, to the tensordict. :class:`InteractionType` and
:class:`set_interaction_type` choose how the module gets its output from
the distribution: a random sample, the mode, the median, the mean, or
another deterministic value. :class:`set_composite_lp_aggregate` chooses
whether a :class:`~tensordict.nn.distributions.CompositeDistribution`
aggregates the log-probabilities and entropies of its parts in one
tensor.

.. autosummary::
    :toctree: generated/
    :template: nn_module.rst

    ProbabilisticTensorDictModule
    ProbabilisticTensorDictSequential

.. autosummary::
    :toctree: generated/
    :template: class.rst

    InteractionType
    set_interaction_type
    set_composite_lp_aggregate

.. autosummary::
    :toctree: generated/
    :template: function.rst

    composite_lp_aggregate
    interaction_type

Ensembles
---------

:class:`EnsembleModule` makes ``num_copies`` copies of a module, each with
its own newly initialized parameters, and calls all of them on the input
with :func:`torch.vmap`. The output has a new leading batch dimension of
size ``num_copies``. To keep parameters as a tensordict on a module, see
:class:`~tensordict.TensorDictParams` on :ref:`api-tensordict`.

:func:`functional_vmap` calls a module once per set of parameters in a
stacked tensordict, such as one built with
:meth:`~tensordict.TensorDict.from_modules`, and stacks the outputs. It runs
the sets with :func:`torch.vmap`, or one at a time for modules that
:func:`torch.vmap` cannot handle.

.. autosummary::
    :toctree: generated/
    :template: nn_module.rst

    EnsembleModule

.. autosummary::
    :toctree: generated/
    :template: function.rst

    functional_vmap

Compiling modules
-----------------

The modules of ``tensordict.nn`` work with :func:`torch.compile`. For
example, you can compile a :class:`TensorDictSequential`. For the limits
and their workarounds, see :ref:`compile`. To capture a module in CUDA
graphs, use :class:`CudaGraphModule`.

Distributions
-------------

.. module:: tensordict.nn.distributions

These distributions and modules build the distributions of probabilistic
modules. :class:`NormalParamExtractor` and
:class:`AddStateIndependentNormalScale` turn the output of a network into
the location and scale of a normal distribution.
:class:`CompositeDistribution` groups several distributions, one for each
entry of a tensordict.

.. autosummary::
    :toctree: generated/
    :template: class.rst

    AddStateIndependentNormalScale
    CompositeDistribution
    Delta
    NormalParamExtractor
    OneHotCategorical
    TruncatedNormal
    truncated_normal.TruncatedStandardNormal

Utilities
---------

.. currentmodule:: tensordict.nn

- :class:`dispatch` lets you call a module with keyword arguments instead
  of a tensordict.
- :class:`set_skip_existing` and :func:`skip_existing` make a module skip
  its computation when its output entries are already in the tensordict.
- :func:`mappings`, :func:`add_custom_mapping`, and :class:`biased_softplus`
  map the output of a network to positive values, for example the scale
  of a distribution. :func:`inv_softplus` is the inverse of the softplus
  function, and :func:`~tensordict.nn.utils.expln` is the positive mapping
  of the ``"expln"`` key of :func:`mappings`.
- :func:`rand_one_hot` samples a one-hot vector from logits.

``tensordict.nn.make_tensordict`` is :func:`tensordict.make_tensordict`.

.. autosummary::
    :toctree: generated/
    :template: class.rst

    dispatch
    set_skip_existing
    biased_softplus

.. autosummary::
    :toctree: generated/
    :template: function.rst

    skip_existing
    mappings
    add_custom_mapping
    inv_softplus
    utils.expln
    rand_one_hot

FX tracing (prototype)
----------------------

.. module:: tensordict.prototype

:func:`symbolic_trace` traces a tensordict module with ``torch.fx`` and
returns a :class:`~tensordict.prototype.fx.TDGraphModule`. Both are
prototypes. See :ref:`fx`.

.. autosummary::
    :toctree: generated/
    :template: function.rst

    symbolic_trace

.. currentmodule:: tensordict.prototype.fx

.. autosummary::
    :toctree: generated/
    :template: class.rst

    TDGraphModule

See also
--------

- :ref:`modules-guide`: how a module reads and writes a tensordict, with
  examples.
- :ref:`compile`: compile modules and capture them in CUDA graphs.
- The tutorials :doc:`/tutorials/tensordict_module`,
  :doc:`/tutorials/functional`, and :doc:`/tutorials/export`.
