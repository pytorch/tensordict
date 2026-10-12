# tensordict.nn

This page lists the modules of `tensordict.nn`, and the probabilistic
modules, distributions, and helpers that go with them. A module reads the
entries named in its `in_keys` from a tensordict, and writes the entries
named in its `out_keys` into it. For a guide with examples, see
[Modules: TensorDictModule and tensordict.nn](../modules.html#modules-guide).

The page of a module class lists its methods and the methods of its
tensordict base classes. For the methods of [`torch.nn.Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module), see
the PyTorch documentation.

## Modules

- [`TensorDictModuleBase`](generated/tensordict.nn.TensorDictModuleBase.html#tensordict.nn.TensorDictModuleBase) is the base class of the tensordict
modules.
- [`TensorDictModule`](generated/tensordict.nn.TensorDictModule.html#tensordict.nn.TensorDictModule) wraps a [`torch.nn.Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module) that takes and
returns tensors, and gives it `in_keys` and `out_keys`.
- [`TensorDictSequential`](generated/tensordict.nn.TensorDictSequential.html#tensordict.nn.TensorDictSequential) runs modules one after the other on the
same tensordict.
- [`WrapModule`](generated/tensordict.nn.WrapModule.html#tensordict.nn.WrapModule) wraps a function that takes and returns a
tensordict.
- [`TensorDictModuleWrapper`](generated/tensordict.nn.TensorDictModuleWrapper.html#tensordict.nn.TensorDictModuleWrapper) wraps a tensordict module and behaves
like it, except for the methods that a subclass overrides.
- [`as_tensordict_module`](generated/tensordict.nn.as_tensordict_module.html#tensordict.nn.as_tensordict_module) is a decorator that turns a function into
a [`TensorDictModule`](generated/tensordict.nn.TensorDictModule.html#tensordict.nn.TensorDictModule).
- [`CudaGraphModule`](generated/tensordict.nn.CudaGraphModule.html#tensordict.nn.CudaGraphModule) captures a module or a function in a CUDA graph
and replays it. See [torch.compile, torch.func, torch.export, and CUDA graphs](../compile.html#compile).

| [`TensorDictModuleBase`](generated/tensordict.nn.TensorDictModuleBase.html#tensordict.nn.TensorDictModuleBase)(*args, **kwargs) | Base class to TensorDict modules. |
| --- | --- |
| [`TensorDictModule`](generated/tensordict.nn.TensorDictModule.html#tensordict.nn.TensorDictModule)(module, in_keys, out_keys, *) | Wraps a callable, such as a [`torch.nn.Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module), so that it reads its inputs from a tensordict and writes its outputs to it. |
| [`TensorDictSequential`](generated/tensordict.nn.TensorDictSequential.html#tensordict.nn.TensorDictSequential)(*modules[, ...]) | Runs TensorDictModules one after the other on the same tensordict. |
| [`WrapModule`](generated/tensordict.nn.WrapModule.html#tensordict.nn.WrapModule)(func, *[, inplace, in_keys, out_keys]) | A wrapper around any callable that processes TensorDict instances. |
| [`TensorDictModuleWrapper`](generated/tensordict.nn.TensorDictModuleWrapper.html#tensordict.nn.TensorDictModuleWrapper)(td_module) | Wrapper class for TensorDictModule objects. |

| [`as_tensordict_module`](generated/tensordict.nn.as_tensordict_module.html#tensordict.nn.as_tensordict_module)(*, in_keys, out_keys) | A decorator that converts a function into a TensorDictModule. |
| --- | --- |
| [`CudaGraphModule`](generated/tensordict.nn.CudaGraphModule.html#tensordict.nn.CudaGraphModule)(module[, warmup, in_keys, ...]) | Records a function, or a TensorDictModule, as a CUDA graph and replays it. |

## Modules with tensorclass inputs and outputs

A [`TensorClassModuleBase`](generated/tensordict.nn.TensorClassModuleBase.html#tensordict.nn.TensorClassModuleBase) subclass declares the tensorclass types
of its input and output, so that a type checker can report a wrong field
name or type.
[`TensorClassModuleBase.as_td_module()`](generated/tensordict.nn.TensorClassModuleBase.html#tensordict.nn.TensorClassModuleBase.as_td_module) returns a
[`TensorClassModuleWrapper`](generated/tensordict.nn.TensorClassModuleWrapper.html#tensordict.nn.TensorClassModuleWrapper), which reads and writes a tensordict, so
that you can put the module in a [`TensorDictSequential`](generated/tensordict.nn.TensorDictSequential.html#tensordict.nn.TensorDictSequential).

| [`TensorClassModuleBase`](generated/tensordict.nn.TensorClassModuleBase.html#tensordict.nn.TensorClassModuleBase)(*args, **kwargs) | A TensorClassModuleBase is a base class for modules that operate on TensorClass instances. |
| --- | --- |
| [`TensorClassModuleWrapper`](generated/tensordict.nn.TensorClassModuleWrapper.html#tensordict.nn.TensorClassModuleWrapper)(module) | Wrapper class for TensorClassModuleBase objects. |

## Probabilistic modules

A [`ProbabilisticTensorDictModule`](generated/tensordict.nn.ProbabilisticTensorDictModule.html#tensordict.nn.ProbabilisticTensorDictModule) builds a distribution from
entries of the tensordict, and writes a sample, and optionally its
log-probability, to the tensordict. [`InteractionType`](generated/tensordict.nn.InteractionType.html#tensordict.nn.InteractionType) and
[`set_interaction_type`](generated/tensordict.nn.set_interaction_type.html#tensordict.nn.set_interaction_type) choose how the module gets its output from
the distribution: a random sample, the mode, the median, the mean, or
another deterministic value. [`set_composite_lp_aggregate`](generated/tensordict.nn.set_composite_lp_aggregate.html#tensordict.nn.set_composite_lp_aggregate) chooses
whether a [`CompositeDistribution`](generated/tensordict.nn.distributions.CompositeDistribution.html#tensordict.nn.distributions.CompositeDistribution)
aggregates the log-probabilities and entropies of its parts in one
tensor.

| [`ProbabilisticTensorDictModule`](generated/tensordict.nn.ProbabilisticTensorDictModule.html#tensordict.nn.ProbabilisticTensorDictModule)(in_keys, ...) | A module that builds a distribution from entries of a tensordict and writes a value from it. |
| --- | --- |
| [`ProbabilisticTensorDictSequential`](generated/tensordict.nn.ProbabilisticTensorDictSequential.html#tensordict.nn.ProbabilisticTensorDictSequential)(*modules) | A [`TensorDictSequential`](generated/tensordict.nn.TensorDictSequential.html#tensordict.nn.TensorDictSequential) that contains one or more [`ProbabilisticTensorDictModule`](generated/tensordict.nn.ProbabilisticTensorDictModule.html#tensordict.nn.ProbabilisticTensorDictModule) instances, and can return their distribution. |

| [`InteractionType`](generated/tensordict.nn.InteractionType.html#tensordict.nn.InteractionType)(value[, names, module, ...]) | The rule that a probabilistic module uses to get a value from a distribution. |
| --- | --- |
| [`set_interaction_type`](generated/tensordict.nn.set_interaction_type.html#tensordict.nn.set_interaction_type)([type]) | Sets the interaction type of all probabilistic modules inside a block of code, or inside a decorated function. |
| [`set_composite_lp_aggregate`](generated/tensordict.nn.set_composite_lp_aggregate.html#tensordict.nn.set_composite_lp_aggregate)([mode]) | Controls whether [`CompositeDistribution`](generated/tensordict.nn.distributions.CompositeDistribution.html#tensordict.nn.distributions.CompositeDistribution) log-probabilities and entropies will be aggregated in a single tensor. |

| [`composite_lp_aggregate`](generated/tensordict.nn.composite_lp_aggregate.html#tensordict.nn.composite_lp_aggregate)([nowarn]) | Returns whether a [`CompositeDistribution`](generated/tensordict.nn.distributions.CompositeDistribution.html#tensordict.nn.distributions.CompositeDistribution) log-probabilities and entropies will be aggregated in a single tensor. |
| --- | --- |
| [`probabilistic.interaction_type`](generated/tensordict.nn.probabilistic.interaction_type.html#tensordict.nn.probabilistic.interaction_type)() | Returns the interaction type set by [`set_interaction_type`](generated/tensordict.nn.set_interaction_type.html#tensordict.nn.set_interaction_type), or `None` outside of it. |

## Ensembles

[`EnsembleModule`](generated/tensordict.nn.EnsembleModule.html#tensordict.nn.EnsembleModule) makes `num_copies` copies of a module, each with
its own newly initialized parameters, and calls all of them on the input
with [`torch.vmap()`](https://docs.pytorch.org/docs/stable/generated/torch.vmap.html#torch.vmap). The output has a new leading batch dimension of
size `num_copies`. To keep parameters as a tensordict on a module, see
[`TensorDictParams`](generated/tensordict.TensorDictParams.html#tensordict.TensorDictParams) on [tensordict: containers and functions](td.html#api-tensordict).

| [`EnsembleModule`](generated/tensordict.nn.EnsembleModule.html#tensordict.nn.EnsembleModule)(module, num_copies[, ...]) | Module that wraps a module and repeats it to form an ensemble. |
| --- | --- |

## Compiling modules

The modules of `tensordict.nn` work with [`torch.compile()`](https://docs.pytorch.org/docs/stable/generated/torch.compile.html#torch.compile). For
example, you can compile a [`TensorDictSequential`](generated/tensordict.nn.TensorDictSequential.html#tensordict.nn.TensorDictSequential). For the limits
and their workarounds, see [torch.compile, torch.func, torch.export, and CUDA graphs](../compile.html#compile). To capture a module in CUDA
graphs, use [`CudaGraphModule`](generated/tensordict.nn.CudaGraphModule.html#tensordict.nn.CudaGraphModule).

## Distributions

These distributions and modules build the distributions of probabilistic
modules. [`NormalParamExtractor`](generated/tensordict.nn.distributions.NormalParamExtractor.html#tensordict.nn.distributions.NormalParamExtractor) and
[`AddStateIndependentNormalScale`](generated/tensordict.nn.distributions.AddStateIndependentNormalScale.html#tensordict.nn.distributions.AddStateIndependentNormalScale) turn the output of a network into
the location and scale of a normal distribution.
[`CompositeDistribution`](generated/tensordict.nn.distributions.CompositeDistribution.html#tensordict.nn.distributions.CompositeDistribution) groups several distributions, one for each
entry of a tensordict.

| [`AddStateIndependentNormalScale`](generated/tensordict.nn.distributions.AddStateIndependentNormalScale.html#tensordict.nn.distributions.AddStateIndependentNormalScale)([...]) | A nn.Module that adds trainable state-independent scale parameters. |
| --- | --- |
| [`CompositeDistribution`](generated/tensordict.nn.distributions.CompositeDistribution.html#tensordict.nn.distributions.CompositeDistribution)(params, ...[, ...]) | A composite distribution that groups multiple distributions together using the TensorDict interface. |
| [`Delta`](generated/tensordict.nn.distributions.Delta.html#tensordict.nn.distributions.Delta)(param[, atol, rtol, batch_shape, ...]) | Delta distribution. |
| [`NormalParamExtractor`](generated/tensordict.nn.distributions.NormalParamExtractor.html#tensordict.nn.distributions.NormalParamExtractor)([scale_mapping, scale_lb]) | A non-parametric nn.Module that splits its input into loc and scale parameters. |
| [`OneHotCategorical`](generated/tensordict.nn.distributions.OneHotCategorical.html#tensordict.nn.distributions.OneHotCategorical)([logits, probs]) | One-hot categorical distribution. |
| [`TruncatedNormal`](generated/tensordict.nn.distributions.TruncatedNormal.html#tensordict.nn.distributions.TruncatedNormal)(loc, scale, a, b[, ...]) | Truncated Normal distribution. |
| [`truncated_normal.TruncatedStandardNormal`](generated/tensordict.nn.distributions.truncated_normal.TruncatedStandardNormal.html#tensordict.nn.distributions.truncated_normal.TruncatedStandardNormal)(a, b) | Truncated Standard Normal distribution. |

## Utilities

- [`dispatch`](generated/tensordict.nn.dispatch.html#tensordict.nn.dispatch) lets you call a module with keyword arguments instead
of a tensordict.
- [`set_skip_existing`](generated/tensordict.nn.set_skip_existing.html#tensordict.nn.set_skip_existing) and [`skip_existing()`](generated/tensordict.nn.skip_existing.html#tensordict.nn.skip_existing) make a module skip
its computation when its output entries are already in the tensordict.
- [`mappings()`](generated/tensordict.nn.mappings.html#tensordict.nn.mappings), [`add_custom_mapping()`](generated/tensordict.nn.add_custom_mapping.html#tensordict.nn.add_custom_mapping), and [`biased_softplus`](generated/tensordict.nn.biased_softplus.html#tensordict.nn.biased_softplus)
map the output of a network to positive values, for example the scale
of a distribution. [`inv_softplus()`](generated/tensordict.nn.inv_softplus.html#tensordict.nn.inv_softplus) is the inverse of the softplus
function, and [`expln()`](generated/tensordict.nn.utils.expln.html#tensordict.nn.utils.expln) is the positive mapping
of the `"expln"` key of [`mappings()`](generated/tensordict.nn.mappings.html#tensordict.nn.mappings).
- [`rand_one_hot()`](generated/tensordict.nn.rand_one_hot.html#tensordict.nn.rand_one_hot) samples a one-hot vector from logits.

`tensordict.nn.make_tensordict` is [`tensordict.make_tensordict()`](generated/tensordict.make_tensordict.html#tensordict.make_tensordict).

| [`dispatch`](generated/tensordict.nn.dispatch.html#tensordict.nn.dispatch)([separator, source, dest, ...]) | Allows for a function expecting a TensorDict to be called using kwargs. |
| --- | --- |
| [`set_skip_existing`](generated/tensordict.nn.set_skip_existing.html#tensordict.nn.set_skip_existing)([mode, in_key_attr, ...]) | A context manager for skipping existing nodes in a TensorDict graph. |
| [`biased_softplus`](generated/tensordict.nn.biased_softplus.html#tensordict.nn.biased_softplus)(bias[, min_val]) | A biased softplus module. |

| [`skip_existing`](generated/tensordict.nn.skip_existing.html#tensordict.nn.skip_existing)() | Returns which existing entries should not be re-computed by a module. |
| --- | --- |
| [`mappings`](generated/tensordict.nn.mappings.html#tensordict.nn.mappings)(key) | Returns the function named by `key`. |
| [`add_custom_mapping`](generated/tensordict.nn.add_custom_mapping.html#tensordict.nn.add_custom_mapping)(name, mapping) | Adds a custom mapping to be used in mapping classes. |
| [`inv_softplus`](generated/tensordict.nn.inv_softplus.html#tensordict.nn.inv_softplus)(bias) | Inverse softplus function. |
| [`utils.expln`](generated/tensordict.nn.utils.expln.html#tensordict.nn.utils.expln)(x) | A smooth, continuous positive mapping presented in "State-Dependent Exploration for Policy Gradient Methods". |
| [`rand_one_hot`](generated/tensordict.nn.rand_one_hot.html#tensordict.nn.rand_one_hot)(values[, do_softmax]) | Draws a one-hot sample along the last dimension of `values`. |

## FX tracing (prototype)

[`symbolic_trace()`](generated/tensordict.prototype.symbolic_trace.html#tensordict.prototype.symbolic_trace) traces a tensordict module with `torch.fx` and
returns a [`TDGraphModule`](generated/tensordict.prototype.fx.TDGraphModule.html#tensordict.prototype.fx.TDGraphModule). Both are
prototypes. See [FX tracing (prototype)](../fx.html#fx).

| [`symbolic_trace`](generated/tensordict.prototype.symbolic_trace.html#tensordict.prototype.symbolic_trace)(td_module) | Traces a TensorDictModule or a TensorDictSequential into an FX graph. |
| --- | --- |

| [`TDGraphModule`](generated/tensordict.prototype.fx.TDGraphModule.html#tensordict.prototype.fx.TDGraphModule)(graph_module, out_keys) | The module that `symbolic_trace()` returns: an FX graph that reads and writes a tensordict. |
| --- | --- |

## See also

- [Modules: TensorDictModule and tensordict.nn](../modules.html#modules-guide): how a module reads and writes a tensordict, with
examples.
- [torch.compile, torch.func, torch.export, and CUDA graphs](../compile.html#compile): compile modules and capture them in CUDA graphs.
- The tutorials [Building models with TensorDictModule](../tutorials/tensordict_module.html),
[Functional calls and ensembles with TensorDict](../tutorials/functional.html), and [Exporting tensordict modules](../tutorials/export.html).