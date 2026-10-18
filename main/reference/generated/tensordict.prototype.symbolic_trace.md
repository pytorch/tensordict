# symbolic_trace

tensordict.prototype.symbolic_trace(*td_module: [TensorDictModuleBase](tensordict.nn.TensorDictModuleBase.html#tensordict.nn.TensorDictModuleBase)*) → [TDGraphModule](tensordict.prototype.fx.TDGraphModule.html#tensordict.prototype.fx.TDGraphModule)

Traces a TensorDictModule or a TensorDictSequential into an FX graph.

This function is a prototype: its API can change without a deprecation
period. See [FX tracing (prototype)](../../fx.html#fx) for a guide.

It supports a [`TensorDictModule`](tensordict.nn.TensorDictModule.html#tensordict.nn.TensorDictModule) that wraps an
[`Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module) that [`torch.fx.symbolic_trace()`](https://docs.pytorch.org/docs/stable/fx.html#torch.fx.symbolic_trace) can trace,
and a [`TensorDictSequential`](tensordict.nn.TensorDictSequential.html#tensordict.nn.TensorDictSequential) of such modules. A
module that wraps a plain function or a lambda raises a `RuntimeError`
("Unsupported type ... passed for root!").

Parameters:

**td_module** ([*TensorDictModuleBase*](tensordict.nn.TensorDictModuleBase.html#tensordict.nn.TensorDictModuleBase)) - the module to trace.

Returns:

A `TDGraphModule` with the same inputs and outputs as
`td_module`.

Raises:

[**TypeError**](https://docs.python.org/3/builtins/exceptions.html#TypeError) - if `td_module` is not a
 [`TensorDictModule`](tensordict.nn.TensorDictModule.html#tensordict.nn.TensorDictModule) or a
 [`TensorDictSequential`](tensordict.nn.TensorDictSequential.html#tensordict.nn.TensorDictSequential).

Examples

```
>>> import torch
>>> from torch import nn
>>> from tensordict import TensorDict
>>> from tensordict.nn import TensorDictModule, TensorDictSequential
>>> from tensordict.prototype.fx import symbolic_trace
>>> seq = TensorDictSequential(
... TensorDictModule(nn.Linear(3, 4), in_keys=["x"], out_keys=["hidden"]),
... TensorDictModule(nn.ReLU(), in_keys=["hidden"], out_keys=["y"]),
... )
>>> graph_module = symbolic_trace(seq)
>>> td = graph_module(TensorDict(x=torch.ones(2, 3), batch_size=[2]))
>>> sorted(td.keys())
['hidden', 'x', 'y']
>>> expected = seq(TensorDict(x=torch.ones(2, 3), batch_size=[2]))["y"]
>>> torch.equal(td["y"], expected)
True
```