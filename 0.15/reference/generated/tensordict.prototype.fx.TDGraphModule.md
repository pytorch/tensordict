# TDGraphModule

*class*tensordict.prototype.fx.TDGraphModule(*graph_module: [GraphModule](https://docs.pytorch.org/docs/stable/fx.html#torch.fx.GraphModule)*, *out_keys: [list](https://docs.python.org/3/builtins/stdtypes.html#list)[[NestedKey](tensordict.NestedKey.html#tensordict.NestedKey)]*)

Bases: [`Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)

The module that `symbolic_trace()` returns: an FX graph that reads and writes a tensordict.

A call reads the `in_keys` of the traced module from the input
tensordict, runs the graph, and writes the outputs at `out_keys`, into
`tensordict_out` if given, else into the input. The `graph` and
`code` attributes are those of the underlying
[`torch.fx.GraphModule`](https://docs.pytorch.org/docs/stable/fx.html#torch.fx.GraphModule).

This class is a prototype: its API can change without a deprecation
period.

Parameters:

- **graph_module** ([*GraphModule*](https://docs.pytorch.org/docs/stable/fx.html#torch.fx.GraphModule)) - the traced graph.
- **out_keys** ([*list*](https://docs.python.org/3/builtins/stdtypes.html#list)*of*[*NestedKey*](tensordict.NestedKey.html#tensordict.NestedKey)) - the keys where the outputs are written.

forward(*tensordict: [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*, *tensordict_out: [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, ***kwargs*) → [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase)

Define the computation performed at every call.

Should be overridden by all subclasses.

Note

Although the recipe for forward pass needs to be defined within
this function, one should call the `Module` instance afterwards
instead of this since the former takes care of running the
registered hooks while the latter silently ignores them.