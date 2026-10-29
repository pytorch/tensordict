Note

Go to the end
to download the full example code.

# Exporting tensordict modules

**Author**: [Vincent Moens](https://github.com/vmoens)

After you write a model with `tensordict.nn`, you can export its
computational graph with [`torch.export.export()`](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/export/api_reference.html#torch.export.export). The exported program
takes and returns plain tensors, and it does not need tensordict at run time.
Use it to run the model on other hardware (robots, drones, edge devices), or
to remove the dependency on tensordict. This tutorial is for readers who
deploy a `tensordict.nn` model.

What you will learn:

- How to call a `tensordict.nn` module with tensors instead of a
[`TensorDict`](../reference/generated/tensordict.TensorDict.html#tensordict.TensorDict).
- How to export the module with `torch.export` and save the result to a file.
- How to export a module that reads and writes nested keys.
- How to select the outputs of the model before you export it.

Before you start, read the [TensorDictModule](tensordict_module.html#tensordictmodule)
tutorial.

```
import tempfile
from pathlib import Path

import torch
from tensordict.nn import (
 NormalParamExtractor,
 TensorDictModule as Mod,
 TensorDictSequential as Seq,
)
from torch import nn
```

## Building the model

Build a small network with `tensordict.nn`. The network has:

- A linear layer that maps the input to a hidden representation.
- A ReLU activation.
- A second linear layer that writes a latent representation.

A [`NormalParamExtractor`](../reference/generated/tensordict.nn.distributions.NormalParamExtractor.html#tensordict.nn.distributions.NormalParamExtractor) then splits the latent
tensor into two outputs, `"loc"` and `"scale"`.

```
model = Seq(
 # 1. A small network for embedding
 Mod(nn.Linear(3, 4), in_keys=["x"], out_keys=["hidden"]),
 Mod(nn.ReLU(), in_keys=["hidden"], out_keys=["hidden"]),
 Mod(nn.Linear(4, 4), in_keys=["hidden"], out_keys=["latent"]),
 # 2. Extracting params (splits into loc and scale)
 Mod(NormalParamExtractor(), in_keys=["latent"], out_keys=["loc", "scale"]),
)
```

Run the model with a tensor as a keyword argument, and look at the output:

```
torch.manual_seed(0)
x = torch.randn(1, 3)
print(model(x=x))
```

```
(tensor([[0.0000, 0.8067, 0.0791, 0.0000]], grad_fn=<ReluBackward0>), tensor([[-0.1955, 0.2908, -0.2976, 0.2212]], grad_fn=<AddmmBackward0>), tensor([[-0.1955, 0.2908]], grad_fn=<SplitBackward0>), tensor([[0.8236, 1.1446]], grad_fn=<ClampMinBackward0>))
```

Called with a tensor, the model returns one tensor per output key: here
`"hidden"`, `"latent"`, `"loc"`, and `"scale"`. For a large model,
you usually need fewer outputs. The section "Selecting the outputs" below
shows how to keep only some of them.

## Using `torch.export` with a `TensorDictModule`

`torch.export` captures the graph of a module in a standard form, which
does not depend on `tensordict`. The [`export()`](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/export/api_reference.html#torch.export.export)
function returns an `ExportedProgram`, which has these attributes:

- `graph_module`: the FX graph that `export` captured.
- `graph_signature`: the inputs and outputs of the graph.
- `module()`: a method that returns a callable, which you can use in
place of the original module.

The model accepts positional and keyword arguments. This tutorial uses
keyword arguments, because they name the input keys.

```
from torch.export import export

model_export = export(model, args=(), kwargs={"x": x})
```

Look at the module:

```
print("module:", model_export.module())
```

```
module: GraphModule(
 (module): Module(
 (0): Module(
 (module): Module()
 )
 (2): Module(
 (module): Module()
 )
 )
 (_guards_fn): GuardsFn()
)

def forward(self, x):
 x, = fx_pytree.tree_flatten_spec(([], {'x':x}), self._in_spec)
 module_0_module_weight = getattr(self.module, "0").module.weight
 module_0_module_bias = getattr(self.module, "0").module.bias
 module_2_module_weight = getattr(self.module, "2").module.weight
 module_2_module_bias = getattr(self.module, "2").module.bias
 _guards_fn = self._guards_fn(x); _guards_fn = None
 linear = torch.ops.aten.linear.default(x, module_0_module_weight, module_0_module_bias); x = module_0_module_weight = module_0_module_bias = None
 relu = torch.ops.aten.relu.default(linear); linear = None
 linear_1 = torch.ops.aten.linear.default(relu, module_2_module_weight, module_2_module_bias); module_2_module_weight = module_2_module_bias = None
 chunk = torch.ops.aten.chunk.default(linear_1, 2, -1)
 getitem = chunk[0]
 getitem_1 = chunk[1]; chunk = None
 add = torch.ops.aten.add.Tensor(getitem_1, 0.5254586935043335); getitem_1 = None
 softplus = torch.ops.aten.softplus.default(add); add = None
 add_1 = torch.ops.aten.add.Tensor(softplus, 0.01); softplus = None
 clamp_min = torch.ops.aten.clamp_min.default(add_1, 0.0001); add_1 = None
 return pytree.tree_unflatten((relu, linear_1, getitem, clamp_min), self._out_spec)

# To see more debug info, please use `graph_module.print_readable()`
```

You call this module like the original module, and it returns the same
tensors:

```
exported = model_export.module()
for out_model, out_exported in zip(model(x=x), exported(x=x)):
 torch.testing.assert_close(out_model, out_exported)
```

Export does not optimize the graph. The exported module runs the same tensor
operations as the original module, one by one, without the tensordict reads
and writes. To make it faster, compile it (see [torch.compile, torch.func, torch.export, and CUDA graphs](../compile.html#compile)), or compile
it ahead of time with AOTInductor (see "Further reading").

Print the FX graph:

```
print(model_export.graph_module.print_readable(print_output=False))
```

```
class GraphModule(torch.nn.Module):
 def forward(self, p_module_0_module_weight: "f32[4, 3]", p_module_0_module_bias: "f32[4]", p_module_2_module_weight: "f32[4, 4]", p_module_2_module_bias: "f32[4]", x: "f32[1, 3]"):
 # File: /__w/tensordict/tensordict/env/lib/python3.12/site-packages/torch/nn/modules/linear.py:134 in forward, code: return F.linear(input, self.weight, self.bias)
 linear: "f32[1, 4]" = torch.ops.aten.linear.default(x, p_module_0_module_weight, p_module_0_module_bias); x = p_module_0_module_weight = p_module_0_module_bias = None

 # File: /__w/tensordict/tensordict/env/lib/python3.12/site-packages/torch/nn/modules/activation.py:143 in forward, code: return F.relu(input, inplace=self.inplace)
 relu: "f32[1, 4]" = torch.ops.aten.relu.default(linear); linear = None

 # File: /__w/tensordict/tensordict/env/lib/python3.12/site-packages/torch/nn/modules/linear.py:134 in forward, code: return F.linear(input, self.weight, self.bias)
 linear_1: "f32[1, 4]" = torch.ops.aten.linear.default(relu, p_module_2_module_weight, p_module_2_module_bias); p_module_2_module_weight = p_module_2_module_bias = None

 # File: /__w/tensordict/tensordict/tensordict/nn/distributions/continuous.py:80 in forward, code: loc, scale = tensor.chunk(2, -1)
 chunk = torch.ops.aten.chunk.default(linear_1, 2, -1)
 getitem: "f32[1, 2]" = chunk[0]
 getitem_1: "f32[1, 2]" = chunk[1]; chunk = None

 # File: /__w/tensordict/tensordict/tensordict/nn/utils.py:96 in forward, code: return torch.nn.functional.softplus(x + self.bias) + self.min_val
 add: "f32[1, 2]" = torch.ops.aten.add.Tensor(getitem_1, 0.5254586935043335); getitem_1 = None
 softplus: "f32[1, 2]" = torch.ops.aten.softplus.default(add); add = None
 add_1: "f32[1, 2]" = torch.ops.aten.add.Tensor(softplus, 0.01); softplus = None

 # File: /__w/tensordict/tensordict/tensordict/nn/distributions/continuous.py:81 in forward, code: scale = self.scale_mapping(scale).clamp_min(self.scale_lb)
 clamp_min: "f32[1, 2]" = torch.ops.aten.clamp_min.default(add_1, 0.0001); add_1 = None
 return (relu, linear_1, getitem, clamp_min)
```

### Working with nested keys

Nested keys are tuples of strings, such as `("some", "key")`. A keyword
argument must be a single string, so [`dispatch`](../reference/generated/tensordict.nn.dispatch.html#tensordict.nn.dispatch)
joins the parts of a nested key with an underscore (`"_"`). In the
example below, the input key `("some", "key")` becomes the keyword
argument `some_key`.

```
model_nested = Seq(
 Mod(lambda x: x + 1, in_keys=[("some", "key")], out_keys=["hidden"]),
 Mod(lambda x: x - 1, in_keys=["hidden"], out_keys=[("some", "output")]),
).select_out_keys(("some", "output"))

model_nested_export = export(model_nested, args=(), kwargs={"some_key": x})
print("exported module with nested input:", model_nested_export.module())
```

```
exported module with nested input: GraphModule(
 (_guards_fn): GuardsFn()
)

def forward(self, some_key):
 some_key, = fx_pytree.tree_flatten_spec(([], {'some_key':some_key}), self._in_spec)
 _guards_fn = self._guards_fn(some_key); _guards_fn = None
 add = torch.ops.aten.add.Tensor(some_key, 1); some_key = None
 sub = torch.ops.aten.sub.Tensor(add, 1); add = None
 return pytree.tree_unflatten((sub,), self._out_spec)

# To see more debug info, please use `graph_module.print_readable()`
```

The `forward` method of the exported module takes `some_key`:

```
out = model_nested_export.module()(some_key=x)
torch.testing.assert_close(out, x)
```

The callable that `module()` returns is a plain Python callable. You can
compile it with [`compile()`](https://docs.pytorch.org/docs/stable/generated/torch.compile.html#torch.compile).

### Saving the exported module

`torch.export` has its own file format. Save the program with
[`save()`](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/export/api_reference.html#torch.export.save) and load it with [`load()`](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/export/api_reference.html#torch.export.load). By
convention, the file name ends with `".pt2"`:

```
path = Path(tempfile.mkdtemp()) / "model.pt2"
torch.export.save(model_export, path)
model_loaded = torch.export.load(path)
for out_model, out_loaded in zip(model(x=x), model_loaded.module()(x=x)):
 torch.testing.assert_close(out_model, out_loaded)
```

## Selecting the outputs

By default, a `tensordict.nn` module keeps every intermediate value in its
output. During training, you can log these values, or rebuild a
distribution from its saved parameters instead of saving the
[`Distribution`](https://docs.pytorch.org/docs/stable/distributions.html#torch.distributions.distribution.Distribution) object. The extra memory is small
during training, because autograd keeps these tensors for the backward pass
anyway.

An exported inference model usually needs one or two outputs. Select them
before you export, so that the exported graph returns only these tensors.
You can select outputs in three ways:

1. Pass the `selected_out_keys` keyword argument when you build the
[`TensorDictSequential`](../reference/generated/tensordict.nn.TensorDictSequential.html#tensordict.nn.TensorDictSequential).
2. Call the [`select_out_keys()`](../reference/generated/tensordict.nn.TensorDictModuleBase.html#tensordict.nn.TensorDictModuleBase.select_out_keys) method.
It changes the `out_keys` attribute in place. To undo it, call
[`reset_out_keys()`](../reference/generated/tensordict.nn.TensorDictModuleBase.html#tensordict.nn.TensorDictModuleBase.reset_out_keys).
3. Wrap the module in a [`TensorDictSequential`](../reference/generated/tensordict.nn.TensorDictSequential.html#tensordict.nn.TensorDictSequential) that
selects the keys:

```
>>> module_filtered = Seq(module, selected_out_keys=["loc"])
```

Select the `"loc"` output with the second method. With an `x` input,
the model now returns one tensor, the `"loc"` output:

```
model.select_out_keys("loc")
print(model(x=x))
```

```
tensor([[-0.1955, 0.2908]], grad_fn=<SplitBackward0>)
```

Export the model again. The graph now returns a single tensor:

```
model_export = export(model, args=(), kwargs={"x": x})
print("module:", model_export.module())
```

```
module: GraphModule(
 (module): Module(
 (0): Module(
 (module): Module()
 )
 (2): Module(
 (module): Module()
 )
 )
 (_guards_fn): GuardsFn()
)

def forward(self, x):
 x, = fx_pytree.tree_flatten_spec(([], {'x':x}), self._in_spec)
 module_0_module_weight = getattr(self.module, "0").module.weight
 module_0_module_bias = getattr(self.module, "0").module.bias
 module_2_module_weight = getattr(self.module, "2").module.weight
 module_2_module_bias = getattr(self.module, "2").module.bias
 _guards_fn = self._guards_fn(x); _guards_fn = None
 linear = torch.ops.aten.linear.default(x, module_0_module_weight, module_0_module_bias); x = module_0_module_weight = module_0_module_bias = None
 relu = torch.ops.aten.relu.default(linear); linear = None
 linear_1 = torch.ops.aten.linear.default(relu, module_2_module_weight, module_2_module_bias); relu = module_2_module_weight = module_2_module_bias = None
 chunk = torch.ops.aten.chunk.default(linear_1, 2, -1); linear_1 = None
 getitem = chunk[0]
 getitem_1 = chunk[1]; chunk = None
 add = torch.ops.aten.add.Tensor(getitem_1, 0.5254586935043335); getitem_1 = None
 softplus = torch.ops.aten.softplus.default(add); add = None
 add_1 = torch.ops.aten.add.Tensor(softplus, 0.01); softplus = None
 clamp_min = torch.ops.aten.clamp_min.default(add_1, 0.0001); add_1 = clamp_min = None
 return pytree.tree_unflatten((getitem,), self._out_spec)

# To see more debug info, please use `graph_module.print_readable()`
```

## Conclusion

You can now export a `tensordict.nn` model with `torch.export` into a
program that takes and returns plain tensors. You can pass nested keys as
keyword arguments joined with `"_"`, and save the program to a `.pt2`
file and load it again. You can also select the outputs of the model before
the export.

## Further reading

- [torch.compile, torch.func, torch.export, and CUDA graphs](../compile.html#compile) explains how tensordicts go through `torch.export`,
including dynamic batch sizes.
- [Modules: TensorDictModule and tensordict.nn](../modules.html#modules-guide) explains `tensordict.nn` modules and
[`dispatch`](../reference/generated/tensordict.nn.dispatch.html#tensordict.nn.dispatch).
- The [torch.export documentation](https://pytorch.org/docs/stable/export)
and the [torch.export tutorial](https://pytorch.org/tutorials/intermediate/torch_export_tutorial.html)
explain the exporter.
- The [ONNX tutorials](https://pytorch.org/tutorials/beginner/onnx/export_simple_model_to_onnx_tutorial.html)
explain the ONNX exporter. `torch.onnx.export(..., dynamo=True)` is
built on `torch.export`.
- To run PyTorch code on servers without Python, read the
[AOTInductor](https://pytorch.org/docs/main/torch.compiler_aot_inductor.html)
documentation.

**Total running time of the script:** (0 minutes 0.178 seconds)

[`Download Jupyter notebook: export.ipynb`](../_downloads/4e8ac58ef63f1e596d49d1b7366ef9bc/export.ipynb)

[`Download Python source code: export.py`](../_downloads/bbb4a09f4b0d139b93b8e77c84568b01/export.py)

[`Download zipped: export.zip`](../_downloads/150528e38f6816824f1e81ed67476a9f/export.zip)

[Gallery generated by Sphinx-Gallery](https://sphinx-gallery.github.io)