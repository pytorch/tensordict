# from_safetensors

tensordict.from_safetensors(*path: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | Path*, ***, *device: DeviceType | [None](https://docs.python.org/3/builtins/constants.html#None) = None*) → [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase)

Loads a tensordict from a safetensors file.

A file written by [`to_safetensors()`](tensordict.to_safetensors.html#tensordict.to_safetensors) loads with its
structure: the batch sizes, dim names, tensorclasses, lazy stacks and
non-tensor values that it was saved with. A lazy stack loads with the
same stack dim, and its entries load as dense tensordicts. A tensorclass
loads only if the module that defines it is imported: tensordict never
imports a module by the name in a file.

Any other safetensors file, such as a model checkpoint from the Hugging
Face Hub, loads as a nested [`TensorDict`](tensordict.TensorDict.html#tensordict.TensorDict) with an empty
batch size: each name is split on `"."`. The result has the layout of
[`from_module()`](tensordict.TensorDict.html#tensordict.TensorDict.from_module), so that
[`to_module()`](tensordict.TensorDictBase.to_module.html#tensordict.TensorDictBase.to_module) writes it into the model.
Requires the `safetensors` package.

Parameters:

**path** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*or**Path*) - the path of the file.

Keyword Arguments:

**device** ([*device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*or**compatible**,**optional*) - the device to load the
tensors on, which is also the device of the result. Defaults to
`None`: the tensors load on the CPU, and the result has no
device.

Returns:

a [`TensorDict`](tensordict.TensorDict.html#tensordict.TensorDict), a
[`LazyStackedTensorDict`](tensordict.LazyStackedTensorDict.html#tensordict.LazyStackedTensorDict), or the tensorclass that
was saved.

Examples

```
>>> import tempfile
>>> from pathlib import Path
>>> import torch
>>> from safetensors.torch import save_file
>>> from tensordict import from_safetensors
>>> with tempfile.TemporaryDirectory() as tmp:
... save_file(
... {"0.weight": torch.ones(4, 3), "0.bias": torch.zeros(4)},
... Path(tmp) / "model.safetensors",
... )
... params = from_safetensors(Path(tmp) / "model.safetensors")
>>> params["0", "weight"].shape
torch.Size([4, 3])
>>> _ = params.to_module(torch.nn.Sequential(torch.nn.Linear(3, 4)))
```