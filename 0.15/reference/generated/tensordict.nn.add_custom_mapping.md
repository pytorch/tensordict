# add_custom_mapping

tensordict.nn.add_custom_mapping(*name: [str](https://docs.python.org/3/builtins/stdtypes.html#str)*, *mapping: [Callable](https://docs.python.org/3/library/typing.html#typing.Callable)[[[Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)], [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)]*)

Registers `mapping` under `name`, so that [`mappings()`](tensordict.nn.mappings.html#tensordict.nn.mappings) returns it.

The name can then be passed wherever a module takes a mapping name, such
as the `scale_mapping` argument of
`NormalParamExtractor`. A name that exists already
is replaced.

Parameters:

- **name** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)) - the name of the mapping.
- **mapping** (*Callable*) - a function that takes a tensor and returns a
tensor of the same shape.

Examples

```
>>> import torch
>>> from tensordict.nn import add_custom_mapping, NormalParamExtractor
>>> add_custom_mapping("my_mapping", lambda x: torch.zeros_like(x))
>>> npe = NormalParamExtractor(scale_mapping="my_mapping", scale_lb=0.0)
>>> npe(torch.randn(10))[1]
tensor([0., 0., 0., 0., 0.])
```