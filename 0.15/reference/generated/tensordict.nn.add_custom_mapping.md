# add_custom_mapping

tensordict.nn.add_custom_mapping(*name: [str](https://docs.python.org/3/builtins/stdtypes.html#str)*, *mapping: [Callable](https://docs.python.org/3/library/typing.html#typing.Callable)[[[Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)], [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)]*)

Adds a custom mapping to be used in mapping classes.

Parameters:

- **name** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)) - a mapping name.
- **mapping** (*callable*) - a callable that takes a tensor as input and outputs a tensor
with the same shape.

Examples

```
>>> import torch
>>> from tensordict.nn import add_custom_mapping, NormalParamExtractor
>>> add_custom_mapping("my_mapping", lambda x: torch.zeros_like(x))
>>> npe = NormalParamExtractor(scale_mapping="my_mapping", scale_lb=0.0)
>>> assert (npe(torch.randn(10))[1] == torch.zeros(5)).all()
```