# AddStateIndependentNormalScale

*class*tensordict.nn.distributions.AddStateIndependentNormalScale(*scale_shape: [Size](https://docs.pytorch.org/docs/stable/size.html#torch.Size) | [int](https://docs.python.org/3/builtins/functions.html#int) | [tuple](https://docs.python.org/3/builtins/stdtypes.html#tuple) = None*, ***, *scale_mapping: [str](https://docs.python.org/3/builtins/stdtypes.html#str) = 'exp'*, *scale_lb: [Number](https://docs.python.org/3/library/numbers.html#numbers.Number) = 0.0001*, *device: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *make_param: [bool](https://docs.python.org/3/builtins/functions.html#bool) = True*, *init_value: [float](https://docs.python.org/3/builtins/functions.html#float) = 0.0*)

Bases: [`Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)

A module that returns its input as a location, with a learned scale that does not depend on the input.

The module holds one tensor, `state_independent_scale`, of shape
`scale_shape`. `module(loc, *others)` returns
`(loc, scale, *others)`, where `scale` is that tensor expanded to the
shape of `loc`, mapped by `scale_mapping` and clamped to at least
`scale_lb`. Unlike [`NormalParamExtractor`](tensordict.nn.distributions.NormalParamExtractor.html#tensordict.nn.distributions.NormalParamExtractor), the input is not split:
all of it is the location.

Parameters:

**scale_shape** ([*Size*](https://docs.pytorch.org/docs/stable/size.html#torch.Size)*,*[*int*](https://docs.python.org/3/builtins/functions.html#int)*or*[*tuple*](https://docs.python.org/3/builtins/stdtypes.html#tuple)*of*[*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the shape of
the scale tensor. It must match the trailing dimensions of
`loc`, or the call raises a `RuntimeError`. Defaults to
`torch.Size(())`: one scale for all the elements.

Keyword Arguments:

- **scale_mapping** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - the function that maps the scale tensor
to positive values, by name, for example `"softplus"`,
`"exp"` or `"biased_softplus_1.0"`. See
[`mappings()`](tensordict.nn.mappings.html#tensordict.nn.mappings). Defaults to `"exp"`.
- **scale_lb** (*Number**,**optional*) - the minimum value of the scale (the
standard deviation). Defaults to `1e-4`.
- **device** ([*device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,**optional*) - the device of the scale tensor.
Defaults to `None`, the default device.
- **make_param** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, the scale tensor is a
trainable `Parameter`. If `False`, it is a
buffer. Defaults to `True`.
- **init_value** ([*float*](https://docs.python.org/3/builtins/functions.html#float)*,**optional*) - the initial value of the scale tensor,
before `scale_mapping`. With the default `"exp"` mapping,
`0.0` gives a scale of `1`. Defaults to `0.0`.

Examples

```
>>> from torch import nn
>>> import torch
>>> from tensordict.nn.distributions import AddStateIndependentNormalScale
>>> num_outputs = 4
>>> module = nn.Linear(3, num_outputs)
>>> module_normal = AddStateIndependentNormalScale(num_outputs)
>>> tensor = torch.randn(3)
>>> loc, scale = module_normal(module(tensor))
>>> print(loc.shape, scale.shape)
torch.Size([4]) torch.Size([4])
>>> assert (scale > 0).all()
>>> # with modules that return more than one tensor
>>> module = nn.LSTM(3, num_outputs)
>>> module_normal = AddStateIndependentNormalScale(num_outputs)
>>> tensor = torch.randn(4, 2, 3)
>>> loc, scale, others = module_normal(*module(tensor))
>>> print(loc.shape, scale.shape)
torch.Size([4, 2, 4]) torch.Size([4, 2, 4])
>>> assert (scale > 0).all()
```

forward(*loc: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*, **others: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*) → [tuple](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor), ...]

Returns `loc`, the scale expanded to the shape of `loc`, and `others`.

Parameters:

- **loc** ([*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) - the location. Its trailing dimensions must
match `scale_shape`.
- ***others** - other values, returned unchanged.

Returns:

The tuple `(loc, scale, *others)`.

Raises:

[**RuntimeError**](https://docs.python.org/3/builtins/exceptions.html#RuntimeError) - if the trailing dimensions of `loc` do not match
 `scale_shape`.