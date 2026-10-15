# AddStateIndependentNormalScale

*class*tensordict.nn.distributions.AddStateIndependentNormalScale(*scale_shape: [Size](https://docs.pytorch.org/docs/stable/size.html#torch.Size) | [int](https://docs.python.org/3/builtins/functions.html#int) | [tuple](https://docs.python.org/3/builtins/stdtypes.html#tuple) = None*, ***, *scale_mapping: [str](https://docs.python.org/3/builtins/stdtypes.html#str) = 'exp'*, *scale_lb: [Number](https://docs.python.org/3/library/numbers.html#numbers.Number) = 0.0001*, *device: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *make_param: [bool](https://docs.python.org/3/builtins/functions.html#bool) = True*, *init_value: [float](https://docs.python.org/3/builtins/functions.html#float) = 0.0*)

Bases: [`Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)

A nn.Module that adds trainable state-independent scale parameters.

The scale parameters are mapped onto positive values using the specified `scale_mapping`.

Parameters:

**scale_shape** ([*Size*](https://docs.pytorch.org/docs/stable/size.html#torch.Size)*or**equivalent**,**optional*) - the shape of the scale parameter.
Defaults to `torch.Size(())`.

Keyword Arguments:

- **scale_mapping** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - positive mapping function to be used with the std.
Defaults to `"exp"`,
choices: `"softplus"`, `"exp"`, `"relu"`, `"biased_softplus_1"`.
- **scale_lb** (*Number**,**optional*) - The minimum value of the scale (the standard deviation).
Defaults to `1e-4`.
- **device** ([*device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,**optional*) - the device of the module.
- **make_param** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)*,**optional*) - whether the scale should be a parameter (`True`)
or a buffer (`False`).
Defaults to `True`.
- **init_value** ([*float*](tensordict.TensorDictBase.float.html#tensordict.TensorDictBase.float)*,**optional*) - Initial value of state independent scale.
Defaults to 0.0.

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

Forward of AddStateIndependentNormalScale.

Parameters:

- **loc** ([*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) - a location parameter.
- ***others** - other unused parameters.

Returns:

a tuple of two or more tensors containing the `(loc, scale, *others)` values.