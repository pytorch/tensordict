# TensorDictBase.softmax

TensorDictBase.softmax(*dim: [int](https://docs.python.org/3/builtins/functions.html#int)*, *dtype: [dtype](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*)

Returns a new tensordict with [`torch.softmax()`](https://docs.pytorch.org/docs/stable/generated/torch.softmax.html#torch.softmax) applied to every leaf along a batch dimension.

Parameters:

- **dim** ([*int*](https://docs.python.org/3/builtins/functions.html#int)) - the batch dimension along which softmax is computed. A
negative `dim` counts from the end of the batch dimensions.
- **dtype** ([*dtype*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype)*,**optional*) - if given, each leaf is cast to this
dtype before the operation, which helps to prevent overflows.
Defaults to `None`.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict(x=torch.zeros(2, 3), batch_size=[2])
>>> td.softmax(0)["x"]
tensor([[0.5000, 0.5000, 0.5000],
 [0.5000, 0.5000, 0.5000]])
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).