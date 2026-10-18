# biased_softplus

*class*tensordict.nn.biased_softplus(*bias: [float](https://docs.python.org/3/builtins/functions.html#float)*, *min_val: [float](https://docs.python.org/3/builtins/functions.html#float) = 0.01*)

Bases: [`Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)

A softplus shifted so that it returns `bias` for a zero input and stays above `min_val`.

The module computes `softplus(x + inv_softplus(bias - min_val)) + min_val`.
[`mappings()`](tensordict.nn.mappings.html#tensordict.nn.mappings) returns it for the keys `"biased_softplus_{bias}"`
and `"biased_softplus_{bias}_{min_val}"`.

Parameters:

- **bias** ([*float*](https://docs.python.org/3/builtins/functions.html#float)) - the output for a zero input.
- **min_val** ([*float*](https://docs.python.org/3/builtins/functions.html#float)*,**optional*) - the lower bound of the output. Defaults to `0.01`.

Examples

```
>>> import torch
>>> from tensordict.nn import biased_softplus
>>> f = biased_softplus(1.0)
>>> f(torch.zeros(2))
tensor([1., 1.])
```

forward(*x: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*) → [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

Define the computation performed at every call.

Should be overridden by all subclasses.

Note

Although the recipe for forward pass needs to be defined within
this function, one should call the `Module` instance afterwards
instead of this since the former takes care of running the
registered hooks while the latter silently ignores them.