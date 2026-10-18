# expln

tensordict.nn.utils.expln(*x*)

Maps real numbers to positive numbers: `exp(x)` for `x <= 0`, `log1p(x) + 1` for `x > 0`.

The function is smooth, and grows like a logarithm for large inputs. It
comes from Rückstieß et al., "State-Dependent Exploration for Policy
Gradient Methods" (ECML 2008,
[https://people.idsia.ch/~juergen/ecml2008rueckstiess.pdf](https://people.idsia.ch/~juergen/ecml2008rueckstiess.pdf)).
[`mappings()`](tensordict.nn.mappings.html#tensordict.nn.mappings) returns it for `"expln"`.

Parameters:

**x** ([*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) - the input.

Returns:

A new tensor with the shape of `x`.

Examples

```
>>> import torch
>>> from tensordict.nn.utils import expln
>>> expln(torch.tensor([-1.0, 0.0, 1.0]))
tensor([0.3679, 1.0000, 1.6931])
```