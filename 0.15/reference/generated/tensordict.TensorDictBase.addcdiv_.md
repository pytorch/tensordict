# TensorDictBase.addcdiv

TensorDictBase.addcdiv_(*other1*, *other2*, ***, *value: [float](https://docs.python.org/3/builtins/functions.html#float) | [None](https://docs.python.org/3/builtins/constants.html#None) = 1*)

Adds `value * other1 / other2` to every leaf in place and returns `self`.

The in-place version of [`addcdiv()`](tensordict.TensorDictBase.addcdiv.html#tensordict.TensorDictBase.addcdiv).

Parameters:

- **other1** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*or*[*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) - the numerator, paired with
the leaves as in [`addcdiv()`](tensordict.TensorDictBase.addcdiv.html#tensordict.TensorDictBase.addcdiv).
- **other2** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*or*[*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) - the denominator, paired
like `other1`.

Keyword Arguments:

**value** (*Number**,**optional*) - the multiplier for
\(\text{other1} / \text{other2}\). Defaults to `1`.

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).