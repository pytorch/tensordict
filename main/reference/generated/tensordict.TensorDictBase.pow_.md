# TensorDictBase.pow

TensorDictBase.pow_(*other: [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase) | [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*) → Self

Raises every leaf to the power `other` in place and returns `self`.

The in-place version of [`pow()`](tensordict.TensorDictBase.pow.html#tensordict.TensorDictBase.pow), called by `td **= other`. It
has no `default`: a tensordict `other` must have every key of the
tensordict, and its other keys are ignored.

Parameters:

**other** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*,*[*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*or**Number*) - the exponent,
paired with the leaves as in [`pow()`](tensordict.TensorDictBase.pow.html#tensordict.TensorDictBase.pow).

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).