# TensorDictBase.mul

TensorDictBase.mul_(*other: [TensorCollection](tensordict.TensorCollection.html#tensordict.TensorCollection) | [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*) → Self

Multiplies every leaf by `other` in place and returns `self`.

The in-place version of [`mul()`](tensordict.TensorDictBase.mul.html#tensordict.TensorDictBase.mul), called by `td *= other`. It
has no `default`: a tensordict `other` must have every key of the
tensordict, and its other keys are ignored.

Parameters:

**other** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*,*[*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*or**Number*) - the multiplier,
paired with the leaves as in [`mul()`](tensordict.TensorDictBase.mul.html#tensordict.TensorDictBase.mul).

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).