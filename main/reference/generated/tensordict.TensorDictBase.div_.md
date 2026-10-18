# TensorDictBase.div

TensorDictBase.div_(*other: [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase) | [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*) → Self

Divides every leaf by `other` in place and returns `self`.

The in-place version of [`div()`](tensordict.TensorDictBase.div.html#tensordict.TensorDictBase.div), called by `td /= other`. It
has no `default`: a tensordict `other` must have every key of the
tensordict, and its other keys are ignored.

Parameters:

**other** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*,*[*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*or**Number*) - the divisor,
paired with the leaves as in [`div()`](tensordict.TensorDictBase.div.html#tensordict.TensorDictBase.div).

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).