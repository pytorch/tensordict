# TensorDictBase.clamp_min

TensorDictBase.clamp_min_(*other: [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase) | [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*) → Self

Clamps every leaf to at least `other` in place and returns `self`.

The in-place version of [`clamp_min()`](tensordict.TensorDictBase.clamp_min.html#tensordict.TensorDictBase.clamp_min). It has no `default`: a
tensordict `other` must have every key of the tensordict, and its
other keys are ignored.

Parameters:

**other** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*or**Number*) - the lower bound, paired with
the leaves as in [`clamp_min()`](tensordict.TensorDictBase.clamp_min.html#tensordict.TensorDictBase.clamp_min).

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).