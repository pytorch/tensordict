# TensorDictBase.clamp_max

TensorDictBase.clamp_max_(*other: [TensorCollection](tensordict.TensorCollection.html#tensordict.TensorCollection) | [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*) → Self

Clamps every leaf to at most `other` in place and returns `self`.

The in-place version of [`clamp_max()`](tensordict.TensorDictBase.clamp_max.html#tensordict.TensorDictBase.clamp_max). It has no `default`: a
tensordict `other` must have every key of the tensordict, and its
other keys are ignored.

Parameters:

**other** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*or**Number*) - the upper bound, paired with
the leaves as in [`clamp_max()`](tensordict.TensorDictBase.clamp_max.html#tensordict.TensorDictBase.clamp_max).

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).