# TensorDictBase.clamp_min

TensorDictBase.clamp_min_(*other: [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase) | [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*) → Self

Clamps every leaf to at least `other` in place and returns `self`.

The in-place version of [`clamp_min()`](tensordict.TensorDictBase.clamp_min.html#tensordict.TensorDictBase.clamp_min). It has no `default`: a
tensordict `other` must have every key of the tensordict, and its
other keys are ignored.

Parameters:

**other** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*,*[*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*or**Number*) - the lower bound,
paired with the leaves as in [`clamp_min()`](tensordict.TensorDictBase.clamp_min.html#tensordict.TensorDictBase.clamp_min). A tensor is
aligned with the batch dimensions of each leaf, except when
a leaf is an [`UnbatchedTensor`](tensordict.UnbatchedTensor.html#tensordict.UnbatchedTensor) or inside a
[`torch.func`](https://docs.pytorch.org/docs/stable/func.api.html#module-torch.func) transform such as [`torch.vmap()`](https://docs.pytorch.org/docs/stable/generated/torch.vmap.html#torch.vmap): then
every leaf broadcasts with it from the right, as in torch.

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).