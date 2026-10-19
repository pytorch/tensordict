# TensorDictBase.maximum

TensorDictBase.maximum_(*other: [TensorCollection](tensordict.TensorCollection.html#tensordict.TensorCollection) | [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*) → Self

Replaces every leaf with its element-wise maximum with `other` in place and returns `self`.

The in-place version of [`maximum()`](tensordict.TensorDictBase.maximum.html#tensordict.TensorDictBase.maximum). It has no `default`: a
tensordict `other` must have every key of the tensordict, and its
other keys are ignored.

Parameters:

**other** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*,*[*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*or**Number*) - the other operand,
paired with the leaves as in [`maximum()`](tensordict.TensorDictBase.maximum.html#tensordict.TensorDictBase.maximum). A tensor is
aligned with the batch dimensions of each leaf, except when
a leaf is an [`UnbatchedTensor`](tensordict.UnbatchedTensor.html#tensordict.UnbatchedTensor) or inside a
[`torch.func`](https://docs.pytorch.org/docs/stable/func.api.html#module-torch.func) transform such as [`torch.vmap()`](https://docs.pytorch.org/docs/stable/generated/torch.vmap.html#torch.vmap): then
every leaf broadcasts with it from the right, as in torch.

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).