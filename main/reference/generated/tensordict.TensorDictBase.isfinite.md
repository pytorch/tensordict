# TensorDictBase.isfinite

TensorDictBase.isfinite() → Self

Returns a new tensordict with [`torch.isfinite()`](https://docs.pytorch.org/docs/stable/generated/torch.isfinite.html#torch.isfinite) applied to every leaf.

The leaves of the result are `True` where an element is neither NaN
nor infinite.

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).