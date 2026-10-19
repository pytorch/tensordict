# TensorDictBase.isnan

TensorDictBase.isnan() → Self

Returns a new tensordict with [`torch.isnan()`](https://docs.pytorch.org/docs/stable/generated/torch.isnan.html#torch.isnan) applied to every leaf.

The leaves of the result are `True` where an element is NaN.

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).