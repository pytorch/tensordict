# TensorDictBase.to_padded_tensor

TensorDictBase.to_padded_tensor(*padding=0.0*, *mask_key: [NestedKey](tensordict.NestedKey.html#tensordict.NestedKey) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*) → Self

Converts all nested tensors to a padded version and adapts the batch-size accordingly.

Parameters:

- **padding** ([*float*](https://docs.python.org/3/builtins/functions.html#float)) - the padding value for the tensors in the tensordict.
Defaults to `0.0`.
- **mask_key** ([*NestedKey*](tensordict.NestedKey.html#tensordict.NestedKey)*,**optional*) - if provided, the key where a
mask for valid values will be written.
Will result in an error if the heterogeneous dimension
isn't part of the tensordict batch-size.
Defaults to `None`

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).