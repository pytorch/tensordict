# TensorDictBase.contiguous

*abstract*TensorDictBase.contiguous(***, *canonical: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *inplace: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*) → Self

Returns a new tensordict of the same type with contiguous values (or self if values are already contiguous).

Parameters:

- **canonical** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, every dense tensor leaf
whose strides do not match the canonical C-row-major strides
for its shape is rematerialized into a freshly allocated
contiguous tensor (using `torch.contiguous_format`), even if
[`torch.Tensor.is_contiguous()`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.is_contiguous.html#torch.Tensor.is_contiguous) returns `True` (e.g.
because of size-1 dimensions). When `False` (the default),
the historical behavior of [`torch.Tensor.contiguous()`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.contiguous.html#torch.Tensor.contiguous) is
preserved. Defaults to `False`.
- **inplace** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - If `True`, this tensordict's identity
and key set are preserved; each non-contiguous leaf is
replaced by its contiguous counterpart one at a time.
`Tensor.contiguous()` returns `self` when a leaf is
already contiguous, so for fully-contiguous tensordicts the
operation is effectively a no-op aside from the key walk.
Defaults to `False`.

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).