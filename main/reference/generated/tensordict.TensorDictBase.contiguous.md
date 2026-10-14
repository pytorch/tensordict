# TensorDictBase.contiguous

TensorDictBase.contiguous(***, *canonical: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *inplace: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*) → Self

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

Note

On a [`LazyStackedTensorDict`](tensordict.LazyStackedTensorDict.html#tensordict.LazyStackedTensorDict), a
[`PersistentTensorDict`](tensordict.PersistentTensorDict.html#tensordict.PersistentTensorDict) (h5, zarr) or a store,
the result is a [`TensorDict`](tensordict.TensorDict.html#tensordict.TensorDict). If the tensordicts
of a lazy stack have no names, the result has no names either,
even when the stack dimension has a name (from `stack_dim_name`
or the `names` setter). [`densify()`](tensordict.TensorDictBase.densify.html#tensordict.TensorDictBase.densify) and
[`clone()`](tensordict.TensorDictBase.clone.html#tensordict.TensorDictBase.clone) keep that name. To keep it here, name the dimension
again with [`refine_names()`](tensordict.TensorDictBase.refine_names.html#tensordict.TensorDictBase.refine_names).

Examples

```
>>> import torch
>>> from tensordict import TensorDict, lazy_stack
>>> stack = lazy_stack(
... [TensorDict(a=torch.zeros(3), batch_size=[3]) for _ in range(2)],
... stack_dim_name="s",
... )
>>> stack.names, stack.contiguous().names
(['s', None], [None, None])
>>> stack.contiguous().refine_names("s", ...).names
['s', None]
>>> stack.densify().names
['s', None]
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).