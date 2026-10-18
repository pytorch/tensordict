# TensorDictBase.contiguous

TensorDictBase.contiguous(***, *canonical: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *inplace: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*) → Self

Returns a new tensordict with contiguous leaves.

Each leaf is replaced by [`torch.Tensor.contiguous()`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.contiguous.html#torch.Tensor.contiguous): a leaf that
is already contiguous is shared with the tensordict, the others are
copied.

Keyword Arguments:

- **canonical** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, a leaf whose strides are
not the row-major strides of its shape is copied too, even if
[`torch.Tensor.is_contiguous()`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.is_contiguous.html#torch.Tensor.is_contiguous) returns `True` for it (as
it can with dimensions of size 1). Defaults to `False`.
- **inplace** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, the tensordict keeps its
identity and keys, each leaf that is not contiguous is replaced
by a contiguous copy, and `self` is returned. See
[In-place operations](../../concepts/operations.html#inplace-conventions). Defaults to `False`.

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
>>> td = TensorDict(a=torch.zeros(4, 3).T, batch_size=[3])
>>> td.is_contiguous(), td.contiguous().is_contiguous()
(False, True)
```

On a lazy stack, the result loses the name of the stack dimension:

```
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