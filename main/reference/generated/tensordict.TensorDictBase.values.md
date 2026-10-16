# TensorDictBase.values

TensorDictBase.values(*include_nested: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *leaves_only: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *is_leaf=None*, ***, *sort: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*) → [Iterator](https://docs.python.org/3/library/typing.html#typing.Iterator)[[Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | [TensorCollection](tensordict.TensorCollection.html#tensordict.TensorCollection)]

Returns a generator of the values of the tensordict.

By default, only the top-level values are returned, and a
sub-tensordict is returned as one value. See [Keys, nested keys, and locking](../../concepts/keys.html#keys) for nested
keys and leaves.

Parameters:

- **include_nested** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, the entries of the
sub-tensordicts are also returned, with nested keys (tuples of
strings). Defaults to `False`.
- **leaves_only** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, only leaves are returned,
and sub-tensordicts are skipped. Defaults to `False`.
- **is_leaf** (*callable**,**optional*) - a function that takes a class and
returns `True` if values of that class count as leaves. It
only filters the result when `leaves_only=True`. It does not
stop the recursion: with `include_nested=True`, a
sub-tensordict is still searched even if `is_leaf` returns
`True` for it. By default, tensordicts and tensorclasses are
not leaves. See [`is_leaf_nontensor()`](tensordict.is_leaf_nontensor.html#tensordict.is_leaf_nontensor) and
[`default_is_leaf()`](tensordict.default_is_leaf.html#tensordict.default_is_leaf).

Keyword Arguments:

**sort** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)*,**optional*) - if `True`, the keys are sorted. Nested keys
are sorted by their joined name: `("a", "key")` sorts as
`"a.key"`. Sorting takes extra time on large tensordicts.
Defaults to `False`.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict(
... a=torch.zeros(2),
... nested=TensorDict(b=torch.ones(2, 3), batch_size=[2]),
... batch_size=[2],
... )
>>> [type(value).__name__ for value in td.values()]
['Tensor', 'TensorDict']
>>> [value.shape for value in td.values(include_nested=True, leaves_only=True)]
[torch.Size([2]), torch.Size([2, 3])]
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).

## Tutorials that use `values`

![](../../_images/sphx_glr_serialization_speed_thumb.png)

[Benchmarking TensorDict serialization speed](../../tutorials/serialization_speed.html)

Benchmarking TensorDict serialization speed