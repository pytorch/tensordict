# TensorDictBase.exclude

TensorDictBase.exclude(**keys: [NestedKey](tensordict.NestedKey.html#tensordict.NestedKey)*, *inplace: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*) → Self

Returns a tensordict without the entries of `keys`.

The values are not copied: the result holds the same tensors as the
tensordict, so an in-place change to a tensor shows in both. A nested
key removes one entry of a nested tensordict, and the key of a nested
tensordict removes all its entries. Unlike [`select()`](tensordict.TensorDictBase.select.html#tensordict.TensorDictBase.select), a missing
key is ignored. The result is not locked. See [Keys, nested keys, and locking](../../concepts/keys.html#keys).

Parameters:

***keys** ([*NestedKey*](tensordict.NestedKey.html#tensordict.NestedKey)) - the keys to remove.

Keyword Arguments:

**inplace** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, removes the entries from the
tensordict and returns it. Defaults to `False`.

Returns:

A tensordict of the same type, or `self` if `inplace=True`.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict(
... obs=torch.zeros(3, 4),
... action=torch.zeros(3, 2),
... next=TensorDict(obs=torch.zeros(3, 4), reward=torch.zeros(3, 1)),
... batch_size=[3],
... )
>>> sub = td.exclude("action", ("next", "obs"))
>>> list(sub.keys(include_nested=True, leaves_only=True))
['obs', ('next', 'reward')]
>>> sub["obs"] is td["obs"]
True
>>> sorted(td.exclude("missing").keys())
['action', 'next', 'obs']
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).

## Tutorials that use `exclude`

![](../../_images/sphx_glr_tensordict_keys_thumb.png)

[Getting, setting, and nesting keys](../../tutorials/tensordict_keys.html)

Getting, setting, and nesting keys