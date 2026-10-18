# TensorDictBase.select

TensorDictBase.select(**keys: [NestedKey](tensordict.NestedKey.html#tensordict.NestedKey)*, *inplace: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *strict: [bool](https://docs.python.org/3/builtins/functions.html#bool) = True*, *as_tensordict: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*) → Self

Returns a tensordict that holds only the entries of `keys`.

The values are not copied: the result holds the same tensors as the
tensordict, so an in-place change to a tensor shows in both. A nested
key such as `("next", "obs")` keeps one entry of a nested tensordict,
and the key of a nested tensordict keeps all its entries. The result is
not locked. See [Keys, nested keys, and locking](../../concepts/keys.html#keys).

Parameters:

***keys** ([*NestedKey*](tensordict.NestedKey.html#tensordict.NestedKey)) - the keys to keep.

Keyword Arguments:

- **inplace** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, removes the other entries from
the tensordict and returns it. Defaults to `False`.
- **strict** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, a missing key raises a
`KeyError`. If `False`, missing keys are ignored. Defaults to
`True`.
- **as_tensordict** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, a tensorclass returns
a [`TensorDict`](tensordict.TensorDict.html#tensordict.TensorDict) instead of an instance of its
class, whose other fields are `None`. Has no effect on other
tensordicts. Defaults to `False`.

Returns:

A tensordict of the same type, or `self` if `inplace=True`.

See also

[`exclude()`](tensordict.TensorDictBase.exclude.html#tensordict.TensorDictBase.exclude) keeps the other entries, and
[`split_keys()`](tensordict.TensorDictBase.split_keys.html#tensordict.TensorDictBase.split_keys) returns both parts.

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
>>> sub = td.select("obs", ("next", "reward"))
>>> list(sub.keys(include_nested=True, leaves_only=True))
['obs', ('next', 'reward')]
>>> td.select("obs")["obs"] is td["obs"]
True
>>> list(td.select("obs", "missing", strict=False).keys())
['obs']
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).

## Tutorials that use `select`

![](../../_images/sphx_glr_streamed_tensordict_thumb.png)

[Building tensordicts from streams](../../tutorials/streamed_tensordict.html)

Building tensordicts from streams
![](../../_images/sphx_glr_tensordict_keys_thumb.png)

[Getting, setting, and nesting keys](../../tutorials/tensordict_keys.html)

Getting, setting, and nesting keys
![](../../_images/sphx_glr_tensordict_module_thumb.png)

[Building models with TensorDictModule](../../tutorials/tensordict_module.html)

Building models with TensorDictModule