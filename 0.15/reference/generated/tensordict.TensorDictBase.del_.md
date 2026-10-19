# TensorDictBase.del

TensorDictBase.del_(*key: [NestedKey](tensordict.NestedKey.html#tensordict.NestedKey)*) → Self

Deletes an entry of the tensordict in place and returns the tensordict.

A locked tensordict raises `RuntimeError` (see [`lock_()`](tensordict.TensorDictBase.lock_.html#tensordict.TensorDictBase.lock_)).
`del td[key]` does the same.

Parameters:

**key** ([*NestedKey*](tensordict.NestedKey.html#tensordict.NestedKey)) - the key to delete. A missing key raises
`KeyError`, and a key whose path runs through an entry that
is not a tensordict raises `ValueError`. Until TensorDict
0.17, the stores and the zarr backend of
[`PersistentTensorDict`](tensordict.PersistentTensorDict.html#tensordict.PersistentTensorDict) warn and do nothing
for a missing key.

Returns:

`self`.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict(a=torch.zeros(3), b=TensorDict(c=torch.ones(3)))
>>> list(td.del_(("b", "c")).keys(True))
['a', 'b']
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).

## Tutorials that use `del_`

![](../../_images/sphx_glr_tensordict_keys_thumb.png)

[Getting, setting, and nesting keys](../../tutorials/tensordict_keys.html)

Getting, setting, and nesting keys