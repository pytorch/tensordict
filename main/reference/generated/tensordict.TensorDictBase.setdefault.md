# TensorDictBase.setdefault

TensorDictBase.setdefault(*key: [NestedKey](tensordict.NestedKey.html#tensordict.NestedKey)*, *default: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | [TensorCollection](tensordict.TensorCollection.html#tensordict.TensorCollection) | [Any](https://docs.python.org/3/library/typing.html#typing.Any)*, *inplace: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*) → [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | [TensorCollection](tensordict.TensorCollection.html#tensordict.TensorCollection)

Inserts the `key` entry with a value of `default` if `key` is not in the tensordict.

Returns the value for `key` if `key` is in the tensordict, else `default`.

Parameters:

- **key** ([*NestedKey*](tensordict.NestedKey.html#tensordict.NestedKey)) - the name of the value.
- **default** ([*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*or**compatible type**,*[*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)) - value
to be stored in the tensordict if the key is not already present.
- **inplace** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - passed to [`set()`](tensordict.TensorDictBase.set.html#tensordict.TensorDictBase.set) when `default`
is written. Defaults to `False`.

Returns:

The value of key in the tensordict. Will be default if the key was not
previously set.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict({}, batch_size=[3, 4])
>>> val = td.setdefault("a", torch.zeros(3, 4))
>>> assert (val == 0).all()
>>> val = td.setdefault("a", torch.ones(3, 4))
>>> assert (val == 0).all() # output is still 0
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).

## Tutorials that use `setdefault`

![](../../_images/sphx_glr_tensordict_keys_thumb.png)

[Getting, setting, and nesting keys](../../tutorials/tensordict_keys.html)

Getting, setting, and nesting keys