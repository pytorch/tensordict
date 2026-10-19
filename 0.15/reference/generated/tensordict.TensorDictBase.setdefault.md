# TensorDictBase.setdefault

TensorDictBase.setdefault(*key: [NestedKey](tensordict.NestedKey.html#tensordict.NestedKey)*, *default: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | [TensorCollection](tensordict.TensorCollection.html#tensordict.TensorCollection) | [Any](https://docs.python.org/3/library/typing.html#typing.Any)*, *inplace: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*) → [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | [TensorCollection](tensordict.TensorCollection.html#tensordict.TensorCollection)

Returns the entry at `key`, after writing `default` there if the key is missing.

As for [`dict.setdefault()`](https://docs.python.org/3/builtins/stdtypes.html#dict.setdefault), an existing entry is not changed. The
result is read with [`get()`](tensordict.TensorDictBase.get.html#tensordict.TensorDictBase.get), so a non-tensor entry returns its
[`NonTensorData`](tensordict.NonTensorData.html#tensordict.NonTensorData).

Parameters:

- **key** ([*NestedKey*](tensordict.NestedKey.html#tensordict.NestedKey)) - the key of the entry.
- **default** ([*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*,*[*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*or**any object*) - the value
written if the key is missing.
- **inplace** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - passed to [`set()`](tensordict.TensorDictBase.set.html#tensordict.TensorDictBase.set) when `default`
is written. Defaults to `False`.

Raises:

[**RuntimeError**](https://docs.python.org/3/builtins/exceptions.html#RuntimeError) - if the key is missing and the tensordict is locked,
 or if the leading dimensions of `default` do not match the
 batch size.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict({}, batch_size=[3])
>>> td.setdefault("a", torch.zeros(3))
tensor([0., 0., 0.])
>>> td.setdefault("a", torch.ones(3)) # "a" exists: not changed
tensor([0., 0., 0.])
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).

## Tutorials that use `setdefault`

![](../../_images/sphx_glr_tensordict_keys_thumb.png)

[Getting, setting, and nesting keys](../../tutorials/tensordict_keys.html)

Getting, setting, and nesting keys