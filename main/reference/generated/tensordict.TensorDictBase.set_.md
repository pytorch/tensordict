# TensorDictBase.set

TensorDictBase.set_(*key: [NestedKey](tensordict.NestedKey.html#tensordict.NestedKey)*, *item: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | [TensorCollection](tensordict.TensorCollection.html#tensordict.TensorCollection)*, ***, *non_blocking: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*) → Self

Copies a value into an existing entry, which keeps its storage.

The existing tensor keeps its `data_ptr()`, so other references to
it see the new values. `set_` works on a locked tensordict. See
[In-place operations](../../concepts/operations.html#inplace-conventions).

Parameters:

- **key** ([*NestedKey*](tensordict.NestedKey.html#tensordict.NestedKey)) - the key of the entry. The key must exist.
- **item** ([*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*or**compatible type**,*[*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)) - value to
be copied into the entry. Its shape must match the entry.

Keyword Arguments:

**non_blocking** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)*,**optional*) - if `True` and this copy is between
different devices, the copy may occur asynchronously with respect
to the host.

Returns:

self

Raises:

- [**KeyError**](https://docs.python.org/3/builtins/exceptions.html#KeyError) - if the key is missing.
- [**RuntimeError**](https://docs.python.org/3/builtins/exceptions.html#RuntimeError) - if the shape of the value does not match the batch size.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict({}, batch_size=[3, 4])
>>> x = torch.randn(3, 4)
>>> td = td.set("x", x)
>>> td = td.set_("x", torch.zeros_like(x))
>>> assert (x == 0).all()
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).

## Tutorials that use `set_`

![](../../_images/sphx_glr_tensordict_keys_thumb.png)

[Getting, setting, and nesting keys](../../tutorials/tensordict_keys.html)

Getting, setting, and nesting keys