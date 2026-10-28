# TensorDictBase.set

TensorDictBase.set_(*key: [NestedKey](tensordict.NestedKey.html#tensordict.NestedKey)*, *item: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | [TensorCollection](tensordict.TensorCollection.html#tensordict.TensorCollection)*, ***, *non_blocking: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*) → Self

Copies a value into an existing entry, which keeps its storage, and returns `self`.

The existing tensor keeps its `data_ptr()`, so other references to
it see the new values. `set_` works on a locked tensordict. See
[In-place operations](../../concepts/operations.html#inplace-conventions).

Parameters:

- **key** ([*NestedKey*](tensordict.NestedKey.html#tensordict.NestedKey)) - the key of the entry. The key must exist.
- **item** ([*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*,*[*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*or**any compatible value*) - the
value to copy into the entry. Its leading dimensions must match
the batch size, and it is broadcast to the shape of the entry.

Keyword Arguments:

**non_blocking** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, copies between devices
can run asynchronously with respect to the host. Defaults to
`False`.

Returns:

`self`.

Raises:

- [**KeyError**](https://docs.python.org/3/builtins/exceptions.html#KeyError) - if the key is missing. If an entry on the path of a
 nested key is missing, a tensordict, a lazy stack and a
 tensorclass create it and emit a [`FutureWarning`](https://docs.python.org/3/builtins/exceptions.html#FutureWarning):
 starting with TensorDict 0.17, the default will change to
 raising the `KeyError`. A locked tensordict, a
 `TensorDictParams` and a view of a
 tensordict, such as a sub-tensordict, raise it now.
- [**RuntimeError**](https://docs.python.org/3/builtins/exceptions.html#RuntimeError) - if the leading dimensions of the value do not match
 the batch size.
- [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) - if the value does not fit the shape of the entry.
 The error is also a `RuntimeError`, the type that
 [`update_()`](tensordict.TensorDictBase.update_.html#tensordict.TensorDictBase.update_) raises.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> x = torch.ones(3, 4)
>>> td = TensorDict(x=x, batch_size=[3, 4])
>>> td = td.set_("x", torch.zeros(3, 4))
>>> td["x"] is x, x.sum()
(True, tensor(0.))
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).

## Tutorials that use `set_`

![](../../_images/sphx_glr_tensordict_keys_thumb.png)

[Getting, setting, and nesting keys](../../tutorials/tensordict_keys.html)

Getting, setting, and nesting keys