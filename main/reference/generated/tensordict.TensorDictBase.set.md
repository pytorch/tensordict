# TensorDictBase.set

TensorDictBase.set(*key: [NestedKey](tensordict.NestedKey.html#tensordict.NestedKey)*, *item: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | [TensorCollection](tensordict.TensorCollection.html#tensordict.TensorCollection)*, *inplace: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, ***, *non_blocking: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, ***kwargs: [Any](https://docs.python.org/3/library/typing.html#typing.Any)*) → Self

Sets an entry, and returns the tensordict.

`td.set(key, value)` is the same as `td[key] = value`. By default,
the value replaces the existing entry (no data is copied). A tuple key
sets a nested entry, and creates the missing sub-tensordicts. The
leading dimensions of the value must match [`batch_size`](tensordict.TensorDictBase.batch_size.html#tensordict.TensorDictBase.batch_size).
Arrays and numbers are converted to tensors. Strings and other
objects are stored as [`NonTensorData`](tensordict.NonTensorData.html#tensordict.NonTensorData).

A `bool`, `int` or `float`, or a NumPy scalar of one of these
kinds, written over an existing [`NonTensorData`](tensordict.NonTensorData.html#tensordict.NonTensorData)
entry is stored as a 0-d tensor, and the write emits a
[`FutureWarning`](https://docs.python.org/3/builtins/exceptions.html#FutureWarning). Starting with TensorDict 0.17, the default will
change to storing it as [`NonTensorData`](tensordict.NonTensorData.html#tensordict.NonTensorData), like the
entry it replaces.

Parameters:

- **key** ([*NestedKey*](tensordict.NestedKey.html#tensordict.NestedKey)) - the key to set: a string, or a tuple of strings
for a nested entry. See [Keys, nested keys, and locking](../../concepts/keys.html#keys).
- **item** ([*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*,*[*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*or**any object*) - the value to
store.
- **inplace** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True` and the key exists, the value
is copied into the existing tensor, which keeps its storage. If
the key does not exist, the entry is added. For a write that
raises on a missing key, use [`set_()`](tensordict.TensorDictBase.set_.html#tensordict.TensorDictBase.set_). Defaults to
`False`. See [In-place operations](../../concepts/operations.html#inplace-conventions).

Keyword Arguments:

- **non_blocking** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, copies between devices
can run asynchronously with respect to the host. Defaults to
`False`.
- ****kwargs** - ignored, with a [`FutureWarning`](https://docs.python.org/3/builtins/exceptions.html#FutureWarning). Starting with
TensorDict 0.17, the default will change to raising a
`TypeError` for them, as [`set_()`](tensordict.TensorDictBase.set_.html#tensordict.TensorDictBase.set_) does.

Returns:

`self`.

Raises:

- [**RuntimeError**](https://docs.python.org/3/builtins/exceptions.html#RuntimeError) - if the tensordict is locked (see [Locking and temporary changes](../../concepts/keys.html#keys-locking))
 and the call adds or replaces an entry, or if the leading
 dimensions of the value do not match the batch size.
- [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) - if `inplace=True` and the value does not fit the
 shape of the existing entry. The error is also a
 `RuntimeError`, the type that [`update_()`](tensordict.TensorDictBase.update_.html#tensordict.TensorDictBase.update_) raises.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict({}, batch_size=[3, 4])
>>> td = td.set("x", torch.randn(3, 4))
>>> y = torch.randn(3, 4, 5)
>>> td = td.set("y", y, inplace=True) # works, even if 'y' is not present yet
>>> td = td.set("y", torch.zeros_like(y), inplace=True)
>>> assert (y==0).all() # y values are overwritten
>>> td = td.set(("nested", "z"), torch.zeros(3, 4, 2)) # creates td["nested"]
>>> td["nested"].batch_size
torch.Size([3, 4])
>>> td.set("y", torch.ones(5), inplace=True) # the shape does not match
Traceback (most recent call last):
...
RuntimeError: batch dimension mismatch, got self.batch_size=torch.Size([3, 4]) and value.shape=torch.Size([5]) ...
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).

## Tutorials that use `set`

![](../../_images/sphx_glr_tensordict_keys_thumb.png)

[Getting, setting, and nesting keys](../../tutorials/tensordict_keys.html)

Getting, setting, and nesting keys
![](../../_images/sphx_glr_tensordict_shapes_thumb.png)

[Changing the shape of a TensorDict](../../tutorials/tensordict_shapes.html)

Changing the shape of a TensorDict