# TensorDictBase.set_at

TensorDictBase.set_at_(*key: [NestedKey](tensordict.NestedKey.html#tensordict.NestedKey)*, *value: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | [TensorCollection](tensordict.TensorCollection.html#tensordict.TensorCollection)*, *index: [None](https://docs.python.org/3/builtins/constants.html#None) | [int](https://docs.python.org/3/builtins/functions.html#int) | [slice](https://docs.python.org/3/builtins/functions.html#slice) | [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | [List](https://docs.python.org/3/library/typing.html#typing.List)[[Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[Any](https://docs.python.org/3/library/typing.html#typing.Any), ...]*, ***, *non_blocking: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*) → Self

Writes `value` into the existing entry `key` at the index `index`, in place.

`td.set_at_(key, value, index)` is the same as `td[key][index] = value`.

A list, or a `bool`, `int` or `float` (also as a NumPy scalar),
written into a non-tensor entry is cast to a tensor, and the write
emits a [`FutureWarning`](https://docs.python.org/3/builtins/exceptions.html#FutureWarning). Starting with TensorDict 0.17, the
default will change to storing it as non-tensor data. Wrap the value
in [`NonTensorData`](tensordict.NonTensorData.html#tensordict.NonTensorData) to store it as non-tensor data
now, or pass a tensor to keep the tensor. `td[index] = value` writes
through `set_at_` too, but announces another change: see
`__setitem__()`.

The entry keeps its storage, so `set_at_` works on a locked
tensordict.

Parameters:

- **key** ([*NestedKey*](tensordict.NestedKey.html#tensordict.NestedKey)) - the key of the entry. The key must exist.
- **value** ([*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*or**any compatible value*) - the value to write.
It must fit `td[key][index]`.
- **index** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,*[*slice*](https://docs.python.org/3/builtins/functions.html#slice)*,*[*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*or*[*tuple*](https://docs.python.org/3/builtins/stdtypes.html#tuple)) - the index to write at,
along the batch dimensions.

Keyword Arguments:

**non_blocking** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, copies between devices
can run asynchronously with respect to the host. Defaults to
`False`.

Returns:

`self`.

Raises:

- [**KeyError**](https://docs.python.org/3/builtins/exceptions.html#KeyError) - if the key is missing.
- [**RuntimeError**](https://docs.python.org/3/builtins/exceptions.html#RuntimeError) - if the value does not fit `td[key][index]`.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> x = torch.zeros(3, 4)
>>> td = TensorDict(x=x, batch_size=[3, 4])
>>> td = td.set_at_("x", value=torch.ones(1, 4), index=slice(1))
>>> x[:, 0]
tensor([1., 0., 0.])
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).