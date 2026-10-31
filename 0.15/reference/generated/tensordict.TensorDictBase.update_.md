# TensorDictBase.update

TensorDictBase.update_(*input_dict_or_td: [dict](https://docs.python.org/3/builtins/stdtypes.html#dict)[[str](https://docs.python.org/3/builtins/stdtypes.html#str), [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | [TensorCollection](tensordict.TensorCollection.html#tensordict.TensorCollection)] | T | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *clone: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, ***, *non_blocking: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *keys_to_update: [Sequence](https://docs.python.org/3/library/typing.html#typing.Sequence)[[NestedKey](tensordict.NestedKey.html#tensordict.NestedKey)] | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, ***kwargs*) → Self

Copies the entries of a dictionary or tensordict into the existing entries, and returns `self`.

Each value is copied into the existing tensor, which keeps its
storage, as with [`set_()`](tensordict.TensorDictBase.set_.html#tensordict.TensorDictBase.set_). `update_` therefore works on a
locked tensordict. Every key of the input must exist in the tensordict.
Keyword arguments are top-level entries, and win over an entry of the
input with the same key. A key named like an argument of `update_`
must go in the input dictionary. See [In-place operations](../../concepts/operations.html#inplace-conventions).

On a shared or memory-mapped tensordict, a non-tensor value that its
shared buffer cannot store (for example `None`, or an int where a
string is shared) is skipped, with a [`UserWarning`](https://docs.python.org/3/builtins/exceptions.html#UserWarning) that names
its key.

Parameters:

- **input_dict_or_td** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*or*[*dict*](https://docs.python.org/3/builtins/stdtypes.html#dict)*,**optional*) - the entries to
copy. If `None`, only the keyword arguments are copied.
Defaults to `None`.
- **clone** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, clones the tensors of the
input before copying them. Defaults to `False`.

Keyword Arguments:

- **non_blocking** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, copies between devices
can run asynchronously with respect to the host. Defaults to
`False`.
- **keys_to_update** (*sequence**of*[*NestedKey*](tensordict.NestedKey.html#tensordict.NestedKey)*,**optional*) - if passed, only
these keys are copied, without the cost of
`td.update_(other.select(*keys_to_update))`. Defaults to
`None`, which copies every entry. If it lists a
sub-tensordict key and a key under it, the whole sub-tensordict
is copied and the call warns, while `select` keeps only the
deeper keys; starting with TensorDict 0.17, the default will
change to copying only the deeper keys, as `select` does.
An entry that is not a key, such as `None`, an int or bytes,
selects every entry, and the call warns (a tensordict that
holds a lazy stack, a sub-tensordict view or a file raises a
`RuntimeError` instead, as before); starting with
TensorDict 0.18, the default will change to raising a
`RuntimeError`, as [`update()`](tensordict.TensorDictBase.update.html#tensordict.TensorDictBase.update) does.
- ****kwargs** - entries to copy at the top level, by key.

Returns:

`self`.

Raises:

- [**KeyError**](https://docs.python.org/3/builtins/exceptions.html#KeyError) - if a key of the input is missing from the tensordict.
- [**RuntimeError**](https://docs.python.org/3/builtins/exceptions.html#RuntimeError) - if a value does not fit the shape of its entry.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> a = torch.ones(3)
>>> td = TensorDict(a=a, b=torch.ones(3, 4), batch_size=[3])
>>> _ = td.update_(TensorDict(a=torch.zeros(3), b=torch.zeros(3, 4), batch_size=[3]))
>>> td["a"] is a, a.sum()
(True, tensor(0.))
>>> _ = td.update_(a=torch.ones(3)) # keyword form
>>> a.sum()
tensor(3.)
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).