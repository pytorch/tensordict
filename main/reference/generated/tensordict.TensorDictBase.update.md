# TensorDictBase.update

TensorDictBase.update(*input_dict_or_td: [dict](https://docs.python.org/3/builtins/stdtypes.html#dict)[[str](https://docs.python.org/3/builtins/stdtypes.html#str), [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | [TensorCollection](tensordict.TensorCollection.html#tensordict.TensorCollection)] | T | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *clone: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *inplace: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, ***, *non_blocking: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *keys_to_update: [Sequence](https://docs.python.org/3/library/typing.html#typing.Sequence)[[NestedKey](tensordict.NestedKey.html#tensordict.NestedKey)] | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *is_leaf: [Callable](https://docs.python.org/3/library/typing.html#typing.Callable)[[[Type](https://docs.python.org/3/library/typing.html#typing.Type)], [bool](https://docs.python.org/3/builtins/functions.html#bool)] | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *update_batch_size: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *ignore_lock: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, ***kwargs*) → Self

Writes the entries of a dictionary or another tensordict into the tensordict, and returns `self`.

Each entry is written as with [`set()`](tensordict.TensorDictBase.set.html#tensordict.TensorDictBase.set): by reference, unless
`clone=True`. Nested tensordicts are merged, not replaced: the
entries of the tensordict that the input does not have are kept.
Keyword arguments are top-level entries: `td.update(a=x)` writes
`x` at `"a"`, and wins over an entry `"a"` of the input. A key
named like an argument of `update` (`clone`, `inplace`...) must
go in the input dictionary.

Warning

`update` is not atomic. If one entry fails (for
example, its shape does not match the batch size), the entries
before it are already written. Do not catch the exception and
continue with the same tensordict. Update a copy, or make sure
that the input is valid first.

Parameters:

- **input_dict_or_td** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*or*[*dict*](https://docs.python.org/3/builtins/stdtypes.html#dict)*,**optional*) - the entries to
write. A dictionary can be nested and have tuple keys. If
`None`, only the keyword arguments are written. Defaults to
`None`.
- **clone** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, clones the tensors of the
input before writing them. Defaults to `False`.
- **inplace** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, a value whose key exists is
copied into the existing tensor, which keeps its storage. A
new key is added. Defaults to `False`. See
[In-place operations](../../concepts/operations.html#inplace-conventions).

Keyword Arguments:

- **non_blocking** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, copies between devices
can run asynchronously with respect to the host. Defaults to
`False`.
- **keys_to_update** (*sequence**of*[*NestedKey*](tensordict.NestedKey.html#tensordict.NestedKey)*,**optional*) - if passed, only
these keys are written, without the cost of
`td.update(other.select(*keys_to_update))`. Defaults to
`None`, which writes every entry.
- **is_leaf** (*Callable**[**[*[*type*](https://docs.python.org/3/builtins/functions.html#type)*]**,*[*bool*](https://docs.python.org/3/builtins/functions.html#bool)*]**,**optional*) - tells which types of
values are written whole. The other values are merged key by
key. Defaults to `None`, which is
[`is_leaf_nontensor()`](tensordict.is_leaf_nontensor.html#tensordict.is_leaf_nontensor). See also
[`default_is_leaf()`](tensordict.default_is_leaf.html#tensordict.default_is_leaf).
- **update_batch_size** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, sets the batch
size of the tensordict to that of the input when they differ.
A lazy stack is then refilled with copies of the tensordicts of
the input. It needs `keys_to_update=None` and
`inplace=False`, and the keys of the tensordict must all be in
the input. Defaults to `False`.
- **ignore_lock** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - deprecated: `True` warns, and the
argument will be removed in TensorDict 0.17. On a locked
[`TensorDict`](tensordict.TensorDict.html#tensordict.TensorDict), it does not lift the lock: a
new or replaced entry still raises `RuntimeError`
([#2260](https://github.com/pytorch/tensordict/issues/2260)).
Use `inplace=True`, or unlock the tensordict first, as in
`with td.unlock_(): td.update(other)`. Defaults to `False`.
- ****kwargs** - entries to write at the top level, by key.

Returns:

`self`.

Raises:

[**RuntimeError**](https://docs.python.org/3/builtins/exceptions.html#RuntimeError) - if the tensordict is locked (see [Locking and temporary changes](../../concepts/keys.html#keys-locking))
 and the call adds or replaces an entry, or if the leading
 dimensions of a value do not match the batch size.

Note

With `update_batch_size=True`, a lazy stack of N
tensordicts updated with a lazy stack of M > N tensordicts along
the stack dimension gets copies of the M - N extra tensordicts
appended, so `update` can grow a lazy stack. Without it, the
extra tensordicts are left out.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict(a=torch.zeros(3), batch_size=[3])
>>> other = TensorDict(a=torch.ones(3), b=torch.ones(3, 4), batch_size=[3])
>>> td = td.update(other)
>>> td["a"] is other["a"] # written by reference unless clone=True
True
>>> td = td.update(c=torch.zeros(3)) # keyword form
>>> sorted(td.keys())
['a', 'b', 'c']
```

Nested tensordicts are merged:

```
>>> td = td.update({"nested": {"x": torch.zeros(3)}})
>>> td = td.update({"nested": {"y": torch.ones(3)}})
>>> sorted(td["nested"].keys())
['x', 'y']
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).

## Tutorials that use `update`

![](../../_images/sphx_glr_tensordict_keys_thumb.png)

[Getting, setting, and nesting keys](../../tutorials/tensordict_keys.html)

Getting, setting, and nesting keys