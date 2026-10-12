# TensorDictBase.update

TensorDictBase.update(*input_dict_or_td: [dict](https://docs.python.org/3/builtins/stdtypes.html#dict)[[str](https://docs.python.org/3/builtins/stdtypes.html#str), [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | TensorCollection] | T | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *clone: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *inplace: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, ***, *non_blocking: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *keys_to_update: [Sequence](https://docs.python.org/3/library/typing.html#typing.Sequence)[[NestedKey](tensordict.NestedKey.html#tensordict.NestedKey)] | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *is_leaf: [Callable](https://docs.python.org/3/library/typing.html#typing.Callable)[[[Type](https://docs.python.org/3/library/typing.html#typing.Type)], [bool](https://docs.python.org/3/builtins/functions.html#bool)] | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *update_batch_size: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *ignore_lock: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, ***kwargs*) → Self

Writes the entries of a dictionary or another tensordict into this tensordict, and returns it.

Each entry is written as with [`set()`](tensordict.TensorDictBase.set.html#tensordict.TensorDictBase.set): by reference, unless
`clone=True`. Sub-tensordicts are merged, not replaced: the entries
of `self` that the input does not have are kept.

Warning

`update` is not atomic. If one entry fails (for
example, its shape does not match the batch size), the entries
before it are already written. Do not catch the exception and
continue with the same tensordict. Update a copy, or make sure
that the input is valid first.

Parameters:

- **input_dict_or_td** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*or*[*dict*](https://docs.python.org/3/builtins/stdtypes.html#dict)*,**optional*) - input data to be written
in self. If `None`, only the keyword arguments are used.
- **clone** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - whether the tensors in the input (
tensor) dict should be cloned before being set.
Defaults to `False`.
- **inplace** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True` and if a key matches an existing
key in the tensordict, the value is copied into the existing
tensor, which keeps its storage. If the entry cannot be found,
it will be added. Defaults to `False`. See
[In-place operations](../../concepts/operations.html#inplace-conventions).

Keyword Arguments:

- **keys_to_update** (*sequence**of**NestedKeys**,**optional*) - if provided, only
the list of keys in `key_to_update` will be updated.
This is aimed at avoiding calls to
`data_dest.update(data_src.select(*keys_to_update))`.
- **non_blocking** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)*,**optional*) - if `True` and this copy is between
different devices, the copy may occur asynchronously with respect
to the host.
- **is_leaf** (*Callable**[**[**Type**]**,*[*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)*]**,**optional*) - 

a callable that indicates
whether an object type is to be considered a leaf and swapped
or a tensor collection.

See also

[`is_leaf_nontensor()`](tensordict.is_leaf_nontensor.html#tensordict.is_leaf_nontensor) and [`default_is_leaf()`](tensordict.default_is_leaf.html#tensordict.default_is_leaf).
- **update_batch_size** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)*,**optional*) - 

if `True`, `update` will attempt to update the batch-size
of the destination (self) if it mismatches the source's batch size. Defaults to `False`.

Note

In cases where the batch size does not match, [`LazyStackedTensorDict`](tensordict.LazyStackedTensorDict.html#tensordict.LazyStackedTensorDict)
instances will be emptied of their content and copies of the tensordicts from the source
will be used to repopulate the container.

Note

This argument assumes that keys_to_update is left empty, and that inplace=False.
If the keys of the destination (self) is not a subset of the keys of the source,
an exception will be raised, as TensorDict will be unable to infer what to do with the extra
destination entries.
- **ignore_lock** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)*,**optional*) - meant to update a tensordict
regardless of its locked status. Defaults to `False`.
On a locked [`TensorDict`](tensordict.TensorDict.html#tensordict.TensorDict), `ignore_lock=True`
does not lift the lock in the current version: a new or
replaced entry still raises `RuntimeError`
([#2260](https://github.com/pytorch/tensordict/issues/2260)). Use
`inplace=True`, or unlock the tensordict with
[`unlock_()`](tensordict.TensorDictBase.unlock_.html#tensordict.TensorDictBase.unlock_) first.
- ****kwargs** - additional key-value pairs to write into `self` as top-level entries. When both
`input_dict_or_td` and `kwargs` are provided, kwargs win on key conflict.

Note

When updating a [`LazyStackedTensorDict`](tensordict.LazyStackedTensorDict.html#tensordict.LazyStackedTensorDict) with N elements with another
[`LazyStackedTensorDict`](tensordict.LazyStackedTensorDict.html#tensordict.LazyStackedTensorDict) with M elements, with M > N, along the stack dimension,
the `update` method will append copies of the extra tensordicts to the dest (self) lazy stack.
This allows users to rely on `update` to increment lazy stacks progressively.

Note

Keyword arguments are treated as top-level keys only. Nested structures still work when
the kwarg value is itself a dict or tensor collection (e.g. `td.update(outer={"inner": val})`).
For deeper tuple keys, use the positional dict form: `td.update({("a", "b"): val})`.
Keys whose names collide with reserved parameters (`clone`, `inplace`, `non_blocking`,
`keys_to_update`, `is_leaf`, `update_batch_size`, `ignore_lock`) must be passed through
the positional dict form.

Returns:

self

Raises:

[**RuntimeError**](https://docs.python.org/3/builtins/exceptions.html#RuntimeError) - if the tensordict is locked (see [`lock_()`](tensordict.TensorDictBase.lock_.html#tensordict.TensorDictBase.lock_)) and
 `inplace` is `False`, or if the leading dimensions of a
 value do not match the batch size.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict({}, batch_size=[3])
>>> a = torch.randn(3)
>>> b = torch.randn(3, 4)
>>> other_td = TensorDict({"a": a, "b": b}, batch_size=[])
>>> td = td.update(other_td, inplace=True) # writes "a" and "b" even though they can't be found
>>> assert td['a'] is other_td['a']
>>> other_td = other_td.clone().zero_()
>>> td = td.update(other_td)
>>> assert td['a'] is other_td['a'] # entries are written by reference unless clone=True
>>> # keyword form for top-level entries
>>> td = td.update(monkey=torch.zeros(3))
>>> assert (td["monkey"] == 0).all()
>>> # sub-tensordicts are merged
>>> td = td.update({"nested": {"x": torch.zeros(3)}})
>>> td = td.update({"nested": {"y": torch.ones(3)}})
>>> sorted(td["nested"].keys())
['x', 'y']
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).

## Tutorials that use `update`

![](../../_images/sphx_glr_tensordict_keys_thumb.png)

[Getting, setting and nesting keys](../../tutorials/tensordict_keys.html)

Getting, setting and nesting keys