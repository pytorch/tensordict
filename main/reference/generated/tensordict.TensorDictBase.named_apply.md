# TensorDictBase.named_apply

TensorDictBase.named_apply(*fn: [Callable](https://docs.python.org/3/library/typing.html#typing.Callable)*, **others: T*, *nested_keys: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *batch_size: [Sequence](https://docs.python.org/3/library/typing.html#typing.Sequence)[[int](https://docs.python.org/3/builtins/functions.html#int)] | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *device: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | [None](https://docs.python.org/3/builtins/constants.html#None) = NO_DEFAULT*, *names: [Sequence](https://docs.python.org/3/library/typing.html#typing.Sequence)[[str](https://docs.python.org/3/builtins/stdtypes.html#str)] | [None](https://docs.python.org/3/builtins/constants.html#None) = NO_DEFAULT*, *inplace: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *default: [Any](https://docs.python.org/3/library/typing.html#typing.Any) = NO_DEFAULT*, *filter_empty: [bool](https://docs.python.org/3/builtins/functions.html#bool) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *propagate_lock: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *call_on_nested: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *is_leaf: [Callable](https://docs.python.org/3/library/typing.html#typing.Callable)[[[type](https://docs.python.org/3/builtins/functions.html#type)], [bool](https://docs.python.org/3/builtins/functions.html#bool)] | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *out: [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, ***constructor_kwargs*) → Self | [None](https://docs.python.org/3/builtins/constants.html#None)

Applies a callable to every leaf and its key, and returns a new tensordict with the results.

Same as [`apply()`](tensordict.TensorDictBase.apply.html#tensordict.TensorDictBase.apply), except that `fn` also receives the key of
each leaf: the last string of its nested key, or the whole nested key
with `nested_keys=True`. If `fn` returns `None`, the entry is left
out of the result, which can be used to filter entries by key.

Parameters:

- **fn** (*Callable*) - the function called on each leaf, as
`fn(key, leaf, *other_leaves)`. It returns a tensor, a
tensordict, or `None` to leave the entry out.
- ***others** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)) - other tensordicts with the same structure
as the tensordict. An entry that one of them lacks raises
`KeyError`, unless `default` is given.

Keyword Arguments:

- **nested_keys** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, `fn` receives the whole
nested key of each leaf, a tuple of strings for an entry of a
nested tensordict. If `False`, it receives the last string of
the key. Defaults to `False`.
- **batch_size** ([*Size*](https://docs.pytorch.org/docs/stable/size.html#torch.Size)*or*[*list*](https://docs.python.org/3/builtins/stdtypes.html#list)*of*[*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the batch size of
the result, for a `fn` that changes the shape of the leaves:
the results must have it as their leading dimensions. `fn` is
then not called on [`UnbatchedTensor`](tensordict.UnbatchedTensor.html#tensordict.UnbatchedTensor) entries,
which keep their data and take the new batch size. Defaults to
`None`, which keeps the batch size of the tensordict.
- **device** ([*device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,*[*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*or*[*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the device of the
result. The results of `fn` are moved to it. `None` gives a
result with no device, which leaves each result where `fn`
put it. By default, the result has the device of the
tensordict.
- **names** ([*list*](https://docs.python.org/3/builtins/stdtypes.html#list)*of*[*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - the names of the batch dimensions of
the result, one per dimension. By default, the names of the
tensordict are kept, or dropped if `batch_size` is given.
- **inplace** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, the results are written into
the existing leaves of the tensordict (as [`set_()`](tensordict.TensorDictBase.set_.html#tensordict.TensorDictBase.set_) does),
and `self` is returned. See [In-place operations](../../concepts/operations.html#inplace-conventions).
Defaults to `False`.
- **default** (*Any**,**optional*) - the value passed to `fn` for an entry
that one of `others` lacks. By default, a missing entry raises
`KeyError`.
- **filter_empty** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, a nested tensordict
with no entry in the result (because `fn` returned `None`
for all its leaves, or because it was already empty) is left
out. If `False`, it is kept, empty. Defaults to `None`,
which leaves out the nested tensordicts that `fn` emptied and
keeps those that were already empty. Non-tensor data that
`fn` is not called on is kept in all cases.
- **propagate_lock** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, the result of a locked
tensordict is locked too. Defaults to `False`.
- **call_on_nested** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, `fn` is called on the
top-level entries, nested tensordicts and tensorclasses
included, and is responsible for going into them. If `False`,
`fn` is called on the leaves only. Defaults to `False`.
- **is_leaf** (*Callable**,**optional*) - called with the type of each entry:
`fn` is called on the entries for which it returns `True`,
and `named_apply` goes into the others. Defaults to `None`, which
uses [`default_is_leaf()`](tensordict.default_is_leaf.html#tensordict.default_is_leaf): `fn` is called on
tensors, and non-tensor data is kept unchanged. Pass
[`is_leaf_nontensor()`](tensordict.is_leaf_nontensor.html#tensordict.is_leaf_nontensor) to call `fn` on
non-tensor data too.
- **out** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*,**optional*) - a tensordict to write the results
into, instead of a new one. When `batch_size` or `device` is
given, `out` must have it. See the warning below about
`out=self`. Defaults to `None`.
- ****constructor_kwargs** - accepted and ignored.

Returns:

A new tensordict, `out` if it is given, or `self` if
`inplace=True`. `None` if `fn` returns `None` for every
leaf and `filter_empty` is not `False`. With
`filter_empty=None`, a tensordict that is already empty gives an
empty tensordict.

The result has the type of the tensordict for a
[`TensorDict`](tensordict.TensorDict.html#tensordict.TensorDict), a lazy stack and a tensorclass.
Other types, for example [`PersistentTensorDict`](tensordict.PersistentTensorDict.html#tensordict.PersistentTensorDict)
or a memory-mapped tensordict, give a [`TensorDict`](tensordict.TensorDict.html#tensordict.TensorDict)
that lives in memory. A locked tensordict gives an unlocked result
unless `propagate_lock=True`.

Warning

With `out=self`, the results are silently wrong if
`fn` reads other entries of the tensordict, which may already be
overwritten.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict(
... a=torch.zeros(3),
... nested=TensorDict(a=torch.ones(3), b=torch.ones(3), batch_size=[3]),
... batch_size=[3],
... )
>>> out = td.named_apply(lambda key, x: x if key == "a" else None)
>>> list(out.keys(include_nested=True, leaves_only=True))
['a', ('nested', 'a')]
>>> out = td.named_apply(
... lambda key, x: x + 1 if key == ("nested", "b") else x,
... nested_keys=True,
... )
>>> out["nested", "b"], out["nested", "a"]
(tensor([2., 2., 2.]), tensor([1., 1., 1.]))
```

With another tensordict, `fn` receives one leaf of each:

```
>>> td.named_apply(lambda key, x, y: x + y, td)["nested", "a"]
tensor([2., 2., 2.])
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).