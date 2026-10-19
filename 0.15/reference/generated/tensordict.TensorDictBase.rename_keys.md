# TensorDictBase.rename_keys

TensorDictBase.rename_keys(*mapping: [Mapping](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[NestedKey](tensordict.NestedKey.html#tensordict.NestedKey), [NestedKey](tensordict.NestedKey.html#tensordict.NestedKey)]*, ***, *strict: [bool](https://docs.python.org/3/builtins/functions.html#bool) = True*) → Self

Returns a tensordict with several keys renamed at once.

The tensordict is not modified. The result has a new tree of nested
tensordicts and shares every leaf with the tensordict, tensors and
non-tensor data alike. Its batch size, device and dimension names are
those of the tensordict, and it is not locked.

Every source is read before any destination is written, so swaps
(`{"a": "b", "b": "a"}`) and chains (`{"a": "b", "b": "c"}`) work.
A nested tensordict that the renaming leaves empty is removed. Other
empty nested tensordicts are kept.

A lazy stack renames the keys in each of its tensordicts, and removes a
nested tensordict only if the renaming empties it in all of them. A
tensorclass returns the same class, its destinations must be fields,
and its fields are never removed.
[`PersistentTensorDict`](tensordict.PersistentTensorDict.html#tensordict.PersistentTensorDict) and the stores raise
`NotImplementedError`: call [`to_tensordict()`](tensordict.TensorDictBase.to_tensordict.html#tensordict.TensorDictBase.to_tensordict) first.

To rename a single key in place, see [`rename_key_()`](tensordict.TensorDictBase.rename_key_.html#tensordict.TensorDictBase.rename_key_).

Parameters:

**mapping** (*Mapping**[*[*NestedKey*](tensordict.NestedKey.html#tensordict.NestedKey)*,*[*NestedKey*](tensordict.NestedKey.html#tensordict.NestedKey)*]*) - the old key of each entry
mapped to its new key. Both sides can be nested keys, so an
entry can move to another nested tensordict and a whole nested
tensordict can be renamed.

Keyword Arguments:

**strict** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, a source missing from the
tensordict raises `KeyError`. If `False`, it is skipped.
Defaults to `True`.

Returns:

A new tensordict of the same type as `self`, except that a
sub-tensordict view and a `TensorDictParams`
return a [`TensorDict`](tensordict.TensorDict.html#tensordict.TensorDict), as [`exclude()`](tensordict.TensorDictBase.exclude.html#tensordict.TensorDictBase.exclude) does.

Raises:

- [**KeyError**](https://docs.python.org/3/builtins/exceptions.html#KeyError) - if `strict=True` and a source is missing, or if a key is
 empty.
- [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) - if two sources have the same destination, if a source
 lies inside another source or a destination inside another
 destination, or if a destination is already present and is not
 renamed away.
- [**AttributeError**](https://docs.python.org/3/builtins/exceptions.html#AttributeError) - if `self` is a tensorclass and a destination is
 not one of its fields.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict(a=torch.zeros(()), b=torch.ones(()))
>>> out = td.rename_keys({"a": "b", "b": "a"})
>>> out["a"], td["a"]
(tensor(1.), tensor(0.))
>>> td = TensorDict(raw=TensorDict(p=torch.zeros(3), u=torch.ones(3)))
>>> out = td.rename_keys({("raw", "p"): "pressure", ("raw", "u"): "velocity"})
>>> sorted(out.keys(include_nested=True)) # "raw" was emptied and removed
['pressure', 'velocity']
>>> out = td.rename_keys({"raw": "solution"})
>>> sorted(out.keys(include_nested=True, leaves_only=True))
[('solution', 'p'), ('solution', 'u')]
>>> td.rename_keys({("raw", "zz"): "z", ("raw", "p"): "p"}, strict=False)["p"]
tensor([0., 0., 0.])
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).