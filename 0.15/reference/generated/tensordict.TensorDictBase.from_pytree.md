# TensorDictBase.from_pytree

*classmethod*TensorDictBase.from_pytree(*pytree*, ***, *batch_size: [Size](https://docs.pytorch.org/docs/stable/size.html#torch.Size) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *auto_batch_size: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *batch_dims: [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*)

Converts a pytree to a tensordict that [`to_pytree()`](tensordict.TensorDictBase.to_pytree.html#tensordict.TensorDictBase.to_pytree) converts back.

Each list, tuple, named tuple and dict of the pytree becomes a nested
tensordict, and its type is stored in a `"_pytree_type"` entry so
that [`to_pytree()`](tensordict.TensorDictBase.to_pytree.html#tensordict.TensorDictBase.to_pytree) can rebuild the pytree. The elements of a list
or tuple get the keys `"0"`, `"1"`, ... The subclasses of these
types, such as [`collections.OrderedDict`](https://docs.python.org/3/library/collections.html#collections.OrderedDict),
[`collections.defaultdict`](https://docs.python.org/3/library/collections.html#collections.defaultdict) and the `torch.return_types`
results of `torch.max` and the like, are accepted too. A
`defaultdict` is rebuilt without its `default_factory`.

A tensor or a tensor collection is returned as it is. Python ints and
floats and NumPy arrays become tensors, so [`to_pytree()`](tensordict.TensorDictBase.to_pytree.html#tensordict.TensorDictBase.to_pytree) returns
them as tensors. A dict key that is not a [`NestedKey`](tensordict.NestedKey.html#tensordict.NestedKey) (such as an
int or an object) is stored in a [`NonTensorData`](tensordict.NonTensorData.html#tensordict.NonTensorData)
entry next to its value. Other types raise a `NotImplementedError`.

Parameters:

**pytree** - the pytree to convert: a list, tuple, named tuple or dict,
nested to any depth, whose leaves are tensors, tensor
collections, Python numbers or NumPy arrays.

Keyword Arguments:

- **batch_size** ([*Size*](https://docs.pytorch.org/docs/stable/size.html#torch.Size)*or*[*list*](https://docs.python.org/3/builtins/stdtypes.html#list)*of*[*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the batch size of
the result: the leading dimensions that all its entries share.
Defaults to `None`, which gives an empty batch size unless
`auto_batch_size=True`.
- **auto_batch_size** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, sets the batch size
to the largest leading shape that all the entries share.
Defaults to `False`.
- **batch_dims** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - with `auto_batch_size`, the maximum
number of batch dimensions. Ignored without
`auto_batch_size`, with a `FutureWarning` when it would
change the result. Starting with TensorDict 0.17, the default
will change to turning on `auto_batch_size` when
`batch_dims` is given, as `from_dict` does. Defaults to
`None`.

Note

A module `state_dict()` round-trips through `from_pytree`, and its
[`memmap()`](tensordict.TensorDictBase.memmap.html#tensordict.TensorDictBase.memmap) output loads with `load_memmap(path, allow_pickle=False)`.
The state dicts of an optimizer and of a learning-rate scheduler hold
`None`, `str` values and Python numbers, which `from_pytree` rejects
(`None` and `str`) or returns as tensors (numbers): save them with
`torch.save(sd, path)` and load them with `torch.load(path, weights_only=True)`,
which round-trips them exactly without unpickling arbitrary objects.

Note

The list fields of a named tuple, and some of its dict and tuple
fields, are converted with [`from_any()`](tensordict.TensorDictBase.from_any.html#tensordict.TensorDictBase.from_any) and do not
round-trip through [`to_pytree()`](tensordict.TensorDictBase.to_pytree.html#tensordict.TensorDictBase.to_pytree):

- a list of numbers comes back as a tensor;
- an empty tuple, an empty dict or named tuple, and a dict that
holds a str or a list come back as a `GenericDict` named
tuple;
- a dict that holds a tuple, a tuple subclass (such as a
`torch.return_types` result), and a tuple that holds anything
other than tensors, Python numbers, dicts and named tuples make
[`to_pytree()`](tensordict.TensorDictBase.to_pytree.html#tensordict.TensorDictBase.to_pytree) raise a `ValueError`.

The other dict, named tuple and tuple fields come back with their
own type.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> pytree = {"obs": torch.zeros(3, 4), "pair": (torch.zeros(3), torch.ones(3))}
>>> td = TensorDict.from_pytree(pytree, auto_batch_size=True)
>>> td.batch_size
torch.Size([3])
>>> sorted(td.keys()), sorted(td["pair"].keys())
(['_pytree_type', 'obs', 'pair'], ['0', '1', '_pytree_type'])
>>> type(td.to_pytree()["pair"]).__name__
'tuple'
```

A pytree with lists, dicts, a named tuple and a dict key that is
not a string round-trips:

```
>>> class Key:
... pass
>>> key = Key()
>>> pytree = (
... [torch.arange(3), torch.zeros(2)],
... {"tensor": torch.ones(2), key: torch.arange(2), "list": [1, 2, 3]},
... {"named_tuple": TensorDict(two=torch.ones(1)).to_namedtuple()},
... )
>>> pytree_recon = TensorDict.from_pytree(pytree).to_pytree()
>>> def check(v1, v2):
... assert (v1 == v2).all()
>>> _ = torch.utils._pytree.tree_map(check, pytree, pytree_recon)
>>> key in pytree_recon[1]
True
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).