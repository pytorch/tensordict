# TensorDictBase.from_pytree

*classmethod*TensorDictBase.from_pytree(*pytree*, ***, *batch_size: [Size](https://docs.pytorch.org/docs/stable/size.html#torch.Size) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *auto_batch_size: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *batch_dims: [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*)

Converts a pytree to a TensorDict instance.

This method is designed to keep the pytree nested structure as much as possible.

Additional non-tensor keys are added to keep track of each level's identity, providing
a built-in pytree-to-tensordict bijective transform API.

Accepted classes currently include lists, tuples, named tuples and dict,
and subclasses of these such as [`collections.OrderedDict`](https://docs.python.org/3/library/collections.html#collections.OrderedDict),
[`collections.defaultdict`](https://docs.python.org/3/library/collections.html#collections.defaultdict) and the results of `torch.max` and
the other `torch.return_types`. A `defaultdict` is rebuilt without
its `default_factory`.

Note

For dictionaries, non-NestedKey keys are registered separately as [`NonTensorData`](tensordict.NonTensorData.html#tensordict.NonTensorData)
instances.

Note

Tensor-castable types (such as int, float or np.ndarray) will be converted to torch.Tensor instances.
Note that this transformation is surjective: transforming back the tensordict to a pytree will not
recover the original types.

Note

The list fields of a named tuple, and some of its dict and tuple
fields, are converted with [`from_any()`](tensordict.TensorDictBase.from_any.html#tensordict.TensorDictBase.from_any) and do not
round-trip through [`to_pytree()`](tensordict.TensorDictBase.to_pytree.html#tensordict.TensorDictBase.to_pytree):

- a list of numbers comes back as a tensor;
- a dict holding a str, list or tuple, and an empty dict or named
tuple (or a dict or named tuple holding one), come back as a
`GenericDict` named tuple;
- a tuple subclass, such as a `torch.return_types` result, and
a plain tuple holding anything but tensors, Python ints and
floats, dicts and named tuples (such as a str, a list or a NumPy
number), or holding one of the dicts or named tuples above, make
[`to_pytree()`](tensordict.TensorDictBase.to_pytree.html#tensordict.TensorDictBase.to_pytree) raise a `ValueError`, and an empty tuple
comes back as an empty `GenericDict`.

The other dict, named tuple and plain tuple fields come back with
their own type.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> # Create a pytree with tensor leaves, and one "weird"-looking dict key
>>> class WeirdLookingClass:
... pass
...
>>> weird_key = WeirdLookingClass()
>>> # Make a pytree with tuple, lists, dict and namedtuple
>>> pytree = (
... [torch.randint(10, (3,)), torch.zeros(2)],
... {
... "tensor": torch.randn(
... 2,
... ),
... "td": TensorDict({"one": 1}),
... weird_key: torch.randint(10, (2,)),
... "list": [1, 2, 3],
... },
... {"named_tuple": TensorDict({"two": torch.ones(1) * 2}).to_namedtuple()},
... )
>>> # Build a TensorDict from that pytree
>>> td = TensorDict.from_pytree(pytree)
>>> # Recover the pytree
>>> pytree_recon = td.to_pytree()
>>> # Check that the leaves match
>>> def check(v1, v2):
... assert (v1 == v2).all()
...
>>> _ = torch.utils._pytree.tree_map(check, pytree, pytree_recon)
>>> assert weird_key in pytree_recon[1]
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).