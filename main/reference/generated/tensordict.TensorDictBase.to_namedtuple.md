# TensorDictBase.to_namedtuple

TensorDictBase.to_namedtuple(*dest_cls: [type](https://docs.python.org/3/builtins/functions.html#type) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*) → [Any](https://docs.python.org/3/library/typing.html#typing.Any)

Converts the tensordict to a named tuple, recursively.

Each nested tensordict becomes a nested named tuple. Without
`dest_cls`, each level is an instance of a new named tuple class,
`GenericDict`, whose fields are the keys. The `None` fields of
tensorclasses are dropped.

On a tensorclass, the top level is a plain [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple), also with
`dest_cls`: it holds the values of `to_dict(retain_none=False)`,
in the same order, without the field names. The nested tensordicts
and tensorclasses in it are still named tuples. A plain tuple of
tensors can be pickled and loaded with
`torch.load(weights_only=True)`, while a `GenericDict` cannot be
pickled, since its class is made at each call. To keep the field
names, call `to_dict()`, or `to_tensordict().to_namedtuple()`.

Parameters:

**dest_cls** ([*type*](https://docs.python.org/3/builtins/functions.html#type)*,**optional*) - the named tuple class of the result. A
nested tensordict is converted to `dest_cls` too if
`dest_cls` takes its keys, and to a `GenericDict`
otherwise. Defaults to `None`.

See also

[`from_namedtuple()`](tensordict.TensorDictBase.from_namedtuple.html#tensordict.TensorDictBase.from_namedtuple), which
converts back.

Examples

```
>>> from tensordict import TensorDict
>>> import torch
>>> data = TensorDict({
... "a_tensor": torch.zeros((3)),
... "nested": {"a_tensor": torch.zeros((3)), "a_string": "zero!"}}, [3])
>>> data.to_namedtuple()
GenericDict(a_tensor=tensor([0., 0., 0.]), nested=GenericDict(a_tensor=tensor([0., 0., 0.]), a_string='zero!'))
```

On a tensorclass, the top level is a plain tuple:

```
>>> from tensordict import tensorclass
>>> @tensorclass
... class Data:
... a_tensor: torch.Tensor
... a_string: str
>>> Data(a_tensor=torch.zeros(3), a_string="zero!", batch_size=[3]).to_namedtuple()
(tensor([0., 0., 0.]), 'zero!')
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).