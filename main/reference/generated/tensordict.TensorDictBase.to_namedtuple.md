# TensorDictBase.to_namedtuple

TensorDictBase.to_namedtuple(*dest_cls: [type](https://docs.python.org/3/builtins/functions.html#type) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*) → [Any](https://docs.python.org/3/library/typing.html#typing.Any)

Converts the tensordict to a named tuple, recursively.

Each nested tensordict becomes a nested named tuple. Without
`dest_cls`, each level is an instance of a new named tuple class,
`GenericDict`, whose fields are the keys. The `None` fields of
tensorclasses are dropped.

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

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).