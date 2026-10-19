# TensorDictBase.state_dict

TensorDictBase.state_dict(*destination=None*, *prefix=''*, *keep_vars=False*, *flatten=True*) → [OrderedDict](https://docs.python.org/3/library/typing.html#typing.OrderedDict)[[str](https://docs.python.org/3/builtins/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)]

Returns the leaves of the tensordict in an `OrderedDict`, as [`torch.nn.Module.state_dict()`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.state_dict) does.

The state dict is flat by default: the key of a nested leaf joins its
keys with `"."` (`"n.b"` for `("n", "b")`), following
[`torch.nn.Module.state_dict()`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.state_dict). With `flatten=False`, each
nested tensordict becomes a nested `OrderedDict`. Read it back with
[`load_state_dict()`](tensordict.TensorDictBase.load_state_dict.html#tensordict.TensorDictBase.load_state_dict).

The batch size and device of each level are stored in a
`_metadata` attribute of the result, as in
[`torch.nn.Module.state_dict()`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.state_dict). In a flat state dict, they are
stored under the `"."`-joined key of the level (`""` for the root,
`"n"` for the nested tensordict `"n"`). In a nested state dict,
each nested `OrderedDict` has its own `_metadata`.

Parameters:

- **destination** (*OrderedDict**,**optional*) - an `OrderedDict` to write the
entries into, and to return. A plain `dict` raises an
`AttributeError`. Defaults to `None`, which creates a new
`OrderedDict`.
- **prefix** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - a string prepended to every key. Defaults
to `""`.
- **keep_vars** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `False`, the tensors are detached
from autograd. If `True`, the tensors of the tensordict are
returned unchanged. Defaults to `False`.
- **flatten** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, nested keys are joined with
`"."`. If `False`, nested tensordicts become nested
`OrderedDict` instances. Defaults to `True`.

Returns:

`destination`, or a new `OrderedDict`.

Examples

```
>>> from tensordict import TensorDict
>>> data = TensorDict({"1": 1, "2": 2, "3": {"3": 3}}, [])
>>> sd = data.state_dict()
>>> print(dict(sd))
{'1': tensor(1), '2': tensor(2), '3.3': tensor(3)}
>>> print(dict(sd._metadata))
{'': {'batch_size': torch.Size([]), 'device': None}, '3': {'batch_size': torch.Size([]), 'device': None}}
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).