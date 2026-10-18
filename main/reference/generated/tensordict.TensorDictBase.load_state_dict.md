# TensorDictBase.load_state_dict

TensorDictBase.load_state_dict(*state_dict: [OrderedDict](https://docs.python.org/3/library/typing.html#typing.OrderedDict)[[str](https://docs.python.org/3/builtins/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)]*, *strict=True*, *assign=False*, *from_flatten=None*) → Self

Loads a state dict, as [`state_dict()`](tensordict.TensorDictBase.state_dict.html#tensordict.TensorDictBase.state_dict) returns it, into the tensordict, and returns `self`.

Reads the flat format (the default of [`state_dict()`](tensordict.TensorDictBase.state_dict.html#tensordict.TensorDictBase.state_dict)), the
nested format (`flatten=False`), and the legacy format, with
`"__batch_size"` and `"__device"` keys. The batch size of the
tensordict is set to the one stored in the state dict, if any.

Parameters:

- **state_dict** (*OrderedDict*) - the state dict to load.
- **strict** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, a state dict whose keys are
not those of the tensordict raises a `RuntimeError` (empty
nested tensordicts excepted). If `False`, the keys of the
state dict are written and the other entries of the
tensordict are kept. Defaults to `True`.
- **assign** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `False`, the values are copied in
place into the existing tensors, which keep their storage,
dtype and device. If `True`, the tensors of the state dict
replace them. Defaults to `False`.
- **from_flatten** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, the state dict is read
as flat, with `"."`-joined keys. If `False`, as nested.
Defaults to `None`, which reads it as flat if it has a
`_metadata` attribute and its keys differ from those of the
tensordict.

Returns:

`self`.

Raises:

[**RuntimeError**](https://docs.python.org/3/builtins/exceptions.html#RuntimeError) - if `strict=True` and the keys do not match, or if
 the state dict was saved on another device than that of the
 tensordict.

Examples

```
>>> from tensordict import TensorDict
>>> data = TensorDict({"1": 1, "2": 2, "3": {"3": 3}}, [])
>>> data_zeroed = TensorDict({"1": 0, "2": 0, "3": {"3": 0}}, [])
>>> sd = data.state_dict()
>>> _ = data_zeroed.load_state_dict(sd)
>>> print(data_zeroed["3", "3"])
tensor(3)
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).