# TensorDictBase.load_state_dict

TensorDictBase.load_state_dict(*state_dict: [OrderedDict](https://docs.python.org/3/library/typing.html#typing.OrderedDict)[[str](https://docs.python.org/3/builtins/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)]*, *strict=True*, *assign=False*, *from_flatten=None*) → Self

Loads a state-dict, formatted as in [`state_dict()`](tensordict.TensorDictBase.state_dict.html#tensordict.TensorDictBase.state_dict), into the tensordict.

Supports the flat format (with `_metadata`, the default output of
[`state_dict()`](tensordict.TensorDictBase.state_dict.html#tensordict.TensorDictBase.state_dict)), the nested format (with per-level `_metadata`),
and the legacy format (with `__batch_size`/`__device` sentinel
keys).

When `from_flatten` is `None` (the default), the format is
auto-detected: if `_metadata` is present and the state_dict keys
don't match this tensordict's keys, the state_dict is unflattened
before loading.

Parameters:

- **state_dict** (*OrderedDict*) - the state_dict of to be copied.
- **strict** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - whether to strictly enforce that the keys
in [`state_dict`](tensordict.TensorDictBase.state_dict.html#tensordict.TensorDictBase.state_dict) match the keys returned by this tensordict's
[`torch.nn.Module.state_dict()`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.state_dict) function. Default: `True`
- **assign** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - whether to assign items in the state
dictionary to their corresponding keys in the tensordict instead
of copying them inplace into the tensordict's current tensors.
When `False`, the properties of the tensors in the current
module are preserved while when `True`, the properties of the
Tensors in the state dict are preserved.
Default: `False`
- **from_flatten** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, the input state_dict is
assumed to be flattened and will be unflattened before loading.
If `None` (default), auto-detects based on `_metadata` and
key comparison.

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