# from_module

tensordict.from_module(*module*, *as_module: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *lock: [bool](https://docs.python.org/3/builtins/functions.html#bool) = True*, *use_state_dict: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *filter_empty: [bool](https://docs.python.org/3/builtins/functions.html#bool) = True*)

Returns a locked tensordict that holds the parameters and buffers of a module.

Same as [`tensordict.TensorDict.from_module()`](tensordict.TensorDict.html#tensordict.TensorDict.from_module), except for the default
of `lock`: this function locks the result by default, and the method
does not. The entries are the parameters and buffers of the
module themselves, not copies. Submodules become nested tensordicts.

Parameters:

- **module** ([*Module*](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)) - the module to get the parameters from.
- **as_module** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, a
`TensorDictParams` instance is returned,
which can be used to store parameters within a
[`torch.nn.Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module). Defaults to `False`.
- **lock** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, the resulting tensordict is
locked. Defaults to `True`.
- **use_state_dict** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, the state-dict from the
module is used and unflattened into a TensorDict with the tree
structure of the model. Use it when the module has state-dict
hooks. Defaults to `False`.
- **filter_empty** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, submodules without
parameters or buffers (`nn.ReLU`, for example) get no entry.
If `False`, they get an empty tensordict. Defaults to
`True`.

Returns:

A [`TensorDict`](tensordict.TensorDict.html#tensordict.TensorDict), or a
`TensorDictParams` if `as_module=True`.

Examples

```
>>> from torch import nn
>>> import tensordict
>>> module = nn.Sequential(nn.Linear(3, 4), nn.ReLU())
>>> params = tensordict.from_module(module)
>>> params["0", "weight"] is module[0].weight
True
>>> params.is_locked
True
```