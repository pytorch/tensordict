# TensorDictParams

`TensorDictParams` has the methods of [`TensorDictBase`](tensordict.TensorDictBase.html#tensordict.TensorDictBase),
listed by task in [Methods by task](../td.html#api-tensordict-methods). This page lists only the
methods that `TensorDictParams` adds, and the overrides that have their own
documentation.

*class*tensordict.TensorDictParams(*parameters: [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase) | [dict](https://docs.python.org/3/builtins/stdtypes.html#dict) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, ***, *no_convert=False*, *lock: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, ***kwargs*)

Bases: [`TensorDictBase`](tensordict.TensorDictBase.html#tensordict.TensorDictBase), [`Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)

A TensorDict whose entries are registered as the parameters and buffers of an [`Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module).

Use it to keep a TensorDict of parameters on a module, for example the
stacked weights of an ensemble or a copy of the weights for a target
network. A plain TensorDict attribute is not seen by `module.parameters()`,
`module.state_dict()` or `module.to()`. A TensorDictParams attribute is
seen by all three, so an optimizer can train its entries.

A TensorDictParams behaves like the TensorDict that it wraps. You can index
it, set entries, and call [`to_module()`](tensordict.TensorDictBase.to_module.html#tensordict.TensorDictBase.to_module) to
load the values into a module. `named_parameters()` names each entry
by its nested key joined with `"."`, for example `"0.weight"`, as
[`Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module) does.

When you set an entry, float16, float32 and float64 tensors become
`Parameter`. Tensors of all other dtypes (including
bfloat16, complex and bool) become buffers. An optimizer built from
`parameters()` therefore does not see bfloat16 entries. Pass
`no_convert=True` to register all tensors as buffers.

Warning

TensorDictParams wraps the TensorDict that you pass,
without a copy, and turns its leaves into parameters and
buffers in place. It registers the leaves that are present at
construction. If you later add or replace an entry of the
original TensorDict, indexing the TensorDictParams shows the
new value, but `parameters()` and `buffers()` keep
the old one. Write through the TensorDictParams, or pass
`lock=True`.

Parameters:

**parameters** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*or*[*dict*](https://docs.python.org/3/builtins/stdtypes.html#dict)*,**optional*) - the tensordict to wrap.
A `dict` is wrapped in a [`TensorDict`](tensordict.TensorDict.html#tensordict.TensorDict) first.
Exclusive with `**kwargs`. Defaults to `None`.

Keyword Arguments:

- **no_convert** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, tensors that are not
parameters or buffers become `Buffer` instances.
If `False`, float16, float32 and float64 tensors become
`Parameter`, and tensors of all other dtypes
become buffers. Tensors that already are parameters or buffers
stay as they are. Defaults to `False`.
- **lock** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, locks the wrapped tensordict (see
[Locking and temporary changes](../../concepts/keys.html#keys-locking)), so that entries can no longer be added to it
or removed from it directly. Writes through the TensorDictParams
still work. Defaults to `False`.
- ****kwargs** - the entries of the tensordict, as `key=value` pairs, when
`parameters` is not given. Exclusive with `parameters`.

Examples

```
>>> import torch
>>> from torch import nn
>>> from tensordict import TensorDict
>>> from tensordict.nn import TensorDictParams
>>> module = nn.Sequential(nn.Linear(3, 4), nn.Linear(4, 4))
>>> params = TensorDict.from_module(module)
>>> _ = params.lock_()
>>> p = TensorDictParams(params)
>>> print(p)
TensorDictParams(params=TensorDict(
 fields={
 0: TensorDict(
 fields={
 bias: Parameter(shape=torch.Size([4]), device=cpu, dtype=torch.float32, is_shared=False),
 weight: Parameter(shape=torch.Size([4, 3]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False),
 1: TensorDict(
 fields={
 bias: Parameter(shape=torch.Size([4]), device=cpu, dtype=torch.float32, is_shared=False),
 weight: Parameter(shape=torch.Size([4, 4]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False))
>>> class CustomModule(nn.Module):
... def __init__(self, params):
... super().__init__()
... self.params = params
>>> m = CustomModule(p)
>>> [name for name, _ in m.named_parameters()]
['params.0.weight', 'params.0.bias', 'params.1.weight', 'params.1.bias']
>>> # The wrapper supports assignment, and values are converted to Parameters
>>> m.params['other'] = torch.randn(3)
>>> isinstance(m.params['other'], nn.Parameter)
True
>>> # A bfloat16 entry becomes a buffer
>>> m.params['half'] = torch.zeros(3, dtype=torch.bfloat16)
>>> [name for name, _ in m.named_buffers()]
['params.half']
```

register_get_post_hook(*hook*)

Registers a hook that is called after each read of a leaf.

Parameters:

**hook** (*Callable*) - called as `hook(tensordict_params, value)`
after each read of a leaf. If it returns a value other than
`None`, that value replaces the one read.

## Tutorials that use `TensorDictParams`

![](../../_images/sphx_glr_functional_thumb.png)

[Functional calls and ensembles with TensorDict](../../tutorials/functional.html)

Functional calls and ensembles with TensorDict