# TensorAttrs

`TensorAttrs` has the methods of [`TensorDictBase`](tensordict.TensorDictBase.html#tensordict.TensorDictBase),
listed by task in [Methods by task](../td.html#api-tensordict-methods). This page lists only the
methods that `TensorAttrs` adds, and the overrides that have their own
documentation.

*class*tensordict.TensorAttrs(*tgt_device: [Any](https://docs.python.org/3/library/typing.html#typing.Any) = None*, *tgt_dtype: [Any](https://docs.python.org/3/library/typing.html#typing.Any) = None*, *tgt_shape: [Any](https://docs.python.org/3/library/typing.html#typing.Any) = None*, ***, *batch_size*, *device=None*, *names=None*)

Bases: [`TensorClass`](tensordict.TensorClass.html#tensordict.TensorClass)

Per-leaf carrier describing a tensor's attributes -- device, dtype, and shape.

Mirrors the [PyTorch Tensor Attributes](https://pytorch.org/docs/stable/tensor_attributes.html)
terminology. Instances are produced by [`attrs()`](tensordict.TensorDictBase.attrs.html#tensordict.TensorDictBase.attrs)
-- one per leaf of the source tensordict -- and consumed by
[`to()`](tensordict.TensorDictBase.to.html#tensordict.TensorDictBase.to) to drive per-leaf device/dtype casting
when the source tensordict is heterogeneous (e.g. `device=None` with leaves
on different devices).

The field names are prefixed with `tgt_` to avoid shadowing the tensordict
attributes `device`/`dtype`/`shape`.

Fields populated via `from_tensor()` are controlled by the `fields` argument
so the API scales as new attributes are added.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict({"a": torch.zeros(3, device="cpu"),
... "b": torch.zeros(3, dtype=torch.int32)}, batch_size=[3])
>>> attrs = td.attrs()
>>> attrs["a"].tgt_device, attrs["a"].tgt_dtype
(device(type='cpu'), torch.float32)
```

*property*device*: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*

Retrieves the device type of tensor class.

extend(*tensordict: [list](https://docs.python.org/3/builtins/stdtypes.html#list)[[TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase)] | [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*) → [None](https://docs.python.org/3/builtins/constants.html#None)

Appends tensordicts to the lazy stack that the instance holds.

Deprecated since version 0.15: Build a new instance with [`tensordict.lazy_stack()`](tensordict.lazy_stack.html#tensordict.lazy_stack) or
[`torch.cat()`](https://docs.pytorch.org/docs/stable/generated/torch.cat.html#torch.cat) instead.

*classmethod*fields() → [tuple](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[Field](https://docs.python.org/3/library/dataclasses.html#dataclasses.Field), ...]

Returns the fields of the tensorclass.

Deprecated since version 0.15: Use `dataclasses.fields(cls)` instead. It returns the same tuple of
[`dataclasses.Field`](https://docs.python.org/3/library/dataclasses.html#dataclasses.Field) objects.

from_module(*as_module: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *lock: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *use_state_dict: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *filter_empty: [bool](https://docs.python.org/3/builtins/functions.html#bool) = True*)

Puts the parameters and buffers of a module in a tensordict.

The entries are the parameters and buffers of the module themselves,
not copies. Submodules become nested tensordicts.

Note

This method does not lock the result by default (`lock=False`).
The function [`tensordict.from_module()`](tensordict.from_module.html#tensordict.from_module) locks it by default
(`lock=True`).

Parameters:

- **module** (*nn.Module*) - the module to get the parameters from.
- **as_module** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, a `TensorDictParams`
instance is returned, which can be used to store parameters
within a [`torch.nn.Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module). Defaults to `False`.
- **lock** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, the resulting tensordict is
locked. Defaults to `False`.
- **use_state_dict** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, the state-dict from the
module is used and unflattened into a TensorDict with
the tree structure of the model. Use it when the module has
state-dict hooks. Defaults to `False`.
- **filter_empty** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, submodules without
parameters or buffers (`nn.ReLU`, for example) get no
entry. If `False`, they get an empty tensordict.
Defaults to `True`.

Returns:

a [`TensorDict`](tensordict.TensorDict.html#tensordict.TensorDict), or a
`TensorDictParams` if `as_module=True`.

Examples

```
>>> from torch import nn
>>> from tensordict import TensorDict
>>> module = nn.Sequential(nn.Linear(3, 4), nn.ReLU())
>>> params = TensorDict.from_module(module)
>>> params["0", "weight"] is module[0].weight
True
>>> list(params.keys())
['0']
>>> params.is_locked
False
```

*classmethod*from_tensor(*tensor*, ***, *fields=('device', 'dtype', 'shape')*)

Build a `TensorAttrs` from a tensor, populating only the requested fields.

`fields` accepts the short names `"device"`, `"dtype"`, `"shape"`. Unrequested
fields remain `None`.

*classmethod*from_tensordict(*tensordict: [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*, *non_tensordict: [dict](https://docs.python.org/3/builtins/stdtypes.html#dict) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *safe: [bool](https://docs.python.org/3/builtins/functions.html#bool) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*) → Self

Wraps a tensordict in a new instance of the tensorclass, without copying the leaves.

If `tensordict` is a [`TensorDict`](tensordict.TensorDict.html#tensordict.TensorDict), the entries of the fields
annotated with a tensorclass (or `Optional` of one) that are tensordicts become
instances of that tensorclass, recursively. They are set in a shallow copy of
`tensordict`, which keeps its own entries. An entry with keys that the annotated
class does not declare stays a tensordict. Other backends are wrapped as they are:
setting a tensorclass in them would write to their storage (e.g. an H5 file).

Parameters:

- **tensordict** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)) - the tensordict that holds the tensor fields.
- **non_tensordict** ([*dict*](https://docs.python.org/3/builtins/stdtypes.html#dict)*,**optional*) - the values of the non-tensor fields. The
fields that neither argument holds are set to `None`.
- **safe** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - Deprecated in 0.15, to be removed in TensorDict 0.17. Whether to
raise an error if `tensordict` is not a [`TensorDictBase`](tensordict.TensorDictBase.html#tensordict.TensorDictBase).
Without it, such a tensordict raises an error.

Examples

```
>>> import torch
>>> from tensordict import TensorClass
>>> class Pose(TensorClass):
... q: torch.Tensor
>>> class Obs(TensorClass):
... a: Pose
>>> obs = Obs(a=Pose(q=torch.zeros(5, 4), batch_size=[5]), batch_size=[5])
>>> type(Obs.from_tensordict(obs.to_tensordict()).a).__name__
'Pose'
```

rename_keys(*mapping*, ***, *strict: [bool](https://docs.python.org/3/builtins/functions.html#bool) = True*)

TensorClass-specific rename_keys: the destinations must be fields, which are never removed.

select(**keys*, *inplace: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *strict: [bool](https://docs.python.org/3/builtins/functions.html#bool) = True*, *as_tensordict: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*)

TensorClass-specific select that supports `as_tensordict`.