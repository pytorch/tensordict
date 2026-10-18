# MetaData

`MetaData` has the methods of [`TensorDictBase`](tensordict.TensorDictBase.html#tensordict.TensorDictBase),
listed by task in [Methods by task](../td.html#api-tensordict-methods). This page lists only the
methods that `MetaData` adds, and the overrides that have their own
documentation.

*class*tensordict.MetaData(*data: [Any](https://docs.python.org/3/library/typing.html#typing.Any)*, *_metadata: [dict](https://docs.python.org/3/builtins/stdtypes.html#dict) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *_is_non_tensor: [bool](https://docs.python.org/3/builtins/functions.html#bool) = True*, ***, *batch_size*, *device=None*, *names=None*)

Bases: [`NonTensorDataBase`](tensordict.NonTensorDataBase.html#tensordict.NonTensorDataBase)

A tensordict entry that holds one Python object for the whole batch.

Use it for information about the batch as a whole, such as a
configuration dict or the name of a dataset. It behaves as
[`NonTensorData`](tensordict.NonTensorData.html#tensordict.NonTensorData), except for indexing, stacking,
squeezing, unsqueezing and similar operations:

- When stacking, `MetaData` checks whether the items hold the same
object (with `is`, not `==`). If they do, the result is one
`MetaData` with the stacked batch size. If they do not, the
result is a [`NonTensorStack`](tensordict.NonTensorStack.html#tensordict.NonTensorStack).
[`clone()`](tensordict.TensorDictBase.clone.html#tensordict.TensorDictBase.clone) copies the value, so a clone
and its source hold different objects.
- [`unsqueeze()`](https://docs.pytorch.org/docs/stable/generated/torch.unsqueeze.html#torch.unsqueeze) and indexing return a `MetaData`, not a
stack.

`MetaData[cls]` makes a subclass that raises a `TypeError` if the
value is not an instance of `cls`. See [Non-tensor data and unbatched entries](../../concepts/non_tensor.html#non-tensor-data).

Parameters:

**data** (*Any*) - the value. A [`NonTensorData`](tensordict.NonTensorData.html#tensordict.NonTensorData) is
unwrapped: its value is used.

Keyword Arguments:

- **batch_size** ([*Size*](https://docs.pytorch.org/docs/stable/size.html#torch.Size)*or*[*list*](https://docs.python.org/3/builtins/stdtypes.html#list)*of*[*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the batch size of
the entry, usually that of the tensordict that holds it. Defaults
to `torch.Size([])`.
- **device** ([*device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,*[*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*or*[*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the device of the entry.
The value itself is not moved. Defaults to `None`.
- **names** ([*list*](https://docs.python.org/3/builtins/stdtypes.html#list)*of*[*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - the names of the batch dimensions, one
per dimension. Defaults to `None`.

Examples

```
>>> import torch
>>> from tensordict import MetaData, TensorDict
>>> config = {"lr": 0.1}
>>> td = TensorDict(a=torch.zeros(3), batch_size=[3])
>>> td["config"] = MetaData(config, batch_size=[3])
>>> td["config"] is config
True
>>> stacked = torch.stack([td, td])
>>> type(stacked.get("config")).__name__
'MetaData'
>>> stacked["config"] is config
True
>>> MetaData[str](1)
Traceback (most recent call last):
 ...
TypeError: Expected data of type str, got int
```

*classmethod*fields() → [tuple](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[Field](https://docs.python.org/3/library/dataclasses.html#dataclasses.Field), ...]

Returns the fields of the tensorclass.

Deprecated since version 0.15: Use `dataclasses.fields(cls)` instead. It returns the same tuple of
[`dataclasses.Field`](https://docs.python.org/3/library/dataclasses.html#dataclasses.Field) objects.

from_module(*as_module: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *lock: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *use_state_dict: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *filter_empty: [bool](https://docs.python.org/3/builtins/functions.html#bool) = True*)

Returns a tensordict that holds the parameters and buffers of a module.

The entries are the parameters and buffers of the module themselves,
not copies. Submodules become nested tensordicts.

Note

This method does not lock the result by default (`lock=False`).
The function [`tensordict.from_module()`](tensordict.from_module.html#tensordict.from_module) locks it by default
(`lock=True`).

Parameters:

- **module** ([*Module*](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)) - the module to get the parameters from.
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

A [`TensorDict`](tensordict.TensorDict.html#tensordict.TensorDict), or a
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
fields that neither argument holds are set to `None`. Defaults
to `None`.
- **safe** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - deprecated in 0.15, will be removed in TensorDict 0.17.
If `False`, skips the check that `tensordict` is a
[`TensorDictBase`](tensordict.TensorDictBase.html#tensordict.TensorDictBase). By default, the check runs, and
another type raises a `RuntimeError`.

Returns:

A new instance of the class, whose fields are the entries of
`tensordict`.

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