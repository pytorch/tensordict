# NonTensorDataBase

`NonTensorDataBase` has the methods of [`TensorDictBase`](tensordict.TensorDictBase.html#tensordict.TensorDictBase),
listed by task in [Methods by task](../td.html#api-tensordict-methods). This page lists only the
methods that `NonTensorDataBase` adds, and the overrides that have their own
documentation.

*class*tensordict.NonTensorDataBase(*data: [Any](https://docs.python.org/3/library/typing.html#typing.Any)*, *_metadata: [dict](https://docs.python.org/3/builtins/stdtypes.html#dict) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *_is_non_tensor: [bool](https://docs.python.org/3/builtins/functions.html#bool) = True*, ***, *batch_size*, *device=None*, *names=None*)

Bases: [`TensorClass`](tensordict.TensorClass.html#tensordict.TensorClass)

A base class for tensordict entries that hold a Python object instead of a tensor.

There are two subclasses:

- [`NonTensorData`](tensordict.NonTensorData.html#tensordict.NonTensorData) holds one value per batch element.
Stacking or `unsqueeze` gives a [`NonTensorStack`](tensordict.NonTensorStack.html#tensordict.NonTensorStack).
- [`MetaData`](tensordict.MetaData.html#tensordict.MetaData) holds one value for the whole batch.
Stacking items that hold the same object (checked with `is`) gives
one [`MetaData`](tensordict.MetaData.html#tensordict.MetaData) with the new batch size. Other items
give a [`NonTensorStack`](tensordict.NonTensorStack.html#tensordict.NonTensorStack). `unsqueeze` and indexing
keep a [`MetaData`](tensordict.MetaData.html#tensordict.MetaData).

See [Non-tensor data and unbatched entries](../../concepts/non_tensor.html#non-tensor-data).

copy()

Return a shallow copy, equivalent to `_clone(recurse=False)`.

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

maybe_to_stack()

Converts the NonTensorDataBase object to a NonTensorStack object if it has a non-empty batch-size.

rename_keys(*mapping*, ***, *strict: [bool](https://docs.python.org/3/builtins/functions.html#bool) = True*)

TensorClass-specific rename_keys: the destinations must be fields, which are never removed.

select(**keys*, *inplace: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *strict: [bool](https://docs.python.org/3/builtins/functions.html#bool) = True*, *as_tensordict: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*)

TensorClass-specific select that supports `as_tensordict`.

tolist(***, *convert_tensors: [bool](https://docs.python.org/3/builtins/functions.html#bool) | [Literal](https://docs.python.org/3/library/typing.html#typing.Literal)['numpy'] = False*, *tolist_first: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *as_linked_list: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*)

Converts the data in a list if the batch-size is non-empty.

If the batch-size is empty, returns the data.

Keyword Arguments:

- **convert_tensors** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)*,**"numpy"*) - if `True`, tensors will be converted to lists when creating the dictionary.
If "numpy", tensors will be converted to numpy arrays.
Otherwise, they will remain as tensors. Default: `False`.
- **tolist_first** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)*,**optional*) - if `True`, the tensordict will be converted to a list first when
it has batch dimensions. Default: `False`.
- **as_linked_list** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)*,**optional*) - if `True`, the list will be converted to a [`tensordict.utils.LinkedList`](tensordict.utils.LinkedList.html#tensordict.utils.LinkedList)
which will automatically update the tensordict when the list is modified. Default: `False`.