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

copy()

Returns a shallow copy: a new entry that holds the same value, as `clone(recurse=False)` does.

A `list`, `tuple` or `dict` value is copied, but not its items.

*property*device*: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*

The device of the tensorclass, or `None` if its leaves may be on different devices.

It is read-only: use [`to()`](tensordict.TensorDictBase.to.html#tensordict.TensorDictBase.to) to move the
tensorclass to another device.

extend(*tensordict: [list](https://docs.python.org/3/builtins/stdtypes.html#list)[[TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase)] | [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*) → [None](https://docs.python.org/3/builtins/constants.html#None)

Appends tensordicts to the lazy stack that the instance holds.

Deprecated since version 0.15: Build a new instance with [`tensordict.lazy_stack()`](tensordict.lazy_stack.html#tensordict.lazy_stack) or
[`torch.cat()`](https://docs.pytorch.org/docs/stable/generated/torch.cat.html#torch.cat) instead.

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

maybe_to_stack()

Returns a [`NonTensorStack`](tensordict.NonTensorStack.html#tensordict.NonTensorStack) with the value in every batch element.

The elements share the value: it is not copied. If the batch size is
empty, returns `self`.

Examples

```
>>> from tensordict import NonTensorData
>>> stack = NonTensorData("a", batch_size=[2]).maybe_to_stack()
>>> type(stack).__name__, stack.tolist()
('NonTensorStack', ['a', 'a'])
```

rename_keys(*mapping*, ***, *strict: [bool](https://docs.python.org/3/builtins/functions.html#bool) = True*)

Returns a tensorclass with several keys renamed at once.

Works as [`rename_keys()`](tensordict.TensorDictBase.rename_keys.html#tensordict.TensorDictBase.rename_keys), which describes
`mapping` and `strict`, with two differences: each destination must be
a field of the class, or the call raises an `AttributeError`, and a
field that the renaming empties is kept.

Examples

```
>>> import torch
>>> from tensordict import TensorClass
>>> class Foo(TensorClass):
... x: torch.Tensor
... y: torch.Tensor
>>> foo = Foo(x=torch.zeros(3), y=torch.ones(3), batch_size=[3])
>>> foo.rename_keys({"x": "y", "y": "x"}).x
tensor([1., 1., 1.])
```

select(**keys*, *inplace: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *strict: [bool](https://docs.python.org/3/builtins/functions.html#bool) = True*, *as_tensordict: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*)

Returns a tensorclass that holds only the fields of `keys`, with the other fields set to `None`.

Works as [`select()`](tensordict.TensorDictBase.select.html#tensordict.TensorDictBase.select), which describes
`keys`: the leaves are not copied.

Keyword Arguments:

- **inplace** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, sets the other fields of the
tensorclass to `None` and returns `self`. Defaults to
`False`.
- **strict** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, a missing key raises a
`KeyError`. If `False`, missing keys are ignored. Defaults to
`True`.
- **as_tensordict** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, returns the selected
entries in a [`TensorDict`](tensordict.TensorDict.html#tensordict.TensorDict) instead of a
tensorclass. Defaults to `False`.

Examples

```
>>> import torch
>>> from tensordict import TensorClass
>>> class Foo(TensorClass):
... x: torch.Tensor
... y: torch.Tensor
>>> foo = Foo(x=torch.zeros(3), y=torch.ones(3), batch_size=[3])
>>> print(foo.select("x").y)
None
>>> type(foo.select("x", as_tensordict=True)).__name__
'TensorDict'
```

tolist(***, *convert_tensors: [bool](https://docs.python.org/3/builtins/functions.html#bool) | [Literal](https://docs.python.org/3/library/typing.html#typing.Literal)['numpy'] = False*, *tolist_first: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *as_linked_list: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*)

Returns the value in a nested list with one item per batch element.

If the batch size is empty, returns the value itself.

Keyword Arguments:

- **convert_tensors** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*or**"numpy"**,**optional*) - accepted for
compatibility with [`tolist()`](tensordict.TensorDictBase.tolist.html#tensordict.TensorDictBase.tolist).
It has no effect: the items are the stored value. Defaults to
`False`.
- **tolist_first** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - accepted for compatibility with
[`tolist()`](tensordict.TensorDictBase.tolist.html#tensordict.TensorDictBase.tolist). It has no effect.
Defaults to `False`.
- **as_linked_list** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, returns a
`tensordict.utils.LinkedList` bound to the entry instead of a
list. A [`NonTensorData`](tensordict.NonTensorData.html#tensordict.NonTensorData) holds one value, so setting an
item of the list to a non-tensor value, such as a string,
raises a `ValueError` and leaves the list unchanged. Call
`maybe_to_stack()` first to get a
[`NonTensorStack`](tensordict.NonTensorStack.html#tensordict.NonTensorStack), whose list writes back.
Defaults to `False`.

Examples

```
>>> from tensordict import NonTensorData
>>> NonTensorData("a", batch_size=[2, 1]).tolist()
[['a'], ['a']]
>>> NonTensorData("a").tolist()
'a'
```