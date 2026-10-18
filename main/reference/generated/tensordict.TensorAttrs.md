# TensorAttrs

`TensorAttrs` has the methods of [`TensorDictBase`](tensordict.TensorDictBase.html#tensordict.TensorDictBase),
listed by task in [Methods by task](../td.html#api-tensordict-methods). This page lists only the
methods that `TensorAttrs` adds, and the overrides that have their own
documentation.

*class*tensordict.TensorAttrs(*tgt_device: [Any](https://docs.python.org/3/library/typing.html#typing.Any) = None*, *tgt_dtype: [Any](https://docs.python.org/3/library/typing.html#typing.Any) = None*, *tgt_shape: [Any](https://docs.python.org/3/library/typing.html#typing.Any) = None*, ***, *batch_size*, *device=None*, *names=None*)

Bases: [`TensorClass`](tensordict.TensorClass.html#tensordict.TensorClass)

A tensorclass that holds the device, dtype and shape of one tensor.

[`attrs()`](tensordict.TensorDictBase.attrs.html#tensordict.TensorDictBase.attrs) returns a tensordict with one
`TensorAttrs` per leaf. Pass it to
[`to()`](tensordict.TensorDictBase.to.html#tensordict.TensorDictBase.to) to move or cast each leaf of another
tensordict to the device and dtype of the matching leaf, for example when
the leaves are on different devices. The names follow the [PyTorch Tensor
Attributes](https://pytorch.org/docs/stable/tensor_attributes.html).

The fields start with `tgt_` so that they do not hide the `device`,
`dtype` and `shape` attributes of the tensorclass.

Parameters:

- **tgt_device** ([*device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,**optional*) - the device of the tensor.
Defaults to `None`.
- **tgt_dtype** ([*dtype*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype)*,**optional*) - the dtype of the tensor. Defaults
to `None`.
- **tgt_shape** ([*Size*](https://docs.pytorch.org/docs/stable/size.html#torch.Size)*,**optional*) - the shape of the tensor. Defaults
to `None`.

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

*classmethod*from_tensor(*tensor*, ***, *fields=('device', 'dtype', 'shape')*)

Returns a `TensorAttrs` with the attributes of `tensor`.

Parameters:

**tensor** ([*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) - the tensor to describe.

Keyword Arguments:

**fields** ([*tuple*](https://docs.python.org/3/builtins/stdtypes.html#tuple)*of*[*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - the attributes to read, among
`"device"`, `"dtype"` and `"shape"`. The other fields
are `None`. Defaults to `("device", "dtype", "shape")`.

Examples

```
>>> import torch
>>> from tensordict import TensorAttrs
>>> attrs = TensorAttrs.from_tensor(torch.zeros(2, 3), fields=("dtype",))
>>> attrs.tgt_dtype, attrs.tgt_shape
(torch.float32, None)
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