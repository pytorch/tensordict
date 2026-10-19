# TensorDict

`TensorDict` has the methods of [`TensorDictBase`](tensordict.TensorDictBase.html#tensordict.TensorDictBase),
listed by task in [Methods by task](../td.html#api-tensordict-methods). This page lists only the
methods that `TensorDict` adds, and the overrides that have their own
documentation.

*class*tensordict.TensorDict(*source: T | [Mapping](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[Any](https://docs.python.org/3/library/typing.html#typing.Any), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *batch_size: [Sequence](https://docs.python.org/3/library/typing.html#typing.Sequence)[[int](https://docs.python.org/3/builtins/functions.html#int)] | [Size](https://docs.pytorch.org/docs/stable/size.html#torch.Size) | [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *device: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *names: [Sequence](https://docs.python.org/3/library/typing.html#typing.Sequence)[[str](https://docs.python.org/3/builtins/stdtypes.html#str)] | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *non_blocking: [bool](https://docs.python.org/3/builtins/functions.html#bool) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *lock: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, ***kwargs: [Any](https://docs.python.org/3/library/typing.html#typing.Any)*)

Bases: [`TensorDictBase`](tensordict.TensorDictBase.html#tensordict.TensorDictBase)

A dictionary of tensors that share a batch size.

The batch size is the leading part of the shape that every entry has in
common. With `batch_size=[3, 4]`, every entry has a shape that starts
with `(3, 4)`. Shape operations (indexing, `reshape`, `view`,
`unsqueeze`, `permute`, masking) act on the batch dimensions of every
entry at once. Pointwise operations (`td + other`, `td * 2`),
`td.to(device)` and `td.to(dtype)` apply to every entry. An entry can
be another tensordict, so keys can be nested: `td["a", "b"]` reads the
entry `"b"` of the nested tensordict `"a"`. [Batch size, shapes, and indexing](../../concepts/batch_size.html#batch-size) and
[Keys, nested keys, and locking](../../concepts/keys.html#keys) explain these ideas.

Use `TensorDict` for most data. To stack tensordicts whose keys or
entry shapes differ, use [`LazyStackedTensorDict`](tensordict.LazyStackedTensorDict.html#tensordict.LazyStackedTensorDict). To
declare a fixed set of fields with types, use
[`TensorClass`](tensordict.TensorClass.html#tensordict.TensorClass) or [`TypedTensorDict`](tensordict.TypedTensorDict.html#tensordict.TypedTensorDict).
[Which container should I use?](../../choosing_a_container.html#choosing-a-container) compares the containers.

Build a tensordict from a dict, or from keyword arguments as with
`dict(...)`:

```
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict(
... {"obs": torch.zeros(3, 4, 5), "reward": torch.zeros(3, 4)},
... batch_size=[3, 4],
... )
>>> td[0]["obs"].shape
torch.Size([4, 5])
>>> td.view(-1)["obs"].shape
torch.Size([12, 5])
>>> (td + 1)["reward"].sum()
tensor(12.)
>>> td.to(torch.float64)["obs"].dtype
torch.float64
>>> td = TensorDict(a=torch.zeros(3), b={"c": torch.ones(3, 2)}, batch_size=[3])
>>> td["b", "c"].shape
torch.Size([3, 2])
```

A tensordict has these constraints:

- The batch size is not inferred from the data: it defaults to `[]`.
Call [`auto_batch_size_()`](tensordict.TensorDictBase.auto_batch_size_.html#tensordict.TensorDictBase.auto_batch_size_) to set it to
the largest leading shape that all entries share.
- The shape of every entry must start with the batch size. If it does
not, setting the entry raises a `RuntimeError`.
- If `device` is set, every entry is moved to that device. If it is
not set, the entries can be on different devices. See [Devices, dtypes, and memory](../../concepts/devices.html#devices).
- A key is a string, or a tuple of strings for a nested entry. Other keys
raise a `KeyError`.

The values of `source` are stored as given, not copied, except when
they are moved to `device`. A nested `dict` becomes a nested
tensordict with the same batch size. Numbers and arrays become tensors,
and other values such as strings are stored as non-tensor data (see
[Non-tensor data and unbatched entries](../../concepts/non_tensor.html#non-tensor-data)).

Parameters:

- **source** ([*dict*](https://docs.python.org/3/builtins/stdtypes.html#dict)*,**Mapping**or*[*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*,**optional*) - the entries, read
like a `dict`: string or nested keys to tensors, tensordicts,
nested dicts or other values. Exclusive with `**kwargs`.
Defaults to `None`, which gives an empty tensordict that can be
filled later.
- **batch_size** ([*Size*](https://docs.pytorch.org/docs/stable/size.html#torch.Size)*,*[*list*](https://docs.python.org/3/builtins/stdtypes.html#list)*of*[*int*](https://docs.python.org/3/builtins/functions.html#int)*or*[*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the batch size
of the tensordict: the leading dimensions that all its entries
share. It can be changed later if the entries allow it. Defaults
to `None`, which gives the batch size `[]`.
- **device** ([*device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,*[*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*or*[*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the device of the
tensordict. Entries on another device are moved to it. Defaults to
`None`, which leaves each entry where it is, and the tensordict
has no device.
- **names** ([*list*](https://docs.python.org/3/builtins/stdtypes.html#list)*of*[*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - the names of the batch dimensions, one
per dimension. Defaults to `None`, which leaves the dimensions
unnamed.
- **non_blocking** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - with `device`, if `True`, the
entries are copied to the device asynchronously with respect to
the host, with no synchronization: this is safe only from CPU to
CUDA. If `False`, every copy is synchronous. Defaults to
`None`, which copies asynchronously and synchronizes after the
copies when needed.
- **lock** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, the tensordict is locked: its
entries cannot be added, removed or replaced. See
[Locking and temporary changes](../../concepts/keys.html#keys-locking). Defaults to `False`.
- ****kwargs** - the entries, as keyword arguments, instead of `source`.

Raises:

- [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) - if both `source` and `**kwargs` are given, or if the
 length of `names` differs from that of `batch_size`.
- [**RuntimeError**](https://docs.python.org/3/builtins/exceptions.html#RuntimeError) - if the shape of an entry does not start with
 `batch_size`.

Examples

Name the batch dimensions. Shape operations keep the names:

```
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict(
... obs=torch.zeros(3, 4), action=torch.zeros(3, 2), batch_size=[3], names=["time"]
... )
>>> td.names
['time']
>>> td.unsqueeze(0).names
[None, 'time']
```

Set the batch size from the data:

```
>>> td = TensorDict(obs=torch.zeros(3, 4))
>>> td.batch_size
torch.Size([])
>>> td.auto_batch_size_().batch_size
torch.Size([3, 4])
```

Adding an entry to a locked tensordict raises an error:

```
>>> td = TensorDict(obs=torch.zeros(3), batch_size=[3], lock=True)
>>> td["action"] = torch.zeros(3)
Traceback (most recent call last):
 ...
RuntimeError: Cannot modify locked TensorDict. For in-place modification, consider using the `set_()` method and make sure the key is present.
```

*classmethod*from_module(*module: [Module](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)*, *as_module: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *lock: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *use_state_dict: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *filter_empty: [bool](https://docs.python.org/3/builtins/functions.html#bool) = True*)

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

A `TensorDict`, or a
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

## Tutorials that use `TensorDict`

![](../../_images/sphx_glr_data_fashion_thumb.png)

[Using TensorDict for datasets](../../tutorials/data_fashion.html)

Using TensorDict for datasets
![](../../_images/sphx_glr_export_thumb.png)

[Exporting tensordict modules](../../tutorials/export.html)

Exporting tensordict modules
![](../../_images/sphx_glr_functional_thumb.png)

[Functional calls and ensembles with TensorDict](../../tutorials/functional.html)

Functional calls and ensembles with TensorDict
![](../../_images/sphx_glr_serialization_speed_thumb.png)

[Benchmarking TensorDict serialization speed](../../tutorials/serialization_speed.html)

Benchmarking TensorDict serialization speed
![](../../_images/sphx_glr_streamed_tensordict_thumb.png)

[Building tensordicts from streams](../../tutorials/streamed_tensordict.html)

Building tensordicts from streams
![](../../_images/sphx_glr_tensorclass_fashion_thumb.png)

[Using tensorclasses for datasets](../../tutorials/tensorclass_fashion.html)

Using tensorclasses for datasets
![](../../_images/sphx_glr_tensorclass_imagenet_thumb.png)

[Batched data loading with tensorclasses](../../tutorials/tensorclass_imagenet.html)

Batched data loading with tensorclasses
![](../../_images/sphx_glr_tensordict_keys_thumb.png)

[Getting, setting, and nesting keys](../../tutorials/tensordict_keys.html)

Getting, setting, and nesting keys
![](../../_images/sphx_glr_tensordict_memory_thumb.png)

[Devices and memory-mapped tensors](../../tutorials/tensordict_memory.html)

Devices and memory-mapped tensors
![](../../_images/sphx_glr_tensordict_module_thumb.png)

[Building models with TensorDictModule](../../tutorials/tensordict_module.html)

Building models with TensorDictModule
![](../../_images/sphx_glr_tensordict_preallocation_thumb.png)

[Filling a TensorDict row by row](../../tutorials/tensordict_preallocation.html)

Filling a TensorDict row by row
![](../../_images/sphx_glr_tensordict_shapes_thumb.png)

[Changing the shape of a TensorDict](../../tutorials/tensordict_shapes.html)

Changing the shape of a TensorDict
![](../../_images/sphx_glr_tensordict_slicing_thumb.png)

[Slicing, indexing, and masking](../../tutorials/tensordict_slicing.html)

Slicing, indexing, and masking
![](../../_images/sphx_glr_zarr_storage_thumb.png)

[Storing TensorDicts in zarr](../../tutorials/zarr_storage.html)

Storing TensorDicts in zarr