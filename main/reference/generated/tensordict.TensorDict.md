# TensorDict

`TensorDict` has the methods of [`TensorDictBase`](tensordict.TensorDictBase.html#tensordict.TensorDictBase),
listed by task in [Methods by task](../td.html#api-tensordict-methods). This page lists only the
methods that `TensorDict` adds, and the overrides that have their own
documentation.

*class*tensordict.TensorDict(*source: T | [Mapping](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[NestedKey](tensordict.NestedKey.html#tensordict.NestedKey), [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | TensorCollection] | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *batch_size: [Sequence](https://docs.python.org/3/library/typing.html#typing.Sequence)[[int](https://docs.python.org/3/builtins/functions.html#int)] | [Size](https://docs.pytorch.org/docs/stable/size.html#torch.Size) | [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *device: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *names: [Sequence](https://docs.python.org/3/library/typing.html#typing.Sequence)[[str](https://docs.python.org/3/builtins/stdtypes.html#str)] | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *non_blocking: [bool](https://docs.python.org/3/builtins/functions.html#bool) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *lock: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, ***kwargs: [Any](https://docs.python.org/3/library/typing.html#typing.Any)*)

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
not set, the entries can be on different devices.
- A key is a string, or a tuple of strings for a nested entry.

Parameters:

- **source** (*TensorDict**or**Mapping**[*[*NestedKey*](tensordict.NestedKey.html#tensordict.NestedKey)*,**Union**[**Tensor**,*[*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*]**]*) - a
data source, such as a `dict` or any other
[`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping), read like a `dict`. If empty,
the tensordict can be populated subsequently.
A `TensorDict` can also be built via a sequence of keyword arguments,
as it is the case for `dict(...)`.
- **batch_size** (*iterable**of*[*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - a batch size for the
tensordict. The batch size can be modified subsequently as long
as it is compatible with its content.
If no batch size is provided, an empty batch size is assumed (it
is not inferred automatically from the data). To automatically set
the batch size, refer to [`auto_batch_size_()`](tensordict.TensorDictBase.auto_batch_size_.html#tensordict.TensorDictBase.auto_batch_size_).
- **device** ([*device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*or**compatible type**,**optional*) - a device for the
TensorDict. If provided, all tensors will be stored on that device.
If not, tensors on different devices are allowed.
- **names** ([*list*](https://docs.python.org/3/builtins/stdtypes.html#list)*of*[*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - the names of the dimensions of the
tensordict. If provided, its length must match the one of the
`batch_size`. Defaults to `None` (no dimension name, or `None`
for every dimension).
- **non_blocking** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True` and a device is passed, the
entries are moved to the device without synchronization. This is
safe only from CPU to CUDA. In other cases, you must synchronize
yourself. If `False`, every move is synchronous. If `None`
(default), the moves are asynchronous, and the constructor
synchronizes after them if required.
- **lock** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, the resulting tensordict will be
locked.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> source = {'random': torch.randn(3, 4),
... 'zeros': torch.zeros(3, 4, 5)}
>>> batch_size = [3]
>>> td = TensorDict(source, batch_size=batch_size)
>>> print(td.shape) # equivalent to td.batch_size
torch.Size([3])
>>> td_unsqueeze = td.unsqueeze(-1)
>>> print(td_unsqueeze.get("zeros").shape)
torch.Size([3, 1, 4, 5])
>>> print(td_unsqueeze[0].shape)
torch.Size([1])
>>> print(td_unsqueeze.view(-1).shape)
torch.Size([3])
>>> print((td.clone()==td).all())
True
>>> td_add = td.apply(lambda x, y: x + y, td)
>>> (td_add["random"] == 2 * td["random"]).all()
tensor(True)
```

*classmethod*from_module(*module: [Module](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)*, *as_module: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *lock: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *use_state_dict: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *filter_empty: [bool](https://docs.python.org/3/builtins/functions.html#bool) = True*)

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

a `TensorDict`, or a
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