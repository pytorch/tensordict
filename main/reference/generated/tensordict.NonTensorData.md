# NonTensorData

`NonTensorData` has the methods of [`TensorDictBase`](tensordict.TensorDictBase.html#tensordict.TensorDictBase),
listed by task in [Methods by task](../td.html#api-tensordict-methods). This page lists only the
methods that `NonTensorData` adds, and the overrides that have their own
documentation.

*class*tensordict.NonTensorData(*data: [Any](https://docs.python.org/3/library/typing.html#typing.Any)*, *_metadata: [dict](https://docs.python.org/3/builtins/stdtypes.html#dict) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *_is_non_tensor: [bool](https://docs.python.org/3/builtins/functions.html#bool) = True*, ***, *batch_size*, *device=None*, *names=None*)

Bases: [`NonTensorDataBase`](tensordict.NonTensorDataBase.html#tensordict.NonTensorDataBase)

A tensordict entry that holds a Python object instead of a tensor.

Use it to keep strings, numbers that must not become tensors, or any
other object next to the tensors of a tensordict: a task name, a file
path, a configuration. Each batch element has the value: indexing a
tensordict keeps the value in every element, and stacking gives a
[`NonTensorStack`](tensordict.NonTensorStack.html#tensordict.NonTensorStack) with one value per element. The value
is not copied by stacking, expanding and similar operations. To keep one
value for the whole batch through stacking and unsqueezing, use
[`MetaData`](tensordict.MetaData.html#tensordict.MetaData) instead. [Non-tensor data and unbatched entries](../../concepts/non_tensor.html#non-tensor-data) explains
the non-tensor classes.

Parameters:

**data** (*Any*) - the value. A `NonTensorData` is unwrapped: its
value is used.

Keyword Arguments:

- **batch_size** ([*Size*](https://docs.pytorch.org/docs/stable/size.html#torch.Size)*or*[*list*](https://docs.python.org/3/builtins/stdtypes.html#list)*of*[*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the batch size of
the entry, usually that of the tensordict that holds it. Defaults
to `torch.Size([])`.
- **device** ([*device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,*[*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*or*[*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the device of the entry.
The value itself is not moved. Defaults to `None`.
- **names** ([*list*](https://docs.python.org/3/builtins/stdtypes.html#list)*of*[*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - the names of the batch dimensions, one
per dimension. Defaults to `None`.

You rarely build a `NonTensorData` yourself. Assigning a value
that is not a tensor to a key creates one, and reading the key returns
the value. [`get()`](tensordict.TensorDictBase.get.html#tensordict.TensorDictBase.get) returns the
`NonTensorData` itself:

```
>>> import torch
>>> from tensordict import NonTensorData, TensorDict
>>> td = TensorDict({"a": torch.zeros(3)}, batch_size=[3])
>>> td["b"] = "a string!"
>>> td["b"]
'a string!'
>>> td[0]["b"]
'a string!'
>>> td.get("b")
NonTensorData(data=a string!, batch_size=torch.Size([3]), device=None)
```

[`set_non_tensor()`](tensordict.TensorDictBase.set_non_tensor.html#tensordict.TensorDictBase.set_non_tensor) and
[`get_non_tensor()`](tensordict.TensorDictBase.get_non_tensor.html#tensordict.TensorDictBase.get_non_tensor) write and read the value
of a nested key. Assigning a list to a key gives a
[`NonTensorStack`](tensordict.NonTensorStack.html#tensordict.NonTensorStack) with one item per batch element
instead. `filter_non_tensor()` removes the
non-tensor entries of a tensordict.

Comparisons

Unlike other tensorclasses, `NonTensorData` supports `==`,
`!=`, `^` and `|` between two non-tensor entries. The result is a
boolean tensor with the batch size of the entry. `==` and `!=` also
compare with a [`NonTensorStack`](tensordict.NonTensorStack.html#tensordict.NonTensorStack) of the same batch
size, element by element, unless the value is an array or a tensor. A
comparison with any other object returns an empty
[`TensorDict`](tensordict.TensorDict.html#tensordict.TensorDict), for compatibility with
`<a tensordict> == <float_number>`.

```
>>> a = NonTensorData(True, batch_size=[3])
>>> b = NonTensorData(True, batch_size=[3])
>>> a == b
tensor([True, True, True])
>>> a ^ b
tensor([False, False, False])
```

Saving

[`memmap()`](tensordict.TensorDictBase.memmap.html#tensordict.TensorDictBase.memmap) and the related methods save
a `NonTensorData`, and [`load_memmap()`](tensordict.TensorDictBase.load_memmap.html#tensordict.TensorDictBase.load_memmap)
loads it. A value that JSON can encode goes to the `meta.json` file of
the entry (a tuple is loaded back as a list). Other values, such as a
`set` or an instance of another class, are pickled to an
`other.pickle` file. Pickle can execute arbitrary code when it loads,
so pass `allow_pickle=True` to
[`load_memmap()`](tensordict.TensorDictBase.load_memmap.html#tensordict.TensorDictBase.load_memmap) only for data that you
trust. Without it, loading a pickled value warns. Starting with
TensorDict 0.16, the default will change to `allow_pickle=False`,
which raises a `RuntimeError` for pickled values. The `dict`,
`collections.OrderedDict`, `list` and `tuple` classes that
[`from_pytree()`](tensordict.TensorDictBase.from_pytree.html#tensordict.TensorDictBase.from_pytree) records are also saved by
name, and load without pickle.

```
>>> import tempfile
>>> from fractions import Fraction
>>> data = TensorDict(batch_size=[3])
>>> _ = data.set_non_tensor("picklable", Fraction(1, 3))
>>> with tempfile.TemporaryDirectory() as tmpdir:
... _ = data.memmap(tmpdir)
... loaded = TensorDict.load_memmap(tmpdir, allow_pickle=True)
>>> loaded.get_non_tensor("picklable")
Fraction(1, 3)
```

Preallocation

A tensordict can hold one `NonTensorData` with a batch size, then
receive different values for some batch elements. The entry then
becomes a [`NonTensorStack`](tensordict.NonTensorStack.html#tensordict.NonTensorStack), and `td[key]` returns
a `tensordict.utils.LinkedList` of the values instead of one value:

```
>>> td = TensorDict({"val": NonTensorData(data=0, batch_size=[10])}, [10])
>>> td["val"]
0
>>> td[1::2] = TensorDict({"val": NonTensorData(data=1, batch_size=[5])}, [5])
>>> type(td.get("val")).__name__
'NonTensorStack'
>>> td["val"]
LinkedList([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
```

Sharing between processes

[`memmap_()`](tensordict.TensorDictBase.memmap_.html#tensordict.TensorDictBase.memmap_) and
[`share_memory_()`](tensordict.TensorDictBase.share_memory_.html#tensordict.TensorDictBase.share_memory_) make a
`NonTensorData` that other processes can read. The shared
tensordict is locked, so assigning the key raises an error. To write a
new value, use [`update_()`](tensordict.TensorDictBase.update_.html#tensordict.TensorDictBase.update_),
[`update_at_()`](tensordict.TensorDictBase.update_at_.html#tensordict.TensorDictBase.update_at_) or
[`update()`](tensordict.TensorDictBase.update.html#tensordict.TensorDictBase.update) with `inplace=True`.
`share_memory_` starts a `multiprocessing.Manager` process for
the values, so these examples need a process that can start child
processes.

```
>>> td = TensorDict({"val": NonTensorData(data=0, batch_size=[])}, [])
>>> _ = td.share_memory_() 
>>> _ = td.update_(TensorDict({"val": NonTensorData(data=1)}, [])) 
>>> td["val"] 
1
>>> td["val"] = 2 
Traceback (most recent call last):
 ...
RuntimeError: Cannot modify locked TensorDict. For in-place modification, consider using the `set_()` method and make sure the key is present.
```

A shared value can be written if it is a `str`, `int`, `float`,
`bool`, `dict` or `list`. For a value of another type (a
dataclass instance, for example), `update_` skips the entry and warns.
A shared [`NonTensorStack`](tensordict.NonTensorStack.html#tensordict.NonTensorStack) can be written too, if the
entry was a stack before the tensordict was shared:

```
>>> td = TensorDict({"val": NonTensorData(data=0, batch_size=[10])}, [10])
>>> td[1::2] = TensorDict({"val": NonTensorData(data=1, batch_size=[5])}, [5])
>>> _ = td.share_memory_() 
>>> td[1::2] = TensorDict({"val": NonTensorData(data=2, batch_size=[5])}, [5]) 
>>> td["val"] 
LinkedList([0, 2, 0, 2, 0, 2, 0, 2, 0, 2])
```

If it is still a single `NonTensorData` when the tensordict is
shared, writing different values to some elements raises a
`RuntimeError`.

A memory-mapped `NonTensorData` writes the new value to its JSON
file at each update, which takes time. If the data does not need to stay
on disk, use `share_memory_` instead:

```
>>> with tempfile.TemporaryDirectory() as tmpdir:
... td = TensorDict({"val": NonTensorData(data=0, batch_size=[])}, [])
... _ = td.memmap_(tmpdir)
... _ = td.update_(TensorDict({"val": NonTensorData(data=1, batch_size=[])}, []))
... print(TensorDict.load_memmap(tmpdir, allow_pickle=False)["val"])
1
```

Callables

Calling a `NonTensorData` calls its value. This lets
[`apply()`](tensordict.TensorDictBase.apply.html#tensordict.TensorDictBase.apply) take a tensordict of functions:

```
>>> x = TensorDict({"a": 1, "b": 1})
>>> funcs = TensorDict({"a": lambda v: v - 1, "b": lambda v: v + 1})
>>> out = x.apply(lambda v, func: func(v), funcs)
>>> out["a"], out["b"]
(tensor(0), tensor(2))
```

expand(**args*, ***kwargs*) → T

Returns the entry expanded to a new batch size, as [`torch.Tensor.expand()`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.expand.html#torch.Tensor.expand) does.

If the batch size changes, the result is a
[`NonTensorStack`](tensordict.NonTensorStack.html#tensordict.NonTensorStack) whose elements share the value.
A shape that [`torch.Tensor.expand()`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.expand.html#torch.Tensor.expand) rejects (fewer sizes than
batch dimensions, or a size that differs from a batch dimension that
is not 1) warns, and the batch size of the result can differ from the
shape. Starting with TensorDict 0.17, the default will change to
raising a RuntimeError for such shapes, as
[`expand()`](tensordict.TensorDictBase.expand.html#tensordict.TensorDictBase.expand) does.

Parameters:

***shape** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*or*[*Size*](https://docs.pytorch.org/docs/stable/size.html#torch.Size)) - the new batch size. `-1` keeps the
size of a batch dimension.

Examples

```
>>> from tensordict import NonTensorData
>>> expanded = NonTensorData("a", batch_size=[2]).expand(3, 2)
>>> expanded.batch_size, expanded.tolist()
(torch.Size([3, 2]), [['a', 'a'], ['a', 'a'], ['a', 'a']])
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

where(*condition: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*, *other: [Any](https://docs.python.org/3/library/typing.html#typing.Any)*, ***, *out: T | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *pad: [int](https://docs.python.org/3/builtins/functions.html#int) | [bool](https://docs.python.org/3/builtins/functions.html#bool) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *update_batch_size: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*) → T

Returns the elements of `self` where `condition` is `True`, and those of `other` elsewhere.

When `other` is a `NonTensorData` with another value, or a
[`NonTensorStack`](tensordict.NonTensorStack.html#tensordict.NonTensorStack), the result is a [`NonTensorStack`](tensordict.NonTensorStack.html#tensordict.NonTensorStack) with
the value that `condition` picks for each element, or a
`NonTensorData` when all the elements have the same value. This
reads `condition`, which breaks the graph under [`torch.compile()`](https://docs.pytorch.org/docs/stable/generated/torch.compile.html#torch.compile).
Otherwise, the result has the value of `self` everywhere.

`out` receives the result if it can hold it. A `NonTensorData`
`out` cannot hold a [`NonTensorStack`](tensordict.NonTensorStack.html#tensordict.NonTensorStack): the stack is returned,
and `out` is left as it is. [`TensorDict.where`](tensordict.TensorDictBase.where.html#tensordict.TensorDictBase.where) then writes the stack in place of
the entry of its own `out`. See
[`where()`](tensordict.TensorDictBase.where.html#tensordict.TensorDictBase.where) for the keyword arguments
`out`, `pad` and `update_batch_size`.

Parameters:

- **condition** ([*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) - a boolean tensor with the batch size of
the entry.
- **other** (*Any*) - the non-tensor entry or tensor collection used where
`condition` is `False`.

Examples

```
>>> import torch
>>> from tensordict import NonTensorData
>>> a = NonTensorData("a", batch_size=[3])
>>> b = NonTensorData("b", batch_size=[3])
>>> a.where(torch.tensor([True, False, True]), b).tolist()
['a', 'b', 'a']
```