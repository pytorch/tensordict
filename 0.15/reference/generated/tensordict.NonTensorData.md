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
[`NonTensorStack`](tensordict.NonTensorStack.html#tensordict.NonTensorStack) with one value per element. To keep
one value for the whole batch through stacking and unsqueezing, use
[`MetaData`](tensordict.MetaData.html#tensordict.MetaData) instead. [Non-tensor data and unbatched entries](../../concepts/non_tensor.html#non-tensor-data) explains
the non-tensor classes.

You rarely build a `NonTensorData` yourself. Assigning a value
that is not a tensor to a key creates one, and reading the key returns
the value. [`get()`](tensordict.TensorDictBase.get.html#tensordict.TensorDictBase.get) returns the
`NonTensorData` itself:

```
>>> import torch
>>> from tensordict import NonTensorData, NonTensorStack, TensorDict
>>> td = TensorDict({"a": torch.zeros(3)}, batch_size=[3])
>>> td["b"] = "a string!"
>>> td["b"]
'a string!'
>>> td[0]["b"]
'a string!'
>>> td.get("b")
NonTensorData(data=a string!, batch_size=torch.Size([3]), device=None)
```

You can also build one explicitly, or with
[`set_non_tensor()`](tensordict.TensorDictBase.set_non_tensor.html#tensordict.TensorDictBase.set_non_tensor):

```
>>> non_tensor = NonTensorData("a string!", batch_size=[])
>>> data = TensorDict({}, batch_size=[3])
>>> _ = data.set_non_tensor(("nested", "key"), "a string!")
>>> assert isinstance(data.get(("nested", "key")), NonTensorData)
>>> assert data.get_non_tensor(("nested", "key")) == "a string!"
```

If the data passed to `NonTensorData` is a `NonTensorData`
itself, the data from the nested object is used:

```
>>> non_tensor = NonTensorData(NonTensorData("a string!"))
>>> assert non_tensor.data == "a string!"
```

Note

If you assign a list to a key, each element of the list becomes the
value of one batch element, in a [`NonTensorStack`](tensordict.NonTensorStack.html#tensordict.NonTensorStack).
Reading the key returns a [`LinkedList`](tensordict.utils.LinkedList.html#tensordict.utils.LinkedList) of
the values:

```
>>> td = TensorDict(batch_size=(3,))
>>> td["foo"] = ["a", "b", "c"]
>>> td["foo"]
LinkedList(['a', 'b', 'c'])
>>> td.get("foo").tolist()
['a', 'b', 'c']
```

Note

Unlike other tensorclass classes, `NonTensorData` supports
comparisons of two non-tensor data through [`__eq__()`](tensordict.TensorDictBase.__eq__.html#tensordict.TensorDictBase.__eq__), [`__ne__()`](tensordict.TensorDictBase.__ne__.html#tensordict.TensorDictBase.__ne__),
[`__xor__()`](tensordict.TensorDictBase.__xor__.html#tensordict.TensorDictBase.__xor__) or [`__or__()`](tensordict.TensorDictBase.__or__.html#tensordict.TensorDictBase.__or__). These operations return a tensor
of shape batch_size. [`__eq__()`](tensordict.TensorDictBase.__eq__.html#tensordict.TensorDictBase.__eq__) and [`__ne__()`](tensordict.TensorDictBase.__ne__.html#tensordict.TensorDictBase.__ne__) also compare
with a [`NonTensorStack`](tensordict.NonTensorStack.html#tensordict.NonTensorStack) of the same batch size, element
by element when their values can be compared, unless the data is an array
or a tensor. For compatibility with `<a tensordict> == <float_number>`,
a comparison with any other object returns an empty
[`TensorDict`](tensordict.TensorDict.html#tensordict.TensorDict).

```
>>> a = NonTensorData(True)
>>> b = NonTensorData(True)
>>> assert a == b
>>> assert not (a != b)
>>> assert not (a ^ b)
>>> assert a | b
>>> # The output is a tensor of shape batch-size
>>> a = NonTensorData(True, batch_size=[3])
>>> b = NonTensorData(True, batch_size=[3])
>>> print(a == b)
tensor([True, True, True])
```

Note

Stacking `NonTensorData` instances gives a
[`NonTensorStack`](tensordict.NonTensorStack.html#tensordict.NonTensorStack) instance. The data is not copied
during stacking, expansion and similar operations. To keep a single
value instead of a stack, use [`MetaData`](tensordict.MetaData.html#tensordict.MetaData).

```
>>> stack = torch.stack([NonTensorData(1) for _ in range(10)])
>>> stack
NonTensorStack(
 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
 batch_size=torch.Size([10]),
 device=None)
```

Note

Non-tensor data can be filtered out from a tensordict using
`filter_non_tensor()`.

Saving

[`memmap()`](tensordict.TensorDictBase.memmap.html#tensordict.TensorDictBase.memmap) and the related methods save
a `NonTensorData`, and [`load_memmap()`](tensordict.TensorDictBase.load_memmap.html#tensordict.TensorDictBase.load_memmap)
loads it. If JSON can encode the value, it goes to the `meta.json`
file of the entry (a tuple is loaded back as a list). Other values, such
as a `set` or an instance of another class, are pickled to an
`other.pickle` file. The classes `dict`, `collections.OrderedDict`,
`list` and `tuple`, which [`from_pytree()`](tensordict.TensorDictBase.from_pytree.html#tensordict.TensorDictBase.from_pytree)
records, are pickled for older versions and also saved by name, and
[`load_memmap()`](tensordict.TensorDictBase.load_memmap.html#tensordict.TensorDictBase.load_memmap) reads them without pickle.
Pickle can execute arbitrary code when it loads, so
[`load_memmap()`](tensordict.TensorDictBase.load_memmap.html#tensordict.TensorDictBase.load_memmap) warns when it loads a
pickled value. Pass `allow_pickle=True` for data that you trust. From
TensorDict 0.16, the default is `allow_pickle=False`, which raises a
`RuntimeError` for pickled values.

```
>>> import tempfile
>>> from fractions import Fraction
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
becomes a [`NonTensorStack`](tensordict.NonTensorStack.html#tensordict.NonTensorStack):

```
>>> td = TensorDict({"val": NonTensorData(data=0, batch_size=[10])}, [10])
>>> print(td)
TensorDict(
 fields={
 val: NonTensorData(data=0, batch_size=torch.Size([10]), device=None)},
 batch_size=torch.Size([10]),
 device=None,
 is_shared=False)
>>> print(td["val"])
0
>>> newdata = TensorDict({"val": NonTensorData(data=1, batch_size=[5])}, [5])
>>> td[1::2] = newdata
>>> print(td)
TensorDict(
 fields={
 val: NonTensorStack(
 [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
 batch_size=torch.Size([10]),
 device=None)},
 batch_size=torch.Size([10]),
 device=None,
 is_shared=False)
>>> td["val"]
LinkedList([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
```

If the value is the same for every element, `td[key]` returns that
value. If the entry is a [`NonTensorStack`](tensordict.NonTensorStack.html#tensordict.NonTensorStack),
`td[key]` returns the values as a [`LinkedList`](tensordict.utils.LinkedList.html#tensordict.utils.LinkedList).
The reason is that a single `NonTensorData` with a batch size
carries metadata for the whole tensordict, and a
[`NonTensorStack`](tensordict.NonTensorStack.html#tensordict.NonTensorStack) carries one value per batch element.

Sharing between processes

[`memmap_()`](tensordict.TensorDictBase.memmap_.html#tensordict.TensorDictBase.memmap_) and
[`share_memory_()`](tensordict.TensorDictBase.share_memory_.html#tensordict.TensorDictBase.share_memory_) make a
`NonTensorData` that other processes can read. To write a new
value, use [`update()`](tensordict.TensorDictBase.update.html#tensordict.TensorDictBase.update) with
`inplace=True`, [`update_()`](tensordict.TensorDictBase.update_.html#tensordict.TensorDictBase.update_) or
[`update_at_()`](tensordict.TensorDictBase.update_at_.html#tensordict.TensorDictBase.update_at_). Assigning the key raises
an error, because the shared tensordict is locked:

```
>>> td = TensorDict({"val": NonTensorData(data=0, batch_size=[])}, [])
>>> _ = td.share_memory_()
>>> _ = td.update_(TensorDict({"val": NonTensorData(data=1, batch_size=[])}, []))
>>> _ = td.update(TensorDict({"val": NonTensorData(data=1, batch_size=[])}, []), inplace=True)
>>> td["val"] = 1
Traceback (most recent call last):
 ...
RuntimeError: Cannot modify locked TensorDict. For in-place modification, consider using the `set_()` method and make sure the key is present.
```

A shared `NonTensorData` is writable if its value is a `str`,
`int`, `float`, `bool`, `dict` or `list`. Other values (a
dataclass instance, for example) do not raise an error in `memmap_` or
`share_memory_`, but writing a new value raises a
`NotImplementedError`:

```
>>> from dataclasses import dataclass
>>> @dataclass
... class MyClass:
... string: str
>>> td = TensorDict({"val": MyClass("a string!")}, [])
>>> _ = td.share_memory_()
>>> _ = td.update_(TensorDict({"val": MyClass("another string!")}, []))
Traceback (most recent call last):
 ...
NotImplementedError: Updating MyClass within a shared/memmaped structure is not supported.
```

A shared [`NonTensorStack`](tensordict.NonTensorStack.html#tensordict.NonTensorStack) is writable too, if it was
preallocated: the entry must be a stack before the tensordict is
shared.

```
>>> td = TensorDict({"val": NonTensorData(data=0, batch_size=[10])}, [10])
>>> td[1::2] = TensorDict({"val": NonTensorData(data=1, batch_size=[5])}, [5])
>>> _ = td.share_memory_()
>>> td[1::2] = TensorDict({"val": NonTensorData(data=2, batch_size=[5])}, [5])
>>> td["val"]
LinkedList([0, 2, 0, 2, 0, 2, 0, 2, 0, 2])
```

If the entry is still a single `NonTensorData` when the
tensordict is shared, writing different values to some elements raises
an error:

```
>>> td = TensorDict({"val": NonTensorData(data=0, batch_size=[10])}, [10])
>>> _ = td.share_memory_()
>>> td[1::2] = TensorDict({"val": NonTensorData(data=2, batch_size=[5])}, [5])
Traceback (most recent call last):
 ...
RuntimeError: You're attempting to update a leaf in-place with a shared tensordict, but the new value does not match the previous. If you're using NonTensorData, see the class documentation to see how to properly pre-allocate memory in shared contexts.
```

A memory-mapped `NonTensorData` writes the new value to its JSON
file on disk at each update, which takes time. If the data does not need
to stay on disk, use `share_memory_` instead. A value that JSON can
encode stays in the JSON file, so the directory loads with
`allow_pickle=False`:

```
>>> with tempfile.TemporaryDirectory() as tmpdir:
... td = TensorDict({"val": NonTensorData(data=0, batch_size=[])}, [])
... _ = td.memmap_(tmpdir)
... _ = td.update_(TensorDict({"val": NonTensorData(data=1, batch_size=[])}, []))
... td_load = TensorDict.load_memmap(tmpdir, allow_pickle=False)
... print(td_load["val"])
1
```

Callables

A `NonTensorData` can hold a callable. Calling the
`NonTensorData` calls its `data`:

```
>>> td0 = TensorDict({"a": 0, "b": 0})
>>> td1 = TensorDict({"a": 1, "b": 1})
>>> td_func = TensorDict({"a": lambda x, y: x-y, "b": lambda x, y: x+y})
>>> td = td0.apply(lambda x, y, func: func(x, y), td1, td_func)
>>> assert td["a"] == -1
>>> assert td["b"] == 1
```

expand(**args*, ***kwargs*) → T

Expands the batch dims of the NonTensorData to the given shape, as [`torch.Tensor.expand()`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.expand.html#torch.Tensor.expand) does.

A shape that [`torch.Tensor.expand()`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.expand.html#torch.Tensor.expand) rejects (fewer sizes than
batch dims, or a size that differs from a batch dim that is not 1)
warns, and the batch size of the result can differ from the shape.
Starting with TensorDict 0.17, the default will change to raising a
RuntimeError for such shapes, as [`TensorDict.expand()`](tensordict.TensorDictBase.expand.html#tensordict.TensorDictBase.expand) does.

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
and `out` is left as it is. [`TensorDict.where()`](tensordict.TensorDictBase.where.html#tensordict.TensorDictBase.where) then writes the
stack in place of the entry of its own `out`. See
[`TensorDictBase.where()`](tensordict.TensorDictBase.where.html#tensordict.TensorDictBase.where) for the arguments.

Examples

```
>>> import torch
>>> from tensordict import NonTensorData
>>> a = NonTensorData("a", batch_size=[3])
>>> b = NonTensorData("b", batch_size=[3])
>>> a.where(torch.tensor([True, False, True]), b).tolist()
['a', 'b', 'a']
```