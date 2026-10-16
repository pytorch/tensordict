# TensorClass

`TensorClass` has the methods of [`TensorDictBase`](tensordict.TensorDictBase.html#tensordict.TensorDictBase),
listed by task in [Methods by task](../td.html#api-tensordict-methods). This page lists only the
methods that `TensorClass` adds, and the overrides that have their own
documentation.

*class*tensordict.TensorClass(**args*, ***kwargs*)

Bases: `TensorCollection`

A base class for dataclass-like containers of tensors.

Subclass `TensorClass` and annotate fields, as with
[`dataclasses.dataclass()`](https://docs.python.org/3/library/dataclasses.html#dataclasses.dataclass). The instances behave like a
[`TensorDict`](tensordict.TensorDict.html#tensordict.TensorDict): they have a batch size, and they support
indexing, reshaping, `to(device)`, [`torch.stack()`](https://docs.pytorch.org/docs/stable/generated/torch.stack.html#torch.stack),
[`torch.cat()`](https://docs.pytorch.org/docs/stable/generated/torch.cat.html#torch.cat) and memory-mapping. A field can hold a tensor, a
tensordict, another tensorclass, or any Python object (stored as
[`NonTensorData`](tensordict.NonTensorData.html#tensordict.NonTensorData)). This class replaces the
[`tensorclass()`](tensordict.tensorclass.html#tensordict.tensorclass) decorator, which is kept for existing code.

Use a tensorclass instead of a [`TensorDict`](tensordict.TensorDict.html#tensordict.TensorDict) when the
data has a fixed set of named fields, when you want editors and type
checkers to know the fields, or when you want to add methods. Use a
[`TypedTensorDict`](tensordict.TypedTensorDict.html#tensordict.TypedTensorDict) instead if you need a
[`TensorDictBase`](tensordict.TensorDictBase.html#tensordict.TensorDictBase) subclass (`state["key"]`,
`**state` unpacking). [Which container should I use?](../../choosing_a_container.html#choosing-a-container) compares the
containers, and [TensorClass](../../tensorclass.html#tensorclass-guide) is the user guide.

Examples

```
>>> from typing import Any
>>> import torch
>>> from tensordict import TensorClass
>>> class Foo(TensorClass):
... tensor: torch.Tensor
... non_tensor: Any
... nested: Any = None
>>> foo = Foo(tensor=torch.randn(3), non_tensor="a string!", nested=None, batch_size=[3])
>>> print(foo)
Foo(
 non_tensor=NonTensorData(data=a string!, batch_size=torch.Size([3]), device=None),
 tensor=Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False),
 nested=None,
 batch_size=torch.Size([3]),
 device=None,
 is_shared=False)
>>> foo[0].non_tensor
'a string!'
>>> foo.reshape(1, 3).batch_size
torch.Size([1, 3])
```

Instances support the methods of [`TensorDictBase`](tensordict.TensorDictBase.html#tensordict.TensorDictBase)
(`apply`, `reshape`, `to`, `memmap`, ...). The pages of these
methods are under [`TensorDictBase`](tensordict.TensorDictBase.html#tensordict.TensorDictBase), because the
tensorclass forwards the call to the tensordict that holds its fields.

Keyword Arguments:

- **batch_size** ([*Size*](https://docs.pytorch.org/docs/stable/size.html#torch.Size)*,**optional*) - The batch size of the instance.
Defaults to `torch.Size([])`.
- **device** ([*device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,**optional*) - The device of the instance. Defaults
to `None`.
- **names** ([*list*](https://docs.python.org/3/builtins/stdtypes.html#list)*of*[*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - The names of the batch dimensions.
Defaults to `None`.
- **lock** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)*,**optional*) - If `True`, the instance is locked: writing a
field raises a `RuntimeError`. Defaults to `False`.

These keyword arguments go to the constructor, with the fields:
`Foo(tensor=..., batch_size=[3])`. The class flags below go to the
class definition instead.

Class flags

A flag changes the behavior of every instance of the class. Set it in
one of three equivalent forms:

```
>>> from tensordict import tensorclass
>>> class Foo(TensorClass["autocast"]): # bracket form
... integer: int
>>> class Foo(TensorClass, autocast=True): # keyword form
... integer: int
>>> @tensorclass(autocast=True) # legacy decorator form
... class Foo:
... integer: int
```

To combine flags, list them in the brackets:

```
>>> class Foo(TensorClass["nocast", "frozen"]):
... integer: int
```

The flags are:

- No flag: tensor-compatible values (`int`, `float`, `np.ndarray`)
are cast to tensors. Other values are stored as
[`NonTensorData`](tensordict.NonTensorData.html#tensordict.NonTensorData).

```
>>> class Foo(TensorClass):
... x: int
>>> Foo(x=1).x
tensor(1)
```
- `autocast`: when a field is set, the value is converted to the
annotated type. For example, an `int` field set to
`torch.ones(())` holds `1`.

```
>>> class Foo(TensorClass["autocast"]):
... x: int
>>> Foo(x=torch.ones(())).x
1
```
- `nocast`: tensor-compatible values are stored as they are, not cast
to tensors.

```
>>> class Foo(TensorClass["nocast"]):
... x: int
>>> Foo(x=1).x
1
```
- `tensor_only`: every field annotation must be a tensor type, a tensor
collection, or an optional or union of these. Other annotations raise
a `TypeError` when the class is defined. Values that can be cast to
a tensor (Python scalars, numpy arrays) are cast. Reading a field then
skips the lookup in the non-tensor storage.

```
>>> class Foo(TensorClass["tensor_only"]):
... x: torch.Tensor
>>> Foo(x=1).x
tensor(1)
```
- `frozen`: assigning a field raises `FrozenInstanceError`, as with
`@dataclass(frozen=True)`. Subclasses of a frozen class are frozen
too. A frozen class cannot subclass a non-frozen tensorclass: this
raises a `TypeError`.

```
>>> class Foo(TensorClass["frozen"]):
... x: torch.Tensor
>>> foo = Foo(x=torch.zeros(3))
>>> foo.x = torch.ones(3)
Traceback (most recent call last):
 ...
dataclasses.FrozenInstanceError: cannot assign to field 'x'
```
- `shadow`: by default, a field whose name is a tensorclass member
(`batch_size`, `device`, `names`, `sum`, `apply` and the
other TensorDict methods) raises an `AttributeError` when the class
is defined. `shadow=True` allows these names. The field then hides
the member of the same name.

```
>>> class Foo(TensorClass["shadow"]):
... sum: torch.Tensor
>>> Foo(sum=torch.ones(3), batch_size=[3]).sum
tensor([1., 1., 1.])
```
- `skip_post_init_on_rebuild`: `__post_init__` runs on the data that
comes in (the generated `__init__`, `from_tensordict`,
`from_dict`, the other constructors such as `from_any` or
`fromkeys`, and loads), but not when tensordict makes a new instance
from an existing one (indexing, `clone`, `to`, `apply`, stacking
and the other operations). Subclasses inherit the flag.

```
>>> class Foo(TensorClass["skip_post_init_on_rebuild"]):
... x: torch.Tensor
... def __post_init__(self):
... print("checked")
>>> foo = Foo(x=torch.zeros(3))
checked
>>> foo = foo.clone()
```

`autocast`, `nocast` and `tensor_only` are exclusive: combining two
of them raises a `TypeError`.

A class keyword that is not a flag, `cast_filter`, sets the default
`cast_filter` of [`to()`](tensordict.TensorDictBase.to.html#tensordict.TensorDictBase.to) and of the dtype
shorthands (`float()`, `bfloat16()`, ...): a predicate
`fn(key, value) -> bool` that selects the tensor leaves a dtype cast
applies to. A `cast_filter` passed to the call overrides it, and
`cast_filter=None` there casts every leaf. Subclasses inherit it. It
applies to calls on the tensorclass itself, not when the tensorclass is a
value of another tensordict or tensorclass that is cast.

```
>>> from tensordict import is_inexact
>>> class Mesh(TensorClass, cast_filter=is_inexact):
... points: torch.Tensor
... cells: torch.Tensor
>>> mesh = Mesh(points=torch.zeros(4, 3), cells=torch.zeros(2, 3, dtype=torch.long))
>>> mesh.bfloat16().points.dtype, mesh.bfloat16().cells.dtype
(torch.bfloat16, torch.int64)
```

Subclassing

A class with flags is a regular class. You can subclass it and add
flags:

```
>>> class Base(TensorClass["autocast"]):
... x: int
>>> class Sub(Base, shadow=True): # autocast inherited, shadow added
... y: float
```

Type checking

`TensorClass[...]` is implemented with [`__class_getitem__()`](https://docs.python.org/3/reference/datamodel.html#object.__class_getitem__),
so pyright resolves it to the class with flags, not to a generic
parameter. Mypy does not evaluate `__class_getitem__` in a list of base
classes and rejects the bracket form. In code that mypy checks, use the
keyword form: `class Foo(TensorClass, autocast=True)`. Editors complete
the annotated fields on instances.

Note

`TensorClass` itself is not a tensorclass: only its
subclasses become dataclasses. The reason is that a non-frozen base
cannot have a frozen subclass, and the base cannot know whether a
subclass sets `frozen`.

See also

- [`tensorclass()`](tensordict.tensorclass.html#tensordict.tensorclass) for the decorator form.
- [`from_dataclass()`](tensordict.from_dataclass.html#tensordict.from_dataclass) to migrate an existing `@dataclass` to a tensorclass.
- [`TensorClassModuleBase`](tensordict.nn.TensorClassModuleBase.html#tensordict.nn.TensorClassModuleBase) for typed `nn.Module` I/O backed by
tensorclasses.

*classmethod*from_dict(*input_dict*, ***, *auto_batch_size: [bool](https://docs.python.org/3/builtins/functions.html#bool) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *batch_size=None*, *device=None*, *batch_dims=None*)

Builds an instance of the tensorclass from a dictionary of its fields.

The fields that `input_dict` misses are set to `None`, also those that
declare a default or a default factory. Starting with TensorDict 0.17, the
default will change to filling these fields with their defaults, as the
constructor does. Until then, a call that misses such a field warns with a
[`FutureWarning`](https://docs.python.org/3/builtins/exceptions.html#FutureWarning).

Parameters:

**input_dict** ([*dict*](https://docs.python.org/3/builtins/stdtypes.html#dict)) - the values of the fields. Nested dictionaries become
tensordicts, or tensorclasses for the fields annotated with one.

Keyword Arguments:

- **auto_batch_size** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)*,**optional*) - whether to infer the batch size.
- **batch_size** ([*Size*](https://docs.pytorch.org/docs/stable/size.html#torch.Size)*,**optional*) - the batch size.
- **device** ([*device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,**optional*) - the device.
- **batch_dims** ([*int*](tensordict.TensorDictBase.int.html#tensordict.TensorDictBase.int)*,**optional*) - the number of batch dimensions to infer.

from_tensordict(*tensordict: [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*, *non_tensordict: [dict](https://docs.python.org/3/builtins/stdtypes.html#dict) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *safe: [bool](https://docs.python.org/3/builtins/functions.html#bool) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*) → Self

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

get(*key: [NestedKey](tensordict.NestedKey.html#tensordict.NestedKey)*, **args*, ***kwargs*)

Gets the value stored with the input key.

Parameters:

- **key** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,*[*tuple*](https://docs.python.org/3/builtins/stdtypes.html#tuple)*of*[*str*](https://docs.python.org/3/builtins/stdtypes.html#str)) - key to be queried. If tuple of str it is
equivalent to chained calls of getattr.
- **default** - default value if the key is not found in the tensorclass.

Returns:

value stored with the input key

load_memmap_(*prefix: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [Path](https://docs.python.org/3/library/pathlib.html#pathlib.Path)*, *robust_key: [bool](https://docs.python.org/3/builtins/functions.html#bool) | [None](https://docs.python.org/3/builtins/constants.html#None) = True*, ***, *allow_pickle: [bool](https://docs.python.org/3/builtins/functions.html#bool) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *mode: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*)

Loads the content of a memory-mapped tensorclass within the tensorclass where `load_memmap_` is called.

See [`load_memmap_()`](tensordict.TensorDictBase.load_memmap_.html#tensordict.TensorDictBase.load_memmap_) for more info.

load_state_dict(*state_dict: [dict](https://docs.python.org/3/builtins/stdtypes.html#dict)[[str](https://docs.python.org/3/builtins/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)]*, *strict=True*, *assign=False*, *from_flatten=None*)

Loads a state_dict into the tensorclass.

Supports both the new format (logical keys with `_metadata`) and the
legacy format (`_tensordict`/`_non_tensordict` wrapper keys).

set(*key: [NestedKey](tensordict.NestedKey.html#tensordict.NestedKey)*, *value: [Any](https://docs.python.org/3/library/typing.html#typing.Any)*, *inplace: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *non_blocking: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*)

Sets a new key-value pair.

Parameters:

- **key** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,*[*tuple*](https://docs.python.org/3/builtins/stdtypes.html#tuple)*of*[*str*](https://docs.python.org/3/builtins/stdtypes.html#str)) - name of the key to be set.
If tuple of str it is equivalent to chained calls of getattr
followed by a final setattr.
- **value** (*Any*) - value to be stored in the tensorclass
- **inplace** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, set will tentatively try to
update the value in-place. If `False` or if the key isn't present,
the value will be simply written at its destination.

Returns:

self

state_dict(*destination=None*, *prefix=''*, *keep_vars=False*, *flatten=True*) → [dict](https://docs.python.org/3/builtins/stdtypes.html#dict)[[str](https://docs.python.org/3/builtins/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)]

Returns a state_dict with logical keys, matching TensorDictBase conventions.

Tensor fields appear as data keys. Non-tensor fields (strings, ints, etc.)
and the tensorclass type are stored in `_metadata`. This replaces the
legacy `_tensordict`/`_non_tensordict` wrapper format.

to_tensordict(***, *retain_none: [bool](https://docs.python.org/3/builtins/functions.html#bool) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*) → [TensorDict](tensordict.TensorDict.html#tensordict.TensorDict)

Convert the tensorclass into a regular TensorDict.

Makes a copy of all entries. Memmap and shared memory tensors are converted to
regular tensors.

Keyword Arguments:

**retain_none** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)*,**optional*) - if `True`, fields set to `None` are
written as `None` entries. Defaults to `None`, which drops
them.

Returns:

A new TensorDict object containing the same values as the tensorclass.

Examples

```
>>> from typing import Any
>>> import torch
>>> from tensordict import TensorClass
>>> class Foo(TensorClass):
... x: torch.Tensor
... y: Any = None
>>> foo = Foo(x=torch.zeros(3))
>>> sorted(foo.to_tensordict().keys())
['x']
>>> sorted(foo.to_tensordict(retain_none=True).keys())
['x', 'y']
```

unbind(*dim: [int](https://docs.python.org/3/builtins/functions.html#int)*)

Returns a tuple of indexed tensorclass instances unbound along the indicated dimension.

Resulting tensorclass instances will share the storage of the initial tensorclass instance.

update_(*input_dict_or_td: [dict](https://docs.python.org/3/builtins/stdtypes.html#dict)[[str](https://docs.python.org/3/builtins/stdtypes.html#str), [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | TensorCollection] | T | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *clone: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *inplace: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, ***, *keys_to_update: [Sequence](https://docs.python.org/3/library/typing.html#typing.Sequence)[[NestedKey](tensordict.NestedKey.html#tensordict.NestedKey)] | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *non_blocking: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, ***kwargs*)

Writes the entries of a dictionary, a tensordict or a tensorclass into the existing entries.

See [`update_()`](tensordict.TensorDictBase.update_.html#tensordict.TensorDictBase.update_). A tensorclass or dict
source ignores `keys_to_update`: the entries that it leaves out are
written too, and the call warns. Starting with TensorDict 0.17, the default
will change to writing only the entries in `keys_to_update`, as for a
TensorDict source.

update_at_(*input_dict_or_td: [dict](https://docs.python.org/3/builtins/stdtypes.html#dict)[[str](https://docs.python.org/3/builtins/stdtypes.html#str), [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | TensorCollection] | T*, *idx: [None](https://docs.python.org/3/builtins/constants.html#None) | [int](https://docs.python.org/3/builtins/functions.html#int) | [slice](https://docs.python.org/3/builtins/functions.html#slice) | [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | [List](https://docs.python.org/3/library/typing.html#typing.List)[[Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[Any](https://docs.python.org/3/library/typing.html#typing.Any), ...]*, *clone: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, ***, *keys_to_update: [Sequence](https://docs.python.org/3/library/typing.html#typing.Sequence)[[NestedKey](tensordict.NestedKey.html#tensordict.NestedKey)] | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *non_blocking: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*)

Writes the entries of a dictionary, a tensordict or a tensorclass in place at an index.

See [`update_at_()`](tensordict.TensorDictBase.update_at_.html#tensordict.TensorDictBase.update_at_). A tensorclass or dict
source that [`update()`](tensordict.TensorDictBase.update.html#tensordict.TensorDictBase.update) accepts, such as one with the batch size of
the tensorclass, goes to [`update()`](tensordict.TensorDictBase.update.html#tensordict.TensorDictBase.update) instead: its entries replace
those of the tensorclass whatever the index, and the call warns. Starting
with TensorDict 0.17, the default will change to writing such a source in
place at the index, as [`update_at_()`](tensordict.TensorDictBase.update_at_.html#tensordict.TensorDictBase.update_at_)
does.

## Tutorials that use `TensorClass`

![](../../_images/sphx_glr_tensorclass_fashion_thumb.png)

[Using tensorclasses for datasets](../../tutorials/tensorclass_fashion.html)

Using tensorclasses for datasets
![](../../_images/sphx_glr_tensorclass_imagenet_thumb.png)

[Batched data loading with tensorclasses](../../tutorials/tensorclass_imagenet.html)

Batched data loading with tensorclasses
![](../../_images/sphx_glr_tensordict_shapes_thumb.png)

[Changing the shape of a TensorDict](../../tutorials/tensordict_shapes.html)

Changing the shape of a TensorDict
![](../../_images/sphx_glr_tensordict_slicing_thumb.png)

[Slicing, indexing, and masking](../../tutorials/tensordict_slicing.html)

Slicing, indexing, and masking