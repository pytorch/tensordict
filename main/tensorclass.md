# TensorClass: a dataclass of tensors

A *tensorclass* is a dataclass of tensors. You declare its fields with type
annotations, as in a [`dataclass()`](https://docs.python.org/3/library/dataclasses.html#dataclasses.dataclass), and it has the
operations that such a class otherwise writes by hand: `.to(device)`,
`pin_memory()`, slicing of every field with `batch[idx]`,
`torch.stack` and `torch.cat`, and a `collate_fn` for the
[`DataLoader`](https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) ([`collate()`](reference/generated/tensordict.collate.html#tensordict.collate)). It is
registered as a pytree, so `torch.utils._pytree`, [`torch.func`](https://docs.pytorch.org/docs/stable/func.api.html#module-torch.func) and
[`torch.compile()`](https://docs.pytorch.org/docs/stable/generated/torch.compile.html#torch.compile) accept it. You write its methods once: a method runs
on one item or on a batch of any shape, and compiles with
`torch.compile(fullgraph=True)`. The instances have the other operations of
a [`TensorDict`](reference/generated/tensordict.TensorDict.html#tensordict.TensorDict), such as reshaping and saving to disk.

```
>>> import torch
>>> from tensordict import TensorClass
>>> class Particles(TensorClass):
... pos: torch.Tensor # [*batch, 3]
... vel: torch.Tensor # [*batch, 3]
... mass: torch.Tensor # [*batch]
...
... def step(self, force: torch.Tensor, dt: float) -> "Particles":
... vel = self.vel + dt * force / self.mass.unsqueeze(-1)
... return Particles(
... pos=self.pos + dt * vel, vel=vel, mass=self.mass, batch_size=self.batch_size
... )
>>> _ = torch.manual_seed(0)
>>> systems = Particles( # 64 systems of 1,000 particles each
... pos=torch.randn(64, 1000, 3),
... vel=torch.zeros(64, 1000, 3),
... mass=torch.ones(64, 1000),
... batch_size=[64, 1000],
... )
>>> systems = systems.step(torch.randn(64, 1000, 3), dt=0.01) # all 64 systems
>>> one = systems[0].step(torch.randn(1000, 3), dt=0.01) # one system
>>> type(one).__name__, one.batch_size
('Particles', torch.Size([1000]))
>>> grid = systems.reshape(8, 8, 1000).step(torch.randn(8, 8, 1000, 3), dt=0.01)
>>> grid.batch_size
torch.Size([8, 8, 1000])
>>> step = torch.compile(Particles.step, fullgraph=True)
>>> force = torch.randn(64, 1000, 3)
>>> torch.allclose(step(systems, force, 0.01).pos, systems.step(force, 0.01).pos)
True
```

`step` has no loop over the systems and no shape argument. The *batch size*
(`[64, 1000]` here) is the leading dimensions that all the fields share (see
[Batch size, shapes, and indexing](concepts/batch_size.html#batch-size)). An index or a `reshape` applies to these dimensions and
returns a `Particles`. So the same method runs on one system or on a grid
of systems. The compiled `step` gives the same result as the eager
`step`. The class declares its fields, so your editor completes
`self.pos`, and a type checker reads the fields as the signature of the
constructor (see Static type checking).

Read the quick start and Methods first. Each later section
describes one feature: read it when you need that feature. To compare a
tensorclass with a `TensorDict` and the other containers, see
[Which container should I use?](choosing_a_container.html#choosing-a-container). If you have a dataclass with a hand-written
`to`, `pin_memory`, `__getitem__` or collate function,
[Coming from a hand-written dataclass](coming_from_dataclass.html#coming-from-dataclass) shows what replaces each one.

## Quick start

Subclass [`TensorClass`](reference/generated/tensordict.TensorClass.html#tensordict.TensorClass) and annotate each field. Then pass
the values and the batch size to the constructor:

```
>>> from typing import Optional
>>> class Reading(TensorClass):
... signal: torch.Tensor
... count: torch.Tensor
... sensor: str
... backup: Optional["Reading"] = None
>>> data = Reading(
... signal=torch.randn(3, 4, 5),
... count=torch.randint(10, (3, 4, 1)),
... sensor="lidar",
... batch_size=[3, 4], # for example, 3 robots and 4 time steps
... )
>>> print(data)
Reading(batch_size=[3,4], fields={
 count: i64[1],
 sensor: str,
 signal: f32[5],
 backup: None})
```

The `backup` annotation is a string because the class refers to itself.
Another way is `from __future__ import annotations` at the top of the
module. If you do not pass `batch_size`, it is `torch.Size([])`, and the
tensorclass has no batch dimensions.

Read and assign the fields as attributes. A field that is not a tensor
(`sensor` here) returns its plain value:

```
>>> data.sensor
'lidar'
>>> data.backup = Reading(
... signal=torch.randn(3, 4, 5),
... count=torch.randint(10, (3, 4, 1)),
... sensor="radar",
... batch_size=[3, 4],
... )
```

An index applies to the batch dimensions of every field, also in the nested
tensorclass. Fields that are not tensors keep their value:

```
>>> sub = data[:2]
>>> sub.batch_size
torch.Size([2, 4])
>>> sub.backup.signal.shape
torch.Size([2, 4, 5])
>>> sub.backup.sensor
'radar'
```

The shape and device operations of [`TensorDict`](reference/generated/tensordict.TensorDict.html#tensordict.TensorDict) work on a
tensorclass: `stack`, `cat`, `reshape`, `to`, `clone`, `apply`, and
the others. They return an instance of the same class:

```
>>> data2 = data.clone()
>>> cat_tc = torch.cat([data, data2], 0)
>>> type(cat_tc).__name__, cat_tc.batch_size
('Reading', torch.Size([6, 4]))
```

A tensorclass is not a [`TensorDictBase`](reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase) and not a
[`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping). A string index raises an error, because an
index always applies to the batch dimensions. To read a field by name, use
`get`:

```
>>> data["count"]
Traceback (most recent call last):
...
ValueError: Invalid indexing arguments: 'count'. A tensorclass is indexed along its batch dimensions; to read a field, use the attribute or get('count').
>>> data.get("count").shape
torch.Size([3, 4, 1])
```

Assigning a name that is not a field, as an attribute or with `set()`,
raises `AttributeError`. `set_non_tensor()`, `setdefault()`,
`create_nested()`, `rename_key_()`, `update()` with a `TensorDict`
or another tensorclass, and an indexed assignment of a `TensorDict`
(`data[idx] = td`) still store such a key, and warn with a
`DeprecationWarning`. Starting with TensorDict 0.17, they raise
`AttributeError` too.

To test whether an object is a tensordict or a tensorclass, use
[`is_tensor_collection()`](reference/generated/tensordict.is_tensor_collection.html#tensordict.is_tensor_collection). `to_tensordict()` returns a copy
of the data as a [`TensorDict`](reference/generated/tensordict.TensorDict.html#tensordict.TensorDict).

## Methods

A tensorclass is a Python class: you can add methods, properties, class
methods, and class constants to it. Of the dataclass features,
`__post_init__`, field defaults, `InitVar`, `field(repr=False)`, the
class option `kw_only`, and a custom `__init__` work, with the rules
below.

### Methods and properties

Methods and properties can read and assign the fields. A method that calls a
tensordict operation, such as `apply`, returns an instance of the class. A
property is not a field: it is not stored and not in `keys()`. Use
`typing.ClassVar` for a constant of the class. It is not a field
either:

```
>>> from typing import ClassVar
>>> class Logits(TensorClass):
... logits: torch.Tensor
... temperature: ClassVar[float] = 2.0
...
... @classmethod
... def uniform(cls, batch, n):
... return cls(logits=torch.zeros(batch, n), batch_size=[batch])
...
... @property
... def probs(self):
... return (self.logits / self.temperature).softmax(-1)
...
... @probs.setter
... def probs(self, value):
... self.logits = value.log() * self.temperature
...
... def detached(self):
... return self.apply(lambda t: t.detach())
>>> x = Logits.uniform(2, 4)
>>> x.probs[0]
tensor([0.2500, 0.2500, 0.2500, 0.2500])
>>> x.probs = torch.tensor([0.4, 0.2, 0.2, 0.2]).expand(2, 4)
>>> x.probs[0]
tensor([0.4000, 0.2000, 0.2000, 0.2000])
>>> list(x.keys())
['logits']
>>> type(x.detached()).__name__, x[0].probs.shape
('Logits', torch.Size([4]))
```

### Field defaults

A field can have a default, as in a dataclass. A tensor default, such as
`step: torch.Tensor = torch.tensor(0)`, is one tensor that every instance
shares, and the class too: an in-place change on one instance, such as
`obj.step.add_(1)`, changes the others and the default. The same holds for
a `TensorDict` or tensorclass default. Python 3.11 and later refuse most of
these, as they refuse a `list` default (a `TensorDict`, a tensorclass that
is not frozen), but accept a frozen tensorclass or a `TensorDictParams`
default; Python 3.10 accepts them all. Creating a class
with such a default emits a `FutureWarning` for each such field. Starting
with TensorDict 0.17, the default will change to giving each instance its own
copy (a clone) of the default. To get a copy for each instance now, use a
default factory:

```
>>> import dataclasses
>>> class State(TensorClass):
... step: torch.Tensor = dataclasses.field(
... default_factory=lambda: torch.tensor(0)
... )
>>> a, b = State(), State()
>>> a.step.add_(1)
tensor(1)
>>> b.step
tensor(0)
```

### `__post_init__`

As in a dataclass, `__post_init__` runs after the generated `__init__`.
Use it to compute a default value, or to make sure that the fields are valid:

```
>>> class Batch(TensorClass):
... obs: torch.Tensor
... mask: torch.Tensor = None
...
... def __post_init__(self):
... if self.mask is None:
... self.mask = torch.ones(self.batch_size, dtype=torch.bool)
>>> b = Batch(obs=torch.zeros(3, 2), batch_size=[3])
>>> b.mask
tensor([True, True, True])
```

`__post_init__` also runs each time tensordict rebuilds an instance, that
is, makes a new instance from the data of an existing one: an index,
`clone`, `to`, `torch.stack`, `apply`, and the other operations that
return a tensorclass. It runs as well on the data that
`from_tensordict`, `from_dict` and a load (`load_memmap`, `load`)
bring in. Do little work in it, and make sure that a second run gives the
same result. In the example above, the second run finds a mask and keeps it.

On a lazy stack ([`lazy_stack()`](reference/generated/tensordict.lazy_stack.html#tensordict.lazy_stack), and the operations that
return one), `__post_init__` runs on each stacked instance, not on the
stack, so it sees fields of the shape of that instance, even when the
instances have different shapes. A lazy stack of N instances makes N calls.
The fields that it sets go to the stacked instances. `torch.stack` makes
one call, on the stacked result.

`__post_init__` does not run on the shells that tensordict builds without
fields and fills afterwards: the result of `empty()`, the tensorclass
entries of `empty(recurse=True)`, and the output of
[`pad_sequence()`](reference/generated/tensordict.pad_sequence.html#tensordict.pad_sequence). A shell has no field, not even those that
`__post_init__` would fill. Under `torch.compile`, `__post_init__`
still runs on these shells.

To validate the data once, when it comes in, set the
`skip_post_init_on_rebuild` flag. `__post_init__` then runs in the
generated `__init__`, `from_tensordict`, `from_dict`, the other
constructors (`from_any`, `from_list`, `fromkeys`, ...), and when the
data is loaded or received from another process, but not when tensordict
rebuilds an instance. Each class follows
its own flag: a tensorclass without the flag that holds one with it runs only
its own `__post_init__` on `clone()`. Subclasses inherit the flag:

```
>>> class Ragged(TensorClass["skip_post_init_on_rebuild"]):
... offsets: torch.Tensor
... indices: torch.Tensor
...
... def __post_init__(self):
... if int(self.offsets[-1]) != len(self.indices):
... raise ValueError("the last offset must be len(indices)")
>>> r = Ragged(offsets=torch.tensor([0, 2, 3]), indices=torch.zeros(3))
>>> r.clone().offsets # no check, and no device synchronization on CUDA
tensor([0, 2, 3])
```

### Field options, `kw_only` and `InitVar`

A field declared with [`dataclasses.field()`](https://docs.python.org/3/library/dataclasses.html#dataclasses.field) keeps its `default`,
`default_factory` and `metadata`, and `repr=False` leaves it out of the
printed instance: the default, the compact and the notebook reprs. The class
option `kw_only=True` (`class Config(TensorClass, kw_only=True)`, or
`@tensorclass(kw_only=True)`) makes the fields of the class keyword-only, as
in a dataclass. The fields of a subclass take positional arguments, unless
the subclass sets the option too:

```
>>> class Config(TensorClass, kw_only=True):
... weights: torch.Tensor
... note: str = dataclasses.field(default="", repr=False)
>>> c = Config(weights=torch.zeros(2), note="tuned", batch_size=[2])
>>> "note" in repr(c), c.note
(False, 'tuned')
>>> Config(torch.zeros(2))
Traceback (most recent call last):
 ...
TypeError: Config.__init__() takes 1 positional arguments but 2 were given
```

The other options are ignored, and the class warns about each when it is
defined: `init=False` and `kw_only=True` on a field (`__init__` still
takes the field, by name or in order), and `compare=False` (`==` compares
the fields elementwise and returns a tensorclass).

Positional arguments go to the fields in the order they are declared, as in
earlier releases, whereas the signature of `__init__`, which [`help()`](https://docs.python.org/3/builtins/functions.html#help)
and IDEs show, leaves out the `init=False` and keyword-only fields, as in a
dataclass. A call that the field order makes raise and that the signature
allows follows the signature. A call that the signature maps differently
emits a `FutureWarning`: starting with TensorDict 0.17, the default will
change to positional arguments that follow the signature. Pass those fields by
keyword to get the same result in every release:

```
>>> class Sample(TensorClass):
... label: torch.Tensor = dataclasses.field(default=None, kw_only=True)
... x: torch.Tensor = None
>>> Sample(torch.ones(2), label=torch.zeros(2), batch_size=[2]).x
tensor([1., 1.])
>>> Sample(torch.ones(2), batch_size=[2]).label # FutureWarning
tensor([1., 1.])
>>> Sample(x=torch.ones(2), batch_size=[2]).label is None
True
```

A [`dataclasses.InitVar`](https://docs.python.org/3/library/dataclasses.html#dataclasses.InitVar) is an argument of the generated `__init__`
that is not a field: `__init__` passes it to `__post_init__`, as in a
dataclass. The other calls to `__post_init__` (rebuilds,
`from_tensordict`, loads) pass the defaults of the InitVars, so give each a
default, and set `skip_post_init_on_rebuild` if `__post_init__` must not
run again with the defaults:

```
>>> from dataclasses import InitVar
>>> class Scaled(TensorClass["skip_post_init_on_rebuild"]):
... x: torch.Tensor
... scale: InitVar[float] = 1.0
...
... def __post_init__(self, scale):
... self.x = self.x * scale
>>> s = Scaled(x=torch.ones(2), scale=3.0, batch_size=[2])
>>> s.x
tensor([3., 3.])
>>> s.clone().x # no second call to __post_init__
tensor([3., 3.])
```

This holds for a class whose `__post_init__` cannot be called without its
InitVars. If `__post_init__` gives them defaults of its own, it gets the
InitVars only when they are given, and the other calls pass none. Without
`__post_init__`, the InitVars are taken by name and dropped. In both
cases, positional arguments go to the fields in order, as above, and warn
where the signature, which takes the InitVars, maps them differently. A class
whose `__post_init__` does not take its InitVars does not accept them as
arguments.

### A custom `__init__`

If the class defines `__init__`, it replaces the generated one. When it
runs, the instance already exists, with the `batch_size`, `device`, and
`names` that the caller passed as keyword arguments. Assign the fields as
attributes. To set the batch size from the data, assign `self.batch_size`
after the fields:

```
>>> class Points(TensorClass):
... xyz: torch.Tensor
...
... def __init__(self, n: int):
... self.xyz = torch.zeros(n, 3)
... self.batch_size = [n]
>>> Points(4).batch_size
torch.Size([4])
>>> Points(4)[:2].batch_size
torch.Size([2])
```

These rules apply to a custom `__init__`:

- If it does not set a required field, the constructor raises
`TypeError: Points.__init__() did not set the required field 'xyz'`.
- Indexing, `clone`, `torch.stack`, and the other operations make new
instances without a call to `__init__`. So `__init__` must only fill
the fields, and the fields must be enough to rebuild the instance.
- As in a dataclass, a custom `__init__` does not call `__post_init__`.
If you need it, call it yourself.
- `super().__init__(...)` with fields works when the parent class is a
tensorclass with fields: it runs the generated `__init__` of the parent.
If the parent is [`TensorClass`](reference/generated/tensordict.TensorClass.html#tensordict.TensorClass) itself, a call with fields
raises `NotImplementedError`. Assign the fields instead.

### Overriding `__setattr__` and `__getitem__`

A `__setattr__` or `__getitem__` that the class defines replaces the one
of the tensorclass. For the cases that you do not handle, call the method of
`super()`. A custom `__setattr__` also runs when the constructor sets the
fields:

```
>>> class Clamped(TensorClass):
... x: torch.Tensor
...
... def __setattr__(self, key, value):
... if key == "x":
... value = value.clamp(0, 1)
... super().__setattr__(key, value)
>>> c = Clamped(x=torch.tensor([-1.0, 2.0]), batch_size=[2])
>>> c.x
tensor([0., 1.])
```

## Fields that are not tensors

A field can hold a value that is not a tensor, such as a string. The
tensorclass stores it in a wrapper.

### `NonTensorData` and `MetaData`

Two wrappers exist, and they differ when you stack instances:

- [`NonTensorData`](reference/generated/tensordict.NonTensorData.html#tensordict.NonTensorData) is the default. It keeps one value per
batch element. Stacking instances gives a
[`NonTensorStack`](reference/generated/tensordict.NonTensorStack.html#tensordict.NonTensorStack), and attribute access returns a list of
the values, even when all the values are equal.
- [`MetaData`](reference/generated/tensordict.MetaData.html#tensordict.MetaData) holds one value for the whole batch. Stacking
`MetaData("cfg")` with `MetaData("cfg")` returns `MetaData("cfg")`.
Stacking items with different values returns a `NonTensorStack`, as for
`NonTensorData`. Use `MetaData` for a value that is the same for every
element, such as a label or a configuration dict.

To use `MetaData`, assign a `MetaData` instance to the field:

```
>>> from tensordict import MetaData
>>> class Sample(TensorClass):
... x: torch.Tensor
... label: str
>>> a = Sample(x=torch.zeros(()), label="cfg")
>>> torch.stack([a, a]).label
LinkedList(['cfg', 'cfg'])
>>> m = Sample(x=torch.zeros(()), label=MetaData("cfg"))
>>> torch.stack([m, m]).label
'cfg'
```

`LinkedList` is a subclass of [`list`](https://docs.python.org/3/builtins/stdtypes.html#list). For the full rules of
non-tensor values in tensordicts, see [Non-tensor data and unbatched entries](concepts/non_tensor.html#non-tensor-data).

### Fields set to `None`

A field can be `None` at first and get a value later. The value that you
assign decides how the field is stored: a tensor is stored as a tensor, and a
string as `NonTensorData`:

```
>>> from typing import Any
>>> class Slots(TensorClass):
... X: Any
... y: Any
>>> slots = Slots(X=None, y=None, batch_size=[3, 4])
>>> slots.X = torch.ones(3, 4, 5)
>>> slots.y = "testing"
>>> print(slots)
Slots(batch_size=[3,4], fields={X: f32[5], y: str})
```

A `None` field is a key missing from the tensor storage. Setting a field to
`None`, or deleting it with `del obj.field`, `obj.del_("field")` or
`obj.pop("field")`, removes its entry, and reading it gives `None`.
`to_dict()`, `repr` and `state_dict()` list it as `None`.

### Comparison, item assignment, and concatenation

`==` and `!=` compare two tensorclasses field by field, also the nested
ones. The result is a tensorclass with a boolean tensor for each field. For a
field that is not a tensor, the boolean tensor has the batch shape, and it is
`False` where the values differ:

```
>>> print(data == data2)
Reading(batch_size=[3,4], fields={
 backup: Reading{
 count: bool[1],
 sensor: bool[],
 signal: bool[5],
 backup: None},
 count: bool[1],
 sensor: bool[],
 signal: bool[5]})
>>> data2.sensor = "camera"
>>> (data == data2).sensor.any()
tensor(False)
```

An item assignment writes the values that are not tensors element by element,
with no warning when they differ from the rest of the batch. The field then
holds one value per element (a [`NonTensorStack`](reference/generated/tensordict.NonTensorStack.html#tensordict.NonTensorStack)), and
attribute access returns a nested list of those values, with one level per
batch dimension:

```
>>> data[0] = data2[0]
>>> [row[0] for row in data.sensor]
['camera', 'lidar', 'lidar']
```

`torch.cat` and `torch.stack` keep the non-tensor value of each element.
If the inputs differ on a non-tensor field, the field of the result is a
`NonTensorStack`, and attribute access returns the nested list of values.
`torch.stack` does this even when all the inputs have the same value.
`torch.cat` of inputs that have one same value keeps that value:

```
>>> cat_tc = torch.cat([data, data2], dim=0)
>>> [row[0] for row in cat_tc.sensor]
['camera', 'lidar', 'lidar', 'camera', 'camera', 'camera']
```

## Class options

Flags change how a tensorclass converts and checks the values of its fields.
You can set them in three forms, which have the same effect:

```
>>> from tensordict import tensorclass
>>> class Foo(TensorClass["autocast"]): # bracket form
... x: int
>>> class Foo(TensorClass, autocast=True): # keyword form
... x: int
>>> @tensorclass(autocast=True) # decorator form
... class Foo:
... x: int
```

To set more than one flag, list them in the brackets:

```
>>> class Foo(TensorClass["nocast", "frozen"]):
... x: int
```

mypy rejects the bracket form. See Static type checking.

The flags are:

`autocast`

The constructor and attribute assignment convert each value to the type
annotation of its field. See Auto-casting.

`nocast`

Python numbers and NumPy arrays stay as they are. Without this flag, an
`int`, `float`, `bool`, or NumPy array becomes a tensor. A number
given to a field annotated `int`, `float` or `bool` (or an optional
or union of these) is the exception: with a non-empty batch size it is
stored as [`NonTensorData`](reference/generated/tensordict.NonTensorData.html#tensordict.NonTensorData), and with an empty batch size
it still becomes a 0-d tensor and emits a [`FutureWarning`](https://docs.python.org/3/builtins/exceptions.html#FutureWarning). Starting
with TensorDict 0.17, the default will change to keeping the Python value
of fields annotated int/float/bool, as [`NonTensorData`](reference/generated/tensordict.NonTensorData.html#tensordict.NonTensorData).
Annotate the field `torch.Tensor` to keep the tensor
([#1199](https://github.com/pytorch/tensordict/issues/1199)).

`tensor_only`

Each field must have a tensor annotation: a tensor type, a tensor collection
([`TensorDict`](reference/generated/tensordict.TensorDict.html#tensordict.TensorDict) or a tensorclass), or an `Optional` or
union of these. Any other annotation raises `TypeError` when the class is
defined. Python numbers and NumPy arrays become tensors. A value that cannot
become a tensor, for example a `str`, is stored as
[`NonTensorData`](reference/generated/tensordict.NonTensorData.html#tensordict.NonTensorData) with no error, and attribute access
returns the `NonTensorData` wrapper. Attribute access does not look in
the storage for non-tensor values, so it does less work. On one CPU, reading
a tensor field took 150 ns with `tensor_only` and 230 ns without it.
`benchmarks/tensorclass/test_tensorclass_speed.py` measures it.

`frozen`

Assigning a field raises [`dataclasses.FrozenInstanceError`](https://docs.python.org/3/library/dataclasses.html#dataclasses.FrozenInstanceError), as with
`@dataclass(frozen=True)`. See Immutability with frozen.

`shadow`

Allows field names that are members of a tensorclass, for example
`batch_size`, `device`, `names`, `sum`, `keys`, `get`, or
`to_dict`. Without this flag, such a name raises `AttributeError` when
the class is defined. A field named `data` or `fields` is allowed
without the flag, and it replaces the member of that name.

`skip_post_init_on_rebuild`

`__post_init__` runs only on the data that comes in (the generated
`__init__`, `from_tensordict`, `from_dict`, the other constructors
such as `from_any` or `fromkeys`, and loads), not each time
tensordict makes a new instance from an existing one, as in `clone`,
`to`, indexing, or `torch.stack`. See __post_init__.

`autocast`, `nocast`, and `tensor_only` are mutually exclusive. A class
with two of them raises `TypeError`.

### Auto-casting

Warning

Auto-casting is experimental. Its behavior can change in a
future release.

With `autocast`, the constructor and attribute assignment convert each value
to the annotation of its field:

- A dict becomes a [`TensorDict`](reference/generated/tensordict.TensorDict.html#tensordict.TensorDict), or the tensorclass of the
annotation.
- A value for an `int`, `float`, or `str` field becomes an `int`,
`float`, or `str`. For example, `torch.tensor(3)` becomes `3`.
- A value for a `torch.Tensor` field goes through [`torch.as_tensor()`](https://docs.pytorch.org/docs/stable/generated/torch.as_tensor.html#torch.as_tensor).
A value that it cannot convert, for example a `str`, raises `TypeError`.

In this example, the dict of `td` becomes a `TensorDict`, the dict of
`leaf` becomes a `Leaf`, and the tensor of `count` becomes an `int`:

```
>>> from tensordict import TensorDict
>>> class Leaf(TensorClass):
... value: torch.Tensor
>>> class Node(TensorClass["autocast"]):
... tensor: torch.Tensor
... td: TensorDict
... count: int
... leaf: Leaf
>>> node = Node(
... tensor=torch.zeros(()),
... td={"a": torch.zeros(())},
... count=torch.tensor(3),
... leaf={"value": torch.ones(())},
... )
>>> type(node.td).__name__, type(node.leaf).__name__, node.count
('TensorDict', 'Leaf', 3)
```

`update()` and `from_dict()` also convert a dict into the
`TensorDict` or tensorclass of its field. They do not convert values to
`int`, `float`, or `str`.

A field annotated with `Optional[...]`, `Union[...]`, or `A | B` is not
auto-cast. A dict that you assign to it stays a dict, because a tensorclass
can hold values that are not tensors. The other fields of the class are still
auto-cast:

```
>>> class Pair(TensorClass["autocast"]):
... left: Leaf = None
... right: Optional[Leaf] = None
>>> pair = Pair(left={"value": torch.ones(())}, right={"value": torch.ones(())})
>>> type(pair.left).__name__, type(pair.right).__name__
('Leaf', 'dict')
```

Auto-casting reads the annotations with [`typing.get_type_hints()`](https://docs.python.org/3/library/typing.html#typing.get_type_hints). If one
annotation of the class does not resolve, tensordict warns
`type_hints are none, cannot perform auto-casting` and casts no field of the
class. An annotation does not resolve when it names a class that is not
visible from the module, and the annotation is a string (or the module uses
`from __future__ import annotations`). The usual case is a tensorclass that is defined
inside a function, and that another class refers to. Define such classes at
module level.

### Immutability with `frozen`

With `frozen`, assigning a field raises
[`dataclasses.FrozenInstanceError`](https://docs.python.org/3/library/dataclasses.html#dataclasses.FrozenInstanceError), as with `@dataclass(frozen=True)`.
The instance is also locked, so you cannot add entries to it (see
[Keys, nested keys, and locking](concepts/keys.html#keys)):

```
>>> class State(TensorClass["frozen"]):
... params: torch.Tensor
... step: torch.Tensor
>>> s = State(params=torch.zeros(3), step=torch.zeros((), dtype=torch.long))
>>> s.step = s.step + 1
Traceback (most recent call last):
...
dataclasses.FrozenInstanceError: cannot assign to field 'step'
```

The tensors of a frozen instance are not read-only. An in-place operation on
a field, or `set_()`, writes into the existing tensor:

```
>>> _ = s.step.add_(1)
>>> s.step
tensor(1)
```

A subclass of a frozen tensorclass is frozen, with or without the flag. A
frozen class cannot subclass a tensorclass that is not frozen: this raises
`TypeError: cannot inherit frozen dataclass from a non-frozen one`. Type
checkers read a subclass without the flag as not frozen and report it, so
repeat `frozen=True` on the subclass (for mypy, see
Static type checking).

To change a field of a frozen instance, make a copy with new values.
`s.replace(step=...)` (see [`replace()`](reference/generated/tensordict.TensorDictBase.replace.html#tensordict.TensorDictBase.replace))
writes the new values into a shallow copy. [`copy.replace()`](https://docs.python.org/3/library/copy.html#copy.replace) (Python 3.13
and later) builds the copy with `__init__`, as for a dataclass, and keeps
the batch size and names of the instance unless you pass them,
`__init__` or `__post_init__` sets the batch size, or the new values do
not fit that batch size (the copy then has the default batch size, `[]`);
the device is always kept unless you pass it.
[`dataclasses.replace()`](https://docs.python.org/3/library/dataclasses.html#dataclasses.replace) does not go through `__replace__` on any Python
version, so its result has the default batch size (`[]`), no device and no
names: use `s.replace(...)` or [`copy.replace()`](https://docs.python.org/3/library/copy.html#copy.replace) instead.

## Converting a dataclass

If you have a `@dataclass`, [`from_dataclass()`](reference/generated/tensordict.from_dataclass.html#tensordict.from_dataclass) makes a
tensorclass type or instance from it. You do not change the class definition:

```
>>> from dataclasses import dataclass
>>> from tensordict import from_dataclass
>>> @dataclass
... class X:
... a: int
... b: torch.Tensor
>>> XTc = from_dataclass(X, autocast=True) # convert the type
>>> x_tc = from_dataclass(X(a=0, b=torch.zeros(())), autocast=True) # convert an instance
>>> XTc(a=torch.tensor(1), b=torch.zeros(())).a
1
```

Without `autocast` or `nocast`, `from_dataclass(X(a=0, ...))` casts
`a` to a 0-d tensor and emits the `FutureWarning` of `nocast` above.

It accepts the flags of a tensorclass (`autocast`, `nocast`, `frozen`,
`tensor_only`, `shadow`). When it applies them (not with `dest_cls`,
and not to a type that is already a tensorclass), `autocast` with
`nocast`, or `tensor_only` with either of them on a type, emits a
`FutureWarning`. Starting with TensorDict 0.17, the default will change to
raising the `TypeError` of the decorator.

## Saving and loading

The [`memmap()`](reference/generated/tensordict.TensorDictBase.memmap.html#tensordict.TensorDictBase.memmap) method saves a tensorclass instance
to a directory. Tensors are written as memory-mapped tensors, and non-tensor
values that JSON can encode are written as JSON. Other non-tensor values are
written with Python `pickle` (to an `other.pickle` file, or
`pickle.pkl` for stacked values).

[`load_memmap()`](reference/generated/tensordict.TensorDictBase.load_memmap.html#tensordict.TensorDictBase.load_memmap) loads it. The result has the
original class if the loading process can import that class:

```
>>> import tempfile
>>> with tempfile.TemporaryDirectory() as path:
... _ = data.memmap(path)
... data_loaded = TensorDict.load_memmap(path)
... assert isinstance(data_loaded, Reading)
... assert (data_loaded == data).all()
```

The class is saved by its module and name. Another program cannot find a
class that you defined in `__main__` (a script or a notebook) under that
name. To load such data, call `load_memmap` on the class itself, for example
`Reading.load_memmap(path)`. `load_memmap` uses the class that you call it
on if its name matches the saved name.

If the saved data has pickled non-tensor values and you do not pass
`allow_pickle`, loading emits a `FutureWarning` in tensordict 0.14 and
0.15. In 0.16, the default becomes `allow_pickle=False`, which raises an
error instead of loading them. Pass `allow_pickle=True` only for data that you
trust. For the other formats, see [Save and load tensordicts](saving.html#saving).

## Tensorclasses as module inputs and outputs

For a module whose inputs and outputs are tensorclasses, subclass
[`TensorClassModuleBase`](reference/generated/tensordict.nn.TensorClassModuleBase.html#tensordict.nn.TensorClassModuleBase). Its generic parameters declare
the input and output types for type checkers and IDEs. Its `as_td_module()`
method returns a module that you can put in a
[`TensorDictSequential`](reference/generated/tensordict.nn.TensorDictSequential.html#tensordict.nn.TensorDictSequential). See [Modules: TensorDictModule and tensordict.nn](modules.html#modules-guide) for an
example.

## `torch.compile`

Tensorclasses work with [`torch.compile()`](https://docs.pytorch.org/docs/stable/generated/torch.compile.html#torch.compile), [`torch.func`](https://docs.pytorch.org/docs/stable/func.api.html#module-torch.func), and
[`torch.export.export()`](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/export/api_reference.html#torch.export.export). The `Particles` example at the top of this page
compiles a method with `fullgraph=True`. [torch.compile, torch.func, torch.export, and CUDA graphs](compile.html#compile) lists what makes a
function recompile or break its graph, and the workarounds. Two of them
concern tensorclasses:

- A function that returns a string field recompiles for each new string.
- A compiled function that flattens a pytree recompiles after a new
tensorclass is defined.

## Static type checking

Type checkers read the fields of a [`TensorClass`](reference/generated/tensordict.TensorClass.html#tensordict.TensorClass) subclass
as its constructor signature, as they do for a `@dataclass`. The
keyword-only `batch_size`, `device`, `names`, and `lock` arguments
follow the fields. A missing field, an unknown keyword argument, or a value of
the wrong type is an error:

```
>>> class Obs(TensorClass):
... a: torch.Tensor
... label: str = "x"
>>> obs = Obs(torch.zeros(3), batch_size=[3]) # accepted
>>> obs = Obs(a=[0.0, 1.0, 2.0], batch_size=[3]) # error: a list is not a Tensor
```

The constructor converts a list, a NumPy array, or a Python number that you
pass to a `torch.Tensor` field. A type checker expects a tensor there, so
pass `torch.as_tensor(value)`. Tensordict methods such as `reshape` and
an index return the type of the class.

The three forms of the class options differ for type checkers:

- pyright accepts the bracket form (`TensorClass["autocast"]`) and the
keyword form (`class Foo(TensorClass, autocast=True)`).
- mypy does not evaluate [`__class_getitem__()`](https://docs.python.org/3/reference/datamodel.html#object.__class_getitem__) in a list of base
classes and rejects the bracket form. In code that mypy reads, use the
keyword form.
- Type checkers do not understand the decorator form. They know the fields of
a `@tensorclass` class, but not its tensordict methods or its
`batch_size` and `device` arguments. In code that a type checker reads,
use a subclass of `TensorClass`.

mypy reports each `frozen=True` tensorclass as "Frozen dataclass cannot
inherit from a non-frozen dataclass". These statements hold for pyright
1.1.414 and mypy 2.4.0.

## Runtime type checking

A runtime type checker that decorates a class checks the methods in its
namespace. Tensorclass marks the methods that it adds with
`__no_type_check__`, the attribute that [`typing.no_type_check()`](https://docs.python.org/3/library/typing.html#typing.no_type_check) sets, so
the checkers that honor it, such as beartype and typeguard, check only the
methods written in the class body. `__init__` is not marked: it keeps the
annotations of the fields, or those of a user-defined `__init__`.

Some methods of a tensorclass have the names of builtins, such as `bool`,
`int`, `float`, `set`, and `type`. With
`from __future__ import annotations`, annotations are strings. A checker that
evaluates them in the namespace of the class finds these methods there when
the class uses the `@tensorclass` decorator, so an `int` annotation names
the method. A subclass of [`TensorClass`](reference/generated/tensordict.TensorClass.html#tensordict.TensorClass) inherits these
methods instead. With the decorator, write `builtins.int`.

A field can have a shape annotation from [jaxtyping](https://docs.kidger.site/jaxtyping/), such as
`Float[torch.Tensor, "*batch 3"]`. The class accepts it, with every flag, and
treats it as the array type that it wraps: `tensor_only` accepts it as a
tensor annotation, and `autocast` casts to `torch.Tensor`. The shapes are
not checked when the class is built or when a field is set. To check them,
decorate a function that takes the class with `@jaxtyped`, and check the
fields with `isinstance` in its body. The checks in one call share their
dimension names (`n` below) with the annotations of the function:

```
>>> from beartype import beartype 
>>> from jaxtyping import Float, jaxtyped 
>>> class Points(TensorClass["tensor_only"]): 
... pos: Float[torch.Tensor, "*batch 3"]
... mass: Float[torch.Tensor, "*batch"]
>>> @jaxtyped(typechecker=beartype) 
... def center(points: Points) -> Float[torch.Tensor, "3"]:
... assert isinstance(points.pos, Float[torch.Tensor, "n 3"])
... assert isinstance(points.mass, Float[torch.Tensor, "n"])
... return (points.pos * points.mass[:, None]).sum(0) / points.mass.sum()
>>> points = Points(pos=torch.zeros(4, 3), mass=torch.ones(4), batch_size=[4]) 
>>> center(points).shape 
torch.Size([3])
>>> wrong = Points(pos=torch.zeros(4, 2), mass=torch.ones(4), batch_size=[4]) 
>>> center(wrong) 
Traceback (most recent call last):
...
AssertionError
```

## Legacy: the `@tensorclass` decorator

The [`tensorclass()`](reference/generated/tensordict.tensorclass.html#tensordict.tensorclass) decorator is the original way to declare a
tensorclass. It is kept for backward compatibility, and it is the shortest way
to convert a body of `@dataclass` code. In new code, subclass
[`TensorClass`](reference/generated/tensordict.TensorClass.html#tensordict.TensorClass) instead: it has the same behavior, and type
checkers understand it without a plugin.

```
>>> @tensorclass
... class LegacyReading:
... signal: torch.Tensor
... count: torch.Tensor
... sensor: str
... backup: Optional["LegacyReading"] = None
```

The decorator accepts each flag as a keyword argument:
`@tensorclass(autocast=True)`, `@tensorclass(frozen=True, tensor_only=True)`.

## See also

- [tensorclass and non-tensor data](reference/tc.html#api-tensorclass): the reference of [`TensorClass`](reference/generated/tensordict.TensorClass.html#tensordict.TensorClass),
[`tensorclass()`](reference/generated/tensordict.tensorclass.html#tensordict.tensorclass), [`from_dataclass()`](reference/generated/tensordict.from_dataclass.html#tensordict.from_dataclass), and the
non-tensor classes.
- [TypedTensorDict](typed_tensordict.html#typed-tensordict-guide): typed fields on a
[`TensorDictBase`](reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase) subclass.
- [Which container should I use?](choosing_a_container.html#choosing-a-container): which container to use.
- [Coming from a hand-written dataclass](coming_from_dataclass.html#coming-from-dataclass): the methods of a hand-written dataclass of
tensors, and the tensorclass operations that replace them.
- [Coming from JAX pytrees](coming_from_jax.html#coming-from-jax): the JAX pytree idioms in tensorclass terms.
- [Containers on storage backends](compatibility.html#compatibility): tensorclasses on HDF5, Redis, lazy stacks, and
memory-mapped tensordicts.
- The tutorials [Using tensorclasses for datasets](tutorials/tensorclass_fashion.html) and
[Batched data loading with tensorclasses](tutorials/tensorclass_imagenet.html).