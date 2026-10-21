# TypedTensorDict

Warning

`TypedTensorDict` is experimental: its API can change in a minor release
without a deprecation cycle.

A [`TypedTensorDict`](reference/generated/tensordict.TypedTensorDict.html#tensordict.TypedTensorDict) is a tensordict with declared fields.
It subclasses [`TensorDictBase`](reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase), and you declare its fields
as class annotations, as in a [`typing.TypedDict`](https://docs.python.org/3/library/typing.html#typing.TypedDict). Choose it over a
[`TensorClass`](reference/generated/tensordict.TensorClass.html#tensordict.TensorClass) when the object must be a
`TensorDictBase`: for `state["key"]`, `**state`, or a function that takes
a tensordict. Choose it also for a pipeline that adds fields in stages, with
one subclass for each stage.

## Quick start

Subclass [`TypedTensorDict`](reference/generated/tensordict.TypedTensorDict.html#tensordict.TypedTensorDict) and annotate each field. Then
pass the values and the batch size to the constructor:

```
>>> import torch
>>> from torch import Tensor
>>> from tensordict import TypedTensorDict
>>> class PredictorState(TypedTensorDict):
... eta: Tensor
... X: Tensor
... beta: Tensor
>>> state = PredictorState(
... eta=torch.randn(5, 3),
... X=torch.randn(5, 4),
... beta=torch.randn(5, 1),
... batch_size=[5],
... )
>>> state.eta.shape
torch.Size([5, 3])
>>> state["X"].shape
torch.Size([5, 4])
```

A `TypedTensorDict` is a tensordict: `isinstance(state, TensorDictBase)`
is `True`, and a string index reads an entry, as on a
[`TensorDict`](reference/generated/tensordict.TensorDict.html#tensordict.TensorDict).

## Build state in stages

A pipeline often builds its state one step at a time. Each stage is a
subclass that adds fields to the previous one, and a function moves the state
from one stage to the next:

```
>>> class ObservedState(PredictorState):
... y: Tensor
... mu: Tensor
>>> def gaussian(state: PredictorState, std: float) -> ObservedState:
... eta = state.eta
... y = eta + torch.randn_like(eta) * std
... return ObservedState(**state, y=y, mu=eta, batch_size=state.batch_size)
>>> obs = gaussian(state, std=0.1)
>>> sorted(obs.keys())
['X', 'beta', 'eta', 'mu', 'y']
```

A `TypedTensorDict` is a [`MutableMapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.MutableMapping), so
`**state` passes all its fields as keyword arguments. To add a field to a
stage, add one line to the class definition: the functions that unpack the
state do not change.

## Inheritance

A subclass has the fields of its parents and the fields that it declares:

```
>>> class ObservedState(PredictorState):
... y: Tensor
... mu: Tensor
... noise: Tensor | None = None
>>> class SurvivalState(ObservedState):
... event_time: Tensor
... indicator: Tensor
... observed_time: Tensor
>>> sorted(ObservedState.__required_keys__)
['X', 'beta', 'eta', 'mu', 'y']
>>> sorted(ObservedState.__optional_keys__)
['noise']
```

Inheritance follows the rules of Python: `isinstance(obs, PredictorState)`
is `True` for an `ObservedState` instance, and a function annotated with
`f(state: PredictorState)` accepts any subclass.

## Optional fields

To make a field optional, give it a `None` default:

```
>>> obs = ObservedState(
... eta=torch.randn(5, 3), X=torch.randn(5, 4), beta=torch.randn(5, 1),
... y=torch.randn(5, 3), mu=torch.randn(5, 3),
... batch_size=[5],
... )
>>> "noise" in obs
False
>>> obs.noise is None
True
```

If you do not pass the field, it is not an entry of the tensordict, and
attribute access returns the default. `NotRequired[Tensor]` also marks a
field as optional, with two differences. mypy accepts `NotRequired` only in
a `TypedDict`. Attribute access on a `NotRequired` field that is not set
raises `AttributeError`. In code that a type checker reads, use a default.

## Values that are not tensors

A `TypedTensorDict` stores each value as a [`TensorDict`](reference/generated/tensordict.TensorDict.html#tensordict.TensorDict)
does, whatever the annotation. It does not compare the values with the
annotations:

- A Python number, a `bool`, or a list of numbers becomes a tensor. A number
is a tensor with no dimensions, so it needs `batch_size=[]`.
- A string is stored as [`NonTensorData`](reference/generated/tensordict.NonTensorData.html#tensordict.NonTensorData). Attribute access
and `get` return the `NonTensorData` wrapper, and a string index returns
the string.

```
>>> class Run(TypedTensorDict):
... loss: Tensor
... name: str
... step: int
>>> run = Run(loss=torch.zeros(()), name="baseline", step=3, batch_size=[])
>>> run.step
tensor(3)
>>> run.name
NonTensorData(data=baseline, batch_size=torch.Size([]), device=None)
>>> run["name"]
'baseline'
```

If your fields are strings or other Python objects that you want to read as
plain values, use a [`TensorClass`](reference/generated/tensordict.TensorClass.html#tensordict.TensorClass).

## Methods and properties

You can add methods and properties to a `TypedTensorDict` class. They work
on the instances that tensordict operations return, such as an index:

```
>>> class Gaussian(TypedTensorDict):
... loc: Tensor
... log_scale: Tensor
...
... @property
... def scale(self):
... return self.log_scale.exp()
...
... def sample(self):
... return self.loc + self.scale * torch.randn_like(self.loc)
>>> g = Gaussian(loc=torch.zeros(4, 2), log_scale=torch.zeros(4, 2), batch_size=[4])
>>> g[0].scale
tensor([1., 1.])
>>> g.sample().shape
torch.Size([4, 2])
```

A `TypedTensorDict` does not support a custom `__init__`, and it does not
call `__post_init__`. To build an instance from other arguments, write a
class method that calls the constructor. If you need a custom `__init__`,
use a [`TensorClass`](reference/generated/tensordict.TensorClass.html#tensordict.TensorClass) (see [TensorClass: a dataclass of tensors](tensorclass.html#tensorclass-guide)).

## Class options

`TypedTensorDict` supports the `"shadow"` and `"frozen"` options of
`TensorClass`, with the same bracket syntax:

```
>>> class MyModel(TypedTensorDict["shadow"]):
... data: Tensor # "data" is also a TensorDictBase attribute
>>> class Immutable(TypedTensorDict["frozen"]):
... x: Tensor # locked after construction
>>> class Combined(TypedTensorDict["shadow", "frozen"]):
... data: Tensor
```

`"shadow"`

Allows field names that are also `TensorDictBase` attributes, such as
`data` or `sum`. Without this option, such a name raises
`AttributeError` when the class is defined.

`"frozen"`

Locks the tensordict after construction. Attribute assignment and new keys
raise `RuntimeError`. In-place writes to tensor fields, such as
`set_()` or `state.x.add_(1)`, still work, so the instance is not
read-only. This is
different from `TensorClass["frozen"]`, where attribute assignment raises
`dataclasses.FrozenInstanceError`.

`"autocast"`, `"nocast"`, `"tensor_only"`

Deprecated in 0.15, to be removed in 0.17. The class accepts them with the
syntax of `TensorClass`, and subclasses inherit them. They have no effect:
a `TypedTensorDict` stores values as a `TensorDict` does. For example,
`x=1` is stored as a tensor even with `"nocast"`. Each of them emits a
`DeprecationWarning`. If you need them, use `TensorClass`.

A subclass of a `"frozen"` class is also frozen.

## Wrapping another backend (`from_tensordict`)

`from_tensordict(td)` wraps a tensordict of any `TensorDictBase` class
without a copy. Writes through the typed wrapper go to the storage of `td`:

```
>>> from tensordict import TensorDict
>>> td = TensorDict(
... {"eta": torch.randn(5, 3), "X": torch.randn(5, 4), "beta": torch.randn(5, 1)},
... batch_size=[5],
... )
>>> state = PredictorState.from_tensordict(td)
>>> state.eta.shape
torch.Size([5, 3])
>>> state.eta = torch.ones(5, 3) # writes to td
>>> (td["eta"] == 1).all()
tensor(True)
```

This works with a [`PersistentTensorDict`](reference/generated/tensordict.PersistentTensorDict.html#tensordict.PersistentTensorDict) (HDF5 or Zarr), a
[`TensorDictStore`](reference/generated/tensordict.store.TensorDictStore.html#tensordict.store.TensorDictStore) (Redis), a
[`LazyStackedTensorDict`](reference/generated/tensordict.LazyStackedTensorDict.html#tensordict.LazyStackedTensorDict), and a memory-mapped
`TensorDict`. Some operations behave differently on these backends. See
[Containers on storage backends](compatibility.html#compatibility) for the examples and the limits.

`from_tensordict` raises `TypeError` if a required field is missing. To
wrap an empty or partly filled tensordict, pass `check=False`. A field that
is not set then raises `AttributeError` when you read it. An indexed write
of a full instance creates the missing entries:

```
>>> empty = TensorDict(batch_size=[3])
>>> state = PredictorState.from_tensordict(empty, check=False)
>>> state.X
Traceback (most recent call last):
...
AttributeError: 'PredictorState' has field 'X' declared but not set
>>> state[0] = PredictorState(
... eta=torch.ones(3), X=torch.ones(4), beta=torch.ones(1), batch_size=[]
... )
>>> sorted(empty.keys()), empty["eta"].shape
(['X', 'beta', 'eta'], torch.Size([3, 3]))
```

For the same pattern on a Redis server, with storage that the server
allocates, see [pre-allocating on Redis](storage.html#compat-redis-prealloc).

## TensorDict operations

Every `TensorDictBase` operation works on `TypedTensorDict` instances, and
the operations that return a new tensordict return the same class:

```
>>> state = PredictorState(
... eta=torch.randn(5, 3), X=torch.randn(5, 4), beta=torch.randn(5, 1),
... batch_size=[5],
... )
>>> state.to("cpu").device
device(type='cpu')
>>> type(state.clone()).__name__
'PredictorState'
>>> state[0:3].batch_size
torch.Size([3])
>>> torch.stack([state, state], dim=0).batch_size
torch.Size([2, 5])
```

This includes `.memmap()`, `.apply()`, `torch.cat`, `.unbind()`,
`.select()`, `.exclude()`, and `.update()`.

The declared fields describe a valid instance. They do not change what the
`TensorDictBase` methods do. `pop`, `rename_key_`, and `select` or
`exclude` with `inplace=True` can remove a declared field, and reading it then raises
`AttributeError`. If the key set must not change, use
`TypedTensorDict["frozen"]`.

Only the keyword constructor rejects keys that the class does not declare.
`from_dict`, the positional form `PredictorState({...}, batch_size=...)`,
`from_tensordict`, `set`, and `__setitem__` keep them. `from_dict` and
the positional form warn with a `FutureWarning` when a required field is
missing, and return an instance without it. Starting with TensorDict 0.17, the
default will change to raising the `TypeError` that the keyword constructor
raises. `from_dict(..., check=False)` builds the instance without the
warning. `from_dict_instance` checks no fields, because tensordict calls it
with the entries that a nested write such as
`outer[0:2] = {"inner": {"eta": value}}` changes.

## Type checking

`TypedTensorDict` uses `@dataclass_transform()` ([**PEP 681**](https://peps.python.org/pep-0681/)) on its
metaclass. With pyright 1.1.414 and mypy 2.4.0:

- Both type checkers give each declared field its type: `state.eta` is a
`Tensor`.
- mypy reports a constructor call that misses a required field, also one of
a parent class. pyright does not.
- Neither reports an unknown keyword argument in the constructor, or an
unknown attribute such as `state.etta`.

A string index (`state["eta"]`) works at run time, but type checkers do not
know the type of its result. For typed access, use the attribute
(`state.eta`).

## Performance

tensordict reads the fields and makes the attribute accessors once, when a
`TypedTensorDict` subclass is defined. A plain `TensorDict` runs no code
of `TypedTensorDict`. On a typed instance, attribute access goes through a
property, and the required keys are checked only at construction, in
`from_tensordict(check=True)`, and in `from_dict(check=True)`.
`benchmarks/tensorclass/test_tensorclass_speed.py` measures construction and
access, and `benchmarks/compile/compile_td_test.py` measures attribute
access under `torch.compile`.

## See also

- [TypedTensorDict](reference/ttd.html#api-typed-tensordict): the reference of
[`TypedTensorDict`](reference/generated/tensordict.TypedTensorDict.html#tensordict.TypedTensorDict).
- [TensorClass: a dataclass of tensors](tensorclass.html#tensorclass-guide): typed fields with plain values, custom
`__init__`, and dataclass features.
- [Which container should I use?](choosing_a_container.html#choosing-a-container): which container to use, and the
`TensorClass` and `TypedTensorDict` comparison.
- [Containers on storage backends](compatibility.html#compatibility): `TypedTensorDict` on HDF5, Redis, lazy stacks, and
memory-mapped tensordicts.