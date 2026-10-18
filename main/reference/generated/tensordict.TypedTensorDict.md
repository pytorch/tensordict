# TypedTensorDict

`TypedTensorDict` has the methods of [`TensorDictBase`](tensordict.TensorDictBase.html#tensordict.TensorDictBase),
listed by task in [Methods by task](../td.html#api-tensordict-methods). This page lists only the
methods that `TypedTensorDict` adds, and the overrides that have their own
documentation.

*class*tensordict.TypedTensorDict(**args*, ***kwargs*)

Bases: `_StateFromSource`, [`TensorDictBase`](tensordict.TensorDictBase.html#tensordict.TensorDictBase)

A [`TensorDictBase`](tensordict.TensorDictBase.html#tensordict.TensorDictBase) subclass whose fields are declared as class annotations.

Warning

`TypedTensorDict` is experimental: its API can change in a minor
release without a deprecation cycle.

A subclass declares its fields as annotations, as a `TypedDict` does,
and gets:

- attribute access to the fields (`state.eta`) and `state["eta"]`;
- a constructor that takes the fields by keyword and checks them;
- inheritance: a subclass adds fields to the fields of its parent;
- default values for the fields that a pipeline may not fill;
- every [`TensorDictBase`](tensordict.TensorDictBase.html#tensordict.TensorDictBase) operation (`to`,
`clone`, indexing, `**state` unpacking);
- typed access to an existing tensordict of any backend (H5, Redis, lazy
stacks) with `from_tensordict()`, which does not copy the data.

Use it instead of a [`TensorClass`](tensordict.TensorClass.html#tensordict.TensorClass) when you need a
[`TensorDictBase`](tensordict.TensorDictBase.html#tensordict.TensorDictBase) subclass: `state["key"]` access,
`**state` unpacking, or typed access to an existing backend. Use a
[`TensorClass`](tensordict.TensorClass.html#tensordict.TensorClass) instead when fields hold non-tensor
values that must come back as plain Python objects.
[Which container should I use?](../../choosing_a_container.html#choosing-a-container) compares the containers, and
[TypedTensorDict](../../typed_tensordict.html#typed-tensordict-guide) is the user guide.

Keyword Arguments:

- **batch_size** ([*Size*](https://docs.pytorch.org/docs/stable/size.html#torch.Size)*or*[*list*](https://docs.python.org/3/builtins/stdtypes.html#list)*of*[*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the batch size of
the instance: the leading dimensions that all its entries share.
Defaults to `None`, which gives an empty batch size.
- **device** ([*device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,*[*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*or*[*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the device of the
instance. Leaves on another device are moved to it. Defaults to
`None`, which leaves each leaf where it is.
- **names** ([*list*](https://docs.python.org/3/builtins/stdtypes.html#list)*of*[*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - the names of the batch dimensions, one
per dimension. Defaults to `None`.
- **non_blocking** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, copies between devices can
run asynchronously with respect to the host. Defaults to `None`:
the moves are asynchronous, and the constructor synchronizes after
them if required.
- **lock** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, the instance is locked. See
[Locking and temporary changes](../../concepts/keys.html#keys-locking). Defaults to `False`.
- ****fields** - the value of each field. A missing field without a default,
or a name that is not a field, raises a `TypeError`.

Two options change the behavior of a class. Set them with
`class State(TypedTensorDict["frozen"])` or
`class State(TypedTensorDict, frozen=True)`. Subclasses inherit them.

- `shadow`: allows a field to have the name of a
[`TensorDictBase`](tensordict.TensorDictBase.html#tensordict.TensorDictBase) attribute (`batch_size`,
`sum`, ...), which it then hides. Without it, such a field raises an
`AttributeError` when the class is defined.
- `frozen`: the constructor locks the instance.

The `autocast`, `nocast` and `tensor_only` options of
[`TensorClass`](tensordict.TensorClass.html#tensordict.TensorClass) are accepted, have no effect, and warn.
They are deprecated and will be removed in TensorDict 0.17.

Examples

```
>>> import torch
>>> from tensordict import TensorDict, TypedTensorDict
>>> from torch import Tensor
>>> class PredictorState(TypedTensorDict):
... eta: Tensor
... X: Tensor
>>> state = PredictorState(eta=torch.zeros(5, 3), X=torch.zeros(5, 4), batch_size=[5])
>>> state.eta.shape
torch.Size([5, 3])
>>> state["X"].shape
torch.Size([5, 4])
>>> state[0].batch_size
torch.Size([])
```

`from_tensordict()` wraps an existing tensordict:

```
>>> td = TensorDict(eta=torch.zeros(5, 3), X=torch.zeros(5, 4), batch_size=[5])
>>> state = PredictorState.from_tensordict(td)
>>> state.eta is td["eta"]
True
```

*classmethod*from_dict(*input_dict*, ***, *auto_batch_size=None*, *batch_size=None*, *device=None*, *batch_dims=None*, *names=None*, *check: [bool](https://docs.python.org/3/builtins/functions.html#bool) = True*)

Builds an instance of the class from a dictionary of its fields.

Keys of `input_dict` that the class does not declare are kept, as
`from_tensordict()` and [`set()`](tensordict.TensorDictBase.set.html#tensordict.TensorDictBase.set)
keep them.

Parameters:

**input_dict** ([*dict*](https://docs.python.org/3/builtins/stdtypes.html#dict)) - the values of the fields. Nested dictionaries
become [`TensorDict`](tensordict.TensorDict.html#tensordict.TensorDict) instances.

Keyword Arguments:

- **auto_batch_size** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, sets the batch size
to the largest leading shape that all the entries share.
Defaults to `None`, which behaves like `False`, unless
`batch_dims` is passed: then it behaves like `True`.
- **batch_size** ([*Size*](https://docs.pytorch.org/docs/stable/size.html#torch.Size)*or*[*list*](https://docs.python.org/3/builtins/stdtypes.html#list)*of*[*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the batch size of
the result: the leading dimensions that all its entries share.
Defaults to `None`.
- **device** ([*device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,*[*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*or*[*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the device of the
result. Leaves on another device are moved to it. Defaults to
`None`, which leaves each leaf where it is.
- **batch_dims** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - with `auto_batch_size`, the maximum
number of batch dimensions. Exclusive with `batch_size`.
Defaults to `None`.
- **names** ([*list*](https://docs.python.org/3/builtins/stdtypes.html#list)*of*[*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - the names of the batch dimensions,
one per dimension. Defaults to `None`.
- **check** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, warns with a
[`FutureWarning`](https://docs.python.org/3/builtins/exceptions.html#FutureWarning) when `input_dict` misses a required
field. Starting with TensorDict 0.17, the default will change
to raising the `TypeError` that the constructor raises. Pass
`False` to build an instance without these fields; reading
one then raises `AttributeError`. Defaults to `True`.

Returns:

A new instance of the class.

Examples

```
>>> import torch
>>> from torch import Tensor
>>> from tensordict import TypedTensorDict
>>> class State(TypedTensorDict):
... x: Tensor
... y: Tensor
>>> state = State.from_dict({"x": torch.zeros(3, 4), "y": torch.zeros(3)}, auto_batch_size=True)
>>> state.batch_size
torch.Size([3])
```

from_dict_instance(*input_dict*, ***, *auto_batch_size=None*, *batch_size=None*, *device=None*, *batch_dims=None*, *names=None*)

Builds an instance of the same class from a dictionary, keeping the types of the existing entries.

Unlike [`from_dict()`](tensordict.from_dict.html#tensordict.from_dict), it does not check the declared fields: the
result can miss required fields and hold undeclared keys. tensordict
calls it with the entries that a nested write changes, as in
`outer[0:2] = {"inner": {"eta": value}}` where `outer["inner"]`
is a `TypedTensorDict`.

Parameters:

**input_dict** ([*dict*](https://docs.python.org/3/builtins/stdtypes.html#dict)) - the values of the entries.

Keyword Arguments:

- **auto_batch_size** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, sets the batch size
to the largest leading shape that all the entries share.
Defaults to `None`, which behaves like `False`, unless
`batch_dims` is passed: then it behaves like `True`.
- **batch_size** ([*Size*](https://docs.pytorch.org/docs/stable/size.html#torch.Size)*or*[*list*](https://docs.python.org/3/builtins/stdtypes.html#list)*of*[*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the batch size of
the result. Defaults to `None`.
- **device** ([*device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,*[*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*or*[*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the device of the
result. Defaults to `None`, which leaves each leaf where it
is.
- **batch_dims** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - with `auto_batch_size`, the maximum
number of batch dimensions. Exclusive with `batch_size`.
Defaults to `None`.
- **names** ([*list*](https://docs.python.org/3/builtins/stdtypes.html#list)*of*[*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - the names of the batch dimensions,
one per dimension. Defaults to `None`.

These arguments go to
[`from_dict_instance()`](tensordict.TensorDictBase.from_dict_instance.html#tensordict.TensorDictBase.from_dict_instance) of the wrapped
tensordict.

*classmethod*from_tensordict(*td: [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*, ***, *check: [bool](https://docs.python.org/3/builtins/functions.html#bool) = True*) → TypedTensorDict

Wraps an existing tensordict in an instance of the class, without copying it.

The instance stores `td` itself: a write through either one is seen
by the other.

Parameters:

**td** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)) - the tensordict to wrap, of any backend
([`TensorDict`](tensordict.TensorDict.html#tensordict.TensorDict),
[`PersistentTensorDict`](tensordict.PersistentTensorDict.html#tensordict.PersistentTensorDict), a lazy stack, ...).

Keyword Arguments:

**check** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, checks that `td` holds every
required field. Pass `False` to wrap an empty or partly
filled tensordict, such as a preallocated
[`TensorDictStore`](tensordict.store.TensorDictStore.html#tensordict.store.TensorDictStore): reading a missing
field then raises an `AttributeError`. Defaults to `True`.

Returns:

A new instance of the class backed by `td`.

Raises:

[**TypeError**](https://docs.python.org/3/builtins/exceptions.html#TypeError) - if `check=True` and `td` misses required fields.

Examples

```
>>> import torch
>>> from torch import Tensor
>>> from tensordict import TensorDict, TypedTensorDict
>>> class State(TypedTensorDict):
... x: Tensor
>>> td = TensorDict(x=torch.zeros(3), batch_size=[3])
>>> state = State.from_tensordict(td)
>>> state.x += 1
>>> td["x"]
tensor([1., 1., 1.])
>>> State.from_tensordict(TensorDict())
Traceback (most recent call last):
 ...
TypeError: State.from_tensordict() missing required field(s): x
```