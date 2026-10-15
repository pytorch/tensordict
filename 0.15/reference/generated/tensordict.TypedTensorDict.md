# TypedTensorDict

`TypedTensorDict` has the methods of [`TensorDictBase`](tensordict.TensorDictBase.html#tensordict.TensorDictBase),
listed by task in [Methods by task](../td.html#api-tensordict-methods). This page lists only the
methods that `TypedTensorDict` adds, and the overrides that have their own
documentation.

*class*tensordict.TypedTensorDict(**args*, ***kwargs*)

Bases: `_StateFromSource`, [`TensorDictBase`](tensordict.TensorDictBase.html#tensordict.TensorDictBase)

A TensorDictBase subclass with typed field declarations and backend composition.

Warning

`TypedTensorDict` is experimental: its API can change in a minor
release without a deprecation cycle.

TypedTensorDict combines TensorDict's tensor operations with TypedDict-style
field declarations. Subclasses declare fields as class annotations and get:

- Typed attribute access (`state.eta`)
- Typed construction (`PredictorState(eta=..., X=..., beta=..., batch_size=...)`)
- Inheritance (`class ObservedState(PredictorState): ...`)
- Default-valued fields for optional pipeline branches
- Full TensorDict interop (`.to()`, `.clone()`, slicing, `**state` spreading)
- Backend composition via `from_tensordict` (H5, Redis, lazy stacks, and others)

Use it instead of a [`TensorClass`](tensordict.TensorClass.html#tensordict.TensorClass) when you need a
[`TensorDictBase`](tensordict.TensorDictBase.html#tensordict.TensorDictBase) subclass: `state["key"]` access,
`**state` unpacking, or typed access to an existing backend. Use a
[`TensorClass`](tensordict.TensorClass.html#tensordict.TensorClass) instead when fields hold non-tensor
values that must come back as plain Python objects.
[Which container should I use?](../../choosing_a_container.html#choosing-a-container) compares the containers, and
[TypedTensorDict](../../typed_tensordict.html#typed-tensordict-guide) is the user guide.

Internally, a `TypedTensorDict` delegates all storage to an internal
`_source` attribute (a `TensorDictBase` instance). Direct construction
creates a `TensorDict` as `_source`. `from_tensordict` stores the
given backend directly, so the typed access to that backend makes no copy.

Examples

```
>>> import torch
>>> from tensordict import TensorDict, TypedTensorDict
>>> from torch import Tensor
>>>
>>> class PredictorState(TypedTensorDict):
... eta: Tensor
... X: Tensor
...
>>> state = PredictorState(eta=torch.randn(5, 3), X=torch.randn(5, 4), batch_size=[5])
>>> state.eta.shape
torch.Size([5, 3])
>>> state["X"].shape
torch.Size([5, 4])
>>>
>>> # Wrap an existing TensorDict backend
>>> td = TensorDict({"eta": torch.randn(5, 3), "X": torch.randn(5, 4)}, batch_size=[5])
>>> state = PredictorState.from_tensordict(td)
>>> state.eta.shape
torch.Size([5, 3])
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

- **auto_batch_size** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)*,**optional*) - whether to infer the batch size.
- **batch_size** ([*Size*](https://docs.pytorch.org/docs/stable/size.html#torch.Size)*,**optional*) - the batch size.
- **device** ([*device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,**optional*) - the device.
- **batch_dims** ([*int*](tensordict.TensorDictBase.int.html#tensordict.TensorDictBase.int)*,**optional*) - the number of batch dimensions to infer.
- **names** ([*list*](https://docs.python.org/3/builtins/stdtypes.html#list)*of*[*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - the names of the batch dimensions.
- **check** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)) - If `True` (default), warn with a
[`FutureWarning`](https://docs.python.org/3/builtins/exceptions.html#FutureWarning) when `input_dict` misses a required
field. Starting with TensorDict 0.17, the default will change
to raising the `TypeError` that the constructor raises. Set
it to `False` to build an instance without these fields;
reading one then raises `AttributeError`.

Returns:

A new instance of the class.

from_dict_instance(*input_dict*, ***, *auto_batch_size=None*, *batch_size=None*, *device=None*, *batch_dims=None*, *names=None*)

Builds an instance of the same class from a dictionary, keeping the types of the existing entries.

Unlike [`from_dict()`](tensordict.from_dict.html#tensordict.from_dict), it does not check the declared fields: the
result can miss required fields and hold undeclared keys. tensordict
calls it with the entries that a nested write changes, as in
`outer[0:2] = {"inner": {"eta": value}}` where `outer["inner"]`
is a `TypedTensorDict`.

The arguments are those of
[`from_dict_instance()`](tensordict.TensorDictBase.from_dict_instance.html#tensordict.TensorDictBase.from_dict_instance).

*classmethod*from_tensordict(*td: [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*, ***, *check: [bool](https://docs.python.org/3/builtins/functions.html#bool) = True*) → TypedTensorDict

Wrap an existing TensorDictBase backend as a TypedTensorDict.

The `td` is stored directly (no copy); mutations through the
TypedTensorDict are reflected in the original backend and vice versa.

Parameters:

**td** - Any `TensorDictBase` instance (`TensorDict`,
`PersistentTensorDict`, `LazyStackedTensorDict`, etc.).

Keyword Arguments:

**check** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)) - If `True` (default), validate that all required
fields are present in `td`. Set to `False` to wrap an
empty or partially-filled backend (e.g. a pre-allocated
`TensorDictStore`); missing fields will raise at access time
rather than wrap time.

Raises:

[**TypeError**](https://docs.python.org/3/builtins/exceptions.html#TypeError) - If `check=True` and required fields are missing.

Returns:

A new `TypedTensorDict` instance backed by `td`.