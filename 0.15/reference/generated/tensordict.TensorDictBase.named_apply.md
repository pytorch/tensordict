# TensorDictBase.named_apply

TensorDictBase.named_apply(*fn: [Callable](https://docs.python.org/3/library/typing.html#typing.Callable)*, **others: T*, *nested_keys: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *batch_size: [Sequence](https://docs.python.org/3/library/typing.html#typing.Sequence)[[int](https://docs.python.org/3/builtins/functions.html#int)] | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *device: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | [None](https://docs.python.org/3/builtins/constants.html#None) = NO_DEFAULT*, *names: [Sequence](https://docs.python.org/3/library/typing.html#typing.Sequence)[[str](https://docs.python.org/3/builtins/stdtypes.html#str)] | [None](https://docs.python.org/3/builtins/constants.html#None) = NO_DEFAULT*, *inplace: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *default: [Any](https://docs.python.org/3/library/typing.html#typing.Any) = NO_DEFAULT*, *filter_empty: [bool](https://docs.python.org/3/builtins/functions.html#bool) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *propagate_lock: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *call_on_nested: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *out: [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, ***constructor_kwargs*) → Self | [None](https://docs.python.org/3/builtins/constants.html#None)

Applies a key-conditioned callable to all values stored in the tensordict and sets them in a new tensordict.

The callable signature must be `Callable[Tuple[str, Tensor, ...], Tensor | TensorDictBase | None]`.

Parameters:

- **fn** (*Callable*) - function to be applied to the (name, tensor) pairs in the
tensordict. For each leaf, only its leaf name will be used (not
the full NestedKey).
- ***others** (*TensorDictBase instances**,**optional*) - if provided, these
tensordict instances should have a structure matching the one
of self. The `fn` argument should receive as many
unnamed inputs as the number of tensordicts, including self.
If other tensordicts have missing entries, a default value
can be passed through the `default` keyword argument.
- **nested_keys** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, the complete path
to the leaf will be used. Defaults to `False`, i.e. only the last
string is passed to the function.
- **batch_size** (*sequence**of*[*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - if provided,
the resulting TensorDict will have the desired batch_size.
The [`batch_size`](tensordict.TensorDictBase.batch_size.html#tensordict.TensorDictBase.batch_size) argument should match the batch_size after
the transformation. This is a keyword only argument.
With a `batch_size`, the call is treated as a shape operation:
`fn` is not called on [`UnbatchedTensor`](tensordict.UnbatchedTensor.html#tensordict.UnbatchedTensor)
entries, which keep their data and take the new batch size.
- **device** ([*device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,**optional*) - the resulting device, if any.
- **names** ([*list*](https://docs.python.org/3/builtins/stdtypes.html#list)*of*[*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - the new dimension names, in case the
batch_size is modified.
- **inplace** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, the results are written into
the existing leaves of `self` (as [`set_()`](tensordict.TensorDictBase.set_.html#tensordict.TensorDictBase.set_) does), and
`self` is returned. Defaults to `False`. See
[In-place operations](../../concepts/operations.html#inplace-conventions).
- **default** (*Any**,**optional*) - default value for missing entries in the
other tensordicts. If not provided, missing entries will
raise a KeyError.
- **filter_empty** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, empty tensordicts will be
filtered out. This also comes with a lower computational cost as
empty data structures won't be created and destroyed. If `False`,
empty tensordicts are kept. Defaults to `None`, which behaves like
`True` (tensordicts left empty by `fn` are filtered out, and `None`
is returned if `fn` returns no value at all) except that tensordicts
that were already empty are kept.
- **propagate_lock** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, a locked tensordict will produce
another locked tensordict. Defaults to `False`.
- **call_on_nested** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - 

if `True`, the function will be called on first-level tensors
and containers (TensorDict or tensorclass). In this scenario, `func` is responsible of
propagating its calls to nested levels. This allows a fine-grained behaviour
when propagating the calls to nested tensordicts.
If `False`, the function will only be called on leaves, and `apply` will take care of dispatching
the function to all leaves.

```
>>> from tensordict import TensorDict, is_tensor_collection
>>> td = TensorDict({"a": {"b": [0.0, 1.0]}, "c": [1.0, 2.0]})
>>> def mean_tensor_only(val):
... if is_tensor_collection(val):
... raise RuntimeError("Unexpected!")
... return val.mean()
>>> td_mean = td.apply(mean_tensor_only)
>>> def mean_any(val):
... if is_tensor_collection(val):
... # Recurse
... return val.apply(mean_any, call_on_nested=True)
... return val.mean()
>>> td_mean = td.apply(mean_any, call_on_nested=True)
```
- **out** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*,**optional*) - 

a tensordict where to write the results. This can be used to avoid
creating a new tensordict:

```
>>> td = TensorDict({"a": 0})
>>> _ = td.apply(lambda x: x+1, out=td)
>>> assert (td==1).all()
```

Warning

If the operation executed on the tensordict requires multiple keys to be accessed for
a single computation, providing an `out` argument equal to `self` can cause the operation
to provide silently wrong results.
For instance:

```
>>> td = TensorDict({"a": 1, "b": 1})
>>> td.apply(lambda x: x+td["a"])["b"] # Right!
tensor(2)
>>> td.apply(lambda x: x+td["a"], out=td)["b"] # Wrong!
tensor(3)
```
- ****constructor_kwargs** - additional keyword arguments to be passed to the
TensorDict constructor.

Returns:

a new tensordict with the transformed tensors, or `self` if
`inplace=True`. `None` if `fn` returns `None` for every
leaf and `filter_empty` is not `False`.

The result has the same type as `self` for
[`TensorDict`](tensordict.TensorDict.html#tensordict.TensorDict),
[`LazyStackedTensorDict`](tensordict.LazyStackedTensorDict.html#tensordict.LazyStackedTensorDict) and tensorclasses.
Other types, for example
[`PersistentTensorDict`](tensordict.PersistentTensorDict.html#tensordict.PersistentTensorDict) or a memory-mapped
tensordict, give a [`TensorDict`](tensordict.TensorDict.html#tensordict.TensorDict) that lives in
memory.

A locked tensordict gives an unlocked result unless
`propagate_lock=True`.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict({
... "a": -torch.ones(3),
... "nested": {"a": torch.ones(3), "b": torch.zeros(3)}},
... batch_size=[3])
>>> def name_filter(name, tensor):
... if name == "a":
... return tensor
>>> td.named_apply(name_filter)
TensorDict(
 fields={
 a: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False),
 nested: TensorDict(
 fields={
 a: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([3]),
 device=None,
 is_shared=False)},
 batch_size=torch.Size([3]),
 device=None,
 is_shared=False)
>>> def name_filter(name, *tensors):
... if name == "a":
... r = 0
... for tensor in tensors:
... r = r + tensor
... return r
>>> out = td.named_apply(name_filter, td)
>>> print(out)
TensorDict(
 fields={
 a: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False),
 nested: TensorDict(
 fields={
 a: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([3]),
 device=None,
 is_shared=False)},
 batch_size=torch.Size([3]),
 device=None,
 is_shared=False)
>>> print(out["a"])
tensor([-2., -2., -2.])
```

Note

If `None` is returned by the function, the entry is ignored. This
can be used to filter the data in the tensordict:

```
>>> td = TensorDict({"1": 1, "2": 2, "b": {"2": 2, "1": 1}}, [])
>>> def name_filter(name, tensor):
... if name == "1":
... return tensor
>>> td.named_apply(name_filter)
TensorDict(
 fields={
 1: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int64, is_shared=False),
 b: TensorDict(
 fields={
 1: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int64, is_shared=False)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False)
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).