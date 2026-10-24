# tensorclass and non-tensor data

This page lists [`TensorClass`](generated/tensordict.TensorClass.html#tensordict.TensorClass), the class form of a
tensordict, and the classes for non-tensor data and unbatched entries. A
*tensorclass* is a class that you declare like a dataclass, with typed fields
and methods, and that has the tensor operations of a
[`TensorDict`](generated/tensordict.TensorDict.html#tensordict.TensorDict). For a guide to fields, auto-casting, the
class options, and static type checking, see [TensorClass: a dataclass of tensors](../tensorclass.html#tensorclass-guide).

To declare a tensorclass, subclass [`TensorClass`](generated/tensordict.TensorClass.html#tensordict.TensorClass). The
[`tensorclass()`](generated/tensordict.tensorclass.html#tensordict.tensorclass) decorator is the original way, kept for
existing code (see [Legacy: the @tensorclass decorator](../tensorclass.html#tensorclass-legacy-decorator)).

A tensorclass has the methods of [`TensorDictBase`](generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase),
listed by task in [Methods by task](td.html#api-tensordict-methods). On a tensorclass, the
methods that return a `TensorDict` on a `TensorDict` return an
instance of the tensorclass. Examples are `select`, `clone`, `to`,
`apply`, indexing, and reshaping.
On a tensorclass, [`rename_keys()`](generated/tensordict.TensorDictBase.rename_keys.html#tensordict.TensorDictBase.rename_keys) raises
`AttributeError` for a destination that is not a field, and keeps the
sub-tensordict fields that it empties.

## Tensorclasses

| [`TensorClass`](generated/tensordict.TensorClass.html#tensordict.TensorClass)(*args, **kwargs) | A base class for dataclass-like containers of tensors. |
| --- | --- |

| [`tensorclass`](generated/tensordict.tensorclass.html#tensordict.tensorclass)([cls, autocast, frozen, nocast, ...]) | A decorator that turns a class with annotated fields into a tensorclass. |
| --- | --- |
| [`from_dataclass`](generated/tensordict.from_dataclass.html#tensordict.from_dataclass)(obj, *[, dest_cls, ...]) | Converts a dataclass instance to a tensorclass instance, or a dataclass type to a tensorclass type. |

## Non-tensor data

A tensordict or a tensorclass can hold values that are not tensors, such
as strings or Python objects. It wraps each one in a
[`NonTensorData`](generated/tensordict.NonTensorData.html#tensordict.NonTensorData), and a stack of them in a
[`NonTensorStack`](generated/tensordict.NonTensorStack.html#tensordict.NonTensorStack). If you stack
[`MetaData`](generated/tensordict.MetaData.html#tensordict.MetaData) entries that hold the same object, the
result is one `MetaData`, not a `NonTensorStack`.
[`TensorAttrs`](generated/tensordict.TensorAttrs.html#tensordict.TensorAttrs) describes the device, dtype, and shape of
an entry: [`attrs()`](generated/tensordict.TensorDictBase.attrs.html#tensordict.TensorDictBase.attrs) returns them. See
[Non-tensor data and unbatched entries](../concepts/non_tensor.html#non-tensor-data).

| [`NonTensorDataBase`](generated/tensordict.NonTensorDataBase.html#tensordict.NonTensorDataBase)(data[, _metadata, ...]) | A base class for tensordict entries that hold a Python object instead of a tensor. |
| --- | --- |
| [`NonTensorData`](generated/tensordict.NonTensorData.html#tensordict.NonTensorData)(data[, _metadata, ...]) | A tensordict entry that holds a Python object instead of a tensor. |
| [`NonTensorStack`](generated/tensordict.NonTensorStack.html#tensordict.NonTensorStack)(*args, **kwargs) | A lazy stack of [`NonTensorData`](generated/tensordict.NonTensorData.html#tensordict.NonTensorData), one value per batch element. |
| [`MetaData`](generated/tensordict.MetaData.html#tensordict.MetaData)(data[, _metadata, _is_non_tensor, ...]) | A tensordict entry that holds one Python object for the whole batch. |
| [`TensorAttrs`](generated/tensordict.TensorAttrs.html#tensordict.TensorAttrs)([tgt_device, tgt_dtype, ...]) | A tensorclass that holds the device, dtype and shape of one tensor. |

## Entries that ignore the batch shape

Warning

`UnbatchedTensor` is experimental: its API can change in a minor release
without a deprecation cycle.

An [`UnbatchedTensor`](generated/tensordict.UnbatchedTensor.html#tensordict.UnbatchedTensor) is a tensor that shape operations
on its tensordict (indexing, reshaping, stacking) leave unchanged.
Operations on its data, such as arithmetic or
[`to()`](generated/tensordict.TensorDictBase.to.html#tensordict.TensorDictBase.to), still apply to it. See
[UnbatchedTensor](../concepts/non_tensor.html#unbatched-tensor).

| [`UnbatchedTensor`](generated/tensordict.UnbatchedTensor.html#tensordict.UnbatchedTensor)(data[, batch_size]) | A tensor entry that has no batch dimensions and that batch operations leave unchanged. |
| --- | --- |

## See also

- [TensorClass: a dataclass of tensors](../tensorclass.html#tensorclass-guide): declare and use a tensorclass.
- [Which container should I use?](../choosing_a_container.html#choosing-a-container): tensorclass, `TypedTensorDict`, or
`TensorDict`.
- [TypedTensorDict](ttd.html#api-typed-tensordict): the other container with typed fields.
- The tutorials [Using tensorclasses for datasets](../tutorials/tensorclass_fashion.html),
[Batched data loading with tensorclasses](../tutorials/tensorclass_imagenet.html), and
[Simulating particles with a tensorclass](../tutorials/tensorclass_particles.html).