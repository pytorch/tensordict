# TypedTensorDict

Warning

`TypedTensorDict` is experimental: its API can change in a minor release
without a deprecation cycle.

This page lists [`TypedTensorDict`](generated/tensordict.TypedTensorDict.html#tensordict.TypedTensorDict), a
[`TensorDictBase`](generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase) subclass whose fields you declare as
class annotations, as in a `TypedDict`. A subclass adds fields to the
fields of its parent, and [`from_tensordict()`](generated/tensordict.TypedTensorDict.html#tensordict.TypedTensorDict.from_tensordict)
wraps any tensordict (H5, Redis, lazy stacks) without a copy. For a guide
with examples, see [TypedTensorDict](../typed_tensordict.html#typed-tensordict-guide).

`TypedTensorDict` has the methods of [`TensorDictBase`](generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase),
listed by task in [Methods by task](td.html#api-tensordict-methods).

| [`TypedTensorDict`](generated/tensordict.TypedTensorDict.html#tensordict.TypedTensorDict)(*args, **kwargs) | A [`TensorDictBase`](generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase) subclass whose fields are declared as class annotations. |
| --- | --- |

## See also

- [TypedTensorDict](../typed_tensordict.html#typed-tensordict-guide): declare and use a `TypedTensorDict`.
- [Which container should I use?](../choosing_a_container.html#choosing-a-container): tensorclass, `TypedTensorDict`, or
`TensorDict`.
- [Containers on storage backends](../compatibility.html#compatibility): which containers work on which storage backends.