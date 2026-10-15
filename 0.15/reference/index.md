# API reference

Each page of the API reference lists the public classes and functions of
one part of tensordict, and each name has its own page. To learn how the
parts fit together, start with [What is TensorDict?](../overview.html#overview) and the user guide.

- [tensordict: containers and functions](td.html#api-tensordict): the container classes (`TensorDict`,
`LazyStackedTensorDict`, `PersistentTensorDict`, ...), the methods
that all tensordicts share, grouped by task, and the functions of the
`tensordict` module.
- [tensorclass and non-tensor data](tc.html#api-tensorclass): `TensorClass`, the `@tensorclass`
decorator, and the classes that hold non-tensor data.
- [TypedTensorDict](ttd.html#api-typed-tensordict): `TypedTensorDict`, a tensordict with
declared fields.
- [tensordict.nn](nn.html#api-nn): the modules of `tensordict.nn`, the probabilistic
modules and distributions, and FX tracing.
- [Key-value stores and UCXX](storage.html#api-storage): the tensordicts on key-value servers
(`tensordict.store`), and UCXX transport.
- [Utilities](utils.html#api-utils): the helpers and types of `tensordict.utils`.