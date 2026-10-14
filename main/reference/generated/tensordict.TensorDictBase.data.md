# TensorDictBase.data

*property*TensorDictBase.data*: [Any](https://docs.python.org/3/library/typing.html#typing.Any)*

Returns a tensordict containing the .data attributes of the leaf tensors.

The leaves of the result share storage with the leaves of `self`,
and do not require gradients. An in-place change to the result
changes `self`.

Assigning `td.data = other` rebinds the `.data` of each leaf of
`self` to the matching leaf of `other`, as `torch.Tensor.data`
does. The leaves then share storage with `other`: memory-mapped or
shared-memory leaves stop writing to their file or shared memory.
`other` and each of its sub-tensordicts must have the batch size of
the matching tensordict in `self`, and each key of `other` must be
in `self`.

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).