# TensorDictBase.cat_from_tensordict

TensorDictBase.cat_from_tensordict(*dim: [int](https://docs.python.org/3/builtins/functions.html#int) = 0*, ***, *sorted: [bool](https://docs.python.org/3/builtins/functions.html#bool) | [List](https://docs.python.org/3/library/typing.html#typing.List)[[NestedKey](tensordict.NestedKey.html#tensordict.NestedKey)] | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *out: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*) → [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

Concatenates all entries of a tensordict in a single tensor.

Parameters:

**dim** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the dimension along which the entries should be concatenated.

Keyword Arguments:

- **sorted** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)*or*[*list*](https://docs.python.org/3/builtins/stdtypes.html#list)*of**NestedKeys*) - if `True`, the entries will be concatenated in alphabetical order.
If `False` (default), the dict order will be used. Alternatively, a list of key names can be provided
and the tensors will be concatenated accordingly. This incurs some overhead as the list of keys will
be checked against the list of leaf names in the tensordict.
- **out** ([*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*,**optional*) - an optional destination tensor for the cat operation.

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).