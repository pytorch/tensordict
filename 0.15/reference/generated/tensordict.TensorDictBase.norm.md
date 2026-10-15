# TensorDictBase.norm

TensorDictBase.norm(***, *out=None*, *dtype: [dtype](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *key_transform: [Callable](https://docs.python.org/3/library/typing.html#typing.Callable)[[[NestedKey](tensordict.NestedKey.html#tensordict.NestedKey)], [NestedKey](tensordict.NestedKey.html#tensordict.NestedKey)] | [None](https://docs.python.org/3/builtins/constants.html#None) = None*) → Self

Computes the norm of each tensor in the tensordict.

Keyword Arguments:

- **out** ([*TensorDict*](tensordict.TensorDict.html#tensordict.TensorDict)*,**optional*) - the output tensordict.
- **dtype** ([*dtype*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype)*,**optional*) - the output dtype.
- **key_transform** (*Callable**[**[*[*NestedKey*](tensordict.NestedKey.html#tensordict.NestedKey)*]**,*[*NestedKey*](tensordict.NestedKey.html#tensordict.NestedKey)*]**,**optional*) - A function to transform key names.
If provided, all keys in the result will be transformed using this function.
For string keys, the function receives a string. For tuple keys, it receives a tuple.
Default: `None`.

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).