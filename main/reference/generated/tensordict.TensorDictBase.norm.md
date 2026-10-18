# TensorDictBase.norm

TensorDictBase.norm(***, *out=None*, *dtype: [dtype](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *key_transform: [Callable](https://docs.python.org/3/library/typing.html#typing.Callable)[[[NestedKey](tensordict.NestedKey.html#tensordict.NestedKey)], [NestedKey](tensordict.NestedKey.html#tensordict.NestedKey)] | [None](https://docs.python.org/3/builtins/constants.html#None) = None*) → Self

Returns a tensordict with the norm of each leaf.

Each tensor leaf is replaced by its 2-norm, computed over all its
elements, so the result has an empty batch size. Non-tensor data is
kept as it is.

Keyword Arguments:

- **out** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*,**optional*) - a tensordict to write the norms
into, with the keys of the tensordict and an empty batch size.
Defaults to `None`.
- **dtype** ([*dtype*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype)*,**optional*) - the dtype of the norms. Defaults to
`None`, which keeps the dtype of each leaf.
- **key_transform** (*Callable**,**optional*) - a function that renames the
keys of the result. It is called on each string key, level by
level: `("a", "b")` becomes
`(key_transform("a"), key_transform("b"))`. Defaults to
`None`, which keeps the keys.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict(a=torch.full((3, 4), 2.0), b=TensorDict(c=torch.ones(3)), batch_size=[3])
>>> norms = td.norm()
>>> norms["a"], norms["b", "c"], norms.batch_size
(tensor(6.9282), tensor(1.7321), torch.Size([]))
>>> list(td.norm(key_transform=lambda key: f"{key}_norm").keys(True, True))
['a_norm', ('b_norm', 'c_norm')]
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).