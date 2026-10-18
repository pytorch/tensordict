# TensorDictBase.apply

TensorDictBase.apply_(*fn: [Callable](https://docs.python.org/3/library/typing.html#typing.Callable)*, **others*, ***kwargs*) → Self

Applies a callable to every leaf and writes the results into the existing leaves.

Same as [`apply()`](tensordict.TensorDictBase.apply.html#tensordict.TensorDictBase.apply) with `inplace=True`: each result is copied
into the leaf it comes from (as [`set_()`](tensordict.TensorDictBase.set_.html#tensordict.TensorDictBase.set_) does), so the leaves
keep their storage.

Parameters:

- **fn** (*Callable*) - the function called on each tensor leaf. It returns
a tensor that can be copied into the leaf.
- ***others** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)) - other tensordicts with the same
structure. `fn` receives the leaf of each, after the leaf of
the tensordict, under the same key.
- ****kwargs** - the keyword arguments of [`apply()`](tensordict.TensorDictBase.apply.html#tensordict.TensorDictBase.apply).

Returns:

`self`.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict(a=torch.zeros(3), b=TensorDict(c=torch.ones(3)))
>>> a = td["a"]
>>> _ = td.apply_(lambda x: x + 1)
>>> a
tensor([1., 1., 1.])
>>> td["b", "c"]
tensor([2., 2., 2.])
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).