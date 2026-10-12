# TensorDictBase.refine_names

TensorDictBase.refine_names(**names*) → Self

Refines the dimension names of self according to names.

Refining is a special case of renaming that "lifts" unnamed dimensions.
A None dim can be refined to have any name; a named dim can only be
refined to have the same name.

Because named tensors can coexist with unnamed tensors, refining names
gives a nice way to write named-tensor-aware code that works with both
named and unnamed tensors.

names may contain up to one Ellipsis (...). The Ellipsis is expanded
greedily; it is expanded in-place to fill names to the same length as
self.dim() using names from the corresponding indices of self.names.

Returns: the same tensordict with dimensions named according to the input.

Examples

```
>>> from tensordict import TensorDict
>>> td = TensorDict({}, batch_size=[3, 4, 5, 6])
>>> tdr = td.refine_names(None, None, None, "d")
>>> assert tdr.names == [None, None, None, "d"]
>>> tdr = td.refine_names("a", None, None, "d")
>>> assert tdr.names == ["a", None, None, "d"]
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).