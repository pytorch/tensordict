# default_is_leaf

tensordict.default_is_leaf(*cls: [Type](https://docs.python.org/3/library/typing.html#typing.Type)*) → [bool](https://docs.python.org/3/builtins/functions.html#bool)

Returns `True` if a type is not a tensor collection (tensordict or tensorclass), or is a pass-through type.

Pass-through types (like UnbatchedTensor) have `_pass_through=True` and are considered leaves
because their shape doesn't conform to batch dimensions.

Note: NonTensorData types are NOT considered leaves here (they have `_is_non_tensor=True`
but not `_pass_through=True`), so they are excluded from leaves when `leaves_only=True`.

Examples

```
>>> from tensordict import TensorDict, default_is_leaf
>>> import torch
>>> td = TensorDict(a={}, b="a string!", c=torch.randn(()))
>>> print(td.keys(leaves_only=True, is_leaf=default_is_leaf))
_TensorDictKeysView(['c'],
 include_nested=False,
 leaves_only=True)
```

See also

[`is_leaf_nontensor()`](tensordict.is_leaf_nontensor.html#tensordict.is_leaf_nontensor).