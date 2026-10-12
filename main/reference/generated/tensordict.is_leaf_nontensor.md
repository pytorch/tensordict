# is_leaf_nontensor

tensordict.is_leaf_nontensor(*cls: [Type](https://docs.python.org/3/library/typing.html#typing.Type)*) → [bool](https://docs.python.org/3/builtins/functions.html#bool)

Returns `True` if a type is not a tensor collection (tensordict or tensorclass) or is a non-tensor.

Examples

```
>>> from tensordict import TensorDict, is_leaf_nontensor
>>> import torch
>>> td = TensorDict(a={}, b="a string!", c=torch.randn(()))
>>> print(td.keys(leaves_only=True, is_leaf=is_leaf_nontensor))
_TensorDictKeysView(['b', 'c'],
 include_nested=False,
 leaves_only=True)
```

See also

[`default_is_leaf()`](tensordict.default_is_leaf.html#tensordict.default_is_leaf).