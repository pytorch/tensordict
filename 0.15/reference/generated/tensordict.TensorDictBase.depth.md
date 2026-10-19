# TensorDictBase.depth

*property*TensorDictBase.depth*: [int](https://docs.python.org/3/builtins/functions.html#int)*

the maximum number of levels of nested tensordicts.

A tensordict with no nested tensordict has depth 0. Empty nested
tensordicts do not count.

Examples

```
>>> from tensordict import TensorDict
>>> TensorDict({"a": 0}).depth
0
>>> TensorDict({"a": {"b": {"c": 0}}}).depth
2
```

Type:

The depth of the tensordict

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).