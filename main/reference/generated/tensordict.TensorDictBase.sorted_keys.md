# TensorDictBase.sorted_keys

*property*TensorDictBase.sorted_keys*: [list](https://docs.python.org/3/builtins/stdtypes.html#list)[[NestedKey](tensordict.NestedKey.html#tensordict.NestedKey)]*

The top-level keys of the tensordict, as a list sorted in alphabetical order.

For nested keys, call `keys(include_nested=True, sort=True)`. On a
locked tensordict, the sorted keys are cached until the tensordict is
unlocked.

Examples

```
>>> from tensordict import TensorDict
>>> td = TensorDict(b=0, a=1, c=TensorDict(d=2))
>>> td.sorted_keys
['a', 'b', 'c']
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).