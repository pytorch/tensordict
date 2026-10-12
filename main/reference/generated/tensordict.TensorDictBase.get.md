# TensorDictBase.get

TensorDictBase.get(*key*)

TensorDictBase.get(*key*, *default*)

Returns the value stored at `key`, or `default` if the key is missing.

`td.get(key)` is the same as `td[key]`, except for a missing key:
`td[key]` raises `KeyError`, and `get` returns `default`.

Parameters:

- **key** ([*NestedKey*](tensordict.NestedKey.html#tensordict.NestedKey)) - the key to read. A tuple reads a nested entry:
`td.get(("a", "b"))` is `td["a"]["b"]`. See [Keys, nested keys, and locking](../../concepts/keys.html#keys).
- **default** (*optional*) - the value returned if the key is missing.
Defaults to `None`, as for [`dict.get()`](https://docs.python.org/3/builtins/stdtypes.html#dict.get).

Changed in version 0.7: A missing key returns `None` instead of raising `KeyError`.
`set_get_defaults_to_none(False)` and `TD_GET_DEFAULTS_TO_NONE=0`
restore the old behavior. They are deprecated and will be removed
in 0.17.

Note

Keyword arguments can be passed to `get()` when dealing with ragged tensors.
See [`get()`](tensordict.LazyStackedTensorDict.html#tensordict.LazyStackedTensorDict.get) for a complete overview.

Examples

```
>>> from tensordict import TensorDict
>>> td = TensorDict({"x": 1, "nested": {"y": 2}}, batch_size=[])
>>> td.get("x")
tensor(1)
>>> td.get(("nested", "y"))
tensor(2)
>>> print(td.get("z"))
None
>>> td.get("z", 0)
0
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).

## Tutorials that use `get`

![](../../_images/sphx_glr_tensordict_keys_thumb.png)

[Getting, setting, and nesting keys](../../tutorials/tensordict_keys.html)

Getting, setting, and nesting keys
![](../../_images/sphx_glr_tensordict_shapes_thumb.png)

[Changing the shape of a TensorDict](../../tutorials/tensordict_shapes.html)

Changing the shape of a TensorDict