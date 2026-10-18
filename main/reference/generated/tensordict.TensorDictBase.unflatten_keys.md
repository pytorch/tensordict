# TensorDictBase.unflatten_keys

TensorDictBase.unflatten_keys(*separator: [str](https://docs.python.org/3/builtins/stdtypes.html#str) = '.'*, *inplace: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*) → Self

Returns a tensordict with the keys that hold the separator split into nested keys.

The leaf at `"a.b.c"` moves to `("a", "b", "c")` with the default
separator. The leaves are not copied. The new nested tensordicts take
the batch size, dimension names and device of the tensordict. The
result has the type of the tensordict for a
[`TensorDict`](tensordict.TensorDict.html#tensordict.TensorDict) and a lazy stack. A split key that
would overwrite an existing entry raises a `KeyError`.
[`flatten_keys()`](tensordict.TensorDictBase.flatten_keys.html#tensordict.TensorDictBase.flatten_keys) does the reverse.

It can be used as a context manager: the changes made to the nested
tensordict in the `with` block are written back to the tensordict
when the block exits.

Parameters:

- **separator** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - the string that separates the parts of
a nested key. Defaults to `"."`.
- **inplace** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, the entries are moved in
the tensordict itself, which is returned. Defaults to `False`.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict({"a": 1, "b.c": 2, "e.f.g": 3})
>>> nested = td.unflatten_keys()
>>> list(nested.keys(include_nested=True, leaves_only=True, sort=True))
['a', ('b', 'c'), ('e', 'f', 'g')]
>>> nested["e", "f", "g"]
tensor(3)
>>> td = TensorDict({"a/b": torch.zeros(3)}, batch_size=[3])
>>> td.unflatten_keys("/")["a"].batch_size
torch.Size([3])
```

With [`flatten_keys()`](tensordict.TensorDictBase.flatten_keys.html#tensordict.TensorDictBase.flatten_keys), it converts a state dict to a nested
tensordict and back:

```
>>> model = torch.nn.Sequential(torch.nn.Linear(3, 4))
>>> params = TensorDict(model.state_dict()).unflatten_keys(".")
>>> sorted(params.keys(include_nested=True, leaves_only=True))
[('0', 'bias'), ('0', 'weight')]
>>> _ = model.load_state_dict(params.flatten_keys(".").to_dict())
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).

## Tutorials that use `unflatten_keys`

![](../../_images/sphx_glr_tensordict_keys_thumb.png)

[Getting, setting, and nesting keys](../../tutorials/tensordict_keys.html)

Getting, setting, and nesting keys