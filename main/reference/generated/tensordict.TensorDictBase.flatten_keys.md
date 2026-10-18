# TensorDictBase.flatten_keys

TensorDictBase.flatten_keys(*separator: [str](https://docs.python.org/3/builtins/stdtypes.html#str) = '.'*, *inplace: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *is_leaf: [Callable](https://docs.python.org/3/library/typing.html#typing.Callable)[[[Type](https://docs.python.org/3/library/typing.html#typing.Type)], [bool](https://docs.python.org/3/builtins/functions.html#bool)] | [None](https://docs.python.org/3/builtins/constants.html#None) = None*) → Self

Returns a tensordict with the nested entries moved to the top level, under joined keys.

The leaf at `("a", "b", "c")` gets the key `"a.b.c"` with the
default separator. The leaves are not copied. The result has the batch
size, device and dimension names of the tensordict, and the type of
[`empty()`](tensordict.TensorDictBase.empty.html#tensordict.TensorDictBase.empty): a [`TensorDict`](tensordict.TensorDict.html#tensordict.TensorDict) for most types, a
lazy stack for a lazy stack. Two keys that become the same once joined
raise a `KeyError`. [`unflatten_keys()`](tensordict.TensorDictBase.unflatten_keys.html#tensordict.TensorDictBase.unflatten_keys) does the reverse.

It can be used as a context manager: the changes made to the flat
tensordict in the `with` block are written back to the tensordict
when the block exits.

Parameters:

- **separator** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - the string that joins the parts of a
nested key. Defaults to `"."`.
- **inplace** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, the entries are moved in
the tensordict itself, which is returned. Defaults to `False`.
- **is_leaf** (*Callable**,**optional*) - a function that takes a class and
returns `True` if values of that class count as leaves. Here
it stops the flattening: a nested tensordict whose class is a
leaf is kept as one value under its key. Defaults to `None`,
which uses [`is_leaf_nontensor()`](tensordict.is_leaf_nontensor.html#tensordict.is_leaf_nontensor): tensors and
non-tensor data are leaves, tensordicts and tensorclasses are
flattened.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict({"a": 1, ("b", "c"): 2, ("e", "f", "g"): 3})
>>> list(td.flatten_keys().keys())
['a', 'b.c', 'e.f.g']
>>> list(td.flatten_keys(separator="/").keys())
['a', 'b/c', 'e/f/g']
>>> with td.flatten_keys() as flat:
... flat["b.c"] += 1
>>> td["b", "c"]
tensor(3)
```

With [`unflatten_keys()`](tensordict.TensorDictBase.unflatten_keys.html#tensordict.TensorDictBase.unflatten_keys), it converts a state dict to a
nested tensordict and back:

```
>>> model = torch.nn.Sequential(torch.nn.Linear(3, 4))
>>> params = TensorDict(model.state_dict()).unflatten_keys(".")
>>> sorted(params.keys(include_nested=True, leaves_only=True))
[('0', 'bias'), ('0', 'weight')]
>>> _ = model.load_state_dict(params.flatten_keys(".").to_dict())
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).

## Tutorials that use `flatten_keys`

![](../../_images/sphx_glr_functional_thumb.png)

[Functional calls and ensembles with TensorDict](../../tutorials/functional.html)

Functional calls and ensembles with TensorDict
![](../../_images/sphx_glr_serialization_speed_thumb.png)

[Benchmarking TensorDict serialization speed](../../tutorials/serialization_speed.html)

Benchmarking TensorDict serialization speed
![](../../_images/sphx_glr_tensordict_keys_thumb.png)

[Getting, setting, and nesting keys](../../tutorials/tensordict_keys.html)

Getting, setting, and nesting keys