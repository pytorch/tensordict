# TensorDictBase.flatten_keys

TensorDictBase.flatten_keys(*separator: [str](https://docs.python.org/3/builtins/stdtypes.html#str) = '.'*, *inplace: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *is_leaf: [Callable](https://docs.python.org/3/library/typing.html#typing.Callable)[[[Type](https://docs.python.org/3/library/typing.html#typing.Type)], [bool](https://docs.python.org/3/builtins/functions.html#bool)] | [None](https://docs.python.org/3/builtins/constants.html#None) = None*) → Self

Converts a nested tensordict into a flat one, recursively.

The TensorDict type will be lost and the result will be a simple TensorDict instance.

Parameters:

- **separator** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - the separator between the nested items.
- **inplace** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, the resulting tensordict will
have the same identity as the one where the call has been made.
Defaults to `False`.
- **is_leaf** (*callable**,**optional*) - a function that takes a class and
returns `True` if values of that class count as leaves. Here
it stops the flattening: a sub-tensordict whose class is a
leaf is kept as one value under its key. By default,
tensordicts and tensorclasses are not leaves. See
[`is_leaf_nontensor()`](tensordict.is_leaf_nontensor.html#tensordict.is_leaf_nontensor) and
[`default_is_leaf()`](tensordict.default_is_leaf.html#tensordict.default_is_leaf).

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> data = TensorDict({"a": 1, ("b", "c"): 2, ("e", "f", "g"): 3}, batch_size=[])
>>> data.flatten_keys(separator=" - ")
TensorDict(
 fields={
 a: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int64, is_shared=False),
 b - c: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int64, is_shared=False),
 e - f - g: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int64, is_shared=False)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False)
```

This method and [`unflatten_keys()`](tensordict.TensorDictBase.unflatten_keys.html#tensordict.TensorDictBase.unflatten_keys) are particularly useful when
handling state-dicts, as they make it possible to seamlessly convert
flat dictionaries into data structures that mimic the structure of the
model.

Examples

```
>>> model = torch.nn.Sequential(torch.nn.Linear(3 ,4))
>>> ddp_model = torch.ao.quantization.QuantWrapper(model)
>>> state_dict = TensorDict(ddp_model.state_dict(), batch_size=[]).unflatten_keys(".")
>>> print(state_dict)
TensorDict(
 fields={
 module: TensorDict(
 fields={
 0: TensorDict(
 fields={
 bias: Tensor(shape=torch.Size([4]), device=cpu, dtype=torch.float32, is_shared=False),
 weight: Tensor(shape=torch.Size([4, 3]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False)
>>> model_state_dict = state_dict.get("module")
>>> print(model_state_dict)
TensorDict(
 fields={
 0: TensorDict(
 fields={
 bias: Tensor(shape=torch.Size([4]), device=cpu, dtype=torch.float32, is_shared=False),
 weight: Tensor(shape=torch.Size([4, 3]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False)
>>> _ = model.load_state_dict(dict(model_state_dict.flatten_keys(".")))
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