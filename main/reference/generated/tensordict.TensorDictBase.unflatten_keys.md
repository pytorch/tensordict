# TensorDictBase.unflatten_keys

TensorDictBase.unflatten_keys(*separator: [str](https://docs.python.org/3/builtins/stdtypes.html#str) = '.'*, *inplace: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*) → Self

Converts a flat tensordict into a nested one, recursively.

The TensorDict type will be lost and the result will be a simple TensorDict instance.
The metadata of the nested tensordicts will be inferred from the root:
all instances across the data tree will share the same batch-size,
dimension names and device.

Parameters:

- **separator** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - the separator between the nested items.
- **inplace** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, the resulting tensordict will
have the same identity as the one where the call has been made.
Defaults to `False`.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> data = TensorDict({"a": 1, "b - c": 2, "e - f - g": 3}, batch_size=[])
>>> data.unflatten_keys(separator=" - ")
TensorDict(
 fields={
 a: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int64, is_shared=False),
 b: TensorDict(
 fields={
 c: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int64, is_shared=False)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False),
 e: TensorDict(
 fields={
 f: TensorDict(
 fields={
 g: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int64, is_shared=False)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False)
```

This method and `unflatten_keys()` are particularly useful when
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

## Tutorials that use `unflatten_keys`

![](../../_images/sphx_glr_tensordict_keys_thumb.png)

[Getting, setting and nesting keys](../../tutorials/tensordict_keys.html)

Getting, setting and nesting keys