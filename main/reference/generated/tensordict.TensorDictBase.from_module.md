# TensorDictBase.from_module

*classmethod*TensorDictBase.from_module(*module*, *as_module: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *lock: [bool](https://docs.python.org/3/builtins/functions.html#bool) = True*, *use_state_dict: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*)

Returns a tensordict of the parameters and buffers of a module.

The leaves are the parameters and buffers of the module, not copies:
an in-place change to a leaf changes the module. Use
`TensorDict.from_module(module).data.clone()` to get independent
tensors. Submodules become sub-tensordicts: `module.layers[0].weight`
is the entry `("layers", "0", "weight")`. Write the parameters back
with [`to_module()`](tensordict.TensorDictBase.to_module.html#tensordict.TensorDictBase.to_module).

Parameters:

- **module** (*nn.Module*) - the module to get the parameters from.
- **as_module** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, a `TensorDictParams`
instance will be returned which can be used to store parameters
within a [`torch.nn.Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module). Defaults to `False`.
- **lock** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, the resulting tensordict will be locked.
Defaults to `True`.
- **use_state_dict** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - 

if `True`, the state-dict from the
module will be used and unflattened into a TensorDict with
the tree structure of the model. Defaults to `False`.

Note

This is particularly useful when state-dict hooks have to be used.

Examples

```
>>> from tensordict import TensorDict
>>> from torch import nn
>>> module = nn.TransformerDecoder(
... decoder_layer=nn.TransformerDecoderLayer(nhead=4, d_model=4),
... num_layers=1
... )
>>> params = TensorDict.from_module(module)
>>> print(params["layers", "0", "linear1"])
TensorDict(
 fields={
 bias: Parameter(shape=torch.Size([2048]), device=cpu, dtype=torch.float32, is_shared=False),
 weight: Parameter(shape=torch.Size([2048, 4]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False)
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).

## Tutorials that use `from_module`

![](../../_images/sphx_glr_functional_thumb.png)

[Functional calls and ensembles with TensorDict](../../tutorials/functional.html)

Functional calls and ensembles with TensorDict
![](../../_images/sphx_glr_tensordict_keys_thumb.png)

[Getting, setting and nesting keys](../../tutorials/tensordict_keys.html)

Getting, setting and nesting keys