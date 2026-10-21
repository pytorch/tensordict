# TensorDictBase.from_module

*classmethod*TensorDictBase.from_module(*module*, *as_module: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *lock: [bool](https://docs.python.org/3/builtins/functions.html#bool) = True*, *use_state_dict: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*)

Returns a tensordict of the parameters and buffers of a module.

The leaves are the parameters and buffers of the module, not copies:
an in-place change to a leaf changes the module. Use
`TensorDict.from_module(module).data.clone()` to get independent
tensors. Submodules become sub-tensordicts: `module.layers[0].weight`
is the entry `("layers", "0", "weight")`. Write the parameters back
with [`to_module()`](tensordict.TensorDictBase.to_module.html#tensordict.TensorDictBase.to_module).

The result is a [`TensorDict`](tensordict.TensorDict.html#tensordict.TensorDict) (or a
`TensorDictParams` with `as_module=True`),
whatever class this method is called on.

Note

This method locks the result by default (`lock=True`), but its
override [`TensorDict.from_module`](tensordict.TensorDict.html#tensordict.TensorDict.from_module) does not.

Parameters:

- **module** (*nn.Module*) - the module to get the parameters from.
- **as_module** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, returns a
`TensorDictParams`, which can be stored
as a submodule of a [`torch.nn.Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module). Defaults to
`False`.
- **lock** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, the result is locked (see
[Locking and temporary changes](../../concepts/keys.html#keys-locking)). Defaults to `True`.
- **use_state_dict** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, reads the module's
`state_dict()`, so that its state-dict hooks run, and
unflattens it into the tree of the submodules. The leaves are
then the tensors of the state dict, not the parameters.
Defaults to `False`.

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
TensorDict(batch_size=[], fields={bias: f32[2048], weight: f32[2048,4]})
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).

## Tutorials that use `from_module`

![](../../_images/sphx_glr_functional_thumb.png)

[Functional calls and ensembles with TensorDict](../../tutorials/functional.html)

Functional calls and ensembles with TensorDict
![](../../_images/sphx_glr_tensordict_keys_thumb.png)

[Getting, setting, and nesting keys](../../tutorials/tensordict_keys.html)

Getting, setting, and nesting keys