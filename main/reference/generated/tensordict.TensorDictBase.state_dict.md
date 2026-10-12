# TensorDictBase.state_dict

TensorDictBase.state_dict(*destination=None*, *prefix=''*, *keep_vars=False*, *flatten=True*) → [OrderedDict](https://docs.python.org/3/library/typing.html#typing.OrderedDict)[[str](https://docs.python.org/3/builtins/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)]

Produces a state_dict from the tensordict.

The state-dict is flat by default (dot-separated keys), following the
convention of [`torch.nn.Module.state_dict()`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.state_dict). Set `flatten=False`
for a nested structure.

Metadata (batch_size, device) is stored in a `_metadata` attribute on
the returned OrderedDict, following the same convention as
[`torch.nn.Module.state_dict()`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.state_dict). In flat mode, metadata for each
nesting level is stored under its dot-separated prefix (`""` for root,
`"sub"` for a nested tensordict at key `"sub"`, etc.). In nested
mode, each nested OrderedDict carries its own `_metadata`.

Parameters:

- **destination** ([*dict*](https://docs.python.org/3/builtins/stdtypes.html#dict)*,**optional*) - If provided, the state of tensordict will
be updated into the dict and the same object is returned.
Otherwise, an `OrderedDict` will be created and returned.
Default: `None`.
- **prefix** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - a prefix added to tensor
names to compose the keys in state_dict. Default: `''`.
- **keep_vars** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - by default the [`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) items
returned in the state dict are detached from autograd. If it's
set to `True`, detaching will not be performed.
Default: `False`.
- **flatten** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - whether the structure should be flattened
with the `"."` character or not.
Defaults to `True`.

Examples

```
>>> from tensordict import TensorDict
>>> data = TensorDict({"1": 1, "2": 2, "3": {"3": 3}}, [])
>>> sd = data.state_dict()
>>> print(dict(sd))
{'1': tensor(1), '2': tensor(2), '3.3': tensor(3)}
>>> print(dict(sd._metadata))
{'': {'batch_size': torch.Size([]), 'device': None}, '3': {'batch_size': torch.Size([]), 'device': None}}
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).