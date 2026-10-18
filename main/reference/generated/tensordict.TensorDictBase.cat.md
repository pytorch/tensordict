# TensorDictBase.cat

*classmethod*TensorDictBase.cat(*input*, *dim: [int](https://docs.python.org/3/builtins/functions.html#int) = 0*, ***, *out=None*)

Concatenates tensordicts along an existing batch dimension `dim`.

It is equivalent to [`torch.cat()`](https://docs.pytorch.org/docs/stable/generated/torch.cat.html#torch.cat), and also works under
[`torch.compile()`](https://docs.pytorch.org/docs/stable/generated/torch.compile.html#torch.compile). The tensordicts must have the same keys, or a
`KeyError` is raised, and the same batch size except along `dim`.

Parameters:

- **input** (*sequence**of*[*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)) - the tensordicts to concatenate.
- **dim** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the batch dimension to concatenate along.
Defaults to `0`.

Keyword Arguments:

**out** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*,**optional*) - a tensordict to write the result
into. Defaults to `None`.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> a = TensorDict(x=torch.zeros(3), batch_size=[3])
>>> b = TensorDict(x=torch.ones(3), batch_size=[3])
>>> TensorDict.cat([a, b]).batch_size
torch.Size([6])
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).