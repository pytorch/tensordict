# TensorDictBase.logsumexp

TensorDictBase.logsumexp(*dim=None*, *keepdim=False*, ***, *out=None*)

Returns the log of the summed exponentials of every leaf along batch dimensions.

The computation is numerically stable, as with [`torch.logsumexp()`](https://docs.pytorch.org/docs/stable/generated/torch.logsumexp.html#torch.logsumexp).
Unlike [`sum()`](tensordict.TensorDictBase.sum.html#tensordict.TensorDictBase.sum), it never reduces the feature dimensions: without
`dim`, it reduces every batch dimension.

Parameters:

- **dim** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*or*[*tuple*](https://docs.python.org/3/builtins/stdtypes.html#tuple)*of*[*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the batch dimensions to reduce. A
negative `dim` counts from the end of the batch dimensions.
Defaults to `None`, which reduces all the batch dimensions.
- **keepdim** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, the reduced batch dimensions are
kept with size 1. Defaults to `False`.

Keyword Arguments:

**out** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*,**optional*) - the tensordict to write the result into,
which is returned. Defaults to `None`, which returns a new
tensordict.

Raises:

[**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) - if `dim` is not a batch dimension.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict(a=torch.zeros(2, 3), batch_size=[2])
>>> td.logsumexp()["a"] # log(exp(0) + exp(0))
tensor([0.6931, 0.6931, 0.6931])
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).