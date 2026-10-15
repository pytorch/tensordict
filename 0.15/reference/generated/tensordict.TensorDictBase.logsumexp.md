# TensorDictBase.logsumexp

TensorDictBase.logsumexp(*dim=None*, *keepdim=False*, ***, *out=None*)

Returns the log of summed exponentials of each row of the input tensordict in the given dimension `dim`. The computation is numerically stabilized.

If keepdim is `True`, the output tensor is of the same size as input except in the dimension(s) `dim` where it is of size `1`.
Otherwise, `dim` is squeezed (see [`squeeze()`](https://docs.pytorch.org/docs/stable/generated/torch.squeeze.html#torch.squeeze)), resulting in the output tensor having 1 (or len(dim)) fewer dimension(s).

Parameters:

- **dim** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*or*[*tuple*](https://docs.python.org/3/builtins/stdtypes.html#tuple)*of**ints**,**optional*) - the dimension or dimensions to reduce. If `None`, all batch dimensions of the
tensordict are reduced.
- **keepdim** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)) - whether the output tensordict has dim retained or not.

Keyword Arguments:

**out** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*,**optional*) - the output tensordict.

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).