# TensorDictBase.where

TensorDictBase.where(*condition: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*, *other: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*, ***, *out: [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *pad: [int](https://docs.python.org/3/builtins/functions.html#int) | [bool](https://docs.python.org/3/builtins/functions.html#bool) = None*, *update_batch_size: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*) → Self

Return a `TensorDict` of elements selected from either self or other, depending on condition.

Parameters:

- **condition** (*BoolTensor*) - When `True` (nonzero), yields `self`,
otherwise yields `other`.
- **other** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*or**Scalar*) - value (if `other` is a scalar)
or values selected at indices where condition is `False`.

Keyword Arguments:

- **out** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*,**optional*) - the output `TensorDictBase` instance.
- **pad** (*scalar**,**optional*) - if provided, missing keys from the source
or destination tensordict will be written as torch.where(mask, self, pad)
or torch.where(mask, pad, other). Defaults to `None`, ie
missing keys are not tolerated.
- **update_batch_size** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)*,**optional*) - if `True` and `out` is provided, the batch size of the output will be
updated to match the batch size of the condition. Defaults to `False`.

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).