# TensorDictBase.addcdiv

TensorDictBase.addcdiv(*other1: [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase) | [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*, *other2: [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase) | [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*, *value: [float](https://docs.python.org/3/builtins/functions.html#float) | [None](https://docs.python.org/3/builtins/constants.html#None) = 1*) → Self

Performs the element-wise division of `other1` by `other2`, multiplies the result by the scalar `value` and adds it to `self`.

\[\text{out}_i = \text{input}_i + \text{value} \times \frac{\text{other1}_i}{\text{other2}_i}\]

The shapes of the elements of `self`, `other1`, and `other2` must be
broadcastable.

For inputs of type FloatTensor or DoubleTensor, `value` must be
a real number, otherwise an integer.

Parameters:

- **other1** ([*TensorDict*](tensordict.TensorDict.html#tensordict.TensorDict)*or**Tensor*) - the numerator tensordict (or tensor)
- **other2** ([*TensorDict*](tensordict.TensorDict.html#tensordict.TensorDict)*or**Tensor*) - the denominator tensordict (or tensor)

Keyword Arguments:

**value** (*Number**,**optional*) - multiplier for \(\text{other1} / \text{other2}\)

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).