# TensorDictBase.sub

TensorDictBase.sub_(*other: [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase) | [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | [float](https://docs.python.org/3/builtins/functions.html#float)*, *alpha: [float](https://docs.python.org/3/builtins/functions.html#float) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*)

Subtracts `other`, scaled by `alpha`, from every leaf in place and returns `self`.

The in-place version of [`sub()`](tensordict.TensorDictBase.sub.html#tensordict.TensorDictBase.sub), called by `td -= other`. It
has no `default`: a tensordict `other` must have every key of the
tensordict, and its other keys are ignored.

Parameters:

- **other** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*,*[*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*or**Number*) - the value to
subtract, paired with the leaves as in [`sub()`](tensordict.TensorDictBase.sub.html#tensordict.TensorDictBase.sub).
- **alpha** (*Number**,**optional*) - the multiplier for `other`. Defaults to
`None`, which subtracts `other` as it is.

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).