# TensorDictBase.add

TensorDictBase.add_(*other: [TensorCollection](tensordict.TensorCollection.html#tensordict.TensorCollection) | [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | [float](https://docs.python.org/3/builtins/functions.html#float)*, ***, *alpha: [float](https://docs.python.org/3/builtins/functions.html#float) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*) → Self

Adds `other`, scaled by `alpha`, to every leaf in place and returns `self`.

The in-place version of [`add()`](tensordict.TensorDictBase.add.html#tensordict.TensorDictBase.add), called by `td += other`. It
has no `default`: a tensordict `other` must have every key of the
tensordict, and its other keys are ignored.

Parameters:

**other** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*,*[*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*or**Number*) - the value to add,
paired with the leaves as in [`add()`](tensordict.TensorDictBase.add.html#tensordict.TensorDictBase.add).

Keyword Arguments:

**alpha** (*Number**,**optional*) - the multiplier for `other`. Defaults to
`None`, which adds `other` as it is.

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).