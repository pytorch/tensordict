# IndexType

tensordict.utils.IndexType

The type of the indices that a tensordict accepts in `td[index]`.

An int, a slice, `None`, a tensor (of indices or a boolean mask), a list,
or a tuple of these, as for a tensor; a str or a tuple of str is a key.

alias of [`Union`](https://docs.python.org/3/library/typing.html#typing.Union)[[`None`](https://docs.python.org/3/builtins/constants.html#None), [`int`](https://docs.python.org/3/builtins/functions.html#int), [`slice`](https://docs.python.org/3/builtins/functions.html#slice), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor), [`List`](https://docs.python.org/3/library/typing.html#typing.List)[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any)], `Tuple`[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any), ...]]