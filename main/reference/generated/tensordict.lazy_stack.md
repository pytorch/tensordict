# lazy_stack

tensordict.lazy_stack(*input: Sequence['T']*, *dim: [int](https://docs.python.org/3/builtins/functions.html#int) = 0*, ***, *out=None*, ***kwargs*) → T

Stacks tensordicts into a lazy stack ([`LazyStackedTensorDict`](tensordict.LazyStackedTensorDict.html#tensordict.LazyStackedTensorDict)).

The result keeps each input as a separate tensordict, so the inputs can
have different keys, entry shapes and batch sizes. Use it for data whose
size changes from one sample to the next. [Stacking and variable-size data](../../concepts/stacking.html#stacking) compares this
function with [`stack()`](tensordict.stack.html#tensordict.stack) and
[`maybe_dense_stack()`](tensordict.maybe_dense_stack.html#tensordict.maybe_dense_stack). Same as
[`tensordict.LazyStackedTensorDict.lazy_stack()`](tensordict.LazyStackedTensorDict.html#tensordict.LazyStackedTensorDict.lazy_stack), which takes the same
arguments and documents the ones for lists of tensors.

Parameters:

- **input** (*sequence**of*[*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*or**tensorclass instances*) - the
tensordicts to stack. It must not be empty. The batch sizes must
have the same number of dimensions. If a dimension differs
between inputs, it is `-1` in the stacked batch size.
- **dim** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the new dimension. Defaults to `0`. A `dim`
below `-batch_dims - 1` warns and wraps around once more (and
raises if it is still out of range).
Starting with TensorDict 0.17, the default will change to raising
an `IndexError`, as for a tensor.

Keyword Arguments:

- **out** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*,**optional*) - a lazy stack to write the result
into. Its batch size must be the stacked batch size. Defaults to
`None`.
- **stack_dim_name** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - the name of the stack dimension.
Defaults to `None`.
- **strict_shape** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, the batch sizes of the
inputs must be equal, or a `RuntimeError` is raised. Defaults
to `False`.

Returns:

A [`LazyStackedTensorDict`](tensordict.LazyStackedTensorDict.html#tensordict.LazyStackedTensorDict). Tensorclass inputs give
an instance of their class that holds a lazy stack.

Examples

```
>>> import torch
>>> import tensordict
>>> from tensordict import TensorDict
>>> td = tensordict.lazy_stack(
... [TensorDict(a=torch.zeros(3), batch_size=[3]),
... TensorDict(a=torch.zeros(4), batch_size=[4])]
... )
>>> type(td).__name__
'LazyStackedTensorDict'
>>> td.batch_size
torch.Size([2, -1])
>>> td[1]["a"].shape
torch.Size([4])
```

## Tutorials that use `lazy_stack`

![](../../_images/sphx_glr_streamed_tensordict_thumb.png)

[Building tensordicts from streams](../../tutorials/streamed_tensordict.html)

Building tensordicts from streams
![](../../_images/sphx_glr_tensordict_shapes_thumb.png)

[Changing the shape of a TensorDict](../../tutorials/tensordict_shapes.html)

Changing the shape of a TensorDict