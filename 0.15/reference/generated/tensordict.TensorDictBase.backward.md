# TensorDictBase.backward

TensorDictBase.backward(*gradient: [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *retain_graph: [bool](https://docs.python.org/3/builtins/functions.html#bool) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *create_graph: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *inputs: [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase) | [Sequence](https://docs.python.org/3/library/typing.html#typing.Sequence)[[Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)] | [None](https://docs.python.org/3/builtins/constants.html#None) = None*) → [None](https://docs.python.org/3/builtins/constants.html#None)

Computes the gradients of the leaves of the tensordict that require gradients.

Like [`torch.Tensor.backward()`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.backward.html#torch.Tensor.backward) for every leaf at once: the leaves
that require gradients, nested ones included, are differentiated in a
single [`torch.autograd.backward()`](https://docs.pytorch.org/docs/stable/generated/torch.autograd.backward.html#torch.autograd.backward) call, and the others are
ignored. The gradients are accumulated in the `.grad` of the graph
leaves, as usual. Returns `None`.

Without `gradient`, every leaf that requires gradients must have a
single element, and its gradient is `1`: `td.backward()` is then
the same as `td.sum(reduce=True).backward()`.

Parameters:

- **gradient** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*,**optional*) - the gradient of each leaf, under
the same nested key and with the same shape. The leaves can
differ in shape, dtype and device. Defaults to `None`, which
needs single-element leaves.
- **retain_graph** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `False`, the graph used to
compute the gradients is freed. Defaults to `None`, which
takes the value of `create_graph`.
- **create_graph** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, the graph of the
derivative is built, for higher-order derivatives. Defaults to
`False`.
- **inputs** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*or**sequence**of*[*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*,**optional*) - the
tensors whose `.grad` receives the gradients; the others are
ignored. Defaults to `None`, which accumulates into all the
graph leaves used to compute the leaves of the tensordict.

Raises:

- [**RuntimeError**](https://docs.python.org/3/builtins/exceptions.html#RuntimeError) - if no leaf requires gradients, if `gradient` is not
 given and a leaf has more than one element, or if a gradient
 has the wrong shape.
- [**KeyError**](https://docs.python.org/3/builtins/exceptions.html#KeyError) - if `gradient` has no entry for a leaf that requires
 gradients.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> x = torch.ones(3, requires_grad=True)
>>> losses = TensorDict(actor_loss=x.sum(), critic_loss=x.pow(2).sum())
>>> losses.backward() # a gradient of 1 for each single-element leaf
>>> x.grad
tensor([3., 3., 3.])
```

Weight the losses with a gradient tensordict:

```
>>> x.grad = None
>>> losses = TensorDict(actor_loss=x.sum(), critic_loss=x.pow(2).sum())
>>> weights = TensorDict(actor_loss=torch.tensor(0.5), critic_loss=torch.tensor(1.0))
>>> losses.backward(weights)
>>> x.grad
tensor([2.5000, 2.5000, 2.5000])
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).