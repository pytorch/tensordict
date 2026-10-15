# TensorDictBase.backward

TensorDictBase.backward(*gradient: [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *retain_graph: [bool](https://docs.python.org/3/builtins/functions.html#bool) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *create_graph: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *inputs: [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase) | [Sequence](https://docs.python.org/3/library/typing.html#typing.Sequence)[[Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)] | [None](https://docs.python.org/3/builtins/constants.html#None) = None*) → [None](https://docs.python.org/3/builtins/constants.html#None)

Computes the gradient of the differentiable leaves of the tensordict w.r.t. graph leaves.

This method mirrors [`torch.Tensor.backward()`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.backward.html#torch.Tensor.backward): it recursively collects the
leaf tensors that require gradients and differentiates the graph through a single
[`torch.autograd.backward()`](https://docs.pytorch.org/docs/stable/generated/torch.autograd.backward.html#torch.autograd.backward) call. Non-differentiable leaves are ignored.

If `gradient` is `None`, every differentiable leaf must be scalar (a single
element) and an implicit gradient of `1` is used, such that for scalar leaves
`td.backward()` is equivalent to `td.sum(reduce=True).backward()`. Otherwise,
`gradient` must be a tensordict whose entries match the differentiable leaves
by nested key. Leaves may have heterogeneous shapes, dtypes and devices, as long
as each gradient entry matches the leaf it is associated with.

Parameters:

- **gradient** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*,**optional*) - the gradient w.r.t. the tensordict.
Entries are matched to the differentiable leaves by nested key and must
have the same shape as the leaf they match. Can be omitted if all
differentiable leaves are scalar. Defaults to `None`.
- **retain_graph** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `False`, the graph used to compute the
grads will be freed. Defaults to the value of `create_graph`.
- **create_graph** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, graph of the derivative will be
constructed, allowing to compute higher order derivative products.
Defaults to `False`.
- **inputs** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*or**sequence**of**Tensor**,**optional*) - inputs w.r.t. which
the gradient will be accumulated in `.grad`. All other tensors will be
ignored. If not provided, the gradient is accumulated into all the leaf
tensors that were used to compute the differentiated tensors.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> x = torch.randn(3, requires_grad=True)
>>> loss_td = TensorDict(
... actor_loss=x.sum(),
... critic_loss=x.pow(2).sum(),
... )
>>> loss_td.backward() # implicit gradient of 1 for scalar leaves
>>> assert x.grad is not None
>>> # weighted losses through a matching gradient tensordict
>>> x.grad = None
>>> loss_td = TensorDict(
... actor_loss=x.sum(),
... critic_loss=x.pow(2).sum(),
... )
>>> weights = TensorDict(
... actor_loss=torch.tensor(0.5),
... critic_loss=torch.tensor(1.0),
... )
>>> loss_td.backward(weights)
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).