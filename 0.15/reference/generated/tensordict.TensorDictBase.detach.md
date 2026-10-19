# TensorDictBase.detach

TensorDictBase.detach() → Self

Returns a new tensordict with every tensor leaf detached from the autograd graph.

The leaves are [`torch.Tensor.detach()`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.detach.html#torch.Tensor.detach) views: they share their
storage with the leaves of the tensordict, and do not require
gradients. A locked tensordict gives a locked result.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict(a=torch.zeros(3, requires_grad=True) + 1)
>>> out = td.detach()
>>> out["a"].requires_grad, td["a"].requires_grad
(False, True)
>>> out["a"].data_ptr() == td["a"].data_ptr()
True
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).

## Tutorials that use `detach`

![](../../_images/sphx_glr_functional_thumb.png)

[Functional calls and ensembles with TensorDict](../../tutorials/functional.html)

Functional calls and ensembles with TensorDict