# TensorDictBase.from_modules

*classmethod*TensorDictBase.from_modules(**modules*, *as_module: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *lock: [bool](https://docs.python.org/3/builtins/functions.html#bool) = True*, *use_state_dict: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *lazy_stack: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *expand_identical: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*)

Retrieves the parameters of several modules for ensebmle learning/feature of expects applications through vmap.

Parameters:

**modules** (*sequence**of**nn.Module*) - the modules to get the parameters from.
If the modules differ in their structure, a lazy stack is needed
(see the `lazy_stack` argument below).

Keyword Arguments:

- **as_module** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)*,**optional*) - if `True`, a `TensorDictParams`
instance will be returned which can be used to store parameters
within a [`torch.nn.Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module). Defaults to `False`.
- **lock** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)*,**optional*) - if `True`, the resulting tensordict will be locked.
Defaults to `True`.
- **use_state_dict** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)*,**optional*) - 

if `True`, the state-dict from the
module will be used and unflattened into a TensorDict with
the tree structure of the model. Defaults to `False`.

Note

This is particularly useful when state-dict hooks have to be used.
- **lazy_stack** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)*,**optional*) - 

whether parameters should be densly or
lazily stacked. Defaults to `False` (dense stack).

Note

`lazy_stack` and `as_module` are exclusive features.

Warning

There is a crucial difference between lazy and non-lazy outputs
in that non-lazy output will reinstantiate parameters with the
desired batch-size, while `lazy_stack` will just represent
the parameters as lazily stacked. This means that whilst the
original parameters can safely be passed to an optimizer
when `lazy_stack=True`, the new parameters need to be passed
when it is set to `True`.

Warning

Whilst it can be tempting to use a lazy stack to keep the
orignal parameter references, remember that lazy stack
perform a stack each time [`get()`](tensordict.TensorDictBase.get.html#tensordict.TensorDictBase.get) is called. This will
require memory (N times the size of the parameters, more if a
graph is built) and time to be computed.
It also means that the optimizer(s) will contain more
parameters, and operations like [`step()`](https://docs.pytorch.org/docs/stable/generated/torch.optim.Optimizer.step.html#torch.optim.Optimizer.step)
or [`zero_grad()`](https://docs.pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html#torch.optim.Optimizer.zero_grad) will take longer
to be executed. In general, `lazy_stack` should be reserved
to very few use cases.
- **expand_identical** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)*,**optional*) - if `True` and the same parameter (same
identity) is being stacked to itself, an expanded version of this parameter
will be returned instead. This argument is ignored when `lazy_stack=True`.

Examples

```
>>> import torch
>>> from torch import nn
>>> from tensordict import TensorDict
>>> _ = torch.manual_seed(0)
>>> empty_module = nn.Linear(3, 4, device="meta")
>>> n_models = 2
>>> modules = [nn.Linear(3, 4) for _ in range(n_models)]
>>> params = TensorDict.from_modules(*modules)
>>> print(params)
TensorDict(
 fields={
 bias: Parameter(shape=torch.Size([2, 4]), device=cpu, dtype=torch.float32, is_shared=False),
 weight: Parameter(shape=torch.Size([2, 4, 3]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([2]),
 device=None,
 is_shared=False)
>>> # example of batch execution
>>> def exec_module(params, x):
... with params.to_module(empty_module):
... return empty_module(x)
>>> x = torch.randn(3)
>>> y = torch.vmap(exec_module, (0, None))(params, x)
>>> assert y.shape == (n_models, 4)
>>> # since lazy_stack = False, backprop leaves the original params untouched
>>> y.sum().backward()
>>> assert params["weight"].grad.norm() > 0
>>> assert modules[0].weight.grad is None
```

With `lazy_stack=True`, things are slightly different:

```
>>> params = TensorDict.from_modules(*modules, lazy_stack=True)
>>> print(params)
LazyStackedTensorDict(
 fields={
 bias: Tensor(shape=torch.Size([2, 4]), device=cpu, dtype=torch.float32, is_shared=False),
 weight: Tensor(shape=torch.Size([2, 4, 3]), device=cpu, dtype=torch.float32, is_shared=False)},
 exclusive_fields={
 },
 batch_size=torch.Size([2]),
 device=None,
 is_shared=False,
 stack_dim=0)
>>> # example of batch execution
>>> y = torch.vmap(exec_module, (0, None))(params, x)
>>> assert y.shape == (n_models, 4)
>>> y.sum().backward()
>>> assert modules[0].weight.grad is not None
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).

## Tutorials that use `from_modules`

![](../../_images/sphx_glr_functional_thumb.png)

[Functional calls and ensembles with TensorDict](../../tutorials/functional.html)

Functional calls and ensembles with TensorDict