# set_interaction_type

*class*tensordict.nn.set_interaction_type(*type: [InteractionType](tensordict.nn.InteractionType.html#tensordict.nn.InteractionType) | [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [None](https://docs.python.org/3/builtins/constants.html#None) = InteractionType.DETERMINISTIC*)

Bases: `_DecoratorContextManager`

Sets the interaction type of all probabilistic modules inside a block of code, or inside a decorated function.

Inside the block, [`interaction_type()`](tensordict.nn.probabilistic.interaction_type.html#tensordict.nn.probabilistic.interaction_type)
returns `type`, and every [`ProbabilisticTensorDictModule`](tensordict.nn.ProbabilisticTensorDictModule.html#tensordict.nn.ProbabilisticTensorDictModule) uses it
instead of its `default_interaction_type`. When the block ends, the
previous value comes back.

Parameters:

**type** ([*InteractionType*](tensordict.nn.InteractionType.html#tensordict.nn.InteractionType)*or*[*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - the interaction type, for
example `InteractionType.RANDOM` or `"random"` (in any case).
`None` makes each module use its `default_interaction_type`.
Defaults to `InteractionType.DETERMINISTIC`.

Examples

```
>>> import torch
>>> from torch import distributions as d
>>> from tensordict import TensorDict
>>> from tensordict.nn import (
... InteractionType,
... ProbabilisticTensorDictModule,
... set_interaction_type,
... )
>>> module = ProbabilisticTensorDictModule(
... in_keys=["loc", "scale"], out_keys=["action"], distribution_class=d.Normal
... )
>>> td = TensorDict(loc=torch.zeros(3), scale=torch.ones(3))
>>> module(td)["action"] # default_interaction_type is DETERMINISTIC: the mean
tensor([0., 0., 0.])
>>> _ = torch.manual_seed(0)
>>> with set_interaction_type("random"):
... module(td)["action"]
tensor([ 1.5410, -0.2934, -2.1788])
```

It also works as a decorator:

```
>>> @set_interaction_type(InteractionType.MODE)
... def act(td):
... return module(td)
>>> act(td)["action"]
tensor([0., 0., 0.])
```

## Tutorials that use `set_interaction_type`

![](../../_images/sphx_glr_tensordict_module_thumb.png)

[Building models with TensorDictModule](../../tutorials/tensordict_module.html)

Building models with TensorDictModule