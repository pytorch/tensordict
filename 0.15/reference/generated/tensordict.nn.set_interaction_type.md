# set_interaction_type

*class*tensordict.nn.set_interaction_type(*type: [InteractionType](tensordict.nn.InteractionType.html#tensordict.nn.InteractionType) | [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [None](https://docs.python.org/3/builtins/constants.html#None) = InteractionType.DETERMINISTIC*, ***, *thread_local: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*)

Bases: `_DecoratorContextManager`

Sets the interaction type of all probabilistic modules inside a block of code, or inside a decorated function.

Inside the block, [`interaction_type()`](tensordict.nn.interaction_type.html#tensordict.nn.interaction_type)
returns `type`, and every [`ProbabilisticTensorDictModule`](tensordict.nn.ProbabilisticTensorDictModule.html#tensordict.nn.ProbabilisticTensorDictModule) uses it
instead of its `default_interaction_type`. When the block ends, the
previous value comes back. By default the interaction type applies to
the whole process (see `thread_local`).

Parameters:

**type** ([*InteractionType*](tensordict.nn.InteractionType.html#tensordict.nn.InteractionType)*or*[*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - the interaction type, for
example `InteractionType.RANDOM` or `"random"` (in any case).
`None` makes each module use its `default_interaction_type`.
Defaults to `InteractionType.DETERMINISTIC`.

Keyword Arguments:

**thread_local** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, the interaction type
applies to the calling thread only, and blocks that other threads
enter do not change it. Inside the block, blocks without this flag
that the same thread enters stay in that thread too. Threads
started inside the block see the process-wide interaction type.
If `False`, the interaction type applies to the whole process.
Defaults to `False`.

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