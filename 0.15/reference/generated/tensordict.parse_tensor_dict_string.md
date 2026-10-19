# parse_tensor_dict_string

tensordict.parse_tensor_dict_string(*s: [str](https://docs.python.org/3/builtins/stdtypes.html#str)*)

Builds a [`TensorDict`](tensordict.TensorDict.html#tensordict.TensorDict) from its printed form.

It is a debugging tool: it rebuilds a tensordict with the keys, shapes,
dtypes, devices and batch sizes of a printed one, so that you can
reproduce a tensordict from a log. The values are not in the repr, so
every tensor of the result is filled with zeros.

Parameters:

**s** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)) - the repr of a tensordict, as printed with the default print
options (see [`set_printoptions()`](tensordict.set_printoptions.html#tensordict.set_printoptions)).

Returns:

A new [`TensorDict`](tensordict.TensorDict.html#tensordict.TensorDict).

Raises:

[**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) - if a field is not a `Tensor` or a `TensorDict` printed
 with the default print options, or if `s` is the repr of a lazy
 stack with exclusive keys.

Examples

```
>>> import torch
>>> from tensordict import TensorDict, parse_tensor_dict_string
>>> td = TensorDict(
... obs=torch.ones(3, 4),
... next=TensorDict(done=torch.ones(3, 1, dtype=torch.bool)),
... batch_size=[3],
... )
>>> rebuilt = parse_tensor_dict_string(repr(td))
>>> rebuilt["next", "done"].dtype, rebuilt.batch_size
(torch.bool, torch.Size([3]))
>>> rebuilt["obs"].sum()
tensor(0.)
```