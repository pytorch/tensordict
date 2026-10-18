# dense_stack_tds

tensordict.dense_stack_tds(*td_list: [Sequence](https://docs.python.org/3/library/typing.html#typing.Sequence)[[TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase)] | [LazyStackedTensorDict](tensordict.LazyStackedTensorDict.html#tensordict.LazyStackedTensorDict)*, *dim: [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*) → T

Stacks a list of tensordicts, or the tensordicts of a lazy stack, densely if possible.

Use it to turn a [`LazyStackedTensorDict`](tensordict.LazyStackedTensorDict.html#tensordict.LazyStackedTensorDict) into a dense
[`TensorDict`](tensordict.TensorDict.html#tensordict.TensorDict) when its tensordicts have the same keys
and entry shapes. It is useful when the tensordicts are lazy stacks
themselves, or hold lazy stacks among their nested entries: then
`lazy_stack(td_list).to_tensordict()` is not possible. This function
calls [`maybe_dense_stack()`](tensordict.maybe_dense_stack.html#tensordict.maybe_dense_stack). So if the tensordicts have
different keys or entry shapes at the top level, the result is a
[`LazyStackedTensorDict`](tensordict.LazyStackedTensorDict.html#tensordict.LazyStackedTensorDict). See [Stacking and variable-size data](../../concepts/stacking.html#stacking).

Parameters:

- **td_list** ([*list*](https://docs.python.org/3/builtins/stdtypes.html#list)*of*[*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*, or*[*LazyStackedTensorDict*](tensordict.LazyStackedTensorDict.html#tensordict.LazyStackedTensorDict)) - the
tensordicts to stack. A [`TensorDict`](tensordict.TensorDict.html#tensordict.TensorDict) is
returned unchanged.
- **dim** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the dimension to stack them along. Required if
`td_list` is a list: without it, a `ValueError` is raised.
Ignored if `td_list` is a
[`LazyStackedTensorDict`](tensordict.LazyStackedTensorDict.html#tensordict.LazyStackedTensorDict), whose `stack_dim` is
used. Defaults to `None`.

Returns:

A [`TensorDict`](tensordict.TensorDict.html#tensordict.TensorDict) (or the class of the stacked
tensordicts) if they can be stacked densely, a
[`LazyStackedTensorDict`](tensordict.LazyStackedTensorDict.html#tensordict.LazyStackedTensorDict) otherwise.

Examples

```
>>> import torch
>>> from tensordict import TensorDict, lazy_stack
>>> from tensordict import assert_allclose_td, dense_stack_tds
>>> td0 = TensorDict({"a": torch.zeros(3)})
>>> td1 = TensorDict({"a": torch.zeros(4), "b": torch.zeros(2)})
>>> td_lazy = lazy_stack([td0, td1], dim=0)
>>> td_container = TensorDict({"lazy": td_lazy})
>>> td_container_clone = td_container.clone()
>>> td_stack = lazy_stack([td_container, td_container_clone], dim=0)
>>> td_dense = dense_stack_tds(td_stack) # uses the stack_dim of td_stack
>>> type(td_dense).__name__, td_dense.batch_size
('TensorDict', torch.Size([2]))
```

The nested lazy stack is now stacked along its dimension `1`, and
each of its tensordicts keeps its own keys:

```
>>> td_dense["lazy"].stack_dim
1
>>> td_dense["lazy"][:, 1]["b"].shape
torch.Size([2, 2])
```

Passing the list of tensordicts and `dim` gives the same result:

```
>>> assert_allclose_td(td_dense, dense_stack_tds([td_container, td_container_clone], dim=0))
True
```