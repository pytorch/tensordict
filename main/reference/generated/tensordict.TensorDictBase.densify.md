# TensorDictBase.densify

TensorDictBase.densify(*layout: [layout](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.layout) = torch.strided*)

Returns the tensordict with its lazy stacks replaced by dense tensordicts.

The entries of a lazy stack are stacked into one tensor when their
shapes match, and into a nested tensor ([`torch.nested`](https://docs.pytorch.org/docs/stable/nested.html#module-torch.nested)) along the
stack dimension otherwise. A key that only some members of a lazy
stack have is filled with an empty entry for the members that lack it,
with the batch size of the member it stands in for (zero rows when
the members that have it agree on its feature dims),
so it appears in the result although `keys()` of the lazy stack
leaves it out. A tensordict that holds no lazy stack is
returned as it is.

Parameters:

**layout** ([*layout*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.layout)*,**optional*) - the layout of the nested tensors,
`torch.strided` or `torch.jagged`. Defaults to
`torch.strided`.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> stack = TensorDict.lazy_stack(
... [TensorDict(a=torch.zeros(3)), TensorDict(a=torch.ones(4))]
... )
>>> dense = stack.densify(layout=torch.jagged)
>>> type(dense).__name__, dense["a"].is_nested
('TensorDict', True)
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).

## Tutorials that use `densify`

![](../../_images/sphx_glr_streamed_tensordict_thumb.png)

[Building tensordicts from streams](../../tutorials/streamed_tensordict.html)

Building tensordicts from streams