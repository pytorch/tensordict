# TensorDictBase.densify

TensorDictBase.densify(*layout: [layout](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.layout) = torch.strided*)

Attempts to represent the lazy stack with contiguous tensors (plain tensors or nested).

Keyword Arguments:

**layout** ([*layout*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.layout)) - the layout of the nested tensors, if any. Defaults to
`strided`.

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).

## Tutorials that use `densify`

![](../../_images/sphx_glr_streamed_tensordict_thumb.png)

[Building tensordicts from streams](../../tutorials/streamed_tensordict.html)

Building tensordicts from streams