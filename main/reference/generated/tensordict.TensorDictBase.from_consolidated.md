# TensorDictBase.from_consolidated

*classmethod*TensorDictBase.from_consolidated(*filename*)

Reconstructs a tensordict from a file written by [`consolidate()`](tensordict.TensorDictBase.consolidate.html#tensordict.TensorDictBase.consolidate).

The batch sizes and names are as they were when the file was written.
To write later changes into the file, call
`consolidate(filename=filename)` again on the consolidated tensordict.
See [`from_consolidated()`](tensordict.from_consolidated.html#tensordict.from_consolidated).

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).

## Tutorials that use `from_consolidated`

![](../../_images/sphx_glr_serialization_speed_thumb.png)

[Benchmarking TensorDict serialization speed](../../tutorials/serialization_speed.html)

Benchmarking TensorDict serialization speed