# TensorDictBase.record_stream

TensorDictBase.record_stream(*stream: [Stream](https://docs.pytorch.org/docs/stable/generated/torch.cuda.streams.Stream.html#torch.cuda.streams.Stream)*) → Self

Marks the tensor leaves of the tensordict as used by `stream`.

Calls [`record_stream()`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.record_stream.html#torch.Tensor.record_stream) on every tensor leaf: when
a leaf is freed, its memory is not reused until the work queued on
`stream` at that time is complete. A tensordict can be marked with
one stream only.

Parameters:

**stream** ([*Stream*](https://docs.pytorch.org/docs/stable/generated/torch.cuda.Stream_class.html#torch.cuda.Stream)) - the stream that uses the leaves.

Returns:

`self`.

Raises:

[**RuntimeError**](https://docs.python.org/3/builtins/exceptions.html#RuntimeError) - if the tensordict is already marked with another
 stream.

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).