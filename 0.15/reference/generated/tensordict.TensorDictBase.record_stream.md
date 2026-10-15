# TensorDictBase.record_stream

TensorDictBase.record_stream(*stream: [Stream](https://docs.pytorch.org/docs/stable/generated/torch.cuda.streams.Stream.html#torch.cuda.streams.Stream)*) → Self

Marks the tensordict as having been used by this stream.

When the tensordict is deallocated, ensure the tensor memory is not reused for other tensors until all work
queued on stream at the time of deallocation is complete.

See [`record_stream()`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.record_stream.html#torch.Tensor.record_stream) for more information.`

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).