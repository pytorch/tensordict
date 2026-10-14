# TensorDictBase.to

TensorDictBase.to(**args*, ***kwargs*) → Self

Maps a TensorDictBase subclass either on another device, dtype or to another TensorDictBase subclass (if permitted).

When a dtype is passed, every tensor leaf is cast to that dtype, whatever its original
dtype (including integer and boolean leaves).

Parameters:

- **device** ([*device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,**optional*) - the desired device of the tensordict.
- **dtype** ([*dtype*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype)*,**optional*) - the desired dtype of all the tensors in
the tensordict.
- **tensor** ([*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*,**optional*) - Tensor whose dtype and device are the desired
dtype and device for all tensors in this TensorDict.

Keyword Arguments:

- **non_blocking** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)*,**optional*) - controls how the leaves are copied.
If `None` (default), the leaves are copied with `non_blocking=True`
and, when the target device is not a CUDA device (e.g. device-to-host
copies), the tensordict synchronizes once after all the copies have been
issued; for a CPU target, only if a leaf comes from another device.
Under [`torch.compile()`](https://docs.pytorch.org/docs/stable/generated/torch.compile.html#torch.compile), these copies are blocking instead.
If `True`, the copies are non-blocking and no synchronization
is performed: the caller is responsible for synchronizing before reading
the results. If `False`, the copies are blocking.
- **memory_format** ([*memory_format*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.memory_format)*,**optional*) - the desired memory
format for the 4D and 5D tensors in this tensordict, as in
[`torch.nn.Module.to()`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.to).
- **batch_size** ([*Size*](https://docs.pytorch.org/docs/stable/size.html#torch.Size)*,**optional*) - resulting batch-size of the
output tensordict. When the device or dtype changes, every
nested tensordict gets exactly this batch size and the names
are erased. Starting with TensorDict 0.17, the default will
change to setting the batch size as `to(batch_size=...)`
does without a device: nested tensordicts with more batch
dimensions keep their batch size, nested lazy stacks stay
lazy stacks (or raise, as there, if they need more batch
dimensions) and the names are kept. A `FutureWarning` is
emitted when the result will change.
- **other** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*,**optional*) - 

TensorDict instance whose dtype
and device are the desired dtype and device for all tensors
in this TensorDict.

Note

Since [`TensorDictBase`](tensordict.TensorDictBase.html#tensordict.TensorDictBase) instances do not have
a dtype, the dtype is gathered from the example leaves.
If there are more than one dtype, then no dtype
casting is done.
- **non_blocking_pin** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)*,**optional*) - 

if `True`, the tensors are pinned before
being sent to device. This will be done asynchronously but can be
controlled via the `num_threads` argument.

Note

Calling `tensordict.pin_memory().to("cuda")` will usually
be much slower than `tensordict.to("cuda", non_blocking_pin=True)` as
the pin_memory is called asynchronously in the second case.
Multithreaded `pin_memory` will usually be beneficial if the tensors
are large and numerous: when there are too few tensors to be sent,
the overhead of spawning threads and collecting data outweighs the benefits
of multithreading, and if the tensors are small the overhead of iterating
over a long list is also prohibitively large.
- **num_threads** ([*int*](tensordict.TensorDictBase.int.html#tensordict.TensorDictBase.int)*or*[*None*](https://docs.python.org/3/builtins/constants.html#None)*,**optional*) - if `non_blocking_pin=True`, the number
of threads to be used for `pin_memory`. By default,
up to `max(1, torch.get_num_threads() // 2)` threads are
started, and no more than one per leaf.
`num_threads=0` will cancel any
multithreading for the pin_memory() calls.
- **inplace** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)*,**optional*) - if `True`, the data will be written in-place in the same tensordict.
This can be significantly faster whenever building a tensordict is CPU-overhead bound.
Defaults to `False`.

Returns:

a new tensordict instance if the device differs from the tensordict
device and/or if the dtype or memory format is passed. The same
tensordict otherwise.
`batch_size` only modifications are done in-place.

Note

If the TensorDict is consolidated, the resulting TensorDict will be consolidated too.
Each new tensor will be a view on the consolidated storage cast to the desired device.

This operation can be used as a context manager too. Inside the
`with` block, you work on the cast tensordict. When the block exits,
its entries are written back to the original tensordict, on the
original device. Use it to run a module that expects its input on a
specific device. See [Devices, dtypes, and memory](../../concepts/devices.html#devices).

Note

On exit, every entry is moved back to the original device. The
entries that already existed get their original dtype back, but
the new entries keep the dtype of the `with` block: after
`with td.to(torch.float64) as td64`, a new entry of `td64` is
`float64` in `td`. The original tensordict must not be
locked. If the block raises an exception, nothing is written back.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> data = TensorDict({"a": 1.0}, [], device=None)
>>> data_int = data.to(torch.int) # casts to int
>>> data_int["a"]
tensor(1, dtype=torch.int32)
```

The next lines need a CUDA device, so they are not run when the
documentation is tested.

```
>>> data_cuda = data.to("cuda:0") 
>>> data_cuda_int = data.to("cuda:0", torch.int) 
>>> data_cuda = data.to(torch.randn(3, device="cuda:0")) 
>>> data_cuda = data.to(other=TensorDict({}, [], device="cuda:0"))
```

Use it as a context manager for a temporary device change (CUDA
needed):

```
>>> from tensordict.nn import TensorDictModule
>>> mod = TensorDictModule(lambda x: x + 1, in_keys=["x"], out_keys=["y"])
>>> td = TensorDict(x=torch.zeros(3), batch_size=[3], device="cpu")
>>> with td.to("cuda") as td_gpu: 
... _ = td_gpu.update(mod(td_gpu)) # runs on the GPU
>>> # The entries are written back on the original device
>>> assert td["y"].device.type == "cpu"
```

A temporary dtype change runs on CPU. The new entry keeps the dtype
of the `with` block:

```
>>> td = TensorDict(x=torch.zeros(3), batch_size=[3])
>>> with td.to(torch.float64) as td64:
... td64["y"] = td64["x"] + 1
>>> td["x"].dtype, td["y"].dtype
(torch.float32, torch.float64)
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).

## Tutorials that use `to`

![](../../_images/sphx_glr_tensordict_memory_thumb.png)

[Devices and memory-mapped tensors](../../tutorials/tensordict_memory.html)

Devices and memory-mapped tensors