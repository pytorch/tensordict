# TensorDictBase.to

TensorDictBase.to(**args*, ***kwargs*) → Self

Returns a tensordict cast to another device, dtype or memory format, or with a new batch size.

It accepts the forms of [`torch.Tensor.to()`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.to.html#torch.Tensor.to), and some more:

- `to(device, dtype=None)`: moves every leaf to `device` and, if
`dtype` is given, casts it.
- `to(dtype)`: casts every tensor leaf to `dtype`, whatever its
original dtype (integer and boolean leaves too), unless
`cast_filter` selects the leaves to cast.
- `to(tensor)`: uses the device and the dtype of `tensor`.
- `to(other=tensordict)`: uses the device of `other`, and its
dtype if all its leaves share one. Otherwise the dtypes are kept.
- `to(batch_size=...)`: only sets the batch size, in place, and
returns `self`.
- `to(td.attrs())`: casts each leaf to the device and the dtype that
[`attrs()`](tensordict.TensorDictBase.attrs.html#tensordict.TensorDictBase.attrs) recorded for the same key. See
[Devices, dtypes, and memory](../../concepts/devices.html#devices).

The result is `self` when the device is already the device of the
tensordict and no dtype or memory format is given. Otherwise it is a
new tensordict, unless `inplace=True`. When only the device changes,
a consolidated tensordict gives a consolidated result, whose leaves
are views of one storage on the new device.

Parameters:

- **device** ([*device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,*[*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*or*[*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the device of the
result. Its nested tensordicts get it too.
- **dtype** ([*dtype*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype)*,**optional*) - the dtype of the tensor leaves of
the result.
- **tensor** ([*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*,**optional*) - a tensor whose device and dtype
are the device and dtype of the result.

Keyword Arguments:

- **non_blocking** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - how the leaves are copied. `None`
copies them with `non_blocking=True` and, when the target
is not a CUDA device (for example a copy to the host),
synchronizes once after all the copies are issued; for a CPU
target, only if a leaf comes from another device. Under
[`torch.compile()`](https://docs.pytorch.org/docs/stable/generated/torch.compile.html#torch.compile), these copies are blocking instead.
`True` copies without synchronizing: synchronize before you
read the results. `False` makes blocking copies. Defaults to
`None`. It can also follow the device, dtype or tensor
positionally, as in `to("cpu", torch.float64, False)`. A
positional `True` is ignored for now, with a
`FutureWarning`: the copies synchronize as for `None`.
Starting with TensorDict 0.17, the default will change to
honoring it as the keyword `True` is.
- **memory_format** ([*memory_format*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.memory_format)*,**optional*) - the memory format of
the 4D and 5D tensors of the result, as in
[`torch.nn.Module.to()`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.to). Defaults to `None`, which keeps
the memory format.
- **batch_size** ([*Size*](https://docs.pytorch.org/docs/stable/size.html#torch.Size)*,**optional*) - the batch size of the result.
When the device or dtype changes, every nested tensordict gets
exactly this batch size and the names are erased. Starting with
TensorDict 0.17, the default will change to setting the batch
size as `to(batch_size=...)` does without a device: nested
tensordicts with more batch dimensions keep their batch size,
nested lazy stacks stay lazy stacks (or raise, as there, if
they need more batch dimensions) and the names are kept. A
`FutureWarning` is emitted when the result will change.
Defaults to `None`, which keeps the batch size.
- **other** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*,**optional*) - a tensordict whose device, and
dtype if all its leaves share one, are those of the result.
Defaults to `None`.
- **cast_filter** (*Callable**,**optional*) - a predicate
`fn(key, value) -> bool` that selects the tensor leaves the
dtype cast applies to. `key` is the nested key of the leaf as
a tuple of str, for example `("obs",)` or
`("next", "done")`. Leaves for which it returns `False`
keep their dtype, but still move to the new device or memory
format. It is not called on non-tensor data, without a dtype,
or with an attrs tensordict. An
[`UnbatchedTensor`](tensordict.UnbatchedTensor.html#tensordict.UnbatchedTensor) leaf is passed as is.
Helpers such as [`is_inexact()`](tensordict.is_inexact.html#tensordict.is_inexact) can be passed
directly. Defaults to the `cast_filter` of a tensorclass that
declares one, and to casting every leaf otherwise; pass
`None` to cast every leaf. The dtype shorthands
([`float()`](tensordict.TensorDictBase.float.html#tensordict.TensorDictBase.float), [`bfloat16()`](tensordict.TensorDictBase.bfloat16.html#tensordict.TensorDictBase.bfloat16), ...) take it too.
- **non_blocking_pin** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - 

if `True`, pins the tensors
before sending them to the device, in `num_threads`
threads. Defaults to `False`.

Note

Each call starts and joins its pinning threads, which costs
the same however little data there is. So
`non_blocking_pin=True` pays off only for large
tensordicts, with leaves of a MiB or more and tens of MiB
in total. For smaller tensordicts, `td.to("cuda")` is
usually the fastest, and
`td.pin_memory().to("cuda", non_blocking=True)` is also
faster than `non_blocking_pin=True`. Measure the
crossover on your machine.
- **num_threads** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - with `non_blocking_pin=True`, the
number of threads that pin the leaves, at most one per leaf.
`0` pins them on the calling thread. Defaults to `None`,
which starts up to `max(1, torch.get_num_threads() // 2)`
threads.
- **inplace** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, writes the cast leaves into
the tensordict, which keeps its identity and keys, and returns
`self`. This can be faster when building a new tensordict is
what costs the most. See [In-place operations](../../concepts/operations.html#inplace-conventions). Defaults
to `False`.

Returns:

`self` when nothing changes, with `inplace=True`, or when only
`batch_size` is given; a new tensordict otherwise.

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

Cast the floating-point leaves only:

```
>>> from tensordict import is_inexact
>>> data = TensorDict(x=torch.zeros(3), done=torch.zeros(3, dtype=torch.bool))
>>> data_bf16 = data.to(torch.bfloat16, cast_filter=is_inexact)
>>> data_bf16["x"].dtype, data_bf16["done"].dtype
(torch.bfloat16, torch.bool)
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