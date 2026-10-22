.. _devices:

Devices, dtypes, and memory
===========================

One call moves or casts every tensor of a tensordict, nested ones too:
``td.to("cuda")``, ``td.to(torch.float16)``, or a device and a dtype for each
tensor, copied from another tensordict. The ``device`` attribute of the
tensordict also decides where new entries go.

Most examples use the ``meta`` device in place of a GPU. Every PyTorch build
has it: a tensor on the ``meta`` device has a shape and a dtype, but no data.
The examples that need a CUDA device are marked ``+SKIP``.

Move and cast with ``to()``
---------------------------

:meth:`~tensordict.TensorDict.to` takes the same arguments as
:meth:`torch.Tensor.to`, and applies them to each leaf (each tensor at the end
of a key path). It returns a new tensordict, and sets its ``device`` attribute
(see `The device of a tensordict`_) when you give a device:

    >>> import torch
    >>> from tensordict import TensorDict
    >>> batch = TensorDict({"images": torch.zeros(4, 3, 8, 8), "labels": torch.arange(4)}, batch_size=[4])
    >>> batch.to("meta").device
    device(type='meta')
    >>> batch64 = batch.to(torch.float64)
    >>> batch64["images"].dtype, batch64["labels"].dtype
    (torch.float64, torch.float64)
    >>> both = batch.to("meta", torch.float16)
    >>> both["images"].device, both["images"].dtype
    (device(type='meta'), torch.float16)

A dtype casts every leaf, also the integer and boolean leaves, such as
``labels`` here. To cast only some of them, see `Per-leaf device and dtype
casting`_. These forms also work:

- ``td.to(tensor)`` uses the device and the dtype of ``tensor``.
- ``td.to(other=other_td)`` uses the device of ``other_td``, and its dtype if
  all its leaves have one dtype.
- :meth:`~tensordict.TensorDict.float`, :meth:`~tensordict.TensorDict.double`,
  :meth:`~tensordict.TensorDict.half`, :meth:`~tensordict.TensorDict.int`,
  :meth:`~tensordict.TensorDict.bool`, and the other dtype methods of
  :class:`torch.Tensor` cast every leaf.
- :meth:`~tensordict.TensorDict.cpu` and :meth:`~tensordict.TensorDict.cuda`
  are the same as ``to("cpu")`` and ``to("cuda")``.

If the tensordict already has the target device and you give no dtype,
``to()`` returns the same tensordict. With ``inplace=True``, ``to()`` writes
the cast leaves into the same tensordict and returns it.

The ``non_blocking`` argument controls how the leaves are copied:

- ``None`` (the default): the leaves are copied with ``non_blocking=True``.
  If the target device is not a CUDA device (for example, the CPU), ``to()``
  then synchronizes once, after all the copies. If the target is the CPU,
  ``to()`` synchronizes only if a leaf comes from another device.
  Under :func:`torch.compile`, these copies are blocking.
- ``True``: the copies are non-blocking and ``to()`` does not synchronize.
  You must synchronize before you read the data.
- ``False``: the copies are blocking.

With ``non_blocking_pin=True``, ``to()`` pins each leaf (see `Pinned
memory`_) before the copy, on the calling thread unless ``num_threads`` asks
for threads. This needs an accelerator.

.. _devices-temporary-moves:

Temporary moves
~~~~~~~~~~~~~~~

``to()`` also works as a context manager. In the ``with`` block, you work on
the moved or cast tensordict. At the end of the block, the values go back
into the entries of the original tensordict, which keep their original device
and dtype. The entries that the block adds are added to
the original tensordict too. They keep the dtype that they have in the block.
If the original tensordict has a device, they go to that device:

    >>> td = TensorDict({"x": torch.zeros(3)}, batch_size=[3], device="cpu")
    >>> with td.to(torch.float64) as td64:
    ...     td64["x"] += 1
    ...     td64["y"] = td64["x"] * 2
    >>> td["x"]
    tensor([1., 1., 1.])
    >>> td["y"].dtype
    torch.float64

With a GPU, the same pattern runs one step on the GPU and keeps the results
on the CPU. This example needs a CUDA device:

    >>> def op(td):
    ...     return td + 1
    >>> td_cpu = TensorDict({"x": torch.zeros(3)}, batch_size=[3], device="cpu")
    >>> with td_cpu.to("cuda:0") as td_gpu:  # doctest: +SKIP
    ...     td_gpu.update(op(td_gpu))
    >>> td_cpu.device  # doctest: +SKIP
    device(type='cpu')
    >>> bool((td_cpu == 1).all())  # doctest: +SKIP
    True

The device of a tensordict
--------------------------

A tensordict has a :attr:`~tensordict.TensorDict.device` attribute. By
default, it is ``None``, also if all the leaves are on the same device:

    >>> td = TensorDict({"a": torch.zeros(3)}, batch_size=[3])
    >>> print(td.device)
    None

The value changes what happens to the entries that you add:

- If ``device`` is a device, each entry that you add, at construction or
  later, is moved to that device. Nested tensordicts get the same device.
- If ``device`` is ``None``, each entry keeps its device. The leaves can be
  on different devices.

In this example, ``td`` has the ``meta`` device, and ``mixed`` has no device:

    >>> td = TensorDict({"a": torch.zeros(3)}, batch_size=[3], device="meta")
    >>> td["b"] = torch.ones(3)  # a CPU tensor
    >>> td["b"].device
    device(type='meta')
    >>> td["n"] = {"c": torch.ones(3)}
    >>> td["n"].device, td["n", "c"].device
    (device(type='meta'), device(type='meta'))
    >>> mixed = TensorDict({"a": torch.zeros(3), "b": torch.zeros(3, device="meta")}, batch_size=[3])
    >>> print(mixed.device)
    None

Two methods change the attribute and do not move any leaf:

- :meth:`~tensordict.TensorDict.auto_device_` sets ``device`` to the device
  of the leaves if they are all on one device, and to ``None`` otherwise.
- :meth:`~tensordict.TensorDict.clear_device_` sets ``device`` to ``None``.

Here, all the leaves are on the CPU:

    >>> TensorDict({"a": torch.zeros(3)}, batch_size=[3]).auto_device_().device
    device(type='cpu')

Per-leaf device and dtype casting
---------------------------------

If a tensordict has ``device=None``, its leaves can be on different devices
and have different dtypes. ``td.to(device)`` moves all the leaves to one
device. To give each leaf the device and the dtype of the matching leaf in
another tensordict, pass the result of
:meth:`~tensordict.TensorDict.attrs` to ``to()``. Use this, for example, to
match FSDP or tensor-parallel shards, data from other ranks, or a
mixed-precision state dict.

:meth:`~tensordict.TensorDict.attrs` returns a tensordict whose leaves are
:class:`~tensordict.TensorAttrs` objects, one for each leaf. Each one records
the device, the dtype, and the shape of its leaf. ``to()`` casts each leaf of
the tensordict to the device and the dtype of the leaf with the same key in
this spec:

    >>> source = TensorDict({"a": torch.zeros(3), "b": torch.zeros(3, dtype=torch.float64)}, batch_size=[3])
    >>> layout = TensorDict(
    ...     {
    ...         "a": torch.zeros(3, dtype=torch.float16, device="meta"),
    ...         "b": torch.zeros(3, dtype=torch.float32),
    ...     },
    ...     batch_size=[3],
    ... )
    >>> out = source.to(layout.attrs())
    >>> out["a"].device, out["a"].dtype
    (device(type='meta'), torch.float16)
    >>> out["b"].device, out["b"].dtype
    (device(type='cpu'), torch.float32)

On a machine with two GPUs, the same call moves each leaf to its own GPU or
to the CPU:

    >>> td0 = TensorDict({"a": torch.zeros(3, device="cuda:0"),  # doctest: +SKIP
    ...                   "b": torch.zeros(3, device="cpu")}, batch_size=[3])
    >>> td2 = TensorDict({"a": torch.zeros(3, device="cpu"),  # doctest: +SKIP
    ...                   "b": torch.zeros(3, device="cuda:1")}, batch_size=[3])
    >>> td3 = td0.to(td2.attrs())  # doctest: +SKIP
    >>> td3["a"].device, td3["b"].device  # doctest: +SKIP
    (device(type='cpu'), device(type='cuda', index=1))

These rules apply:

- By default, the copies are asynchronous. If at least one leaf moves from an
  accelerator to the CPU (D2H, device to host), ``to()`` synchronizes once at
  the end. Copies to an accelerator (H2D) or between accelerators (D2D) do not
  need it, because later kernels on the destination device wait for the copy.
- With ``non_blocking=True``, ``to()`` does not synchronize. You must
  synchronize before you read the data on the host.
- With ``non_blocking=False``, the copies are blocking.
- A spec that changes only dtypes never synchronizes.
- Leaves that are not in the spec do not change. A partial spec changes only
  some leaves.
- ``attrs(fields=...)`` selects the attributes that each
  :class:`~tensordict.TensorAttrs` records. The default is
  ``("device", "dtype", "shape")``. ``to()`` uses only the device and the
  dtype. With ``fields=("dtype",)``, the spec changes only the dtypes.
- ``non_blocking_pin=True`` is not supported with a spec. It raises
  ``NotImplementedError``.

In this example, the spec has only the key ``"a"``, so ``"b"`` keeps its dtype:

    >>> partial = TensorDict({"a": torch.zeros(3, dtype=torch.float16)}, batch_size=[3]).attrs()
    >>> out = source.to(partial)
    >>> out["a"].dtype, out["b"].dtype
    (torch.float16, torch.float64)

For the rules of non-blocking copies and pinned memory in PyTorch, see the
PyTorch tutorial
:external+pytorch_tutorials:doc:`A Guide on Good Usage of non_blocking and pin_memory() <intermediate/pinmem_nonblock>`.

Pinned memory
-------------

Pinned (page-locked) CPU memory makes copies from the CPU to a GPU faster,
and lets them run asynchronously. :meth:`~tensordict.TensorDict.pin_memory`
returns a tensordict whose leaves are in pinned memory. With
``inplace=True``, it replaces the leaves of the same tensordict.
``num_threads`` sets the number of threads that pin the leaves. The default
(``None``), like ``num_threads=0``, pins on the calling thread.
Pinning needs an accelerator, so this example does not run on a CPU-only
machine:

    >>> td = TensorDict({"pos": torch.zeros(64, 3)}, batch_size=[64])
    >>> pinned = td.pin_memory()  # doctest: +SKIP
    >>> td_gpu = pinned.to("cuda", non_blocking=True)  # doctest: +SKIP

``td.to("cuda", non_blocking_pin=True)`` pins and copies in one call: each
leaf's copy starts as soon as that leaf is pinned. With
``td.pin_memory().to("cuda")``, all the leaves are pinned before the first
copy starts. Pinning pays off for leaves of about a MiB or more. For leaves of
a few KiB, a plain ``td.to("cuda")`` is usually faster than either form.
``to()`` pins on the calling thread unless you pass ``num_threads``. Threads
start on every call, and the first pin on each new thread starts its own pool
of ``torch.get_num_threads()`` OpenMP threads, so they are usually slower.
:meth:`~tensordict.TensorDict.pin_memory_` is deprecated, and will be removed
in version 0.17. Use ``pin_memory(inplace=True)``.

Shared memory
-------------

:meth:`~tensordict.TensorDict.share_memory_` moves each CPU leaf to shared
memory. Other processes on the same machine can then read and write the
leaves without a copy, for example the workers of
:mod:`torch.multiprocessing`. After this call, the tensordict is locked (see
:ref:`keys-locking`). You can write into the existing leaves, but you cannot
add, remove, or replace an entry:

    >>> td = TensorDict({"a": torch.zeros(3)}, batch_size=[3]).share_memory_()
    >>> td.is_shared(), td.is_locked
    (True, True)
    >>> td["b"] = torch.zeros(3)
    Traceback (most recent call last):
    ...
    RuntimeError: Cannot modify locked TensorDict. ...
    >>> _ = td.set_("a", torch.ones(3))
    >>> td["a"][0] = 5.0
    >>> td["a"]
    tensor([5., 1., 1.])

``share_memory_()`` keeps non-tensor values of built-in types, such as ints,
floats, bools, strings, bytes, lists and dicts. Two cases lose data and warn.
A shared string has a fixed size, the size of its first value and at least
100 bytes: ``update_`` with a longer string stores only its first bytes. A
shared integer holds the signed 64-bit range, and an integer outside it
wraps: ``2**63`` reads back as ``-2**63``.
:ref:`data-loading` shows how to share tensordicts between processes.

Consolidation
-------------

:meth:`~tensordict.TensorDict.consolidate` copies all the leaves into one
contiguous storage, and makes each leaf a view of it. ``to(device)`` on a
consolidated tensordict then copies one buffer instead of one tensor for
each leaf, and the result is consolidated too. For the same reason, a
consolidated tensordict is faster to send to another process. The result of
``consolidate()`` is locked. If you call
:meth:`~tensordict.TensorDict.unlock_` on it, it is no longer consolidated:

    >>> c = TensorDict({"a": torch.zeros(3), "b": torch.ones(3, 2)}, batch_size=[3]).consolidate()
    >>> c.is_consolidated(), c.is_locked
    (True, True)
    >>> c.to("meta").is_consolidated()
    True

:ref:`saving` describes consolidation for serialization.

See also
--------

- :ref:`api-methods-device`: the reference entries of
  :meth:`~tensordict.TensorDict.to`, :meth:`~tensordict.TensorDict.attrs`,
  :meth:`~tensordict.TensorDict.pin_memory`, and the other methods on this
  page.
- :ref:`operations`: arithmetic, and the in-place conventions that
  ``share_memory_()`` and ``memmap_()`` do not follow.
- :ref:`saving`: memory-mapped tensordicts and consolidation.
- :ref:`compile`: ``to()`` and device copies under :func:`torch.compile`.
- The :doc:`/tutorials/tensordict_memory` tutorial: devices and memory
  placement, step by step.
