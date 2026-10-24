Note

Go to the end
to download the full example code.

# Devices and memory-mapped tensors

**Author**: [Tom Begley](https://github.com/tcbegley)

One call moves all the tensors of a [`TensorDict`](../reference/generated/tensordict.TensorDict.html#tensordict.TensorDict) to a
device, and one call stores them in memory-mapped files. This tutorial is for
readers who move groups of tensors between the CPU and a GPU, or whose
datasets do not fit in RAM.

What you will learn:

- How the `device` of a tensordict applies to its entries.
- How to move a tensordict to another device, and how to remove its device.
- How to convert a tensordict to memory-mapped tensors with `memmap_`, and
how to allocate an empty memory-mapped buffer with `memmap_like`.
- How to save memory-mapped tensors to a directory and load them again.

## Devices

When you create a [`TensorDict`](../reference/generated/tensordict.TensorDict.html#tensordict.TensorDict), you can give it a device
with the `device` keyword argument. If you set the `device`, the
[`TensorDict`](../reference/generated/tensordict.TensorDict.html#tensordict.TensorDict) moves all its entries to this device. If you
do not set it, the entries can be on different devices.

The examples of this section need a CUDA device, so this page shows them
with their output, and does not run them. This example creates a
[`TensorDict`](../reference/generated/tensordict.TensorDict.html#tensordict.TensorDict) with `device="cuda:0"`. The printed
tensordict shows the device once, for all its entries.

```
>>> import torch
>>> from tensordict import TensorDict
>>> tensordict = TensorDict({"a": torch.rand(10)}, [10], device="cuda:0")
>>> print(tensordict)
TensorDict(batch_size=[10], device=cuda:0, fields={a: f32[]})
```

If the device of the [`TensorDict`](../reference/generated/tensordict.TensorDict.html#tensordict.TensorDict) is not `None`, it also
moves the new entries to the device.

```
>>> tensordict["b"] = torch.rand(10, 10)
>>> print(tensordict)
TensorDict(batch_size=[10], device=cuda:0, fields={a: f32[], b: f32[10]})
```

The `device` attribute gives the device of the
[`TensorDict`](../reference/generated/tensordict.TensorDict.html#tensordict.TensorDict).

```
>>> print(tensordict.device)
cuda:0
```

To send the contents of the [`TensorDict`](../reference/generated/tensordict.TensorDict.html#tensordict.TensorDict) to a device, use
[`TensorDict.cuda()`](../reference/generated/tensordict.TensorDictBase.cuda.html#tensordict.TensorDictBase.cuda) or
[`TensorDict.to(device)`](../reference/generated/tensordict.TensorDictBase.to.html#tensordict.TensorDictBase.to), as for a PyTorch
tensor. Like the tensor methods, they return a new tensordict and do not
change the original one.

```
>>> tensordict = tensordict.to(torch.device("cpu"))
>>> print(tensordict)
TensorDict(batch_size=[10], device=cpu, fields={a: f32[], b: f32[10]})
>>> tensordict = tensordict.cuda()
>>> print(tensordict)
TensorDict(batch_size=[10], device=cuda:0, fields={a: f32[], b: f32[10]})
```

[`TensorDict.to`](../reference/generated/tensordict.TensorDictBase.to.html#tensordict.TensorDictBase.to) needs a device as its
argument. To remove the device of the [`TensorDict`](../reference/generated/tensordict.TensorDict.html#tensordict.TensorDict), so
that its entries can be on different devices, use
[`TensorDict.clear_device_`](../reference/generated/tensordict.TensorDictBase.clear_device_.html#tensordict.TensorDictBase.clear_device_). The
entries stay where they are, and the printed form shows the device of each
entry that is not on the CPU.

```
>>> tensordict = tensordict.clear_device_()
>>> print(tensordict)
TensorDict(batch_size=[10], fields={a: f32[]@cuda:0, b: f32[10]@cuda:0})
```

## Memory-mapped tensors

A [`MemoryMappedTensor`](../reference/generated/tensordict.MemoryMappedTensor.html#tensordict.MemoryMappedTensor) is a tensor whose values are stored
in a file. The operating system reads the parts of the file that you index,
so you can read batches from a dataset that does not fit in RAM. The
[Batched data loading with tensorclasses](tensorclass_imagenet.html#sphx-glr-tutorials-tensorclass-imagenet-py) tutorial uses memory-mapped
tensors to load a dataset.

To convert the entries of a [`TensorDict`](../reference/generated/tensordict.TensorDict.html#tensordict.TensorDict) to memory-mapped
tensors, use [`TensorDict.memmap_`](../reference/generated/tensordict.TensorDictBase.memmap_.html#tensordict.TensorDictBase.memmap_). It
copies the values into the memory maps, and locks the tensordict: you can
write into its tensors, but you cannot add or remove keys.

```
import tempfile
from pathlib import Path

import torch
from tensordict import TensorDict

torch.manual_seed(0)

tensordict = TensorDict({"a": torch.rand(10), "b": {"c": torch.rand(10)}}, [10])
tensordict.memmap_()

print(tensordict)
assert tensordict.is_locked
```

```
TensorDict(batch_size=[10], device=cpu, fields={a: f32[], b: {c: f32[]}})
```

[`TensorDict.memmap_like`](../reference/generated/tensordict.TensorDictBase.memmap_like.html#tensordict.TensorDictBase.memmap_like) creates a
new tensordict with the same keys and shapes, with
[`MemoryMappedTensor`](../reference/generated/tensordict.MemoryMappedTensor.html#tensordict.MemoryMappedTensor) values. It does not copy the values:
the output below shows zeros. The new tensordict is also locked.

- Use `memmap_` to move existing data into memory maps.
- Use `memmap_like` to allocate a buffer that you fill later, for example
a dataset that does not fit in RAM.

```
tensordict = TensorDict({"a": torch.rand(10), "b": {"c": torch.rand(10)}}, [10])
mm_tensordict = tensordict.memmap_like()

print(mm_tensordict["a"].contiguous())
```

```
MemoryMappedTensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
```

Write into the buffer with an index, or with an in-place operation:

```
mm_tensordict[:5] = tensordict[:5]
assert (mm_tensordict["a"][:5] == tensordict["a"][:5]).all()
```

## Saving to a directory and loading again

By default, the memory maps are anonymous and temporary. On Linux, they are
in shared memory under `/dev/shm` when it has room. Otherwise, they are in
an unlinked temporary file, or in a shared-memory block on Windows. They
disappear when the process ends, and you cannot load them again. To save
the memory maps to a directory, pass `prefix="/path/to/root"`.

The directory has the structure of the [`TensorDict`](../reference/generated/tensordict.TensorDict.html#tensordict.TensorDict). Each
tensor is a raw `.memmap` file, and the metadata is in `meta.json`
files. For example, the [`TensorDict`](../reference/generated/tensordict.TensorDict.html#tensordict.TensorDict) above is saved as
follows:

```
├── a.memmap
├── b
│ ├── c.memmap
│ └── meta.json
└── meta.json
```

```
prefix = Path(tempfile.mkdtemp())
tensordict.memmap_(prefix=prefix)
print(sorted(str(path.relative_to(prefix)) for path in prefix.rglob("*")))
```

```
['a.memmap', 'b', 'b/c.memmap', 'b/meta.json', 'meta.json']
```

[`TensorDict.load_memmap`](../reference/generated/tensordict.TensorDictBase.load_memmap.html#tensordict.TensorDictBase.load_memmap) loads the
directory again, in this process or in another one. The entries of the
result are memory-mapped tensors that read from the files:

```
loaded = TensorDict.load_memmap(prefix)
print(loaded)
assert (loaded == tensordict).all()
```

```
TensorDict(batch_size=[10], device=cpu, fields={a: f32[], b: {c: f32[]}})
```

## Conclusion

You can now set the device of a [`TensorDict`](../reference/generated/tensordict.TensorDict.html#tensordict.TensorDict), move it to
another device, and remove its device. You can convert a tensordict to
memory-mapped tensors with `memmap_`, or allocate an empty memory-mapped
buffer with `memmap_like`. With `prefix=`, the memory maps go to a
directory, and `load_memmap` loads them again, in this process or in
another one.

## Further reading

- [Devices, dtypes, and memory](../concepts/devices.html#devices) explains devices, dtypes, and memory.
- [Save and load tensordicts](../saving.html#saving) explains all the ways to save and load tensordicts.
- [Large datasets and storage backends](../storage.html#storage) explains how to work with datasets that do not fit in RAM.
- [Batched data loading with tensorclasses](tensorclass_imagenet.html#sphx-glr-tutorials-tensorclass-imagenet-py) loads an image dataset
from memory-mapped tensors.

**Total running time of the script:** (0 minutes 0.005 seconds)

[`Download Jupyter notebook: tensordict_memory.ipynb`](../_downloads/8e7998c9628f210b43934d3aead5e4e6/tensordict_memory.ipynb)

[`Download Python source code: tensordict_memory.py`](../_downloads/8127e96f2f15507b5c5407e4c1ae7b6e/tensordict_memory.py)

[`Download zipped: tensordict_memory.zip`](../_downloads/89ddb148138028541dafab3ced0e0421/tensordict_memory.zip)

[Gallery generated by Sphinx-Gallery](https://sphinx-gallery.github.io)