# MemoryMappedTensor

*class*tensordict.MemoryMappedTensor(*source*, ***, *dtype=None*, *shape=None*, *index=None*, *device=None*, *handler=None*, *filename=None*)

Bases: [`Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

A CPU tensor whose storage is a file on disk, or anonymous shared memory when no filename is given.

Use it to share a tensor between processes, or to work with data that is
larger than RAM: the operating system reads from the file only the pages
that you access. When a MemoryMappedTensor is sent through a
multiprocessing queue or an RPC call, only its file or shared-memory
reference is sent, not the data. [`memmap()`](tensordict.TensorDictBase.memmap.html#tensordict.TensorDictBase.memmap)
turns every leaf of a tensordict into a MemoryMappedTensor. See
[Save and load tensordicts](../../saving.html#saving) and [Load data and share it between processes](../../data_loading.html#data-loading).

Build one with `from_tensor()`, `empty()`, `zeros()`,
`ones()`, `full()`, their `*_like` variants, or
`from_filename()` for an existing file. Indexing a MemoryMappedTensor
gives a MemoryMappedTensor on the same storage. Other operations
(`mmap + 1`, for example) give a regular [`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor).

A MemoryMappedTensor has these constraints:

- It must be on CPU, and it cannot require gradients.
- With RPC, every node must have read access to the file path. If it
does not, the result of sending the tensor is undefined.

Warning

A caller that supplies `filename` must control its parent directory.
Existing symbolic links are rejected, but path-based memory mapping
cannot prevent another process with directory write access from
replacing a path after it has been validated.

MemoryMappedTensor supports multiple construction methods.

Examples

```
>>> import os
>>> import tempfile
>>> import torch
>>> from tensordict import MemoryMappedTensor
>>> # from an existing tensor
>>> tensor = torch.randn(3)
>>> with tempfile.TemporaryDirectory() as tmpdir:
... memmap_tensor = MemoryMappedTensor.from_tensor(tensor, filename=os.path.join(tmpdir, "tensor.memmap"))
... assert memmap_tensor.filename is not None
>>> # without a filename, the tensor lives in anonymous shared memory and has no filename
>>> tensor = torch.randn(3)
>>> memmap_tensor = MemoryMappedTensor.from_tensor(tensor)
>>> memmap_tensor.filename
Traceback (most recent call last):
...
RuntimeError: The MemoryMappedTensor has no file associated.
>>> # one can create an empty tensor too
>>> with tempfile.TemporaryDirectory() as tmpdir:
... memmap_tensor_empty = MemoryMappedTensor.empty_like(tensor, filename=os.path.join(tmpdir, "empty.memmap"))
>>> with tempfile.TemporaryDirectory() as tmpdir:
... memmap_tensor_zero = MemoryMappedTensor.zeros_like(tensor, filename=os.path.join(tmpdir, "zeros.memmap"))
>>> with tempfile.TemporaryDirectory() as tmpdir:
... memmap_tensor = MemoryMappedTensor.ones_like(tensor, filename=os.path.join(tmpdir, "ones.memmap"))
```

chunk(*chunks*, *dim=0*) → List of Tensors

See [`torch.chunk()`](https://docs.pytorch.org/docs/stable/generated/torch.chunk.html#torch.chunk)

*classmethod*empty(**size*, *dtype=None*, *device=None*, *filename=None*, *existsok: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*)

*classmethod*empty(*shape*, ***, *dtype=None*, *device=None*, *filename=None*, *existsok: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*)

Creates a tensor with empty content, specific shape, dtype and filename.

Parameters:

**shape** (*integers**or*[*Size*](https://docs.pytorch.org/docs/stable/size.html#torch.Size)) - the shape of the tensor.

Keyword Arguments:

- **dtype** ([*dtype*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype)) - the dtype of the tensor.
- **device** ([*device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)) - the device of the tensor. Only None and "cpu"
are accepted, any other device will raise an exception.
- **filename** (*path**or**equivalent*) - the path to the file, if any. If
`None`, the tensor lives in anonymous shared memory.
- **existsok** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)*,**optional*) - whether it is ok to overwrite an existing file.
Defaults to `False`.

*classmethod*empty_like(*input*, ***, *filename=None*)

Creates a tensor with no content but the same shape and dtype as the input tensor.

Parameters:

**input** ([*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) - the tensor to use as an example.

Keyword Arguments:

**filename** (*path**or**equivalent*) - the path to the file, if any. If
`None`, the tensor lives in anonymous shared memory.

*property*filename

The filename of the tensor, if it has one.

Raises an exception otherwise.

*classmethod*from_filename(*filename*, *dtype*, *shape*, *index=None*, *mode: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*)

Loads a MemoryMappedTensor from a given filename.

Parameters:

- **filename** (*path**or**equivalent*) - the path to the file.
- **dtype** ([*dtype*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype)) - the dtype of the tensor.
- **shape** ([*Size*](https://docs.pytorch.org/docs/stable/size.html#torch.Size)*or*[*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) - the shape of the tensor. If
a tensor is provided, it is assumed that the tensor is a nested_tensor
instance.
- **index** (*torch-compatible index type*) - an index to use to build the
tensor.
- **mode** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - `"r"` maps the file copy-on-write:
in-place writes stay in memory. `"r+"` maps it shared:
in-place writes reach the file, and a [`PermissionError`](https://docs.python.org/3/builtins/exceptions.html#PermissionError)
is raised if the file is not writable. Defaults to `None`,
which maps the file shared if it is writable and copy-on-write
otherwise.

*classmethod*from_handler(*handler*, *dtype*, *shape*, *index=None*)

Loads a MemoryMappedTensor from a given handler.

Deprecated since version 0.15: The handler is private, and this method has no replacement.

Parameters:

- **handler** (*compatible file handler*) - the handler for the tensor.
- **dtype** ([*dtype*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype)) - the dtype of the tensor.
- **shape** ([*Size*](https://docs.pytorch.org/docs/stable/size.html#torch.Size)*or*[*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) - the shape of the tensor. If
a tensor is provided, it is assumed that the tensor is a nested_tensor
instance.
- **index** (*torch-compatible index type**,**optional*) - an index to use to build the
tensor.

*classmethod*from_tensor(*input*, ***, *filename: [Path](https://docs.python.org/3/library/pathlib.html#pathlib.Path) | [str](https://docs.python.org/3/builtins/stdtypes.html#str) = None*, *existsok: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *copy_existing: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *copy_data: [bool](https://docs.python.org/3/builtins/functions.html#bool) = True*, *shape: [Size](https://docs.pytorch.org/docs/stable/size.html#torch.Size) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*)

Creates a MemoryMappedTensor with the same content as another tensor.

If `input` is a MemoryMappedTensor, it is returned unchanged in two
cases: `filename` is `None` and `input` has no file, or
`filename` is the file of `input` (and `input` was not opened
with `mode="r"`). A file-backed `input` with `filename=None` is
copied to anonymous shared memory. In all other cases, a new
`MemoryMappedTensor` is produced.

Parameters:

**input** ([*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) - the tensor which content must be copied onto
the MemoryMappedTensor.

Keyword Arguments:

- **filename** (*path to a file*) - the path to the file where the tensor
is stored. If `None`, the tensor lives in anonymous
shared memory.
- **existsok** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)*,**optional*) - if `True`, the file will overwrite
an existing file. Defaults to `False`.
- **copy_existing** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)*,**optional*) - if `True` and the provided input
is a MemoryMappedTensor with an associated filename, copying
the content to the new location is permitted. Otherwise, an
exception is thrown. This behaviour exists to prevent
inadvertently duplicating data on disk.
- **copy_data** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)*,**optional*) - if `True`, the content of the tensor
will be copied on the storage. Defaults to `True`.
- **shape** ([*Size*](https://docs.pytorch.org/docs/stable/size.html#torch.Size)*or*[*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) - a shape to override the tensor
shape. If a tensor is passed, it must represent the nested shapes of a
nested tensor.

Raises:

[**RuntimeError**](https://docs.python.org/3/builtins/exceptions.html#RuntimeError) - if `input` requires gradients, or if `input` has
 a file, `filename` is another file and `copy_existing` is
 `False`.

Examples

```
>>> import os
>>> import tempfile
>>> import torch
>>> from tensordict import MemoryMappedTensor
>>> with tempfile.TemporaryDirectory() as tmpdir:
... filename = os.path.join(tmpdir, "tensor.memmap")
... mmap = MemoryMappedTensor.from_tensor(torch.arange(3.0), filename=filename)
... print(mmap)
... print(MemoryMappedTensor.from_tensor(mmap, filename=filename) is mmap)
MemoryMappedTensor([0., 1., 2.])
True
```

*classmethod*full(**size*, *fill_value*, *dtype=None*, *device=None*, *filename=None*, *existsok: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*)

*classmethod*full(*shape*, ***, *fill_value*, *dtype=None*, *device=None*, *filename=None*, *existsok: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*)

Creates a tensor with a single content specified by fill_value, specific shape, dtype and filename.

Parameters:

**shape** (*integers**or*[*Size*](https://docs.pytorch.org/docs/stable/size.html#torch.Size)) - the shape of the tensor.

Keyword Arguments:

- **fill_value** ([*float*](tensordict.TensorDictBase.float.html#tensordict.TensorDictBase.float)*or**equivalent*) - content of the tensor.
- **dtype** ([*dtype*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype)) - the dtype of the tensor.
- **device** ([*device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)) - the device of the tensor. Only None and "cpu"
are accepted, any other device will raise an exception.
- **filename** (*path**or**equivalent*) - the path to the file, if any. If
`None`, the tensor lives in anonymous shared memory.
- **existsok** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)*,**optional*) - whether it is ok to overwrite an existing file.
Defaults to `False`.

*classmethod*full_like(*input*, *fill_value*, ***, *filename=None*)

Creates a tensor with a single content indicated by the fill_value argument, but the same shape and dtype as the input tensor.

Parameters:

- **input** ([*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) - the tensor to use as an example.
- **fill_value** ([*float*](https://docs.python.org/3/builtins/functions.html#float)*or**equivalent*) - content of the tensor.

Keyword Arguments:

**filename** (*path**or**equivalent*) - the path to the file, if any. If
`None`, the tensor lives in anonymous shared memory.

*property*index*: [Any](https://docs.python.org/3/library/typing.html#typing.Any)*

The index of this tensor in the tensor that it was built from.

Deprecated since version 0.15: This is internal to pickling and has no replacement.

*classmethod*ones(**size*, *dtype=None*, *device=None*, *filename=None*, *existsok: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*)

*classmethod*ones(*shape*, ***, *dtype=None*, *device=None*, *filename=None*, *existsok: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*)

Creates a tensor with a 1-filled content, specific shape, dtype and filename.

Parameters:

**shape** (*integers**or*[*Size*](https://docs.pytorch.org/docs/stable/size.html#torch.Size)) - the shape of the tensor.

Keyword Arguments:

- **dtype** ([*dtype*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype)) - the dtype of the tensor.
- **device** ([*device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)) - the device of the tensor. Only None and "cpu"
are accepted, any other device will raise an exception.
- **filename** (*path**or**equivalent*) - the path to the file, if any. If
`None`, the tensor lives in anonymous shared memory.
- **existsok** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)*,**optional*) - whether it is ok to overwrite an existing file.
Defaults to `False`.

*classmethod*ones_like(*input*, ***, *filename=None*)

Creates a tensor with a 1-filled content, but the same shape and dtype as the input tensor.

Parameters:

**input** ([*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) - the tensor to use as an example.

Keyword Arguments:

**filename** (*path**or**equivalent*) - the path to the file, if any. If
`None`, the tensor lives in anonymous shared memory.

*property*parent_shape*: [Size](https://docs.pytorch.org/docs/stable/size.html#torch.Size)*

The shape of the tensor that this tensor was built from.

Deprecated since version 0.15: This is internal to pickling and has no replacement.

split(*split_size*, *dim=0*)

See [`torch.split()`](https://docs.pytorch.org/docs/stable/generated/torch.split.html#torch.split)

unbind(*dim=0*) → seq

See [`torch.unbind()`](https://docs.pytorch.org/docs/stable/generated/torch.unbind.html#torch.unbind)

*classmethod*zeros(**size*, *dtype=None*, *device=None*, *filename=None*, *existsok: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*)

*classmethod*zeros(*shape*, ***, *dtype=None*, *device=None*, *filename=None*, *existsok: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*)

Creates a tensor with a 0-filled content, specific shape, dtype and filename.

Parameters:

**shape** (*integers**or*[*Size*](https://docs.pytorch.org/docs/stable/size.html#torch.Size)) - the shape of the tensor.

Keyword Arguments:

- **dtype** ([*dtype*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype)) - the dtype of the tensor.
- **device** ([*device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)) - the device of the tensor. Only None and "cpu"
are accepted, any other device will raise an exception.
- **filename** (*path**or**equivalent*) - the path to the file, if any. If
`None`, the tensor lives in anonymous shared memory.
- **existsok** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)*,**optional*) - whether it is ok to overwrite an existing file.
Defaults to `False`.

*classmethod*zeros_like(*input*, ***, *filename=None*)

Creates a tensor with a 0-filled content, but the same shape and dtype as the input tensor.

Parameters:

**input** ([*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) - the tensor to use as an example.

Keyword Arguments:

**filename** (*path**or**equivalent*) - the path to the file, if any. If
`None`, the tensor lives in anonymous shared memory.

## Tutorials that use `MemoryMappedTensor`

![](../../_images/sphx_glr_data_fashion_thumb.png)

[Using TensorDict for datasets](../../tutorials/data_fashion.html)

Using TensorDict for datasets
![](../../_images/sphx_glr_tensorclass_imagenet_thumb.png)

[Batched data loading with tensorclasses](../../tutorials/tensorclass_imagenet.html)

Batched data loading with tensorclasses
![](../../_images/sphx_glr_tensordict_memory_thumb.png)

[Devices and memory-mapped tensors](../../tutorials/tensordict_memory.html)

Devices and memory-mapped tensors