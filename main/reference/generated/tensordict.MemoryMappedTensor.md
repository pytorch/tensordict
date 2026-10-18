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

Build one with a class method rather than with the constructor:

- `from_tensor()` copies a tensor;
- `empty()`, `zeros()`, `ones()` and `full()` make a new
tensor of a given shape, and `empty_like()`, `zeros_like()`,
`ones_like()` and `full_like()` one with the shape and dtype of
another tensor;
- `from_filename()` maps an existing file, and `from_storage()`
wraps an existing storage.

The methods that write new data take a `filename`: with one, the data
is in that file; without one, it is in anonymous shared memory, and
`filename` raises an error. Indexing a MemoryMappedTensor, `unbind()`,
`chunk()` and `split()` give MemoryMappedTensors on the same
storage. Other operations (`x + 1`, for example) give a regular
[`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor).

A MemoryMappedTensor has these constraints:

- It must be on CPU, and it cannot require gradients.
- With RPC, every node must have read access to the file path. If it
does not, the result of sending the tensor is undefined.

Warning

A caller that supplies `filename` must control its parent directory.
Existing symbolic links are rejected, but path-based memory mapping
cannot prevent another process with directory write access from
replacing a path after it has been validated.

The `index` and `handler` arguments are internal.

Parameters:

**source** ([*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*or*[*TypedStorage*](https://docs.pytorch.org/docs/stable/storage.html#torch.TypedStorage)*or*[*UntypedStorage*](https://docs.pytorch.org/docs/stable/storage.html#torch.UntypedStorage)) - the data. With `filename`, the tensor is copied to that file, as
`from_tensor()` does. A storage is passed to
`from_storage()` with the other arguments. A tensor without
`filename` is wrapped as it is: the result shares its memory but
has no file or shared-memory handle, so it cannot be pickled.

Keyword Arguments:

- **dtype** ([*dtype*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype)*,**optional*) - with a storage, the dtype of the
tensor, as in `from_storage()`. Not accepted with `filename`.
Defaults to `None`.
- **shape** ([*Size*](https://docs.pytorch.org/docs/stable/size.html#torch.Size)*,**optional*) - the shape of the result, as in
`from_tensor()` or `from_storage()`. Ignored for a tensor
without `filename`. Defaults to `None`, which keeps the shape
of `source`.
- **device** ([*device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,*[*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*or*[*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - `None` or `"cpu"`;
another device raises a `ValueError`. Defaults to `None`.
- **filename** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*or**Path**,**optional*) - the file to copy `source` to.
Defaults to `None`.

Examples

```
>>> import os
>>> import tempfile
>>> import torch
>>> from tensordict import MemoryMappedTensor
>>> with tempfile.TemporaryDirectory() as tmpdir:
... filename = os.path.join(tmpdir, "x.memmap")
... x = MemoryMappedTensor.from_tensor(torch.arange(4.0), filename=filename)
... x[0] = 10.0 # writes to the file
... print(MemoryMappedTensor.from_filename(filename, dtype=torch.float32, shape=(4,)))
MemoryMappedTensor([10., 1., 2., 3.])
```

Without a filename, the data is in anonymous shared memory:

```
>>> z = MemoryMappedTensor.zeros(2, 3)
>>> z.shape, z.dtype
(torch.Size([2, 3]), torch.float32)
>>> type(z[0]).__name__, type(z + 1).__name__
('MemoryMappedTensor', 'Tensor')
```

chunk(*chunks*, *dim=0*)

Splits the tensor into `chunks` views, as [`torch.Tensor.chunk()`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.chunk.html#torch.Tensor.chunk) does.

The chunks are MemoryMappedTensors on the same storage, so they keep
the file and can be pickled. Under [`torch.compile()`](https://docs.pytorch.org/docs/stable/generated/torch.compile.html#torch.compile), they are
regular tensors.

*classmethod*empty(**size*, *dtype=None*, *device=None*, *filename=None*, *existsok: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*)

*classmethod*empty(*shape*, ***, *dtype=None*, *device=None*, *filename=None*, *existsok: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*)

Returns an uninitialized MemoryMappedTensor of the given shape.

Parameters:

***size** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,**sequence**of*[*int*](https://docs.python.org/3/builtins/functions.html#int)*,*[*Size*](https://docs.pytorch.org/docs/stable/size.html#torch.Size)*or*[*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) - the
shape, as ints (`2, 3`) or as one sequence (`(2, 3)`). It
can also be passed as the `shape` keyword. A 2-D tensor gives
a nested tensor, with the shape of each element in one row.

Keyword Arguments:

- **dtype** ([*dtype*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype)*,**optional*) - the dtype of the tensor. Defaults to
`None`, which uses the default dtype
([`torch.get_default_dtype()`](https://docs.pytorch.org/docs/stable/generated/torch.get_default_dtype.html#torch.get_default_dtype)).
- **device** ([*device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,*[*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*or*[*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - `None` or
`"cpu"`; another device raises a `RuntimeError`. Defaults
to `None`.
- **filename** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*or**Path**,**optional*) - the file to write the data to.
Defaults to `None`, which keeps the data in anonymous shared
memory.
- **existsok** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, an existing file at
`filename` is overwritten. If `False`, it raises a
`RuntimeError`. Defaults to `False`.

Examples

```
>>> import os
>>> import tempfile
>>> import torch
>>> from tensordict import MemoryMappedTensor
>>> with tempfile.TemporaryDirectory() as tmpdir:
... filename = os.path.join(tmpdir, "x.memmap")
... x = MemoryMappedTensor.empty((2, 3), dtype=torch.int64, filename=filename)
... print(x.shape, os.path.getsize(filename))
torch.Size([2, 3]) 48
```

*classmethod*empty_like(*input*, ***, *filename=None*)

Returns an uninitialized MemoryMappedTensor with the shape and dtype of `input`.

Parameters:

**input** ([*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) - the tensor whose shape and dtype the result
takes.

Keyword Arguments:

**filename** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*or**Path**,**optional*) - the file to write the data to.
It must not exist. Defaults to `None`, which keeps the data
in anonymous shared memory.

Examples

```
>>> import torch
>>> from tensordict import MemoryMappedTensor
>>> x = MemoryMappedTensor.empty_like(torch.zeros(2, 3, dtype=torch.int64))
>>> x.shape, x.dtype
(torch.Size([2, 3]), torch.int64)
```

*property*filename

The absolute path of the file that holds the data.

Raises a `RuntimeError` if the data is in anonymous shared memory.

*classmethod*from_filename(*filename*, *dtype*, *shape*, *index=None*, *mode: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*)

Returns a MemoryMappedTensor that maps an existing file.

The file holds raw data with no header, so the dtype and the shape
must be given. [`memmap()`](tensordict.TensorDictBase.memmap.html#tensordict.TensorDictBase.memmap) records them
in the `meta.json` file of the directory it writes.

Parameters:

- **filename** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*or**Path*) - the path to the file.
- **dtype** ([*dtype*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype)) - the dtype of the data.
- **shape** ([*Size*](https://docs.pytorch.org/docs/stable/size.html#torch.Size)*,**sequence**of*[*int*](https://docs.python.org/3/builtins/functions.html#int)*or*[*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) - the shape of
the tensor. A 2-D tensor gives a nested tensor, with the shape
of each element in one row.
- **index** (*index**,**optional*) - an index applied to the result, as in
`tensor[index]`. Defaults to `None`.
- **mode** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - `"r"` maps the file copy-on-write:
in-place writes stay in memory. `"r+"` maps it shared:
in-place writes reach the file, and a [`PermissionError`](https://docs.python.org/3/builtins/exceptions.html#PermissionError)
is raised if the file is not writable. Defaults to `None`,
which maps the file shared if it is writable and copy-on-write
otherwise.

Raises:

- [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) - if `mode` is not `None`, `"r"` or `"r+"`.
- [**PermissionError**](https://docs.python.org/3/builtins/exceptions.html#PermissionError) - if `mode="r+"` and the file is not writable.

Examples

```
>>> import os
>>> import tempfile
>>> import torch
>>> from tensordict import MemoryMappedTensor
>>> with tempfile.TemporaryDirectory() as tmpdir:
... filename = os.path.join(tmpdir, "x.memmap")
... _ = MemoryMappedTensor.from_tensor(torch.arange(6), filename=filename)
... x = MemoryMappedTensor.from_filename(filename, torch.int64, (2, 3), mode="r")
... x[0, 0] = 100 # copy-on-write: the file is unchanged
... print(MemoryMappedTensor.from_filename(filename, torch.int64, (2, 3)))
MemoryMappedTensor([[0, 1, 2],
 [3, 4, 5]])
```

*classmethod*from_handler(*handler*, *dtype*, *shape*, *index=None*)

Rebuilds a MemoryMappedTensor from a shared-memory handler.

Deprecated since version 0.15: The handler is private, and this method has no replacement.

Parameters:

- **handler** - the shared-memory handler of the data.
- **dtype** ([*dtype*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype)) - the dtype of the data.
- **shape** ([*Size*](https://docs.pytorch.org/docs/stable/size.html#torch.Size)*or*[*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) - the shape of the tensor, or
a 2-D tensor with the shape of each element of a nested
tensor.
- **index** (*index**,**optional*) - an index applied to the result.
Defaults to `None`.

*classmethod*from_storage(*storage*, ***, *shape: [Size](https://docs.pytorch.org/docs/stable/size.html#torch.Size) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *dtype: [dtype](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *device: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *index: IndexType | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *filename: Path | [str](https://docs.python.org/3/builtins/stdtypes.html#str) = None*, *handler: _handler = None*)

Returns a MemoryMappedTensor that views an existing storage.

The data is not copied: writes to the result change the storage. If
the storage was mapped from a file (for example with
[`torch.UntypedStorage.from_file()`](https://docs.pytorch.org/docs/stable/storage.html#torch.UntypedStorage.from_file)), the result keeps that file
and can be pickled; other storages give a tensor that cannot be
pickled.

Parameters:

**storage** ([*UntypedStorage*](https://docs.pytorch.org/docs/stable/storage.html#torch.UntypedStorage)*or*[*TypedStorage*](https://docs.pytorch.org/docs/stable/storage.html#torch.TypedStorage)) - the storage
to view.

Keyword Arguments:

- **shape** ([*Size*](https://docs.pytorch.org/docs/stable/size.html#torch.Size)*or*[*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*,**optional*) - the shape of the
result. A tensor gives a nested tensor with one shape per
row. Defaults to `None`, which gives a 1-D tensor.
- **dtype** ([*dtype*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype)*,**optional*) - the dtype to read the storage
with. Defaults to `None`, which reads an untyped storage as
`torch.int64` and accepts a typed storage only if its dtype
is the default dtype ([`torch.get_default_dtype()`](https://docs.pytorch.org/docs/stable/generated/torch.get_default_dtype.html#torch.get_default_dtype)). Pass
`dtype` to read another dtype.
- **device** ([*device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,*[*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*or*[*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - `None` or `"cpu"`.
Defaults to `None`.
- **index** (*index**,**optional*) - an index applied to the result. Defaults
to `None`.
- **filename** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*or**Path**,**optional*) - the file of the storage, if it
has one and does not record it. Defaults to `None`.
- **handler** - internal.

Raises:

[**RuntimeError**](https://docs.python.org/3/builtins/exceptions.html#RuntimeError) - if the storage has a file and `filename` names
 another one.

Examples

```
>>> import os
>>> import tempfile
>>> import torch
>>> from tensordict import MemoryMappedTensor
>>> with tempfile.TemporaryDirectory() as tmpdir:
... filename = os.path.join(tmpdir, "x.memmap")
... _ = MemoryMappedTensor.from_tensor(torch.arange(6.0), filename=filename)
... storage = torch.UntypedStorage.from_file(filename, shared=True, nbytes=24)
... x = MemoryMappedTensor.from_storage(storage, dtype=torch.float32, shape=(2, 3))
... print(x, x.filename == filename)
MemoryMappedTensor([[0., 1., 2.],
 [3., 4., 5.]]) True
```

*classmethod*from_tensor(*input*, ***, *filename: [Path](https://docs.python.org/3/library/pathlib.html#pathlib.Path) | [str](https://docs.python.org/3/builtins/stdtypes.html#str) = None*, *existsok: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *copy_existing: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *copy_data: [bool](https://docs.python.org/3/builtins/functions.html#bool) = True*, *shape: [Size](https://docs.pytorch.org/docs/stable/size.html#torch.Size) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*)

Returns a MemoryMappedTensor with a copy of the content of `input`.

If `input` is a MemoryMappedTensor, it is returned unchanged in two
cases: `filename` is `None` and `input` has no file, or
`filename` is the file of `input` (and `input` was not opened
with `mode="r"`). A file-backed `input` with `filename=None` is
copied to anonymous shared memory. In all other cases, a new
`MemoryMappedTensor` is produced.

Parameters:

**input** ([*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) - the tensor to copy. A nested tensor must
have the `torch.strided` layout.

Keyword Arguments:

- **filename** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*or**Path**,**optional*) - the file to write the data to.
Defaults to `None`, which keeps the data in anonymous shared
memory.
- **existsok** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, an existing file at
`filename` is overwritten. If `False`, it raises an error.
Defaults to `False`.
- **copy_existing** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, a MemoryMappedTensor
with a file can be copied to another file. If `False`, this
raises an error, so that data is not duplicated on disk by
mistake. Defaults to `False`.
- **copy_data** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `False`, the result has the
shape and dtype of `input` but its content is not written.
Defaults to `True`.
- **shape** ([*Size*](https://docs.pytorch.org/docs/stable/size.html#torch.Size)*or*[*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*,**optional*) - the shape of the
result, when it differs from that of `input`; `input` is
broadcast to it. For a nested tensor, a tensor with the shape
of each element, one per row. Defaults to `None`, which keeps
the shape of `input`.

Returns:

A MemoryMappedTensor, which is `input` itself in the two cases
above.

Raises:

- [**RuntimeError**](https://docs.python.org/3/builtins/exceptions.html#RuntimeError) - if `input` requires gradients, if `input` has
 a file, `filename` is another file and `copy_existing` is
 `False`, or if `filename` exists and `existsok` is
 `False`.
- [**TypeError**](https://docs.python.org/3/builtins/exceptions.html#TypeError) - if `input` is a NumPy array.

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

Returns a MemoryMappedTensor of the given shape filled with `fill_value`.

Parameters:

***size** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,**sequence**of*[*int*](https://docs.python.org/3/builtins/functions.html#int)*or*[*Size*](https://docs.pytorch.org/docs/stable/size.html#torch.Size)) - the shape, as ints
(`2, 3`) or as one sequence (`(2, 3)`). It can also be
passed as the `shape` keyword.

Keyword Arguments:

- **fill_value** (*number**or*[*bool*](https://docs.python.org/3/builtins/functions.html#bool)) - the value of every element.
- **dtype** ([*dtype*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype)*,**optional*) - the dtype of the tensor. Defaults to
`None`, which uses the default dtype
([`torch.get_default_dtype()`](https://docs.pytorch.org/docs/stable/generated/torch.get_default_dtype.html#torch.get_default_dtype)).
- **device** ([*device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,*[*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*or*[*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - `None` or
`"cpu"`; another device raises a `RuntimeError`. Defaults
to `None`.
- **filename** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*or**Path**,**optional*) - the file to write the data to.
Defaults to `None`, which keeps the data in anonymous shared
memory.
- **existsok** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, an existing file at
`filename` is overwritten. If `False`, it raises a
`RuntimeError`. Defaults to `False`.

Examples

```
>>> import torch
>>> from tensordict import MemoryMappedTensor
>>> MemoryMappedTensor.full(3, fill_value=2, dtype=torch.int64)
MemoryMappedTensor([2, 2, 2])
```

*classmethod*full_like(*input*, *fill_value*, ***, *filename=None*)

Returns a MemoryMappedTensor filled with `fill_value`, with the shape and dtype of `input`.

Parameters:

- **input** ([*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) - the tensor whose shape and dtype the result
takes.
- **fill_value** (*number**or*[*bool*](https://docs.python.org/3/builtins/functions.html#bool)) - the value of every element.

Keyword Arguments:

**filename** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*or**Path**,**optional*) - the file to write the data to.
It must not exist. Defaults to `None`, which keeps the data
in anonymous shared memory.

Examples

```
>>> import torch
>>> from tensordict import MemoryMappedTensor
>>> MemoryMappedTensor.full_like(torch.zeros(3, dtype=torch.int64), 7)
MemoryMappedTensor([7, 7, 7])
```

*property*index*: [Any](https://docs.python.org/3/library/typing.html#typing.Any)*

The index of this tensor in the tensor that it was built from.

Deprecated since version 0.15: This is internal to pickling and has no replacement.

*classmethod*ones(**size*, *dtype=None*, *device=None*, *filename=None*, *existsok: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*)

*classmethod*ones(*shape*, ***, *dtype=None*, *device=None*, *filename=None*, *existsok: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*)

Returns a MemoryMappedTensor of the given shape filled with ones.

Parameters:

***size** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,**sequence**of*[*int*](https://docs.python.org/3/builtins/functions.html#int)*,*[*Size*](https://docs.pytorch.org/docs/stable/size.html#torch.Size)*or*[*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) - the
shape, as ints (`2, 3`) or as one sequence (`(2, 3)`). It
can also be passed as the `shape` keyword. A 2-D tensor gives
a nested tensor, with the shape of each element in one row.

Keyword Arguments:

- **dtype** ([*dtype*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype)*,**optional*) - the dtype of the tensor. Defaults to
`None`, which uses the default dtype
([`torch.get_default_dtype()`](https://docs.pytorch.org/docs/stable/generated/torch.get_default_dtype.html#torch.get_default_dtype)).
- **device** ([*device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,*[*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*or*[*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - `None` or
`"cpu"`; another device raises a `RuntimeError`. Defaults
to `None`.
- **filename** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*or**Path**,**optional*) - the file to write the data to.
Defaults to `None`, which keeps the data in anonymous shared
memory.
- **existsok** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, an existing file at
`filename` is overwritten. If `False`, it raises a
`RuntimeError`. Defaults to `False`.

Examples

```
>>> from tensordict import MemoryMappedTensor
>>> MemoryMappedTensor.ones(2, 3)
MemoryMappedTensor([[1., 1., 1.],
 [1., 1., 1.]])
```

*classmethod*ones_like(*input*, ***, *filename=None*)

Returns a MemoryMappedTensor filled with ones, with the shape and dtype of `input`.

Parameters:

**input** ([*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) - the tensor whose shape and dtype the result
takes.

Keyword Arguments:

**filename** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*or**Path**,**optional*) - the file to write the data to.
It must not exist. Defaults to `None`, which keeps the data
in anonymous shared memory.

Examples

```
>>> import torch
>>> from tensordict import MemoryMappedTensor
>>> MemoryMappedTensor.ones_like(torch.zeros(3))
MemoryMappedTensor([1., 1., 1.])
```

*property*parent_shape*: [Size](https://docs.pytorch.org/docs/stable/size.html#torch.Size)*

The shape of the tensor that this tensor was built from.

Deprecated since version 0.15: This is internal to pickling and has no replacement.

split(*split_size*, *dim=0*)

Splits the tensor into views of `split_size`, as [`torch.Tensor.split()`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.split.html#torch.Tensor.split) does.

The parts are MemoryMappedTensors on the same storage, so they keep
the file and can be pickled. Under [`torch.compile()`](https://docs.pytorch.org/docs/stable/generated/torch.compile.html#torch.compile), they are
regular tensors.

unbind(*dim*)

Removes a dimension and returns the slices along it, as [`torch.Tensor.unbind()`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.unbind.html#torch.Tensor.unbind) does.

The slices are MemoryMappedTensors on the same storage, so they keep
the file and can be pickled. Under [`torch.compile()`](https://docs.pytorch.org/docs/stable/generated/torch.compile.html#torch.compile), they are
regular tensors. Unlike [`torch.Tensor.unbind()`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.unbind.html#torch.Tensor.unbind), `dim` has no
default.

*classmethod*zeros(**size*, *dtype=None*, *device=None*, *filename=None*, *existsok: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*)

*classmethod*zeros(*shape*, ***, *dtype=None*, *device=None*, *filename=None*, *existsok: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*)

Returns a MemoryMappedTensor of the given shape filled with zeros.

Parameters:

***size** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,**sequence**of*[*int*](https://docs.python.org/3/builtins/functions.html#int)*,*[*Size*](https://docs.pytorch.org/docs/stable/size.html#torch.Size)*or*[*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) - the
shape, as ints (`2, 3`) or as one sequence (`(2, 3)`). It
can also be passed as the `shape` keyword. A 2-D tensor gives
a nested tensor, with the shape of each element in one row.

Keyword Arguments:

- **dtype** ([*dtype*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype)*,**optional*) - the dtype of the tensor. Defaults to
`None`, which uses the default dtype
([`torch.get_default_dtype()`](https://docs.pytorch.org/docs/stable/generated/torch.get_default_dtype.html#torch.get_default_dtype)).
- **device** ([*device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,*[*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*or*[*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - `None` or
`"cpu"`; another device raises a `RuntimeError`. Defaults
to `None`.
- **filename** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*or**Path**,**optional*) - the file to write the data to.
Defaults to `None`, which keeps the data in anonymous shared
memory.
- **existsok** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, an existing file at
`filename` is overwritten. If `False`, it raises a
`RuntimeError`. Defaults to `False`.

Examples

```
>>> from tensordict import MemoryMappedTensor
>>> MemoryMappedTensor.zeros(2, 3)
MemoryMappedTensor([[0., 0., 0.],
 [0., 0., 0.]])
```

*classmethod*zeros_like(*input*, ***, *filename=None*)

Returns a MemoryMappedTensor filled with zeros, with the shape and dtype of `input`.

Parameters:

**input** ([*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) - the tensor whose shape and dtype the result
takes.

Keyword Arguments:

**filename** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*or**Path**,**optional*) - the file to write the data to.
It must not exist. Defaults to `None`, which keeps the data
in anonymous shared memory.

Examples

```
>>> import torch
>>> from tensordict import MemoryMappedTensor
>>> MemoryMappedTensor.zeros_like(torch.ones(3))
MemoryMappedTensor([0., 0., 0.])
```

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