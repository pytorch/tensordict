# The on-disk format

This page specifies the files that [`memmap()`](reference/generated/tensordict.TensorDictBase.memmap.html#tensordict.TensorDictBase.memmap),
[`memmap_()`](reference/generated/tensordict.TensorDictBase.memmap_.html#tensordict.TensorDictBase.memmap_),
[`save()`](reference/generated/tensordict.TensorDictBase.save.html#tensordict.TensorDictBase.save) and
[`memmap_like()`](reference/generated/tensordict.TensorDictBase.memmap_like.html#tensordict.TensorDictBase.memmap_like) write, and that
[`load_memmap()`](reference/generated/tensordict.TensorDictBase.load_memmap.html#tensordict.TensorDictBase.load_memmap) reads. It is written for
people who read or write these files without tensordict, for example from
another language. To save and load with tensordict, see [Save and load tensordicts](saving.html#saving).

## Format versions

The `meta.json` file of each saved tensordict records the version of the
format in `"_format_version"`. This page describes version 1. A
`meta.json` without `"_format_version"` was written before the format had
a version (by tensordict 0.15.2 or older), and is version 0. Version 1 has the
same layout as version 0, and adds the `"_format_version"` and `"_class"`
entries, and the `"_names"` entry of a tensordict with dim names.

`load_memmap` reads the versions up to the one it writes, and raises an
error for a newer version. Readers of version 0 ignore the entries that
version 1 adds, so tensordict 0.14 and 0.15 load the files of version 1.

The `meta.json` file of a tensorclass has neither entry (see
Tensorclasses).

## A saved tensordict

A tensordict is saved as a directory, with a `meta.json` file, one file
for each tensor (each *leaf*), and one subdirectory for each nested
tensordict:

```
root/
 meta.json
 obs.memmap
 next/
 meta.json
 obs.memmap
```

The `meta.json` file is a JSON object. It has one entry for each key of
the tensordict, and the entries of the tensordict itself:

```
{
 "obs": {"device": "cpu", "shape": [3, 4], "dtype": "torch.float32", "is_nested": false},
 "next": {"type": "TensorDict"},
 "shape": [3],
 "device": "cpu",
 "_type": "<class 'tensordict._td.TensorDict'>",
 "_format_version": 1,
 "_class": "tensordict:TensorDict"
}
```

The entries of the tensordict are:

- `"shape"`: the batch size, as a list of ints.
- `"device"`: the device, as `str(torch.device)` (such as `"cpu"` or
`"cuda:0"`), or `"None"` for a tensordict without a device.
- `"_type"`: the class, as `str(cls)`. It names the module that
defines the class, which can be a private module.
- `"_class"`: the class, as `"module:qualname"`. For a class that the
`tensordict` package exports, the module is `tensordict`, such as
`"tensordict:TensorDict"` or `"tensordict:LazyStackedTensorDict"`. For
another class, it is the `__module__` of the class. `load_memmap` looks
the class up by `"_class"` first, and by `"_type"` if `"_class"` is
missing or names no registered class.
- `"_format_version"`: the format version, an int.
- `"_names"`: the dim names, a list with one string or `null` for each
batch dim. A tensordict without dim names has no `"_names"` entry.

The entry of a leaf is an object with `"dtype"`, `"shape"`,
`"device"` (the device of the leaf when it was saved) and
`"is_nested"`. The entry of a nested tensordict, or of any other nested
collection, is an object with `"type"`, the `__name__` of its class. The
class of the nested collection is in the `meta.json` of its
subdirectory. A reader skips the entries that are not objects.

The keys of `meta.json` are the keys of the tensordict. The names of the
files and directories are the keys encoded as described in
Keys and file names. A key that is also the name of an entry of the
tensordict, such as `"shape"`, is lost: the entry of the tensordict
replaces the entry of the key. The exceptions are `"_format_version"`,
`"_class"` and `"_names"`: a key of one of these names keeps its entry,
and the `meta.json` then lacks that entry of the tensordict, so it reads as
version 0, its class is looked up by `"_type"`, or its dim names are not
read from it (a named nested tensordict can still give it names).

## Leaves

The file of a leaf is named `<key>.memmap`. It holds the elements of the
tensor in C-contiguous (row-major) order, in the byte order of the machine
that wrote it (little-endian on the platforms that tensordict supports),
with no header. Its size is the number of elements times the size of the
dtype. A leaf with no elements can have no file: the reader creates an empty
tensor from its entry.

The `"dtype"` is `str(torch.dtype)`, such as `"torch.float32"`,
`"torch.bfloat16"`, `"torch.int64"` or `"torch.bool"`.

### Nested tensors

A nested tensor with the `torch.strided` layout has `"is_nested": true`.
Its file holds the elements of each of its tensors, one after another, each
in C-contiguous order. A second file, `<key>.shape.memmap`, holds the
sizes of the tensors: an `int64` array of shape `[n, ndim]` with one row
for each of the `n` tensors. The `"shape"` of the entry is the shape of
that array, `[n, ndim]`, and not the shape of the nested tensor.

`memmap` raises an error for a nested tensor with the `torch.jagged`
layout.

## Keys and file names

By default (`robust_key=True`), a key is encoded before it is used as a
file or directory name. Each of the characters `/ \ < > : " | ? *`, the
space, `%`, the characters 0 to 31 and the character 127 is replaced by
`%` followed by its code in two uppercase hex digits, such as `%2F` for
`/`. The empty key is encoded as `%`, and the keys `.` and `..` as
`%2E` and `%2E%2E`. Other characters, including non-ASCII ones, are kept.

With `robust_key=False`, a key is used as it is. `load_memmap` also
finds files saved under the unencoded name, if no file has the encoded name
and the key is a single path component (no `/` or `\`, and no drive
prefix such as `C:`).

## Non-tensor data

A non-tensor value in a tensordict, [`NonTensorData`](reference/generated/tensordict.NonTensorData.html#tensordict.NonTensorData), is
saved as a nested collection, with a subdirectory and the entry
`{"type": "NonTensorData"}`. The `meta.json` of the subdirectory has the
`"_type"` of the class, the value in `"data"` if JSON can encode it,
and the other fields of [`NonTensorData`](reference/generated/tensordict.NonTensorData.html#tensordict.NonTensorData):

```
{"_type": "<class 'tensordict.tensorclass.NonTensorData'>", "data": "hello", "_metadata": null, "_is_non_tensor": true}
```

A value that JSON cannot encode is pickled to an `other.pickle` file
instead, which holds a pickled dict of the fields that are missing from
`meta.json`. Pickle can execute code when it is loaded: the
`allow_pickle` argument of `load_memmap` controls whether it is read.
When `other.pickle` holds only the classes `dict`, `OrderedDict`,
`list` or `tuple`, as [`from_pytree()`](reference/generated/tensordict.TensorDictBase.from_pytree.html#tensordict.TensorDictBase.from_pytree)
stores, their names are also saved in `json_types.json`, with the
SHA-256 hash of `other.pickle`, so that `load_memmap` reads them without
pickle. A reader ignores `json_types.json` if the hash does not match.

A [`NonTensorStack`](reference/generated/tensordict.NonTensorStack.html#tensordict.NonTensorStack) is saved as a directory whose
`meta.json` has the `"stack_dim"`, the `"device"` (or `null`), and
the values in `"data"`, as nested lists. If JSON cannot encode them,
`"data"` is `"pickle.pkl"`, and the values are pickled to a
`pickle.pkl` file.

## Lazy stacks and views

A [`LazyStackedTensorDict`](reference/generated/tensordict.LazyStackedTensorDict.html#tensordict.LazyStackedTensorDict) is saved as a directory whose
`meta.json` has `"stack_dim"`, the name of the stack dim in
`"stack_dim_name"` (or `null`), and no batch size. Its tensordicts are
saved in the subdirectories `0`, `1`, and so on, in order.

A tensordict indexed lazily (`_SubTensorDict`) is saved with its source,
in a `_source` subdirectory, and the index in `"index"`. A lazy view,
such as the result of `unsqueeze` or `permute` on a lazy stack, is saved
with its source in `_source`, and the operation in `"custom_op"`,
`"inv_op"`, `"custom_op_kwargs"` and `"inv_op_kwargs"`.

## Tensorclasses

A [`tensorclass()`](reference/generated/tensordict.tensorclass.html#tensordict.tensorclass) is saved as a directory. Its tensordict,
which holds the tensor fields and the non-tensor fields stored as
[`NonTensorData`](reference/generated/tensordict.NonTensorData.html#tensordict.NonTensorData), is saved in a `_tensordict`
subdirectory. The `meta.json` of the tensorclass has its `"_type"` and
the other non-tensor fields that JSON can encode. The fields that JSON
cannot encode are pickled to `other.pickle`, as for non-tensor data.

This `meta.json` has no `"_format_version"` or `"_class"` entry,
because tensordict 0.15 and older read each entry other than `"_type"` as
a field of the tensorclass. The `meta.json` of `_tensordict` has them.

To load a tensorclass, `load_memmap` finds the class by its name among
the classes that are imported. A tensorclass saved from `__main__` (a
script or a notebook) loads with the `load_memmap` method of the class.

## Archives

A `.tdz` archive, which [`save()`](reference/generated/tensordict.TensorDictBase.save.html#tensordict.TensorDictBase.save) writes for
a path that ends with `".tdz"` and [`pack_memmap()`](reference/generated/tensordict.pack_memmap.html#tensordict.pack_memmap) makes
from a directory, is a zip file whose entries are the files of the directory
tree, with `/` as the separator. At each level, the files come in name
order, with `meta.json` first, and then the subdirectories in name order.

By default, the entries are stored without compression, and the payload of
each entry starts at an offset that is a multiple of 64 bytes, so that a
reader can map the archive once and view each leaf in place. The local
header of each entry has a zip64 extra block, and before it a padding extra
block with the id `0x6474` (`b"td"` in little-endian) and as many zero
bytes as the alignment needs. Zip readers skip extra blocks that they do not
know.

The entries can be compressed (`compression="deflate"`, `"bzip2"` or
`"lzma"`). `load_memmap` reads an archive from another zip tool: if
the payload of an entry is not aligned or is compressed, it reads the leaf
into memory instead of viewing it in place.

The zip format stores a CRC-32 checksum for each entry. `load_memmap` does
not check them, and a write through `mode="r+"` does not update them:
[`refresh_archive_checksums()`](reference/generated/tensordict.refresh_archive_checksums.html#tensordict.refresh_archive_checksums) rewrites them.

Consolidated files, written by [`consolidate()`](reference/generated/tensordict.TensorDictBase.consolidate.html#tensordict.TensorDictBase.consolidate),
have a different layout that this page does not specify.