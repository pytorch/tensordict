# tensordict: containers and functions

This page lists the tensordict containers, and the methods that they all
share, grouped by task. Each group also lists the functions of the
`tensordict` module for its task. A [`TensorDict`](generated/tensordict.TensorDict.html#tensordict.TensorDict) is a
`dict[str, torch.Tensor]` that behaves like one tensor. Its entries share
their leading dimensions, the *batch size*, and an operation such as
indexing, reshaping, or [`to()`](generated/tensordict.TensorDictBase.to.html#tensordict.TensorDictBase.to) applies to every
entry. For an introduction, see [What is TensorDict?](../overview.html#overview).

## Containers

[`TensorDictBase`](generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase) is the abstract base class of all tensordicts.
Its subclasses keep their entries in different places:

- [`TensorDict`](generated/tensordict.TensorDict.html#tensordict.TensorDict) keeps its tensors in memory. It is the default
container.
- [`LazyStackedTensorDict`](generated/tensordict.LazyStackedTensorDict.html#tensordict.LazyStackedTensorDict) is a stack of tensordicts that keeps each
tensordict separate, so that their entries can have different shapes.
See [Stacking and variable-size data](../concepts/stacking.html#stacking).
- [`PersistentTensorDict`](generated/tensordict.PersistentTensorDict.html#tensordict.PersistentTensorDict) reads and writes its entries in an HDF5 or
zarr file. See [Large datasets and storage backends](../storage.html#storage).
- [`TensorDictParams`](generated/tensordict.TensorDictParams.html#tensordict.TensorDictParams) is a tensordict and a [`torch.nn.Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module).
If a module holds one, the parameters and buffers of the module include
its tensors.

[`MemoryMappedTensor`](generated/tensordict.MemoryMappedTensor.html#tensordict.MemoryMappedTensor) is not a tensordict. It is a tensor whose data
is in a file, and the type of the entries of a memory-mapped tensordict.

Other containers have their own pages: [`TensorClass`](generated/tensordict.TensorClass.html#tensordict.TensorClass), the class
form with typed fields and methods, and non-tensor data
([tensorclass and non-tensor data](tc.html#api-tensorclass)), [`TypedTensorDict`](generated/tensordict.TypedTensorDict.html#tensordict.TypedTensorDict)
([TypedTensorDict](ttd.html#api-typed-tensordict)), and the key-value stores
([Key-value stores and UCXX](storage.html#api-storage)). To choose one, see [Which container should I use?](../choosing_a_container.html#choosing-a-container).

| [`TensorDictBase`](generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)(*args, **kwargs) | TensorDictBase is an abstract parent class for TensorDicts, a torch.Tensor data container. |
| --- | --- |
| [`TensorDict`](generated/tensordict.TensorDict.html#tensordict.TensorDict)([source, batch_size, device, ...]) | A dictionary of tensors that share a batch size. |
| [`LazyStackedTensorDict`](generated/tensordict.LazyStackedTensorDict.html#tensordict.LazyStackedTensorDict)(*tensordicts[, ...]) | A stack of tensordicts that keeps each input as a separate tensordict. |
| [`PersistentTensorDict`](generated/tensordict.PersistentTensorDict.html#tensordict.PersistentTensorDict)(*[, batch_size, ...]) | A tensordict whose entries live in an HDF5 file or a Zarr store. |
| [`TensorDictParams`](generated/tensordict.TensorDictParams.html#tensordict.TensorDictParams)([parameters, no_convert, lock]) | A TensorDict whose entries are registered as the parameters and buffers of an [`Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module). |

| [`MemoryMappedTensor`](generated/tensordict.MemoryMappedTensor.html#tensordict.MemoryMappedTensor)(source, *[, dtype, ...]) | A CPU tensor whose storage is a file on disk, or anonymous shared memory when no filename is given. |
| --- | --- |

## Methods by task

The tables below list the methods of [`TensorDictBase`](generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase), with one
page for each method. Every tensordict class and every tensorclass has
these methods. A class page lists only the methods that the class adds,
and the overrides that have their own documentation. When the
`tensordict` module has a function for the same task, the function is
in the same group.

### Construction and conversion

Create a tensordict from Python, numpy, or pandas data, and convert a
tensordict to other types. Most `from_*` class methods have a function
of the same name in the `tensordict` module. To read a tensordict from a
file, see Saving, loading, and file formats. For the arguments of the constructor,
see [`TensorDict`](generated/tensordict.TensorDict.html#tensordict.TensorDict).

Methods

| [`TensorDictBase.from_any`](generated/tensordict.TensorDictBase.from_any.html#tensordict.TensorDictBase.from_any)(obj, *[, ...]) | Recursively converts any object to a TensorDict. |
| --- | --- |
| [`TensorDictBase.from_dict`](generated/tensordict.TensorDictBase.from_dict.html#tensordict.TensorDictBase.from_dict)(input_dict, *[, ...]) | Returns a TensorDict created from a dictionary or another [`TensorDict`](generated/tensordict.TensorDict.html#tensordict.TensorDict). |
| [`TensorDictBase.from_dict_instance`](generated/tensordict.TensorDictBase.from_dict_instance.html#tensordict.TensorDictBase.from_dict_instance)(input_dict, *) | Instance method version of [`from_dict()`](generated/tensordict.TensorDictBase.from_dict.html#tensordict.TensorDictBase.from_dict). |
| [`TensorDictBase.from_namedtuple`](generated/tensordict.TensorDictBase.from_namedtuple.html#tensordict.TensorDictBase.from_namedtuple)(named_tuple, *) | Converts a namedtuple to a TensorDict recursively. |
| [`TensorDictBase.from_pandas`](generated/tensordict.TensorDictBase.from_pandas.html#tensordict.TensorDictBase.from_pandas)(dataframe, *[, ...]) | Converts a pandas DataFrame to a TensorDict. |
| [`TensorDictBase.from_struct_array`](generated/tensordict.TensorDictBase.from_struct_array.html#tensordict.TensorDictBase.from_struct_array)(struct_array, *) | Converts a structured numpy array to a TensorDict. |
| [`TensorDictBase.from_tuple`](generated/tensordict.TensorDictBase.from_tuple.html#tensordict.TensorDictBase.from_tuple)(obj, *[, ...]) | Converts a tuple to a TensorDict. |
| [`TensorDictBase.from_pytree`](generated/tensordict.TensorDictBase.from_pytree.html#tensordict.TensorDictBase.from_pytree)(pytree, *[, ...]) | Converts a pytree to a TensorDict instance. |
| [`TensorDictBase.from_dataclass`](generated/tensordict.TensorDictBase.from_dataclass.html#tensordict.TensorDictBase.from_dataclass)(dataclass, *) | Converts a dataclass into a TensorDict instance. |
| [`TensorDictBase.from_list`](generated/tensordict.TensorDictBase.from_list.html#tensordict.TensorDictBase.from_list)(input, *[, ...]) | Builds a tensordict by stacking a list of dicts or tensor collections. |
| [`TensorDictBase.from_schema`](generated/tensordict.TensorDictBase.from_schema.html#tensordict.TensorDictBase.from_schema)(schema, *[, ...]) | Pre-allocate a zero-filled TensorDict from a schema. |
| [`TensorDictBase.fromkeys`](generated/tensordict.TensorDictBase.fromkeys.html#tensordict.TensorDictBase.fromkeys)(keys[, value]) | Creates a tensordict from a list of keys and a single value. |
| [`TensorDictBase.empty`](generated/tensordict.TensorDictBase.empty.html#tensordict.TensorDictBase.empty)([recurse, batch_size, ...]) | Returns a new, empty tensordict with the same device and batch size. |
| [`TensorDictBase.new_empty`](generated/tensordict.TensorDictBase.new_empty.html#tensordict.TensorDictBase.new_empty)(*size[, dtype, ...]) | Returns a tensordict of batch size `size` with uninitialized leaves. |
| [`TensorDictBase.new_full`](generated/tensordict.TensorDictBase.new_full.html#tensordict.TensorDictBase.new_full)(size, fill_value, *) | Returns a tensordict of batch size `size` with every leaf filled with `fill_value`. |
| [`TensorDictBase.new_ones`](generated/tensordict.TensorDictBase.new_ones.html#tensordict.TensorDictBase.new_ones)(*size[, dtype, ...]) | Returns a tensordict of batch size `size` with every leaf filled with 1. |
| [`TensorDictBase.new_tensor`](generated/tensordict.TensorDictBase.new_tensor.html#tensordict.TensorDictBase.new_tensor)(data, *[, dtype, ...]) | Returns a new TensorDict with data as the tensor `data`. |
| [`TensorDictBase.new_zeros`](generated/tensordict.TensorDictBase.new_zeros.html#tensordict.TensorDictBase.new_zeros)(*size[, dtype, ...]) | Returns a tensordict of batch size `size` with every leaf filled with 0. |
| [`TensorDictBase.copy`](generated/tensordict.TensorDictBase.copy.html#tensordict.TensorDictBase.copy)() | Return a shallow copy of the tensordict (ie, copies the structure but not the data). |
| [`TensorDictBase.clone`](generated/tensordict.TensorDictBase.clone.html#tensordict.TensorDictBase.clone)([recurse]) | Clones a TensorDictBase subclass instance onto a new TensorDictBase subclass of the same type. |
| [`TensorDictBase.to_dict`](generated/tensordict.TensorDictBase.to_dict.html#tensordict.TensorDictBase.to_dict)(*[, retain_none, ...]) | Returns a dictionary with key-value pairs matching those of the tensordict. |
| [`TensorDictBase.to_namedtuple`](generated/tensordict.TensorDictBase.to_namedtuple.html#tensordict.TensorDictBase.to_namedtuple)([dest_cls]) | Converts a tensordict to a namedtuple. |
| [`TensorDictBase.to_pandas`](generated/tensordict.TensorDictBase.to_pandas.html#tensordict.TensorDictBase.to_pandas)(*[, separator]) | Converts this TensorDict to a pandas DataFrame. |
| [`TensorDictBase.to_pytree`](generated/tensordict.TensorDictBase.to_pytree.html#tensordict.TensorDictBase.to_pytree)() | Converts a tensordict to a PyTree. |
| [`TensorDictBase.to_struct_array`](generated/tensordict.TensorDictBase.to_struct_array.html#tensordict.TensorDictBase.to_struct_array)() | Converts a tensordict to a numpy structured array. |
| [`TensorDictBase.to_tensordict`](generated/tensordict.TensorDictBase.to_tensordict.html#tensordict.TensorDictBase.to_tensordict)(*[, retain_none]) | Returns a [`TensorDict`](generated/tensordict.TensorDict.html#tensordict.TensorDict) copy of this tensordict. |
| [`TensorDictBase.tolist`](generated/tensordict.TensorDictBase.tolist.html#tensordict.TensorDictBase.tolist)(*[, convert_nodes, ...]) | Returns a nested list representation of the tensordict. |
| [`TensorDictBase.numpy`](generated/tensordict.TensorDictBase.numpy.html#tensordict.TensorDictBase.numpy)() | Converts a tensordict to a (possibly nested) dictionary of numpy arrays. |
| [`TensorDictBase.as_tensor`](generated/tensordict.TensorDictBase.as_tensor.html#tensordict.TensorDictBase.as_tensor)() | Converts every leaf of a tensordict to a plain torch.Tensor. |

Functions

| [`from_any`](generated/tensordict.from_any.html#tensordict.from_any)(obj, *[, auto_batch_size, ...]) | Converts any object to a TensorDict. |
| --- | --- |
| [`from_dict`](generated/tensordict.from_dict.html#tensordict.from_dict)(d, *[, auto_batch_size, ...]) | Converts a dictionary to a TensorDict. |
| [`from_namedtuple`](generated/tensordict.from_namedtuple.html#tensordict.from_namedtuple)(named_tuple, *[, ...]) | Converts a namedtuple to a TensorDict. |
| [`from_pandas`](generated/tensordict.from_pandas.html#tensordict.from_pandas)(dataframe, *[, auto_batch_size, ...]) | Converts a pandas DataFrame to a TensorDict. |
| [`from_struct_array`](generated/tensordict.from_struct_array.html#tensordict.from_struct_array)(struct_array, *[, ...]) | Converts a structured numpy array to a TensorDict. |
| [`from_tuple`](generated/tensordict.from_tuple.html#tensordict.from_tuple)(obj, *[, auto_batch_size, ...]) | Converts a tuple to a TensorDict. |
| [`from_pytree`](generated/tensordict.from_pytree.html#tensordict.from_pytree)(pytree, *[, batch_size, ...]) | Converts a pytree to a TensorDict instance. |
| [`fromkeys`](generated/tensordict.fromkeys.html#tensordict.fromkeys)(keys[, value]) | Creates a tensordict from a list of keys and a single value. |
| [`make_tensordict`](generated/tensordict.make_tensordict.html#tensordict.make_tensordict)([input_dict, batch_size, ...]) | Returns a TensorDict created from the keyword arguments or an input dictionary. |

### Keys, values, and entries

Read, write, select, and rename the entries of a tensordict. A key is a
string, or a tuple of strings that points to an entry of a nested
tensordict. [`NestedKey`](generated/tensordict.NestedKey.html#tensordict.NestedKey) is the type of these keys. For the rules,
see [Keys, nested keys, and locking](../concepts/keys.html#keys). For values that are not tensors, see
[Non-tensor data and unbatched entries](../concepts/non_tensor.html#non-tensor-data).

Methods

| [`TensorDictBase.keys`](generated/tensordict.TensorDictBase.keys.html#tensordict.TensorDictBase.keys)([include_nested, ...]) | Returns a view of the keys of the tensordict. |
| --- | --- |
| [`TensorDictBase.values`](generated/tensordict.TensorDictBase.values.html#tensordict.TensorDictBase.values)([include_nested, ...]) | Returns a generator of the values of the tensordict. |
| [`TensorDictBase.items`](generated/tensordict.TensorDictBase.items.html#tensordict.TensorDictBase.items)([include_nested, ...]) | Returns a generator of the `(key, value)` pairs of the tensordict. |
| [`TensorDictBase.sorted_keys`](generated/tensordict.TensorDictBase.sorted_keys.html#tensordict.TensorDictBase.sorted_keys) | Returns the keys sorted in alphabetical order. |
| [`TensorDictBase.get`](generated/tensordict.TensorDictBase.get.html#tensordict.TensorDictBase.get)() | Returns the value stored at `key`, or `default` if the key is missing. |
| [`TensorDictBase.get_at`](generated/tensordict.TensorDictBase.get_at.html#tensordict.TensorDictBase.get_at)() | Get the value of a tensordict from the key key at the index idx. |
| [`TensorDictBase.set`](generated/tensordict.TensorDictBase.set.html#tensordict.TensorDictBase.set)(key, item[, inplace, ...]) | Sets an entry, and returns the tensordict. |
| [`TensorDictBase.set_`](generated/tensordict.TensorDictBase.set_.html#tensordict.TensorDictBase.set_)(key, item, *[, non_blocking]) | Copies a value into an existing entry, which keeps its storage. |
| [`TensorDictBase.set_at_`](generated/tensordict.TensorDictBase.set_at_.html#tensordict.TensorDictBase.set_at_)(key, value, index, *) | Writes `value` into the existing entry `key` at the index `index`, in place. |
| [`TensorDictBase.setdefault`](generated/tensordict.TensorDictBase.setdefault.html#tensordict.TensorDictBase.setdefault)(key, default[, ...]) | Inserts the `key` entry with a value of `default` if `key` is not in the tensordict. |
| [`TensorDictBase.update`](generated/tensordict.TensorDictBase.update.html#tensordict.TensorDictBase.update)([input_dict_or_td, ...]) | Writes the entries of a dictionary or another tensordict into this tensordict, and returns it. |
| [`TensorDictBase.update_`](generated/tensordict.TensorDictBase.update_.html#tensordict.TensorDictBase.update_)([input_dict_or_td, ...]) | Updates the TensorDict in-place with values from either a dictionary or another TensorDict. |
| [`TensorDictBase.update_at_`](generated/tensordict.TensorDictBase.update_at_.html#tensordict.TensorDictBase.update_at_)(input_dict_or_td, idx) | Updates the TensorDict in-place at the specified index with values from either a dictionary or another TensorDict. |
| [`TensorDictBase.copy_`](generated/tensordict.TensorDictBase.copy_.html#tensordict.TensorDictBase.copy_)(tensordict[, non_blocking]) | See [`TensorDictBase.update_`](generated/tensordict.TensorDictBase.update_.html#tensordict.TensorDictBase.update_). |
| [`TensorDictBase.copy_at_`](generated/tensordict.TensorDictBase.copy_at_.html#tensordict.TensorDictBase.copy_at_)(tensordict, idx[, ...]) | Copies values from `tensordict` into `self` at the specified index. |
| [`TensorDictBase.replace`](generated/tensordict.TensorDictBase.replace.html#tensordict.TensorDictBase.replace)(*args, **kwargs) | Creates a shallow copy of the tensordict where entries have been replaced. |
| [`TensorDictBase.pop`](generated/tensordict.TensorDictBase.pop.html#tensordict.TensorDictBase.pop)(key[, default]) | Removes and returns a value from a tensordict. |
| [`TensorDictBase.popitem`](generated/tensordict.TensorDictBase.popitem.html#tensordict.TensorDictBase.popitem)() | Removes the item that was last inserted into the TensorDict. |
| [`TensorDictBase.del_`](generated/tensordict.TensorDictBase.del_.html#tensordict.TensorDictBase.del_)(key) | Deletes a key of the tensordict. |
| [`TensorDictBase.clear`](generated/tensordict.TensorDictBase.clear.html#tensordict.TensorDictBase.clear)() | Erases the content of the tensordict. |
| [`TensorDictBase.select`](generated/tensordict.TensorDictBase.select.html#tensordict.TensorDictBase.select)(*keys[, inplace, ...]) | Selects the keys of the tensordict and returns a new tensordict with only the selected keys. |
| [`TensorDictBase.exclude`](generated/tensordict.TensorDictBase.exclude.html#tensordict.TensorDictBase.exclude)(*keys[, inplace]) | Excludes the keys of the tensordict and returns a new tensordict without these entries. |
| [`TensorDictBase.split_keys`](generated/tensordict.TensorDictBase.split_keys.html#tensordict.TensorDictBase.split_keys)(*key_sets[, ...]) | Splits the tensordict in subsets given one or more set of keys. |
| [`TensorDictBase.separates`](generated/tensordict.TensorDictBase.separates.html#tensordict.TensorDictBase.separates)(*keys[, default, ...]) | Separates the specified keys from the tensordict in-place. |
| [`TensorDictBase.rename_key_`](generated/tensordict.TensorDictBase.rename_key_.html#tensordict.TensorDictBase.rename_key_)(old_key, new_key) | Renames a key with a new string and returns the same tensordict with the updated key name. |
| [`TensorDictBase.flatten_keys`](generated/tensordict.TensorDictBase.flatten_keys.html#tensordict.TensorDictBase.flatten_keys)([separator, ...]) | Converts a nested tensordict into a flat one, recursively. |
| [`TensorDictBase.unflatten_keys`](generated/tensordict.TensorDictBase.unflatten_keys.html#tensordict.TensorDictBase.unflatten_keys)([separator, ...]) | Converts a flat tensordict into a nested one, recursively. |
| [`TensorDictBase.create_nested`](generated/tensordict.TensorDictBase.create_nested.html#tensordict.TensorDictBase.create_nested)(key) | Creates a nested tensordict of the same shape, device and dim names as the current tensordict. |
| [`TensorDictBase.depth`](generated/tensordict.TensorDictBase.depth.html#tensordict.TensorDictBase.depth) | the maximum number of levels of sub-tensordicts. |
| [`TensorDictBase.entry_class`](generated/tensordict.TensorDictBase.entry_class.html#tensordict.TensorDictBase.entry_class)(key) | Returns the class of an entry, possibly without reading the entry. |
| [`TensorDictBase.get_item_shape`](generated/tensordict.TensorDictBase.get_item_shape.html#tensordict.TensorDictBase.get_item_shape)(key) | Returns the shape of the entry, without reading it with [`get()`](generated/tensordict.LazyStackedTensorDict.html#tensordict.LazyStackedTensorDict.get) when possible. |
| [`TensorDictBase.is_empty`](generated/tensordict.TensorDictBase.is_empty.html#tensordict.TensorDictBase.is_empty)() | Returns `True` if the tensordict has no leaf. |
| [`TensorDictBase.fill_`](generated/tensordict.TensorDictBase.fill_.html#tensordict.TensorDictBase.fill_)(key, value) | Fills a tensor pointed by the key with a given scalar value. |
| [`TensorDictBase.set_non_tensor`](generated/tensordict.TensorDictBase.set_non_tensor.html#tensordict.TensorDictBase.set_non_tensor)(key, value) | Registers a non-tensor value in the tensordict using [`tensordict.tensorclass.NonTensorData`](generated/tensordict.NonTensorData.html#tensordict.NonTensorData). |
| [`TensorDictBase.get_non_tensor`](generated/tensordict.TensorDictBase.get_non_tensor.html#tensordict.TensorDictBase.get_non_tensor)(key[, default]) | Gets a non-tensor value, if it exists, or default if the non-tensor value is not found. |
| [`TensorDictBase.is_non_tensor`](generated/tensordict.TensorDictBase.is_non_tensor.html#tensordict.TensorDictBase.is_non_tensor)(key) | Checks if the value associated with a key is a non-tensor ([`NonTensorData`](generated/tensordict.NonTensorData.html#tensordict.NonTensorData)). |
| [`TensorDictBase.non_tensor_items`](generated/tensordict.TensorDictBase.non_tensor_items.html#tensordict.TensorDictBase.non_tensor_items)([include_nested]) | Returns all non-tensor leaves, maybe recursively. |
| [`TensorDictBase.filter_non_tensor_data`](generated/tensordict.TensorDictBase.filter_non_tensor_data.html#tensordict.TensorDictBase.filter_non_tensor_data)() | Filters out all non-tensor-data. |
| [`TensorDictBase.filter_empty_`](generated/tensordict.TensorDictBase.filter_empty_.html#tensordict.TensorDictBase.filter_empty_)() | Filters out all empty tensordicts in-place. |
| [`TensorDictBase.cat_tensors`](generated/tensordict.TensorDictBase.cat_tensors.html#tensordict.TensorDictBase.cat_tensors)(*keys, out_key[, ...]) | Concatenates entries into a new entry and possibly remove the original values. |
| [`TensorDictBase.stack_tensors`](generated/tensordict.TensorDictBase.stack_tensors.html#tensordict.TensorDictBase.stack_tensors)(*keys, out_key) | Stacks entries into a new entry and possibly remove the original values. |
| [`TensorDictBase.cat_from_tensordict`](generated/tensordict.TensorDictBase.cat_from_tensordict.html#tensordict.TensorDictBase.cat_from_tensordict)([dim, ...]) | Concatenates all entries of a tensordict in a single tensor. |
| [`TensorDictBase.stack_from_tensordict`](generated/tensordict.TensorDictBase.stack_from_tensordict.html#tensordict.TensorDictBase.stack_from_tensordict)([dim, ...]) | Stacks all entries of a tensordict in a single tensor. |

Classes and functions

| [`NestedKey`](generated/tensordict.NestedKey.html#tensordict.NestedKey)() | An abstract class for nested keys. |
| --- | --- |

| [`unravel_key`](generated/tensordict.unravel_key.html#tensordict.unravel_key)(key) | Unravel a nested key. |
| --- | --- |
| [`unravel_key_list`](generated/tensordict.unravel_key_list.html#tensordict.unravel_key_list)(keys) | Unravels every key of a list, as [`unravel_key()`](generated/tensordict.unravel_key.html#tensordict.unravel_key) does. |
| [`merge_tensordicts`](generated/tensordict.merge_tensordicts.html#tensordict.merge_tensordicts)(*tensordicts[, callback_exist]) | Merges tensordicts together. |

### Indexing, masking, and splitting

Index every entry along the batch dimensions at once, and split a
tensordict into several tensordicts. `td[0]` and `td[mask]` index the
batch dimensions. `td["key"]` returns an entry: see
Keys, values, and entries. For the rules, see [Batch size, shapes, and indexing](../concepts/batch_size.html#batch-size).

| [`TensorDictBase.__getitem__`](generated/tensordict.TensorDictBase.__getitem__.html#tensordict.TensorDictBase.__getitem__)() | Indexes all tensors according to the provided index. |
| --- | --- |
| [`TensorDictBase.__iter__`](generated/tensordict.TensorDictBase.__iter__.html#tensordict.TensorDictBase.__iter__)() | Iterates over the first batch dimension of the tensordict. |
| [`TensorDictBase.gather`](generated/tensordict.TensorDictBase.gather.html#tensordict.TensorDictBase.gather)(dim, index[, out, inplace]) | Gathers values along an axis specified by dim. |
| [`TensorDictBase.masked_fill`](generated/tensordict.TensorDictBase.masked_fill.html#tensordict.TensorDictBase.masked_fill)(mask, value) | Out-of-place version of masked_fill. |
| [`TensorDictBase.masked_fill_`](generated/tensordict.TensorDictBase.masked_fill_.html#tensordict.TensorDictBase.masked_fill_)(mask, value) | Fills the values corresponding to the mask with the desired value. |
| [`TensorDictBase.masked_select`](generated/tensordict.TensorDictBase.masked_select.html#tensordict.TensorDictBase.masked_select)(mask, *[, inplace]) | Masks all tensors of the TensorDict and return a new TensorDict instance with similar keys pointing to masked values. |
| [`TensorDictBase.where`](generated/tensordict.TensorDictBase.where.html#tensordict.TensorDictBase.where)(condition, other, *[, ...]) | Return a `TensorDict` of elements selected from either self or other, depending on condition. |
| [`TensorDictBase.split`](generated/tensordict.TensorDictBase.split.html#tensordict.TensorDictBase.split)(split_size[, dim]) | Splits each tensor in the TensorDict with the specified size in the given dimension, like torch.split. |
| [`TensorDictBase.chunk`](generated/tensordict.TensorDictBase.chunk.html#tensordict.TensorDictBase.chunk)(chunks[, dim]) | Splits a tensordict into the specified number of chunks, if possible. |
| [`TensorDictBase.tensor_split`](generated/tensordict.TensorDictBase.tensor_split.html#tensordict.TensorDictBase.tensor_split)(indices_or_sections) | Splits a TensorDict into multiple sub-tensordicts, all of which are views of input, along dimension dim according to the indices or number of sections specified by indices_or_sections. |
| [`TensorDictBase.unbind`](generated/tensordict.TensorDictBase.unbind.html#tensordict.TensorDictBase.unbind)(dim) | Returns a tuple of indexed tensordicts, unbound along the indicated dimension. |
| [`TensorDictBase.narrow`](generated/tensordict.TensorDictBase.narrow.html#tensordict.TensorDictBase.narrow)(dim, start, length) | Returns a new tensordict that is a narrowed version of the input. |

### Shape, batch dimensions, and stacking

Read and change the batch size and the dimension names, reshape and expand
a tensordict, and stack or concatenate several tensordicts. These methods
change the batch dimensions. The dimensions of each entry after the batch
dimensions do not change. See [Batch size, shapes, and indexing](../concepts/batch_size.html#batch-size) and [Stacking and variable-size data](../concepts/stacking.html#stacking).

Methods

| [`TensorDictBase.batch_size`](generated/tensordict.TensorDictBase.batch_size.html#tensordict.TensorDictBase.batch_size) | The batch size of the tensordict, as a [`torch.Size`](https://docs.pytorch.org/docs/stable/size.html#torch.Size). |
| --- | --- |
| [`TensorDictBase.batch_dims`](generated/tensordict.TensorDictBase.batch_dims.html#tensordict.TensorDictBase.batch_dims) | The number of batch dimensions, that is `len(self.batch_size)`. |
| [`TensorDictBase.shape`](generated/tensordict.TensorDictBase.shape.html#tensordict.TensorDictBase.shape) | See [`batch_size`](generated/tensordict.TensorDictBase.batch_size.html#tensordict.TensorDictBase.batch_size). |
| [`TensorDictBase.ndim`](generated/tensordict.TensorDictBase.ndim.html#tensordict.TensorDictBase.ndim) | See [`batch_dims()`](generated/tensordict.TensorDictBase.batch_dims.html#tensordict.TensorDictBase.batch_dims). |
| [`TensorDictBase.ndimension`](generated/tensordict.TensorDictBase.ndimension.html#tensordict.TensorDictBase.ndimension)() | See [`batch_dims()`](generated/tensordict.TensorDictBase.batch_dims.html#tensordict.TensorDictBase.batch_dims). |
| [`TensorDictBase.dim`](generated/tensordict.TensorDictBase.dim.html#tensordict.TensorDictBase.dim)() | See [`batch_dims()`](generated/tensordict.TensorDictBase.batch_dims.html#tensordict.TensorDictBase.batch_dims). |
| [`TensorDictBase.size`](generated/tensordict.TensorDictBase.size.html#tensordict.TensorDictBase.size)([dim]) | Returns the batch size, or the size of one batch dimension. |
| [`TensorDictBase.numel`](generated/tensordict.TensorDictBase.numel.html#tensordict.TensorDictBase.numel)() | Returns the number of elements in the batch, which is the product of `batch_size`. |
| [`TensorDictBase.__len__`](generated/tensordict.TensorDictBase.__len__.html#tensordict.TensorDictBase.__len__)() | Returns the length of first dimension, if there is, otherwise 0. |
| [`TensorDictBase.names`](generated/tensordict.TensorDictBase.names.html#tensordict.TensorDictBase.names) | The dimension names of the tensordict, one per batch dimension. |
| [`TensorDictBase.refine_names`](generated/tensordict.TensorDictBase.refine_names.html#tensordict.TensorDictBase.refine_names)(*names) | Refines the dimension names of self according to names. |
| [`TensorDictBase.rename`](generated/tensordict.TensorDictBase.rename.html#tensordict.TensorDictBase.rename)(*names, **rename_map) | Returns a clone of the tensordict with dimensions renamed. |
| [`TensorDictBase.rename_`](generated/tensordict.TensorDictBase.rename_.html#tensordict.TensorDictBase.rename_)(*names, **rename_map) | Same as [`rename()`](generated/tensordict.TensorDictBase.rename.html#tensordict.TensorDictBase.rename), but executes the renaming in-place. |
| [`TensorDictBase.auto_batch_size_`](generated/tensordict.TensorDictBase.auto_batch_size_.html#tensordict.TensorDictBase.auto_batch_size_)([...]) | Sets the maximum batch-size for the tensordict, up to an optional batch_dims. |
| [`TensorDictBase.reshape`](generated/tensordict.TensorDictBase.reshape.html#tensordict.TensorDictBase.reshape)(*args, **kwargs) | Returns a tensordict whose batch dimensions are reshaped to `shape`. |
| [`TensorDictBase.view`](generated/tensordict.TensorDictBase.view.html#tensordict.TensorDictBase.view)(*shape[, size, batch_size]) | Returns a tensordict with views of the tensors according to a new shape, compatible with the tensordict batch_size. |
| [`TensorDictBase.flatten`](generated/tensordict.TensorDictBase.flatten.html#tensordict.TensorDictBase.flatten)([start_dim, end_dim, ...]) | Flattens all the tensors of a tensordict. |
| [`TensorDictBase.unflatten`](generated/tensordict.TensorDictBase.unflatten.html#tensordict.TensorDictBase.unflatten)(dim, ...[, inplace]) | Unflattens a tensordict dim expanding it to a desired shape. |
| [`TensorDictBase.permute`](generated/tensordict.TensorDictBase.permute.html#tensordict.TensorDictBase.permute)(*args, **kwargs) | Returns a view of a tensordict with the batch dimensions permuted according to dims. |
| [`TensorDictBase.transpose`](generated/tensordict.TensorDictBase.transpose.html#tensordict.TensorDictBase.transpose)(dim0, dim1) | Returns a tensordict that is a transposed version of input. |
| [`TensorDictBase.swapaxes`](generated/tensordict.TensorDictBase.swapaxes.html#tensordict.TensorDictBase.swapaxes)(axis0, axis1) | Interchange two axes of the tensordict. |
| [`TensorDictBase.swapdims`](generated/tensordict.TensorDictBase.swapdims.html#tensordict.TensorDictBase.swapdims)(axis0, axis1) | Interchange two axes of the tensordict. |
| [`TensorDictBase.moveaxis`](generated/tensordict.TensorDictBase.moveaxis.html#tensordict.TensorDictBase.moveaxis)(source, destination) | Moves the dimension(s) of input at the position(s) in source to the position(s) in destination. |
| [`TensorDictBase.movedim`](generated/tensordict.TensorDictBase.movedim.html#tensordict.TensorDictBase.movedim)(source, destination) | Moves the dimension(s) of input at the position(s) in source to the position(s) in destination. |
| [`TensorDictBase.squeeze`](generated/tensordict.TensorDictBase.squeeze.html#tensordict.TensorDictBase.squeeze)(*args, **kwargs) | Removes batch dimensions of size 1 from every tensor, and returns them in a new tensordict. |
| [`TensorDictBase.unsqueeze`](generated/tensordict.TensorDictBase.unsqueeze.html#tensordict.TensorDictBase.unsqueeze)(*args, **kwargs) | Inserts a batch dimension of size 1 in every tensor, and returns them in a new tensordict. |
| [`TensorDictBase.expand`](generated/tensordict.TensorDictBase.expand.html#tensordict.TensorDictBase.expand)(*args, **kwargs) | Expands each tensor of the tensordict according to [`torch.Tensor.expand()`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.expand.html#torch.Tensor.expand), ignoring the feature dimensions. |
| [`TensorDictBase.expand_as`](generated/tensordict.TensorDictBase.expand_as.html#tensordict.TensorDictBase.expand_as)(other) | Broadcasts the shape of the tensordict to the shape of other and expands it accordingly. |
| [`TensorDictBase.broadcast_to`](generated/tensordict.TensorDictBase.broadcast_to.html#tensordict.TensorDictBase.broadcast_to)(shape) | Broadcasts the tensordict to a new shape. |
| [`TensorDictBase.repeat`](generated/tensordict.TensorDictBase.repeat.html#tensordict.TensorDictBase.repeat)(*repeats[, inplace]) | Repeats this tensor along the specified dimensions. |
| [`TensorDictBase.repeat_interleave`](generated/tensordict.TensorDictBase.repeat_interleave.html#tensordict.TensorDictBase.repeat_interleave)(repeats[, ...]) | Repeat elements of a TensorDict. |
| [`TensorDictBase.tile`](generated/tensordict.TensorDictBase.tile.html#tensordict.TensorDictBase.tile)(dims) | Construct a tensordict by repeating the elements. |
| [`TensorDictBase.flip`](generated/tensordict.TensorDictBase.flip.html#tensordict.TensorDictBase.flip)(dims) | Reverse the order of elements in the tensordict along the given dimensions. |
| [`TensorDictBase.fliplr`](generated/tensordict.TensorDictBase.fliplr.html#tensordict.TensorDictBase.fliplr)() | Flip the tensordict in the left/right direction. |
| [`TensorDictBase.flipud`](generated/tensordict.TensorDictBase.flipud.html#tensordict.TensorDictBase.flipud)() | Flip the tensordict in the up/down direction. |
| [`TensorDictBase.roll`](generated/tensordict.TensorDictBase.roll.html#tensordict.TensorDictBase.roll)(shifts[, dims, inplace]) | Roll the tensordict along the given dimensions. |
| [`TensorDictBase.rot90`](generated/tensordict.TensorDictBase.rot90.html#tensordict.TensorDictBase.rot90)([k, dims]) | Rotate the tensordict by 90 degrees in the plane specified by dims. |
| [`TensorDictBase.pad`](generated/tensordict.TensorDictBase.pad.html#tensordict.TensorDictBase.pad)(pad_size[, value, ...]) | Pads this tensordict along the batch dimensions with a constant value. |
| [`TensorDictBase.atleast_1d`](generated/tensordict.TensorDictBase.atleast_1d.html#tensordict.TensorDictBase.atleast_1d)() | Returns the tensordict with at least 1 batch dimension. |
| [`TensorDictBase.atleast_2d`](generated/tensordict.TensorDictBase.atleast_2d.html#tensordict.TensorDictBase.atleast_2d)() | Returns the tensordict with at least 2 batch dimensions. |
| [`TensorDictBase.atleast_3d`](generated/tensordict.TensorDictBase.atleast_3d.html#tensordict.TensorDictBase.atleast_3d)() | Returns the tensordict with at least 3 batch dimensions. |
| [`TensorDictBase.contiguous`](generated/tensordict.TensorDictBase.contiguous.html#tensordict.TensorDictBase.contiguous)(*[, canonical, ...]) | Returns a new tensordict of the same type with contiguous values (or self if values are already contiguous). |
| [`TensorDictBase.is_contiguous`](generated/tensordict.TensorDictBase.is_contiguous.html#tensordict.TensorDictBase.is_contiguous)() | Returns a boolean indicating if all the tensors are contiguous. |
| [`TensorDictBase.stack`](generated/tensordict.TensorDictBase.stack.html#tensordict.TensorDictBase.stack)(input[, dim, out]) | Stacks tensordicts into a single tensordict along the given dimension. |
| [`TensorDictBase.cat`](generated/tensordict.TensorDictBase.cat.html#tensordict.TensorDictBase.cat)(input[, dim, out]) | Concatenates tensordicts into a single tensordict along the given dimension. |
| [`TensorDictBase.lazy_stack`](generated/tensordict.TensorDictBase.lazy_stack.html#tensordict.TensorDictBase.lazy_stack)(input[, dim, out]) | Creates a lazy stack of tensordicts. |
| [`TensorDictBase.maybe_dense_stack`](generated/tensordict.TensorDictBase.maybe_dense_stack.html#tensordict.TensorDictBase.maybe_dense_stack)(input[, ...]) | Attempts to make a dense stack of tensordicts, and falls back on lazy stack when required. |
| [`TensorDictBase.to_lazystack`](generated/tensordict.TensorDictBase.to_lazystack.html#tensordict.TensorDictBase.to_lazystack)([dim]) | Converts a TensorDict to a LazyStackedTensorDict or equivalent. |
| [`TensorDictBase.densify`](generated/tensordict.TensorDictBase.densify.html#tensordict.TensorDictBase.densify)([layout]) | Attempts to represent the lazy stack with contiguous tensors (plain tensors or nested). |
| [`TensorDictBase.to_padded_tensor`](generated/tensordict.TensorDictBase.to_padded_tensor.html#tensordict.TensorDictBase.to_padded_tensor)([padding, ...]) | Converts all nested tensors to a padded version and adapts the batch-size accordingly. |

Functions

| [`stack`](generated/tensordict.stack.html#tensordict.stack)(input[, dim, out]) | Stacks tensordicts into one dense tensordict along a new dimension `dim`. |
| --- | --- |
| [`cat`](generated/tensordict.cat.html#tensordict.cat)(input[, dim, out]) | Concatenates tensordicts along an existing batch dimension `dim`. |
| [`lazy_stack`](generated/tensordict.lazy_stack.html#tensordict.lazy_stack)(input[, dim, out]) | Stacks tensordicts in a [`LazyStackedTensorDict`](generated/tensordict.LazyStackedTensorDict.html#tensordict.LazyStackedTensorDict). |
| [`maybe_dense_stack`](generated/tensordict.maybe_dense_stack.html#tensordict.maybe_dense_stack)(input[, dim, out]) | Stacks tensordicts densely if possible, or in a lazy stack otherwise. |
| [`dense_stack_tds`](generated/tensordict.dense_stack_tds.html#tensordict.dense_stack_tds)(td_list[, dim]) | Stacks a list of tensordicts, or the tensordicts of a lazy stack, densely if possible. |
| [`pad`](generated/tensordict.pad.html#tensordict.pad)(tensordict, pad_size[, value, inplace, safe]) | Pads all tensors in a tensordict along the batch dimensions with a constant value. |
| [`pad_sequence`](generated/tensordict.pad_sequence.html#tensordict.pad_sequence)(list_of_tensordicts[, pad_dim, ...]) | Pads a list of tensordicts in order for them to be stacked together in a contiguous format. |

### Device, dtype, and memory

Move a tensordict to a device, cast its entries to another dtype, and
control the memory that holds them. See [Devices, dtypes, and memory](../concepts/devices.html#devices).

Device and memory

| [`TensorDictBase.to`](generated/tensordict.TensorDictBase.to.html#tensordict.TensorDictBase.to)(*args, **kwargs) | Maps a TensorDictBase subclass either on another device, dtype or to another TensorDictBase subclass (if permitted). |
| --- | --- |
| [`TensorDictBase.cpu`](generated/tensordict.TensorDictBase.cpu.html#tensordict.TensorDictBase.cpu)(**kwargs) | Casts a tensordict to CPU. |
| [`TensorDictBase.cuda`](generated/tensordict.TensorDictBase.cuda.html#tensordict.TensorDictBase.cuda)([device]) | Casts a tensordict to a cuda device (if not already on it). |
| [`TensorDictBase.device`](generated/tensordict.TensorDictBase.device.html#tensordict.TensorDictBase.device) | Device of a TensorDict. |
| [`TensorDictBase.auto_device_`](generated/tensordict.TensorDictBase.auto_device_.html#tensordict.TensorDictBase.auto_device_)() | Automatically sets the device, if it is unique. |
| [`TensorDictBase.clear_device_`](generated/tensordict.TensorDictBase.clear_device_.html#tensordict.TensorDictBase.clear_device_)() | Sets the device of the tensordict and its sub-tensordicts to `None`, and returns it. |
| [`TensorDictBase.is_cpu`](generated/tensordict.TensorDictBase.is_cpu.html#tensordict.TensorDictBase.is_cpu) | |
| [`TensorDictBase.is_cuda`](generated/tensordict.TensorDictBase.is_cuda.html#tensordict.TensorDictBase.is_cuda) | |
| [`TensorDictBase.pin_memory`](generated/tensordict.TensorDictBase.pin_memory.html#tensordict.TensorDictBase.pin_memory)([num_threads, inplace]) | Calls [`pin_memory()`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.pin_memory.html#torch.Tensor.pin_memory) on the stored tensors. |
| [`TensorDictBase.share_memory_`](generated/tensordict.TensorDictBase.share_memory_.html#tensordict.TensorDictBase.share_memory_)() | Places all the tensors in shared memory. |
| [`TensorDictBase.is_shared`](generated/tensordict.TensorDictBase.is_shared.html#tensordict.TensorDictBase.is_shared)() | Checks if tensordict is in shared memory. |
| [`TensorDictBase.record_stream`](generated/tensordict.TensorDictBase.record_stream.html#tensordict.TensorDictBase.record_stream)(stream) | Marks the tensordict as having been used by this stream. |
| [`TensorDictBase.consolidate`](generated/tensordict.TensorDictBase.consolidate.html#tensordict.TensorDictBase.consolidate)([filename, ...]) | Copies all the leaves into one contiguous storage, and returns a tensordict of views of it. |
| [`TensorDictBase.is_consolidated`](generated/tensordict.TensorDictBase.is_consolidated.html#tensordict.TensorDictBase.is_consolidated)() | Checks if a TensorDict has a consolidated storage. |
| [`TensorDictBase.from_consolidated`](generated/tensordict.TensorDictBase.from_consolidated.html#tensordict.TensorDictBase.from_consolidated)(filename) | |
| [`TensorDictBase.bytes`](generated/tensordict.TensorDictBase.bytes.html#tensordict.TensorDictBase.bytes)(*[, count_duplicates]) | Counts the number of bytes of the contained tensors. |
| [`TensorDictBase.data_ptr`](generated/tensordict.TensorDictBase.data_ptr.html#tensordict.TensorDictBase.data_ptr)(*[, storage]) | Returns the data_ptr of the tensordict leaves. |
| [`TensorDictBase.param_count`](generated/tensordict.TensorDictBase.param_count.html#tensordict.TensorDictBase.param_count)(*[, count_duplicates]) | Counts the number of parameters (total number of indexable items), accounting for tensors only. |
| [`TensorDictBase.attrs`](generated/tensordict.TensorDictBase.attrs.html#tensordict.TensorDictBase.attrs)(*[, fields, num_threads]) | Return a deviceless tensordict whose leaves are [`TensorAttrs`](generated/tensordict.TensorAttrs.html#tensordict.TensorAttrs). |
| [`TensorDictBase.clear_refs_for_compile_`](generated/tensordict.TensorDictBase.clear_refs_for_compile_.html#tensordict.TensorDictBase.clear_refs_for_compile_)() | Clears the weakrefs in order for the tensordict to get out of the compile region safely. |

Dtype

| [`TensorDictBase.dtype`](generated/tensordict.TensorDictBase.dtype.html#tensordict.TensorDictBase.dtype) | Returns the dtype of the leaves, or `None` if the leaves have different dtypes. |
| --- | --- |
| [`TensorDictBase.type`](generated/tensordict.TensorDictBase.type.html#tensordict.TensorDictBase.type)(dst_type) | Casts all tensors to `dst_type`. |
| [`TensorDictBase.is_floating_point`](generated/tensordict.TensorDictBase.is_floating_point.html#tensordict.TensorDictBase.is_floating_point)() | Checks if all tensors in the tensordict are floating point. |
| [`TensorDictBase.bool`](generated/tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)() | Casts all tensors to `torch.bool`. |
| [`TensorDictBase.half`](generated/tensordict.TensorDictBase.half.html#tensordict.TensorDictBase.half)() | Casts all tensors to `torch.half`. |
| [`TensorDictBase.float`](generated/tensordict.TensorDictBase.float.html#tensordict.TensorDictBase.float)() | Casts all tensors to `torch.float`. |
| [`TensorDictBase.double`](generated/tensordict.TensorDictBase.double.html#tensordict.TensorDictBase.double)() | Casts all tensors to `torch.double` (`torch.float64`). |
| [`TensorDictBase.int`](generated/tensordict.TensorDictBase.int.html#tensordict.TensorDictBase.int)() | Casts all tensors to `torch.int`. |
| [`TensorDictBase.bfloat16`](generated/tensordict.TensorDictBase.bfloat16.html#tensordict.TensorDictBase.bfloat16)() | Casts all tensors to `torch.bfloat16`. |
| [`TensorDictBase.float16`](generated/tensordict.TensorDictBase.float16.html#tensordict.TensorDictBase.float16)() | Casts all tensors to `torch.float16`. |
| [`TensorDictBase.float32`](generated/tensordict.TensorDictBase.float32.html#tensordict.TensorDictBase.float32)() | Casts all tensors to `torch.float32`. |
| [`TensorDictBase.float64`](generated/tensordict.TensorDictBase.float64.html#tensordict.TensorDictBase.float64)() | Casts all tensors to `torch.float64`. |
| [`TensorDictBase.int8`](generated/tensordict.TensorDictBase.int8.html#tensordict.TensorDictBase.int8)() | Casts all tensors to `torch.int8`. |
| [`TensorDictBase.int16`](generated/tensordict.TensorDictBase.int16.html#tensordict.TensorDictBase.int16)() | Casts all tensors to `torch.int16`. |
| [`TensorDictBase.int32`](generated/tensordict.TensorDictBase.int32.html#tensordict.TensorDictBase.int32)() | Casts all tensors to `torch.int32`. |
| [`TensorDictBase.int64`](generated/tensordict.TensorDictBase.int64.html#tensordict.TensorDictBase.int64)() | Casts all tensors to `torch.int64`. |
| [`TensorDictBase.uint8`](generated/tensordict.TensorDictBase.uint8.html#tensordict.TensorDictBase.uint8)() | Casts all tensors to `torch.uint8`. |
| [`TensorDictBase.uint16`](generated/tensordict.TensorDictBase.uint16.html#tensordict.TensorDictBase.uint16)() | Casts all tensors to `torch.uint16`. |
| [`TensorDictBase.uint32`](generated/tensordict.TensorDictBase.uint32.html#tensordict.TensorDictBase.uint32)() | Casts all tensors to `torch.uint32`. |
| [`TensorDictBase.uint64`](generated/tensordict.TensorDictBase.uint64.html#tensordict.TensorDictBase.uint64)() | Casts all tensors to `torch.uint64`. |
| [`TensorDictBase.complex32`](generated/tensordict.TensorDictBase.complex32.html#tensordict.TensorDictBase.complex32)() | Casts all tensors to `torch.complex32`. |
| [`TensorDictBase.complex64`](generated/tensordict.TensorDictBase.complex64.html#tensordict.TensorDictBase.complex64)() | Casts all tensors to `torch.complex64`. |
| [`TensorDictBase.complex128`](generated/tensordict.TensorDictBase.complex128.html#tensordict.TensorDictBase.complex128)() | Casts all tensors to `torch.complex128`. |
| [`TensorDictBase.qint8`](generated/tensordict.TensorDictBase.qint8.html#tensordict.TensorDictBase.qint8)() | Casts all tensors to `torch.qint8`. |
| [`TensorDictBase.qint32`](generated/tensordict.TensorDictBase.qint32.html#tensordict.TensorDictBase.qint32)() | Casts all tensors to `torch.qint32`. |
| [`TensorDictBase.quint8`](generated/tensordict.TensorDictBase.quint8.html#tensordict.TensorDictBase.quint8)() | Casts all tensors to `torch.quint8`. |
| [`TensorDictBase.quint4x2`](generated/tensordict.TensorDictBase.quint4x2.html#tensordict.TensorDictBase.quint4x2)() | Casts all tensors to `torch.quint4x2`. |

Functions

| [`from_consolidated`](generated/tensordict.from_consolidated.html#tensordict.from_consolidated)(filename) | Reconstructs a tensordict from a consolidated file. |
| --- | --- |

### Pointwise math

Apply arithmetic and elementwise functions to every entry. The Python
operators (`+`, `-`, `*`, `/`, `**`, `==`, `&`, ...) call
these methods. A method whose name ends with an underscore, such as
`add_`, writes into the existing tensors: see
[In-place operations](../concepts/operations.html#inplace-conventions). For broadcasting and for tensordicts with
different keys, see [Arithmetic, apply, reductions, and gradients](../concepts/operations.html#operations).

Arithmetic

| [`TensorDictBase.add`](generated/tensordict.TensorDictBase.add.html#tensordict.TensorDictBase.add)(other, *[, alpha, default]) | Adds `other`, scaled by `alpha`, to `self`. |
| --- | --- |
| [`TensorDictBase.add_`](generated/tensordict.TensorDictBase.add_.html#tensordict.TensorDictBase.add_)(other, *[, alpha]) | In-place version of [`add()`](generated/tensordict.TensorDictBase.add.html#tensordict.TensorDictBase.add). |
| [`TensorDictBase.sub`](generated/tensordict.TensorDictBase.sub.html#tensordict.TensorDictBase.sub)(other, *[, alpha, default]) | Subtracts `other`, scaled by `alpha`, from `self`. |
| [`TensorDictBase.sub_`](generated/tensordict.TensorDictBase.sub_.html#tensordict.TensorDictBase.sub_)(other[, alpha]) | In-place version of [`sub()`](generated/tensordict.TensorDictBase.sub.html#tensordict.TensorDictBase.sub). |
| [`TensorDictBase.rsub`](generated/tensordict.TensorDictBase.rsub.html#tensordict.TensorDictBase.rsub)(other, *[, alpha, default]) | Subtracts `self`, scaled by `alpha`, from `other`, leaf by leaf. |
| [`TensorDictBase.mul`](generated/tensordict.TensorDictBase.mul.html#tensordict.TensorDictBase.mul)(other, *[, default]) | Multiplies `other` to `self`. |
| [`TensorDictBase.mul_`](generated/tensordict.TensorDictBase.mul_.html#tensordict.TensorDictBase.mul_)(other) | In-place version of [`mul()`](generated/tensordict.TensorDictBase.mul.html#tensordict.TensorDictBase.mul). |
| [`TensorDictBase.div`](generated/tensordict.TensorDictBase.div.html#tensordict.TensorDictBase.div)(other, *[, default]) | Divides each element of the input `self` by the corresponding element of `other`. |
| [`TensorDictBase.div_`](generated/tensordict.TensorDictBase.div_.html#tensordict.TensorDictBase.div_)(other) | In-place version of [`div()`](generated/tensordict.TensorDictBase.div.html#tensordict.TensorDictBase.div). |
| [`TensorDictBase.pow`](generated/tensordict.TensorDictBase.pow.html#tensordict.TensorDictBase.pow)(other, *[, default]) | Takes the power of each element in `self` with `other` and returns a tensor with the result. |
| [`TensorDictBase.pow_`](generated/tensordict.TensorDictBase.pow_.html#tensordict.TensorDictBase.pow_)(other) | In-place version of [`pow()`](generated/tensordict.TensorDictBase.pow.html#tensordict.TensorDictBase.pow). |
| [`TensorDictBase.mod`](generated/tensordict.TensorDictBase.mod.html#tensordict.TensorDictBase.mod)(other) | Computes the element-wise modulo of `self` and `other`. |
| [`TensorDictBase.lerp`](generated/tensordict.TensorDictBase.lerp.html#tensordict.TensorDictBase.lerp)(end, weight) | Does a linear interpolation of two tensors `start` (given by `self`) and `end` based on a scalar or tensor `weight`. |
| [`TensorDictBase.lerp_`](generated/tensordict.TensorDictBase.lerp_.html#tensordict.TensorDictBase.lerp_)(end, weight) | In-place version of [`lerp()`](generated/tensordict.TensorDictBase.lerp.html#tensordict.TensorDictBase.lerp). |
| [`TensorDictBase.addcdiv`](generated/tensordict.TensorDictBase.addcdiv.html#tensordict.TensorDictBase.addcdiv)(other1, other2[, value]) | Performs the element-wise division of `other1` by `other2`, multiplies the result by the scalar `value` and adds it to `self`. |
| [`TensorDictBase.addcdiv_`](generated/tensordict.TensorDictBase.addcdiv_.html#tensordict.TensorDictBase.addcdiv_)(other1, other2, *[, ...]) | The in-place version of [`addcdiv()`](generated/tensordict.TensorDictBase.addcdiv.html#tensordict.TensorDictBase.addcdiv). |
| [`TensorDictBase.addcmul`](generated/tensordict.TensorDictBase.addcmul.html#tensordict.TensorDictBase.addcmul)(other1, other2, *[, ...]) | Performs the element-wise multiplication of `other1` by `other2`, multiplies the result by the scalar `value` and adds it to `self`. |
| [`TensorDictBase.addcmul_`](generated/tensordict.TensorDictBase.addcmul_.html#tensordict.TensorDictBase.addcmul_)(other1, other2, *[, ...]) | The in-place version of [`addcmul()`](generated/tensordict.TensorDictBase.addcmul.html#tensordict.TensorDictBase.addcmul). |
| [`TensorDictBase.clamp`](generated/tensordict.TensorDictBase.clamp.html#tensordict.TensorDictBase.clamp)([min, max, out]) | Clamps all elements in `self` into the range [ `min`, `max` ]. |
| [`TensorDictBase.clamp_min`](generated/tensordict.TensorDictBase.clamp_min.html#tensordict.TensorDictBase.clamp_min)(other[, default]) | Clamps each element of `self` to at least `other`. |
| [`TensorDictBase.clamp_min_`](generated/tensordict.TensorDictBase.clamp_min_.html#tensordict.TensorDictBase.clamp_min_)(other) | In-place version of [`clamp_min()`](generated/tensordict.TensorDictBase.clamp_min.html#tensordict.TensorDictBase.clamp_min). |
| [`TensorDictBase.clamp_max`](generated/tensordict.TensorDictBase.clamp_max.html#tensordict.TensorDictBase.clamp_max)(other, *[, default]) | Clamps each element of `self` to at most `other`. |
| [`TensorDictBase.clamp_max_`](generated/tensordict.TensorDictBase.clamp_max_.html#tensordict.TensorDictBase.clamp_max_)(other) | In-place version of [`clamp_max()`](generated/tensordict.TensorDictBase.clamp_max.html#tensordict.TensorDictBase.clamp_max). |
| [`TensorDictBase.maximum`](generated/tensordict.TensorDictBase.maximum.html#tensordict.TensorDictBase.maximum)(other, *[, default]) | Computes the element-wise maximum of `self` and `other`. |
| [`TensorDictBase.maximum_`](generated/tensordict.TensorDictBase.maximum_.html#tensordict.TensorDictBase.maximum_)(other) | In-place version of [`maximum()`](generated/tensordict.TensorDictBase.maximum.html#tensordict.TensorDictBase.maximum). |
| [`TensorDictBase.minimum`](generated/tensordict.TensorDictBase.minimum.html#tensordict.TensorDictBase.minimum)(other, *[, default]) | Computes the element-wise minimum of `self` and `other`. |
| [`TensorDictBase.minimum_`](generated/tensordict.TensorDictBase.minimum_.html#tensordict.TensorDictBase.minimum_)(other) | In-place version of [`minimum()`](generated/tensordict.TensorDictBase.minimum.html#tensordict.TensorDictBase.minimum). |
| [`TensorDictBase.zero_`](generated/tensordict.TensorDictBase.zero_.html#tensordict.TensorDictBase.zero_)() | Zeros all tensors in the tensordict in-place. |

Elementwise functions

| [`TensorDictBase.abs`](generated/tensordict.TensorDictBase.abs.html#tensordict.TensorDictBase.abs)() | Computes the absolute value of each element of the TensorDict. |
| --- | --- |
| [`TensorDictBase.abs_`](generated/tensordict.TensorDictBase.abs_.html#tensordict.TensorDictBase.abs_)() | Computes the absolute value of each element of the TensorDict in-place. |
| [`TensorDictBase.neg`](generated/tensordict.TensorDictBase.neg.html#tensordict.TensorDictBase.neg)() | Computes the [`neg()`](https://docs.pytorch.org/docs/stable/generated/torch.neg.html#torch.neg) value of each element of the TensorDict. |
| [`TensorDictBase.neg_`](generated/tensordict.TensorDictBase.neg_.html#tensordict.TensorDictBase.neg_)() | Computes the [`neg()`](https://docs.pytorch.org/docs/stable/generated/torch.neg.html#torch.neg) value of each element of the TensorDict in-place. |
| [`TensorDictBase.reciprocal`](generated/tensordict.TensorDictBase.reciprocal.html#tensordict.TensorDictBase.reciprocal)() | Computes the [`reciprocal()`](https://docs.pytorch.org/docs/stable/generated/torch.reciprocal.html#torch.reciprocal) value of each element of the TensorDict. |
| [`TensorDictBase.reciprocal_`](generated/tensordict.TensorDictBase.reciprocal_.html#tensordict.TensorDictBase.reciprocal_)() | Computes the [`reciprocal()`](https://docs.pytorch.org/docs/stable/generated/torch.reciprocal.html#torch.reciprocal) value of each element of the TensorDict in-place. |
| [`TensorDictBase.sqrt`](generated/tensordict.TensorDictBase.sqrt.html#tensordict.TensorDictBase.sqrt)() | Computes the element-wise square root of `self`. |
| [`TensorDictBase.sqrt_`](generated/tensordict.TensorDictBase.sqrt_.html#tensordict.TensorDictBase.sqrt_)() | In-place version of [`sqrt()`](generated/tensordict.TensorDictBase.sqrt.html#tensordict.TensorDictBase.sqrt). |
| [`TensorDictBase.exp`](generated/tensordict.TensorDictBase.exp.html#tensordict.TensorDictBase.exp)() | Computes the [`exp()`](https://docs.pytorch.org/docs/stable/generated/torch.exp.html#torch.exp) value of each element of the TensorDict. |
| [`TensorDictBase.exp_`](generated/tensordict.TensorDictBase.exp_.html#tensordict.TensorDictBase.exp_)() | Computes the [`exp()`](https://docs.pytorch.org/docs/stable/generated/torch.exp.html#torch.exp) value of each element of the TensorDict in-place. |
| [`TensorDictBase.expm1`](generated/tensordict.TensorDictBase.expm1.html#tensordict.TensorDictBase.expm1)() | Computes the [`expm1()`](https://docs.pytorch.org/docs/stable/generated/torch.expm1.html#torch.expm1) value of each element of the TensorDict. |
| [`TensorDictBase.expm1_`](generated/tensordict.TensorDictBase.expm1_.html#tensordict.TensorDictBase.expm1_)() | Computes the [`expm1()`](https://docs.pytorch.org/docs/stable/generated/torch.expm1.html#torch.expm1) value of each element of the TensorDict in-place. |
| [`TensorDictBase.log`](generated/tensordict.TensorDictBase.log.html#tensordict.TensorDictBase.log)() | Computes the [`log()`](https://docs.pytorch.org/docs/stable/generated/torch.log.html#torch.log) value of each element of the TensorDict. |
| [`TensorDictBase.log_`](generated/tensordict.TensorDictBase.log_.html#tensordict.TensorDictBase.log_)() | Computes the [`log()`](https://docs.pytorch.org/docs/stable/generated/torch.log.html#torch.log) value of each element of the TensorDict in-place. |
| [`TensorDictBase.log10`](generated/tensordict.TensorDictBase.log10.html#tensordict.TensorDictBase.log10)() | Computes the [`log10()`](https://docs.pytorch.org/docs/stable/generated/torch.log10.html#torch.log10) value of each element of the TensorDict. |
| [`TensorDictBase.log10_`](generated/tensordict.TensorDictBase.log10_.html#tensordict.TensorDictBase.log10_)() | Computes the [`log10()`](https://docs.pytorch.org/docs/stable/generated/torch.log10.html#torch.log10) value of each element of the TensorDict in-place. |
| [`TensorDictBase.log1p`](generated/tensordict.TensorDictBase.log1p.html#tensordict.TensorDictBase.log1p)() | Computes the [`log1p()`](https://docs.pytorch.org/docs/stable/generated/torch.log1p.html#torch.log1p) value of each element of the TensorDict. |
| [`TensorDictBase.log1p_`](generated/tensordict.TensorDictBase.log1p_.html#tensordict.TensorDictBase.log1p_)() | Computes the [`log1p()`](https://docs.pytorch.org/docs/stable/generated/torch.log1p.html#torch.log1p) value of each element of the TensorDict in-place. |
| [`TensorDictBase.log2`](generated/tensordict.TensorDictBase.log2.html#tensordict.TensorDictBase.log2)() | Computes the [`log2()`](https://docs.pytorch.org/docs/stable/generated/torch.log2.html#torch.log2) value of each element of the TensorDict. |
| [`TensorDictBase.log2_`](generated/tensordict.TensorDictBase.log2_.html#tensordict.TensorDictBase.log2_)() | Computes the [`log2()`](https://docs.pytorch.org/docs/stable/generated/torch.log2.html#torch.log2) value of each element of the TensorDict in-place. |
| [`TensorDictBase.sin`](generated/tensordict.TensorDictBase.sin.html#tensordict.TensorDictBase.sin)() | Computes the [`sin()`](https://docs.pytorch.org/docs/stable/generated/torch.sin.html#torch.sin) value of each element of the TensorDict. |
| [`TensorDictBase.sin_`](generated/tensordict.TensorDictBase.sin_.html#tensordict.TensorDictBase.sin_)() | Computes the [`sin()`](https://docs.pytorch.org/docs/stable/generated/torch.sin.html#torch.sin) value of each element of the TensorDict in-place. |
| [`TensorDictBase.cos`](generated/tensordict.TensorDictBase.cos.html#tensordict.TensorDictBase.cos)() | Computes the [`cos()`](https://docs.pytorch.org/docs/stable/generated/torch.cos.html#torch.cos) value of each element of the TensorDict. |
| [`TensorDictBase.cos_`](generated/tensordict.TensorDictBase.cos_.html#tensordict.TensorDictBase.cos_)() | Computes the [`cos()`](https://docs.pytorch.org/docs/stable/generated/torch.cos.html#torch.cos) value of each element of the TensorDict in-place. |
| [`TensorDictBase.tan`](generated/tensordict.TensorDictBase.tan.html#tensordict.TensorDictBase.tan)() | Computes the [`tan()`](https://docs.pytorch.org/docs/stable/generated/torch.tan.html#torch.tan) value of each element of the TensorDict. |
| [`TensorDictBase.tan_`](generated/tensordict.TensorDictBase.tan_.html#tensordict.TensorDictBase.tan_)() | Computes the [`tan()`](https://docs.pytorch.org/docs/stable/generated/torch.tan.html#torch.tan) value of each element of the TensorDict in-place. |
| [`TensorDictBase.asin`](generated/tensordict.TensorDictBase.asin.html#tensordict.TensorDictBase.asin)() | Computes the [`asin()`](https://docs.pytorch.org/docs/stable/generated/torch.asin.html#torch.asin) value of each element of the TensorDict. |
| [`TensorDictBase.asin_`](generated/tensordict.TensorDictBase.asin_.html#tensordict.TensorDictBase.asin_)() | Computes the [`asin()`](https://docs.pytorch.org/docs/stable/generated/torch.asin.html#torch.asin) value of each element of the TensorDict in-place. |
| [`TensorDictBase.acos`](generated/tensordict.TensorDictBase.acos.html#tensordict.TensorDictBase.acos)() | Computes the [`acos()`](https://docs.pytorch.org/docs/stable/generated/torch.acos.html#torch.acos) value of each element of the TensorDict. |
| [`TensorDictBase.acos_`](generated/tensordict.TensorDictBase.acos_.html#tensordict.TensorDictBase.acos_)() | Computes the [`acos()`](https://docs.pytorch.org/docs/stable/generated/torch.acos.html#torch.acos) value of each element of the TensorDict in-place. |
| [`TensorDictBase.atan`](generated/tensordict.TensorDictBase.atan.html#tensordict.TensorDictBase.atan)() | Computes the [`atan()`](https://docs.pytorch.org/docs/stable/generated/torch.atan.html#torch.atan) value of each element of the TensorDict. |
| [`TensorDictBase.atan_`](generated/tensordict.TensorDictBase.atan_.html#tensordict.TensorDictBase.atan_)() | Computes the [`atan()`](https://docs.pytorch.org/docs/stable/generated/torch.atan.html#torch.atan) value of each element of the TensorDict in-place. |
| [`TensorDictBase.sinh`](generated/tensordict.TensorDictBase.sinh.html#tensordict.TensorDictBase.sinh)() | Computes the [`sinh()`](https://docs.pytorch.org/docs/stable/generated/torch.sinh.html#torch.sinh) value of each element of the TensorDict. |
| [`TensorDictBase.sinh_`](generated/tensordict.TensorDictBase.sinh_.html#tensordict.TensorDictBase.sinh_)() | Computes the [`sinh()`](https://docs.pytorch.org/docs/stable/generated/torch.sinh.html#torch.sinh) value of each element of the TensorDict in-place. |
| [`TensorDictBase.cosh`](generated/tensordict.TensorDictBase.cosh.html#tensordict.TensorDictBase.cosh)() | Computes the [`cosh()`](https://docs.pytorch.org/docs/stable/generated/torch.cosh.html#torch.cosh) value of each element of the TensorDict. |
| [`TensorDictBase.cosh_`](generated/tensordict.TensorDictBase.cosh_.html#tensordict.TensorDictBase.cosh_)() | Computes the [`cosh()`](https://docs.pytorch.org/docs/stable/generated/torch.cosh.html#torch.cosh) value of each element of the TensorDict in-place. |
| [`TensorDictBase.tanh`](generated/tensordict.TensorDictBase.tanh.html#tensordict.TensorDictBase.tanh)() | Computes the [`tanh()`](https://docs.pytorch.org/docs/stable/generated/torch.tanh.html#torch.tanh) value of each element of the TensorDict. |
| [`TensorDictBase.tanh_`](generated/tensordict.TensorDictBase.tanh_.html#tensordict.TensorDictBase.tanh_)() | Computes the [`tanh()`](https://docs.pytorch.org/docs/stable/generated/torch.tanh.html#torch.tanh) value of each element of the TensorDict in-place. |
| [`TensorDictBase.sigmoid`](generated/tensordict.TensorDictBase.sigmoid.html#tensordict.TensorDictBase.sigmoid)() | Computes the [`sigmoid()`](https://docs.pytorch.org/docs/stable/generated/torch.sigmoid.html#torch.sigmoid) value of each element of the TensorDict. |
| [`TensorDictBase.sigmoid_`](generated/tensordict.TensorDictBase.sigmoid_.html#tensordict.TensorDictBase.sigmoid_)() | Computes the [`sigmoid()`](https://docs.pytorch.org/docs/stable/generated/torch.sigmoid.html#torch.sigmoid) value of each element of the TensorDict in-place. |
| [`TensorDictBase.erf`](generated/tensordict.TensorDictBase.erf.html#tensordict.TensorDictBase.erf)() | Computes the [`erf()`](https://docs.pytorch.org/docs/stable/generated/torch.erf.html#torch.erf) value of each element of the TensorDict. |
| [`TensorDictBase.erf_`](generated/tensordict.TensorDictBase.erf_.html#tensordict.TensorDictBase.erf_)() | Computes the [`erf()`](https://docs.pytorch.org/docs/stable/generated/torch.erf.html#torch.erf) value of each element of the TensorDict in-place. |
| [`TensorDictBase.erfc`](generated/tensordict.TensorDictBase.erfc.html#tensordict.TensorDictBase.erfc)() | Computes the [`erfc()`](https://docs.pytorch.org/docs/stable/generated/torch.erfc.html#torch.erfc) value of each element of the TensorDict. |
| [`TensorDictBase.erfc_`](generated/tensordict.TensorDictBase.erfc_.html#tensordict.TensorDictBase.erfc_)() | Computes the [`erfc()`](https://docs.pytorch.org/docs/stable/generated/torch.erfc.html#torch.erfc) value of each element of the TensorDict in-place. |
| [`TensorDictBase.lgamma`](generated/tensordict.TensorDictBase.lgamma.html#tensordict.TensorDictBase.lgamma)() | Computes the [`lgamma()`](https://docs.pytorch.org/docs/stable/generated/torch.lgamma.html#torch.lgamma) value of each element of the TensorDict. |
| [`TensorDictBase.lgamma_`](generated/tensordict.TensorDictBase.lgamma_.html#tensordict.TensorDictBase.lgamma_)() | Computes the [`lgamma()`](https://docs.pytorch.org/docs/stable/generated/torch.lgamma.html#torch.lgamma) value of each element of the TensorDict in-place. |
| [`TensorDictBase.ceil`](generated/tensordict.TensorDictBase.ceil.html#tensordict.TensorDictBase.ceil)() | Computes the [`ceil()`](https://docs.pytorch.org/docs/stable/generated/torch.ceil.html#torch.ceil) value of each element of the TensorDict. |
| [`TensorDictBase.ceil_`](generated/tensordict.TensorDictBase.ceil_.html#tensordict.TensorDictBase.ceil_)() | Computes the [`ceil()`](https://docs.pytorch.org/docs/stable/generated/torch.ceil.html#torch.ceil) value of each element of the TensorDict in-place. |
| [`TensorDictBase.floor`](generated/tensordict.TensorDictBase.floor.html#tensordict.TensorDictBase.floor)() | Computes the [`floor()`](https://docs.pytorch.org/docs/stable/generated/torch.floor.html#torch.floor) value of each element of the TensorDict. |
| [`TensorDictBase.floor_`](generated/tensordict.TensorDictBase.floor_.html#tensordict.TensorDictBase.floor_)() | Computes the [`floor()`](https://docs.pytorch.org/docs/stable/generated/torch.floor.html#torch.floor) value of each element of the TensorDict in-place. |
| [`TensorDictBase.round`](generated/tensordict.TensorDictBase.round.html#tensordict.TensorDictBase.round)() | Computes the [`round()`](https://docs.pytorch.org/docs/stable/generated/torch.round.html#torch.round) value of each element of the TensorDict. |
| [`TensorDictBase.round_`](generated/tensordict.TensorDictBase.round_.html#tensordict.TensorDictBase.round_)() | Computes the [`round()`](https://docs.pytorch.org/docs/stable/generated/torch.round.html#torch.round) value of each element of the TensorDict in-place. |
| [`TensorDictBase.trunc`](generated/tensordict.TensorDictBase.trunc.html#tensordict.TensorDictBase.trunc)() | Computes the [`trunc()`](https://docs.pytorch.org/docs/stable/generated/torch.trunc.html#torch.trunc) value of each element of the TensorDict. |
| [`TensorDictBase.trunc_`](generated/tensordict.TensorDictBase.trunc_.html#tensordict.TensorDictBase.trunc_)() | Computes the [`trunc()`](https://docs.pytorch.org/docs/stable/generated/torch.trunc.html#torch.trunc) value of each element of the TensorDict in-place. |
| [`TensorDictBase.frac`](generated/tensordict.TensorDictBase.frac.html#tensordict.TensorDictBase.frac)() | Computes the [`frac()`](https://docs.pytorch.org/docs/stable/generated/torch.frac.html#torch.frac) value of each element of the TensorDict. |
| [`TensorDictBase.frac_`](generated/tensordict.TensorDictBase.frac_.html#tensordict.TensorDictBase.frac_)() | Computes the [`frac()`](https://docs.pytorch.org/docs/stable/generated/torch.frac.html#torch.frac) value of each element of the TensorDict in-place. |
| [`TensorDictBase.sign`](generated/tensordict.TensorDictBase.sign.html#tensordict.TensorDictBase.sign)() | Computes the [`sign()`](https://docs.pytorch.org/docs/stable/generated/torch.sign.html#torch.sign) value of each element of the TensorDict. |
| [`TensorDictBase.sign_`](generated/tensordict.TensorDictBase.sign_.html#tensordict.TensorDictBase.sign_)() | Computes the [`sign()`](https://docs.pytorch.org/docs/stable/generated/torch.sign.html#torch.sign) value of each element of the TensorDict in-place. |
| [`TensorDictBase.softmax`](generated/tensordict.TensorDictBase.softmax.html#tensordict.TensorDictBase.softmax)(dim[, dtype]) | Apply a softmax function to the tensordict elements. |

Comparisons and logical operations

| [`TensorDictBase.__eq__`](generated/tensordict.TensorDictBase.__eq__.html#tensordict.TensorDictBase.__eq__)(other) | Compares two tensordicts against each other, for every key. |
| --- | --- |
| [`TensorDictBase.__ne__`](generated/tensordict.TensorDictBase.__ne__.html#tensordict.TensorDictBase.__ne__)(other) | NOT operation over two tensordicts, for evey key. |
| [`TensorDictBase.__lt__`](generated/tensordict.TensorDictBase.__lt__.html#tensordict.TensorDictBase.__lt__)(other) | Compares two tensordicts against each other using the "lower than" operator, for every key. |
| [`TensorDictBase.__le__`](generated/tensordict.TensorDictBase.__le__.html#tensordict.TensorDictBase.__le__)(other) | Compares two tensordicts against each other using the "lower or equal" operator, for every key. |
| [`TensorDictBase.__gt__`](generated/tensordict.TensorDictBase.__gt__.html#tensordict.TensorDictBase.__gt__)(other) | Compares two tensordicts against each other using the "greater than" operator, for every key. |
| [`TensorDictBase.__ge__`](generated/tensordict.TensorDictBase.__ge__.html#tensordict.TensorDictBase.__ge__)(other) | Compares two tensordicts against each other using the "greater or equal" operator, for every key. |
| [`TensorDictBase.__and__`](generated/tensordict.TensorDictBase.__and__.html#tensordict.TensorDictBase.__and__)(other) | Returns a new TensorDict instance with all tensors performing a logical or bitwise AND operation with the given value. |
| [`TensorDictBase.__or__`](generated/tensordict.TensorDictBase.__or__.html#tensordict.TensorDictBase.__or__)(other) | OR operation over two tensordicts, for evey key. |
| [`TensorDictBase.__xor__`](generated/tensordict.TensorDictBase.__xor__.html#tensordict.TensorDictBase.__xor__)(other) | XOR operation over two tensordicts, for evey key. |
| [`TensorDictBase.__invert__`](generated/tensordict.TensorDictBase.__invert__.html#tensordict.TensorDictBase.__invert__)() | Returns a new TensorDict instance with all tensors inverted (i.e., bitwise NOT operation). |
| [`TensorDictBase.bitwise_and`](generated/tensordict.TensorDictBase.bitwise_and.html#tensordict.TensorDictBase.bitwise_and)(other, *[, default]) | Performs a bitwise AND operation between `self` and `other`. |
| [`TensorDictBase.logical_and`](generated/tensordict.TensorDictBase.logical_and.html#tensordict.TensorDictBase.logical_and)(other, *[, default]) | Performs a logical AND operation between `self` and `other`. |
| [`TensorDictBase.isfinite`](generated/tensordict.TensorDictBase.isfinite.html#tensordict.TensorDictBase.isfinite)() | Returns a new tensordict with boolean elements representing if each element is finite or not. |
| [`TensorDictBase.isnan`](generated/tensordict.TensorDictBase.isnan.html#tensordict.TensorDictBase.isnan)() | Returns a new tensordict with boolean elements representing if each element of input is NaN or not. |
| [`TensorDictBase.isneginf`](generated/tensordict.TensorDictBase.isneginf.html#tensordict.TensorDictBase.isneginf)() | Returns a tensordict of boolean leaves that are `True` where the element is negative infinity. |
| [`TensorDictBase.isposinf`](generated/tensordict.TensorDictBase.isposinf.html#tensordict.TensorDictBase.isposinf)() | Returns a tensordict of boolean leaves that are `True` where the element is positive infinity. |
| [`TensorDictBase.isreal`](generated/tensordict.TensorDictBase.isreal.html#tensordict.TensorDictBase.isreal)() | Returns a new tensordict with boolean elements representing if each element of input is real-valued or not. |

### Reductions

Reduce each entry over all its elements, or along batch dimensions with
`dim`. With `reduce=True`, most reductions return one tensor for the
whole tensordict. See [Arithmetic, apply, reductions, and gradients](../concepts/operations.html#operations).

| [`TensorDictBase.all`](generated/tensordict.TensorDictBase.all.html#tensordict.TensorDictBase.all)([dim]) | Checks if all values are True/non-null in the tensordict. |
| --- | --- |
| [`TensorDictBase.any`](generated/tensordict.TensorDictBase.any.html#tensordict.TensorDictBase.any)([dim]) | Checks if any value is True/non-null in the tensordict. |
| [`TensorDictBase.amax`](generated/tensordict.TensorDictBase.amax.html#tensordict.TensorDictBase.amax)([dim, keepdim, reduce]) | Returns the maximum values of all elements in the input tensordict. |
| [`TensorDictBase.amin`](generated/tensordict.TensorDictBase.amin.html#tensordict.TensorDictBase.amin)([dim, keepdim, reduce]) | Returns the minimum values of all elements in the input tensordict. |
| [`TensorDictBase.max`](generated/tensordict.TensorDictBase.max.html#tensordict.TensorDictBase.max)([dim, keepdim, reduce, ...]) | Returns the maximum values of all elements in the input tensordict. |
| [`TensorDictBase.min`](generated/tensordict.TensorDictBase.min.html#tensordict.TensorDictBase.min)([dim, keepdim, reduce, ...]) | Returns the minimum values of all elements in the input tensordict. |
| [`TensorDictBase.cummax`](generated/tensordict.TensorDictBase.cummax.html#tensordict.TensorDictBase.cummax)(dim, *[, reduce, ...]) | Returns the cumulative maximum values of all elements in the input tensordict. |
| [`TensorDictBase.cummin`](generated/tensordict.TensorDictBase.cummin.html#tensordict.TensorDictBase.cummin)(dim, *[, reduce, ...]) | Returns the cumulative minimum values of all elements in the input tensordict. |
| [`TensorDictBase.mean`](generated/tensordict.TensorDictBase.mean.html#tensordict.TensorDictBase.mean)([dim, keepdim, dtype, ...]) | Returns the mean value of all elements in the input tensordict. |
| [`TensorDictBase.nanmean`](generated/tensordict.TensorDictBase.nanmean.html#tensordict.TensorDictBase.nanmean)([dim, keepdim, ...]) | Returns the mean of all non-NaN elements in the input tensordict. |
| [`TensorDictBase.sum`](generated/tensordict.TensorDictBase.sum.html#tensordict.TensorDictBase.sum)([dim, keepdim, dtype, ...]) | Returns the sum value of all elements in the input tensordict. |
| [`TensorDictBase.nansum`](generated/tensordict.TensorDictBase.nansum.html#tensordict.TensorDictBase.nansum)([dim, keepdim, dtype, ...]) | Returns the sum of all non-NaN elements in the input tensordict. |
| [`TensorDictBase.prod`](generated/tensordict.TensorDictBase.prod.html#tensordict.TensorDictBase.prod)([dim, keepdim, dtype, ...]) | Returns the produce of values of all elements in the input tensordict. |
| [`TensorDictBase.std`](generated/tensordict.TensorDictBase.std.html#tensordict.TensorDictBase.std)([dim, keepdim, ...]) | Returns the standard deviation value of all elements in the input tensordict. |
| [`TensorDictBase.var`](generated/tensordict.TensorDictBase.var.html#tensordict.TensorDictBase.var)([dim, keepdim, ...]) | Returns the variance value of all elements in the input tensordict. |
| [`TensorDictBase.quantile`](generated/tensordict.TensorDictBase.quantile.html#tensordict.TensorDictBase.quantile)(q[, dim, keepdim, ...]) | Returns the q-th quantile of all elements in the input tensordict. |
| [`TensorDictBase.logsumexp`](generated/tensordict.TensorDictBase.logsumexp.html#tensordict.TensorDictBase.logsumexp)([dim, keepdim, out]) | Returns the log of summed exponentials of each row of the input tensordict in the given dimension `dim`. |
| [`TensorDictBase.norm`](generated/tensordict.TensorDictBase.norm.html#tensordict.TensorDictBase.norm)(*[, out, dtype, ...]) | Computes the norm of each tensor in the tensordict. |

### Apply, modules, and gradients

Apply a function to every entry, and map a function over the batch. Put
the parameters of a module in a tensordict and back, and compute
gradients.
See [Arithmetic, apply, reductions, and gradients](../concepts/operations.html#operations), [Modules: TensorDictModule and tensordict.nn](../modules.html#modules-guide), and [torch.compile, torch.func, torch.export, and CUDA graphs](../compile.html#compile).

Methods

| [`TensorDictBase.apply`](generated/tensordict.TensorDictBase.apply.html#tensordict.TensorDictBase.apply)(fn, *others[, ...]) | Applies a callable to all values stored in the tensordict and sets them in a new tensordict. |
| --- | --- |
| [`TensorDictBase.apply_`](generated/tensordict.TensorDictBase.apply_.html#tensordict.TensorDictBase.apply_)(fn, *others, **kwargs) | Applies a callable to all values stored in the tensordict and re-writes them in-place. |
| [`TensorDictBase.named_apply`](generated/tensordict.TensorDictBase.named_apply.html#tensordict.TensorDictBase.named_apply)(fn, *others[, ...]) | Applies a key-conditioned callable to all values stored in the tensordict and sets them in a new tensordict. |
| [`TensorDictBase.map`](generated/tensordict.TensorDictBase.map.html#tensordict.TensorDictBase.map)(fn[, dim, num_workers, ...]) | Maps a function to splits of the tensordict across one dimension. |
| [`TensorDictBase.map_iter`](generated/tensordict.TensorDictBase.map_iter.html#tensordict.TensorDictBase.map_iter)(fn[, dim, ...]) | Maps a function to splits of the tensordict across one dimension iteratively. |
| [`TensorDictBase.from_module`](generated/tensordict.TensorDictBase.from_module.html#tensordict.TensorDictBase.from_module)(module[, ...]) | Returns a tensordict of the parameters and buffers of a module. |
| [`TensorDictBase.from_modules`](generated/tensordict.TensorDictBase.from_modules.html#tensordict.TensorDictBase.from_modules)(*modules[, ...]) | Retrieves the parameters of several modules for ensebmle learning/feature of expects applications through vmap. |
| [`TensorDictBase.to_module`](generated/tensordict.TensorDictBase.to_module.html#tensordict.TensorDictBase.to_module)(module, *[, ...]) | Writes the content of the tensordict into the parameters and buffers of a module, recursively. |
| [`TensorDictBase.backward`](generated/tensordict.TensorDictBase.backward.html#tensordict.TensorDictBase.backward)([gradient, ...]) | Computes the gradient of the differentiable leaves of the tensordict w.r.t. |
| [`TensorDictBase.requires_grad`](generated/tensordict.TensorDictBase.requires_grad.html#tensordict.TensorDictBase.requires_grad) | |
| [`TensorDictBase.requires_grad_`](generated/tensordict.TensorDictBase.requires_grad_.html#tensordict.TensorDictBase.requires_grad_)([requires_grad]) | Change if autograd should record operations on this tensor: sets this tensor's requires_grad attribute in-place. |
| [`TensorDictBase.detach`](generated/tensordict.TensorDictBase.detach.html#tensordict.TensorDictBase.detach)() | Detach the tensors in the tensordict. |
| [`TensorDictBase.detach_`](generated/tensordict.TensorDictBase.detach_.html#tensordict.TensorDictBase.detach_)() | Detach the tensors in the tensordict in-place. |
| [`TensorDictBase.grad`](generated/tensordict.TensorDictBase.grad.html#tensordict.TensorDictBase.grad) | Returns a tensordict containing the .grad attributes of the leaf tensors. |
| [`TensorDictBase.data`](generated/tensordict.TensorDictBase.data.html#tensordict.TensorDictBase.data) | Returns a tensordict containing the .data attributes of the leaf tensors. |
| [`TensorDictBase.zero_grad`](generated/tensordict.TensorDictBase.zero_grad.html#tensordict.TensorDictBase.zero_grad)([set_to_none]) | Zeros all the gradients of the TensorDict recursively. |

Functions

| [`from_module`](generated/tensordict.from_module.html#tensordict.from_module)(module[, as_module, lock, ...]) | Puts the parameters and buffers of a module in a tensordict. |
| --- | --- |
| [`from_modules`](generated/tensordict.from_modules.html#tensordict.from_modules)(*modules[, as_module, lock, ...]) | Stacks the parameters and buffers of several modules in one tensordict. |
| [`default_is_leaf`](generated/tensordict.default_is_leaf.html#tensordict.default_is_leaf)(cls) | Returns `True` if a type is not a tensor collection (tensordict or tensorclass), or is a pass-through type. |
| [`is_leaf_nontensor`](generated/tensordict.is_leaf_nontensor.html#tensordict.is_leaf_nontensor)(cls) | Returns `True` if a type is not a tensor collection (tensordict or tensorclass) or is a non-tensor. |

### Saving, loading, and file formats

Save a tensordict to disk and load it again, and write its entries to
memory-mapped files. Other methods read and write key-value stores, and HDF5,
zarr, CSV, JSON, Parquet, and MDS files.
`memmap(return_early=True)` returns a
[`TensorDictFuture`](generated/tensordict.utils.TensorDictFuture.html#tensordict.utils.TensorDictFuture). See [Save and load tensordicts](../saving.html#saving) and
[Large datasets and storage backends](../storage.html#storage).

Methods

| [`TensorDictBase.save`](generated/tensordict.TensorDictBase.save.html#tensordict.TensorDictBase.save)([prefix, copy_existing, ...]) | Saves the tensordict to disk, and returns a memory-mapped copy of it. |
| --- | --- |
| [`TensorDictBase.load`](generated/tensordict.TensorDictBase.load.html#tensordict.TensorDictBase.load)(prefix, *args, **kwargs) | Loads a tensordict that [`save()`](generated/tensordict.TensorDictBase.save.html#tensordict.TensorDictBase.save) or [`memmap()`](generated/tensordict.TensorDictBase.memmap.html#tensordict.TensorDictBase.memmap) wrote to disk. |
| [`TensorDictBase.dumps`](generated/tensordict.TensorDictBase.dumps.html#tensordict.TensorDictBase.dumps)([prefix, ...]) | Saves the tensordict to disk, and returns a memory-mapped copy of it. |
| [`TensorDictBase.memmap`](generated/tensordict.TensorDictBase.memmap.html#tensordict.TensorDictBase.memmap)([prefix, ...]) | Writes all tensors onto a corresponding memory-mapped Tensor in a new tensordict. |
| [`TensorDictBase.memmap_`](generated/tensordict.TensorDictBase.memmap_.html#tensordict.TensorDictBase.memmap_)([prefix, ...]) | Writes all tensors onto a corresponding memory-mapped Tensor, in-place. |
| [`TensorDictBase.memmap_like`](generated/tensordict.TensorDictBase.memmap_like.html#tensordict.TensorDictBase.memmap_like)([prefix, ...]) | Creates a memory-mapped tensordict with the same keys, shapes and dtypes, without copying the values. |
| [`TensorDictBase.memmap_refresh_`](generated/tensordict.TensorDictBase.memmap_refresh_.html#tensordict.TensorDictBase.memmap_refresh_)(*[, allow_pickle]) | Refreshes the content of the memory-mapped tensordict if it has a [`saved_path`](generated/tensordict.TensorDictBase.saved_path.html#tensordict.TensorDictBase.saved_path). |
| [`TensorDictBase.load_memmap`](generated/tensordict.TensorDictBase.load_memmap.html#tensordict.TensorDictBase.load_memmap)(prefix[, device, ...]) | Loads a memory-mapped tensordict from disk. |
| [`TensorDictBase.load_memmap_`](generated/tensordict.TensorDictBase.load_memmap_.html#tensordict.TensorDictBase.load_memmap_)(prefix[, ...]) | Loads the content of a memory-mapped tensordict within the tensordict where `load_memmap_` is called. |
| [`TensorDictBase.make_memmap`](generated/tensordict.TensorDictBase.make_memmap.html#tensordict.TensorDictBase.make_memmap)(key, shape, *[, ...]) | Creates an empty memory-mapped tensor given a shape and possibly a dtype. |
| [`TensorDictBase.make_memmap_from_storage`](generated/tensordict.TensorDictBase.make_memmap_from_storage.html#tensordict.TensorDictBase.make_memmap_from_storage)(key, ...) | Creates an empty memory-mapped tensor given a storage, a shape and possibly a dtype. |
| [`TensorDictBase.make_memmap_from_tensor`](generated/tensordict.TensorDictBase.make_memmap_from_tensor.html#tensordict.TensorDictBase.make_memmap_from_tensor)(key, ...) | Creates an empty memory-mapped tensor given a tensor. |
| [`TensorDictBase.is_memmap`](generated/tensordict.TensorDictBase.is_memmap.html#tensordict.TensorDictBase.is_memmap)() | Checks if tensordict is memory-mapped. |
| [`TensorDictBase.saved_path`](generated/tensordict.TensorDictBase.saved_path.html#tensordict.TensorDictBase.saved_path) | Returns the path where a memmap saved TensorDict is being stored. |
| [`TensorDictBase.state_dict`](generated/tensordict.TensorDictBase.state_dict.html#tensordict.TensorDictBase.state_dict)([destination, ...]) | Produces a state_dict from the tensordict. |
| [`TensorDictBase.load_state_dict`](generated/tensordict.TensorDictBase.load_state_dict.html#tensordict.TensorDictBase.load_state_dict)(state_dict[, ...]) | Loads a state-dict, formatted as in [`state_dict()`](generated/tensordict.TensorClass.html#tensordict.TensorClass.state_dict), into the tensordict. |
| [`TensorDictBase.from_h5`](generated/tensordict.TensorDictBase.from_h5.html#tensordict.TensorDictBase.from_h5)(filename, *[, mode, ...]) | Creates a PersistentTensorDict from a h5 file. |
| [`TensorDictBase.to_h5`](generated/tensordict.TensorDictBase.to_h5.html#tensordict.TensorDictBase.to_h5)(filename, **kwargs) | Converts a tensordict to a PersistentTensorDict with the h5 backend. |
| [`TensorDictBase.from_zarr`](generated/tensordict.TensorDictBase.from_zarr.html#tensordict.TensorDictBase.from_zarr)(filename, *[, ...]) | Creates a PersistentTensorDict from a zarr store. |
| [`TensorDictBase.to_zarr`](generated/tensordict.TensorDictBase.to_zarr.html#tensordict.TensorDictBase.to_zarr)(filename, **kwargs) | Converts a tensordict to a PersistentTensorDict with the zarr backend. |
| [`TensorDictBase.from_csv`](generated/tensordict.TensorDictBase.from_csv.html#tensordict.TensorDictBase.from_csv)(path, *[, ...]) | Creates a TensorDict from a CSV file. |
| [`TensorDictBase.to_csv`](generated/tensordict.TensorDictBase.to_csv.html#tensordict.TensorDictBase.to_csv)(path, *[, separator]) | Writes this TensorDict to a CSV file. |
| [`TensorDictBase.from_json`](generated/tensordict.TensorDictBase.from_json.html#tensordict.TensorDictBase.from_json)(path, *[, ...]) | Creates a TensorDict from a JSON file. |
| [`TensorDictBase.to_json`](generated/tensordict.TensorDictBase.to_json.html#tensordict.TensorDictBase.to_json)(path, *[, separator, ...]) | Writes this TensorDict to a JSON file. |
| [`TensorDictBase.from_parquet`](generated/tensordict.TensorDictBase.from_parquet.html#tensordict.TensorDictBase.from_parquet)(path, *[, ...]) | Creates a TensorDict from a Parquet file. |
| [`TensorDictBase.to_parquet`](generated/tensordict.TensorDictBase.to_parquet.html#tensordict.TensorDictBase.to_parquet)(path, *[, separator]) | Writes this TensorDict to a Parquet file. |
| [`TensorDictBase.to_mds`](generated/tensordict.TensorDictBase.to_mds.html#tensordict.TensorDictBase.to_mds)(*[, columns, ...]) | Writes the content of a TensorCollection to a streaming dataset. |
| [`TensorDictBase.to_store`](generated/tensordict.TensorDictBase.to_store.html#tensordict.TensorDictBase.to_store)(*[, backend, host, ...]) | Upload this TensorDict to a key-value store (Redis, Dragonfly, etc.). |

Functions

| [`save`](generated/tensordict.save.html#tensordict.save)(data[, prefix, copy_existing, ...]) | Saves a tensordict to disk, as memory-mapped files. |
| --- | --- |
| [`load`](generated/tensordict.load.html#tensordict.load)(prefix[, device, non_blocking, out, ...]) | Loads a tensordict that [`save()`](generated/tensordict.save.html#tensordict.save) wrote. |
| [`memmap`](generated/tensordict.memmap.html#tensordict.memmap)(data[, prefix, copy_existing, ...]) | Writes all tensors onto memory-mapped tensors in a new tensordict. |
| [`load_memmap`](generated/tensordict.load_memmap.html#tensordict.load_memmap)(prefix[, device, non_blocking, ...]) | Loads a memory-mapped tensordict from disk. |
| [`pack_memmap`](generated/tensordict.pack_memmap.html#tensordict.pack_memmap)(prefix, archive_path, *[, ...]) | Packs a memory-mapped tensordict directory into a single-file zip archive. |
| [`unpack_memmap`](generated/tensordict.unpack_memmap.html#tensordict.unpack_memmap)(archive_path, prefix) | Extracts a memmap tensordict archive into a memmap directory. |
| [`is_memmap_archive`](generated/tensordict.is_memmap_archive.html#tensordict.is_memmap_archive)(path) | Checks whether `path` points to a memmap tensordict archive. |
| [`refresh_archive_checksums`](generated/tensordict.refresh_archive_checksums.html#tensordict.refresh_archive_checksums)(archive_path) | Recomputes the CRC-32 checksums of a memmap archive after in-place writes. |
| [`from_h5`](generated/tensordict.from_h5.html#tensordict.from_h5)(h5_file, *[, mode, auto_batch_size, ...]) | Converts an HDF5 file to a TensorDict. |
| [`from_zarr`](generated/tensordict.from_zarr.html#tensordict.from_zarr)(zarr_file, *[, mode, ...]) | Converts a zarr store to a TensorDict. |
| [`from_csv`](generated/tensordict.from_csv.html#tensordict.from_csv)(path, *[, auto_batch_size, ...]) | Creates a TensorDict from a CSV file. |
| [`from_json`](generated/tensordict.from_json.html#tensordict.from_json)(path, *[, auto_batch_size, ...]) | Creates a TensorDict from a JSON file. |
| [`from_parquet`](generated/tensordict.from_parquet.html#tensordict.from_parquet)(path, *[, auto_batch_size, ...]) | Creates a TensorDict from a Parquet file. |

### Locking

A locked tensordict raises an error if you add, replace, rename, or delete
an entry. In-place writes with `set_` and `update_` still work, also for
non-tensor entries. `unlock_` also works in a `with` block: the
tensordict is locked again at the end of the block. See [Keys, nested keys, and locking](../concepts/keys.html#keys).

| [`TensorDictBase.lock_`](generated/tensordict.TensorDictBase.lock_.html#tensordict.TensorDictBase.lock_)() | Locks the tensordict and its sub-tensordicts, and returns it. |
| --- | --- |
| [`TensorDictBase.unlock_`](generated/tensordict.TensorDictBase.unlock_.html#tensordict.TensorDictBase.unlock_)() | Unlocks the tensordict and its sub-tensordicts, and returns it. |
| [`TensorDictBase.is_locked`](generated/tensordict.TensorDictBase.is_locked.html#tensordict.TensorDictBase.is_locked) | `True` if the tensordict is locked. |

### Distributed and remote

Send and receive tensordicts between processes with
[`torch.distributed`](https://docs.pytorch.org/docs/stable/distributed.html#module-torch.distributed), run collective operations on them, and stream
them over UCXX (see [Key-value stores and UCXX](storage.html#api-storage)). See [Distributed tensordicts](../distributed.html#distributed).

| [`TensorDictBase.send`](generated/tensordict.TensorDictBase.send.html#tensordict.TensorDictBase.send)([dst, group, group_dst, ...]) | Sends the content of a tensordict to a distant worker. |
| --- | --- |
| [`TensorDictBase.recv`](generated/tensordict.TensorDictBase.recv.html#tensordict.TensorDictBase.recv)([src, group, group_src, ...]) | Receives the content of a tensordict and updates content with it. |
| [`TensorDictBase.isend`](generated/tensordict.TensorDictBase.isend.html#tensordict.TensorDictBase.isend)([dst, group, ...]) | Sends the content of the tensordict asynchronously. |
| [`TensorDictBase.irecv`](generated/tensordict.TensorDictBase.irecv.html#tensordict.TensorDictBase.irecv)([src, group, ...]) | Receives the content of a tensordict and updates content with it asynchronously. |
| [`TensorDictBase.all_reduce`](generated/tensordict.TensorDictBase.all_reduce.html#tensordict.TensorDictBase.all_reduce)([op, group, async_op]) | All-reduces the tensordict across all ranks in-place. |
| [`TensorDictBase.reduce`](generated/tensordict.TensorDictBase.reduce.html#tensordict.TensorDictBase.reduce)(dst[, op, async_op, ...]) | Reduces the tensordict across all machines. |
| [`TensorDictBase.broadcast`](generated/tensordict.TensorDictBase.broadcast.html#tensordict.TensorDictBase.broadcast)(src, *[, group, ...]) | Broadcasts a tensordict from `src` to all ranks. |
| [`TensorDictBase.scatter`](generated/tensordict.TensorDictBase.scatter.html#tensordict.TensorDictBase.scatter)(src[, tensordicts, ...]) | Scatters a list of tensordicts from `src` to all ranks. |
| [`TensorDictBase.all_gather`](generated/tensordict.TensorDictBase.all_gather.html#tensordict.TensorDictBase.all_gather)(*[, group]) | All-gathers tensordicts from every rank. |
| [`TensorDictBase.gather_and_stack`](generated/tensordict.TensorDictBase.gather_and_stack.html#tensordict.TensorDictBase.gather_and_stack)(dst[, group]) | Gathers tensordicts from various workers and stacks them onto self in the destination worker. |
| [`TensorDictBase.init_remote`](generated/tensordict.TensorDictBase.init_remote.html#tensordict.TensorDictBase.init_remote)([dst, group, ...]) | Initializes a remote tensordict by sending its metadata and content. |
| [`TensorDictBase.from_remote_init`](generated/tensordict.TensorDictBase.from_remote_init.html#tensordict.TensorDictBase.from_remote_init)(src[, ...]) | Creates a new tensordict instance initialized from remotely sent metadata. |
| [`TensorDictBase.asend`](generated/tensordict.TensorDictBase.asend.html#tensordict.TensorDictBase.asend)(dst) | Sends the content of a tensordict through a UCXX pipe (async). |
| [`TensorDictBase.arecv`](generated/tensordict.TensorDictBase.arecv.html#tensordict.TensorDictBase.arecv)(src, *[, device]) | Receives content into this tensordict from a UCXX pipe (async). |

### Deprecated methods

These methods still work, and each call raises a `DeprecationWarning`.
Their pages name the replacement and the release that removes them.

| [`TensorDictBase.load_`](generated/tensordict.TensorDictBase.load_.html#tensordict.TensorDictBase.load_)(prefix, *args, **kwargs) | Loads a tensordict from disk within the current tensordict. |
| --- | --- |
| [`TensorDictBase.pin_memory_`](generated/tensordict.TensorDictBase.pin_memory_.html#tensordict.TensorDictBase.pin_memory_)([num_threads]) | Calls [`pin_memory()`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.pin_memory.html#torch.Tensor.pin_memory) on the stored tensors and returns the TensorDict modifies in-place. |

## Other functions

### Type checks and testing

Test whether an object is a tensordict, a tensorclass, or non-tensor data,
and compare two tensordicts in tests.

| [`is_tensor_collection`](generated/tensordict.is_tensor_collection.html#tensordict.is_tensor_collection)(datatype) | Checks if a data object or a type is a tensor container from the tensordict lib. |
| --- | --- |
| [`is_tensorclass`](generated/tensordict.is_tensorclass.html#tensordict.is_tensorclass)(obj) | Returns True if obj is either a tensorclass or an instance of a tensorclass. |
| [`is_non_tensor`](generated/tensordict.is_non_tensor.html#tensordict.is_non_tensor)(data) | Checks if an item is a non-tensor. |
| [`assert_close`](generated/tensordict.assert_close.html#tensordict.assert_close)(actual, expected[, rtol, atol, ...]) | Asserts that two tensordicts, actual and expected, are element-wise equal within a tolerance for all entries. |
| [`assert_allclose_td`](generated/tensordict.assert_allclose_td.html#tensordict.assert_allclose_td)(actual, expected[, rtol, ...]) | Asserts that two tensordicts, actual and expected, are element-wise equal within a tolerance for all entries. |

### Printing

Choose what the representation of a tensordict shows, and rebuild a
tensordict from its printed form. See [Printing and display](../printing.html#printing).

| [`set_printoptions`](generated/tensordict.set_printoptions.html#tensordict.set_printoptions)(*[, verbose]) | Controls which attributes appear in TensorDict's `__repr__` output. |
| --- | --- |
| [`get_printoptions`](generated/tensordict.get_printoptions.html#tensordict.get_printoptions)() | Returns the current TensorDict print options as a dict. |
| [`parse_tensor_dict_string`](generated/tensordict.parse_tensor_dict_string.html#tensordict.parse_tensor_dict_string)(s) | Parse a TensorDict repr to a TensorDict. |

### Global settings

These functions read or change a behavior of tensordict, for the whole
process or inside a `with` block. `set_list_to_stack(False)` and
`set_capture_non_tensor_stack(True)` are deprecated, and their pages say
what to use instead.

| [`list_to_stack`](generated/tensordict.list_to_stack.html#tensordict.list_to_stack)([allow_none]) | Returns whether assigning a list to a key stores a [`NonTensorStack`](generated/tensordict.NonTensorStack.html#tensordict.NonTensorStack). |
| --- | --- |
| [`set_list_to_stack`](generated/tensordict.set_list_to_stack.html#tensordict.set_list_to_stack)(mode) | Context manager and decorator to control list handling in TensorDict. |
| [`capture_non_tensor_stack`](generated/tensordict.capture_non_tensor_stack.html#tensordict.capture_non_tensor_stack)([allow_none]) | Returns whether stacking identical non-tensor data gives one value instead of a stack. |
| [`set_capture_non_tensor_stack`](generated/tensordict.set_capture_non_tensor_stack.html#tensordict.set_capture_non_tensor_stack)(mode) | Controls whether identical non-tensor data should be captured when stacked. |

### Deprecated functions

These functions still work, and each call raises a `DeprecationWarning`.
Their pages name the replacement and the release that removes them.

| [`get_defaults_to_none`](generated/tensordict.get_defaults_to_none.html#tensordict.get_defaults_to_none)([set_to_none]) | Returns the status of get default value. |
| --- | --- |
| [`set_get_defaults_to_none`](generated/tensordict.set_get_defaults_to_none.html#tensordict.set_get_defaults_to_none)([set_to_none]) | Sets the default of get to None and silences deprecation warnings during calls to get that result in a KeyError. |
| [`lazy_legacy`](generated/tensordict.lazy_legacy.html#tensordict.lazy_legacy)([allow_none]) | Returns True if lazy representations will be used for selected methods. |
| [`set_lazy_legacy`](generated/tensordict.set_lazy_legacy.html#tensordict.set_lazy_legacy)(mode) | Sets the behaviour of some methods to a lazy transform. |

## Package constants

tensordict.base.NO_DEFAULT

The sentinel that marks an argument as not given where `None` is a valid
value, such as `default` in [`get_non_tensor()`](generated/tensordict.TensorDictBase.get_non_tensor.html#tensordict.TensorDictBase.get_non_tensor)
or `device` in [`new_zeros()`](generated/tensordict.TensorDictBase.new_zeros.html#tensordict.TensorDictBase.new_zeros). It can also
be imported from `tensordict.tensordict`.

tensordict.__version__

The version string of the installed tensordict package, or `None` if it
cannot be determined, for example when tensordict is imported from a source
tree that is not installed.

## See also

- [What is TensorDict?](../overview.html#overview), [Batch size, shapes, and indexing](../concepts/batch_size.html#batch-size), [Keys, nested keys, and locking](../concepts/keys.html#keys), [Stacking and variable-size data](../concepts/stacking.html#stacking),
[Arithmetic, apply, reductions, and gradients](../concepts/operations.html#operations), and [Devices, dtypes, and memory](../concepts/devices.html#devices): the user guide pages that explain
these methods.
- [Save and load tensordicts](../saving.html#saving), [Large datasets and storage backends](../storage.html#storage), [Load data and share it between processes](../data_loading.html#data-loading), and
[Distributed tensordicts](../distributed.html#distributed): the how-to guides.
- [Utilities](utils.html#api-utils): the helpers of `tensordict.utils`.