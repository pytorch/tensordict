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

[`TensorCollection`](generated/tensordict.TensorCollection.html#tensordict.TensorCollection) is the common base class of the tensordicts and
the subclasses of [`TensorClass`](generated/tensordict.TensorClass.html#tensordict.TensorClass): the type hint for "a tensordict or
a tensorclass".

Other containers have their own pages: [`TensorClass`](generated/tensordict.TensorClass.html#tensordict.TensorClass), the class
form with typed fields and methods, and non-tensor data
([tensorclass and non-tensor data](tc.html#api-tensorclass)), [`TypedTensorDict`](generated/tensordict.TypedTensorDict.html#tensordict.TypedTensorDict)
([TypedTensorDict](ttd.html#api-typed-tensordict)), and the key-value stores
([Key-value stores and UCXX](storage.html#api-storage)). To choose one, see [Which container should I use?](../choosing_a_container.html#choosing-a-container).

| [`TensorDictBase`](generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)(*args, **kwargs) | The abstract base class of every tensordict. |
| --- | --- |
| [`TensorDict`](generated/tensordict.TensorDict.html#tensordict.TensorDict)([source, batch_size, device, ...]) | A dictionary of tensors that share a batch size. |
| [`LazyStackedTensorDict`](generated/tensordict.LazyStackedTensorDict.html#tensordict.LazyStackedTensorDict)(*tensordicts[, ...]) | A stack of tensordicts that keeps each input as a separate tensordict. |
| [`PersistentTensorDict`](generated/tensordict.PersistentTensorDict.html#tensordict.PersistentTensorDict)(*[, batch_size, ...]) | A tensordict whose entries live in an HDF5 file or a Zarr store. |
| [`TensorDictParams`](generated/tensordict.TensorDictParams.html#tensordict.TensorDictParams)([parameters, no_convert, lock]) | A TensorDict whose entries are registered as the parameters and buffers of an [`Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module). |

| [`MemoryMappedTensor`](generated/tensordict.MemoryMappedTensor.html#tensordict.MemoryMappedTensor)(source, *[, dtype, ...]) | A CPU tensor whose storage is a file on disk, or anonymous shared memory when no filename is given. |
| --- | --- |
| [`TensorCollection`](generated/tensordict.TensorCollection.html#tensordict.TensorCollection)(*args, **kwargs) | The abstract base class of tensordicts and tensorclasses. |

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

| [`TensorDictBase.from_any`](generated/tensordict.TensorDictBase.from_any.html#tensordict.TensorDictBase.from_any)(obj, *[, ...]) | Converts a dict, tuple, dataclass, array or file to a tensordict, recursively. |
| --- | --- |
| [`TensorDictBase.from_dict`](generated/tensordict.TensorDictBase.from_dict.html#tensordict.TensorDictBase.from_dict)(input_dict, *[, ...]) | Returns a tensordict built from a dictionary, which can be nested. |
| [`TensorDictBase.from_dict_instance`](generated/tensordict.TensorDictBase.from_dict_instance.html#tensordict.TensorDictBase.from_dict_instance)(input_dict, *) | Builds a tensordict from a dictionary, with the types of the entries of this tensordict. |
| [`TensorDictBase.from_namedtuple`](generated/tensordict.TensorDictBase.from_namedtuple.html#tensordict.TensorDictBase.from_namedtuple)(named_tuple, *) | Converts a named tuple to a tensordict whose keys are its field names. |
| [`TensorDictBase.from_pandas`](generated/tensordict.TensorDictBase.from_pandas.html#tensordict.TensorDictBase.from_pandas)(dataframe, *[, ...]) | Converts a pandas DataFrame to a tensordict, with one entry per column. |
| [`TensorDictBase.from_struct_array`](generated/tensordict.TensorDictBase.from_struct_array.html#tensordict.TensorDictBase.from_struct_array)(struct_array, *) | Converts a NumPy structured array to a tensordict, with one entry per field. |
| [`TensorDictBase.from_tuple`](generated/tensordict.TensorDictBase.from_tuple.html#tensordict.TensorDictBase.from_tuple)(obj, *[, ...]) | Converts a tuple to a tensordict with the keys `"0"`, `"1"`, ... |
| [`TensorDictBase.from_pytree`](generated/tensordict.TensorDictBase.from_pytree.html#tensordict.TensorDictBase.from_pytree)(pytree, *[, ...]) | Converts a pytree to a tensordict that [`to_pytree()`](generated/tensordict.TensorDictBase.to_pytree.html#tensordict.TensorDictBase.to_pytree) converts back. |
| [`TensorDictBase.from_dataclass`](generated/tensordict.TensorDictBase.from_dataclass.html#tensordict.TensorDictBase.from_dataclass)(dataclass, *) | Converts a dataclass instance to a tensordict, with one entry per field. |
| [`TensorDictBase.from_list`](generated/tensordict.TensorDictBase.from_list.html#tensordict.TensorDictBase.from_list)(input, *[, ...]) | Builds a tensordict by stacking a list of dicts or tensor collections. |
| [`TensorDictBase.from_schema`](generated/tensordict.TensorDictBase.from_schema.html#tensordict.TensorDictBase.from_schema)(schema, *[, ...]) | Allocates a zero-filled tensordict from a schema of shapes and dtypes. |
| [`TensorDictBase.fromkeys`](generated/tensordict.TensorDictBase.fromkeys.html#tensordict.TensorDictBase.fromkeys)(keys[, value]) | Creates a tensordict with the same value for every key, like [`dict.fromkeys()`](https://docs.python.org/3/builtins/stdtypes.html#dict.fromkeys). |
| [`TensorDictBase.empty`](generated/tensordict.TensorDictBase.empty.html#tensordict.TensorDictBase.empty)([recurse, batch_size, ...]) | Returns a new tensordict of the same type, with no tensor leaves. |
| [`TensorDictBase.new_empty`](generated/tensordict.TensorDictBase.new_empty.html#tensordict.TensorDictBase.new_empty)(*size[, dtype, ...]) | Returns a tensordict of batch size `size` with uninitialized leaves. |
| [`TensorDictBase.new_full`](generated/tensordict.TensorDictBase.new_full.html#tensordict.TensorDictBase.new_full)(size, fill_value, *) | Returns a tensordict of batch size `size` with every leaf filled with `fill_value`. |
| [`TensorDictBase.new_ones`](generated/tensordict.TensorDictBase.new_ones.html#tensordict.TensorDictBase.new_ones)(*size[, dtype, ...]) | Returns a tensordict of batch size `size` with every leaf filled with 1. |
| [`TensorDictBase.new_tensor`](generated/tensordict.TensorDictBase.new_tensor.html#tensordict.TensorDictBase.new_tensor)(data, *[, dtype, ...]) | Returns a tensordict whose leaves are copies of `data`, each with the dtype of its leaf. |
| [`TensorDictBase.new_zeros`](generated/tensordict.TensorDictBase.new_zeros.html#tensordict.TensorDictBase.new_zeros)(*size[, dtype, ...]) | Returns a tensordict of batch size `size` with every leaf filled with 0. |
| [`TensorDictBase.copy`](generated/tensordict.TensorDictBase.copy.html#tensordict.TensorDictBase.copy)() | Returns a shallow copy of the tensordict: a new structure that holds the same leaves. |
| [`TensorDictBase.clone`](generated/tensordict.TensorDictBase.clone.html#tensordict.TensorDictBase.clone)([recurse]) | Returns a copy of the tensordict, of the same type. |
| [`TensorDictBase.to_dict`](generated/tensordict.TensorDictBase.to_dict.html#tensordict.TensorDictBase.to_dict)(*[, retain_none, ...]) | Returns a nested dictionary with the entries of the tensordict. |
| [`TensorDictBase.to_namedtuple`](generated/tensordict.TensorDictBase.to_namedtuple.html#tensordict.TensorDictBase.to_namedtuple)([dest_cls]) | Converts the tensordict to a named tuple, recursively. |
| [`TensorDictBase.to_pandas`](generated/tensordict.TensorDictBase.to_pandas.html#tensordict.TensorDictBase.to_pandas)(*[, separator]) | Converts the tensordict to a pandas DataFrame, with one column per leaf. |
| [`TensorDictBase.to_pytree`](generated/tensordict.TensorDictBase.to_pytree.html#tensordict.TensorDictBase.to_pytree)() | Rebuilds the pytree that [`from_pytree()`](generated/tensordict.TensorDictBase.from_pytree.html#tensordict.TensorDictBase.from_pytree) converted to this tensordict. |
| [`TensorDictBase.to_struct_array`](generated/tensordict.TensorDictBase.to_struct_array.html#tensordict.TensorDictBase.to_struct_array)() | Converts the tensordict to a NumPy structured array, with one field per entry. |
| [`TensorDictBase.to_tensordict`](generated/tensordict.TensorDictBase.to_tensordict.html#tensordict.TensorDictBase.to_tensordict)(*[, retain_none]) | Returns a [`TensorDict`](generated/tensordict.TensorDict.html#tensordict.TensorDict) copy of this tensordict. |
| [`TensorDictBase.tolist`](generated/tensordict.TensorDictBase.tolist.html#tensordict.TensorDictBase.tolist)(*[, convert_nodes, ...]) | Returns the elements of the tensordict as nested lists, one level per batch dimension. |
| [`TensorDictBase.numpy`](generated/tensordict.TensorDictBase.numpy.html#tensordict.TensorDictBase.numpy)() | Returns a nested dict of NumPy arrays with the entries of the tensordict. |
| [`TensorDictBase.as_tensor`](generated/tensordict.TensorDictBase.as_tensor.html#tensordict.TensorDictBase.as_tensor)() | Returns a tensordict whose leaves are converted with their `as_tensor()` method, if they have one. |

Functions

| [`from_any`](generated/tensordict.from_any.html#tensordict.from_any)(obj, *[, auto_batch_size, ...]) | Converts a dict, tuple, dataclass, array or file to a tensordict, recursively. |
| --- | --- |
| [`from_dict`](generated/tensordict.from_dict.html#tensordict.from_dict)(d, *[, auto_batch_size, ...]) | Converts a dictionary to a tensordict, recursively. |
| [`from_namedtuple`](generated/tensordict.from_namedtuple.html#tensordict.from_namedtuple)(named_tuple, *[, ...]) | Converts a named tuple to a tensordict whose keys are its field names. |
| [`from_pandas`](generated/tensordict.from_pandas.html#tensordict.from_pandas)(dataframe, *[, auto_batch_size, ...]) | Converts a pandas DataFrame to a tensordict, with one entry per column. |
| [`from_struct_array`](generated/tensordict.from_struct_array.html#tensordict.from_struct_array)(struct_array, *[, ...]) | Converts a NumPy structured array to a tensordict, with one entry per field. |
| [`from_tuple`](generated/tensordict.from_tuple.html#tensordict.from_tuple)(obj, *[, auto_batch_size, ...]) | Converts a tuple to a tensordict with the keys `"0"`, `"1"`, ... |
| [`from_pytree`](generated/tensordict.from_pytree.html#tensordict.from_pytree)(pytree, *[, batch_size, ...]) | Converts a pytree (nested lists, tuples and dicts of tensors) to a tensordict. |
| [`fromkeys`](generated/tensordict.fromkeys.html#tensordict.fromkeys)(keys[, value]) | Creates a tensordict with the same value for every key, like [`dict.fromkeys()`](https://docs.python.org/3/builtins/stdtypes.html#dict.fromkeys). |
| [`make_tensordict`](generated/tensordict.make_tensordict.html#tensordict.make_tensordict)([input_dict, batch_size, ...]) | Returns a tensordict built from a dict or from keyword arguments. |

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
| [`TensorDictBase.sorted_keys`](generated/tensordict.TensorDictBase.sorted_keys.html#tensordict.TensorDictBase.sorted_keys) | The top-level keys of the tensordict, as a list sorted in alphabetical order. |
| [`TensorDictBase.get`](generated/tensordict.TensorDictBase.get.html#tensordict.TensorDictBase.get)() | Returns the value stored at `key`, or `default` if the key is missing. |
| [`TensorDictBase.get_at`](generated/tensordict.TensorDictBase.get_at.html#tensordict.TensorDictBase.get_at)() | Returns the entry at `key`, indexed with `index`. |
| [`TensorDictBase.set`](generated/tensordict.TensorDictBase.set.html#tensordict.TensorDictBase.set)(key, item[, inplace, ...]) | Sets an entry, and returns the tensordict. |
| [`TensorDictBase.set_`](generated/tensordict.TensorDictBase.set_.html#tensordict.TensorDictBase.set_)(key, item, *[, non_blocking]) | Copies a value into an existing entry, which keeps its storage, and returns `self`. |
| [`TensorDictBase.set_at_`](generated/tensordict.TensorDictBase.set_at_.html#tensordict.TensorDictBase.set_at_)(key, value, index, *) | Writes `value` into the existing entry `key` at the index `index`, in place. |
| [`TensorDictBase.setdefault`](generated/tensordict.TensorDictBase.setdefault.html#tensordict.TensorDictBase.setdefault)(key, default[, ...]) | Returns the entry at `key`, after writing `default` there if the key is missing. |
| [`TensorDictBase.update`](generated/tensordict.TensorDictBase.update.html#tensordict.TensorDictBase.update)([input_dict_or_td, ...]) | Writes the entries of a dictionary or another tensordict into the tensordict, and returns `self`. |
| [`TensorDictBase.update_`](generated/tensordict.TensorDictBase.update_.html#tensordict.TensorDictBase.update_)([input_dict_or_td, ...]) | Copies the entries of a dictionary or tensordict into the existing entries, and returns `self`. |
| [`TensorDictBase.update_at_`](generated/tensordict.TensorDictBase.update_at_.html#tensordict.TensorDictBase.update_at_)(input_dict_or_td, idx) | Copies the entries of a dictionary or tensordict into the existing entries at `idx`. |
| [`TensorDictBase.copy_`](generated/tensordict.TensorDictBase.copy_.html#tensordict.TensorDictBase.copy_)(tensordict[, non_blocking]) | Copies the entries of `tensordict` into the existing entries, and returns `self`. |
| [`TensorDictBase.copy_at_`](generated/tensordict.TensorDictBase.copy_at_.html#tensordict.TensorDictBase.copy_at_)(tensordict, idx[, ...]) | Copies the tensors of `tensordict` into the existing entries at `idx`, and returns `self`. |
| [`TensorDictBase.replace`](generated/tensordict.TensorDictBase.replace.html#tensordict.TensorDictBase.replace)(*args, **kwargs) | Returns a shallow copy of the tensordict with some entries replaced. |
| [`TensorDictBase.pop`](generated/tensordict.TensorDictBase.pop.html#tensordict.TensorDictBase.pop)(key[, default]) | Removes an entry from the tensordict and returns its value. |
| [`TensorDictBase.popitem`](generated/tensordict.TensorDictBase.popitem.html#tensordict.TensorDictBase.popitem)() | Removes the last top-level entry and returns it as a `(key, value)` pair. |
| [`TensorDictBase.del_`](generated/tensordict.TensorDictBase.del_.html#tensordict.TensorDictBase.del_)(key) | Deletes an entry of the tensordict in place and returns the tensordict. |
| [`TensorDictBase.clear`](generated/tensordict.TensorDictBase.clear.html#tensordict.TensorDictBase.clear)() | Removes every entry of the tensordict and returns `self`. |
| [`TensorDictBase.select`](generated/tensordict.TensorDictBase.select.html#tensordict.TensorDictBase.select)(*keys[, inplace, ...]) | Returns a tensordict that holds only the entries of `keys`. |
| [`TensorDictBase.exclude`](generated/tensordict.TensorDictBase.exclude.html#tensordict.TensorDictBase.exclude)(*keys[, inplace]) | Returns a tensordict without the entries of `keys`. |
| [`TensorDictBase.split_keys`](generated/tensordict.TensorDictBase.split_keys.html#tensordict.TensorDictBase.split_keys)(*key_sets[, ...]) | Splits the tensordict into one tensordict per set of keys, plus one for the remaining entries. |
| [`TensorDictBase.separates`](generated/tensordict.TensorDictBase.separates.html#tensordict.TensorDictBase.separates)(*keys[, default, ...]) | Removes the entries of `keys` from the tensordict in place, and returns them in a new tensordict. |
| [`TensorDictBase.rename_key_`](generated/tensordict.TensorDictBase.rename_key_.html#tensordict.TensorDictBase.rename_key_)(old_key, new_key) | Renames an entry of the tensordict in place and returns the tensordict. |
| [`TensorDictBase.rename_keys`](generated/tensordict.TensorDictBase.rename_keys.html#tensordict.TensorDictBase.rename_keys)(mapping, *[, strict]) | Returns a tensordict with several keys renamed at once. |
| [`TensorDictBase.flatten_keys`](generated/tensordict.TensorDictBase.flatten_keys.html#tensordict.TensorDictBase.flatten_keys)([separator, ...]) | Returns a tensordict with the nested entries moved to the top level, under joined keys. |
| [`TensorDictBase.unflatten_keys`](generated/tensordict.TensorDictBase.unflatten_keys.html#tensordict.TensorDictBase.unflatten_keys)([separator, ...]) | Returns a tensordict with the keys that hold the separator split into nested keys. |
| [`TensorDictBase.create_nested`](generated/tensordict.TensorDictBase.create_nested.html#tensordict.TensorDictBase.create_nested)(key) | Adds an empty nested tensordict at `key`, and returns `self`. |
| [`TensorDictBase.depth`](generated/tensordict.TensorDictBase.depth.html#tensordict.TensorDictBase.depth) | the maximum number of levels of nested tensordicts. |
| [`TensorDictBase.entry_class`](generated/tensordict.TensorDictBase.entry_class.html#tensordict.TensorDictBase.entry_class)(key) | Returns the class of an entry, possibly without reading the entry. |
| [`TensorDictBase.get_item_shape`](generated/tensordict.TensorDictBase.get_item_shape.html#tensordict.TensorDictBase.get_item_shape)(key) | Returns the shape of the entry at `key`, without reading the entry when possible. |
| [`TensorDictBase.is_empty`](generated/tensordict.TensorDictBase.is_empty.html#tensordict.TensorDictBase.is_empty)() | Returns `True` if the tensordict has no leaf. |
| [`TensorDictBase.fill_`](generated/tensordict.TensorDictBase.fill_.html#tensordict.TensorDictBase.fill_)(key, value) | Fills the entry at `key` with `value` in place and returns `self`. |
| [`TensorDictBase.set_non_tensor`](generated/tensordict.TensorDictBase.set_non_tensor.html#tensordict.TensorDictBase.set_non_tensor)(key, value) | Stores `value` as non-tensor data at `key`, and returns `self`. |
| [`TensorDictBase.get_non_tensor`](generated/tensordict.TensorDictBase.get_non_tensor.html#tensordict.TensorDictBase.get_non_tensor)(key[, default]) | Returns the content of the non-tensor entry at `key`. |
| [`TensorDictBase.is_non_tensor`](generated/tensordict.TensorDictBase.is_non_tensor.html#tensordict.TensorDictBase.is_non_tensor)(key) | Returns `True` if the entry at `key` is non-tensor data. |
| [`TensorDictBase.non_tensor_items`](generated/tensordict.TensorDictBase.non_tensor_items.html#tensordict.TensorDictBase.non_tensor_items)([include_nested]) | Returns a tuple of the `(key, value)` pairs of the non-tensor data of the tensordict. |
| [`TensorDictBase.filter_non_tensor_data`](generated/tensordict.TensorDictBase.filter_non_tensor_data.html#tensordict.TensorDictBase.filter_non_tensor_data)() | Returns a tensordict without the non-tensor entries, nested ones included. |
| [`TensorDictBase.filter_empty_`](generated/tensordict.TensorDictBase.filter_empty_.html#tensordict.TensorDictBase.filter_empty_)() | Removes the nested tensordicts that have no leaf, in place, and returns `self`. |
| [`TensorDictBase.cat_tensors`](generated/tensordict.TensorDictBase.cat_tensors.html#tensordict.TensorDictBase.cat_tensors)(*keys, out_key[, ...]) | Concatenates entries of the tensordict into a new entry, and removes them by default. |
| [`TensorDictBase.stack_tensors`](generated/tensordict.TensorDictBase.stack_tensors.html#tensordict.TensorDictBase.stack_tensors)(*keys, out_key) | Stacks entries of the tensordict into a new entry, and removes them by default. |
| [`TensorDictBase.cat_from_tensordict`](generated/tensordict.TensorDictBase.cat_from_tensordict.html#tensordict.TensorDictBase.cat_from_tensordict)([dim, ...]) | Concatenates all the leaves of the tensordict into one tensor, with [`torch.cat()`](https://docs.pytorch.org/docs/stable/generated/torch.cat.html#torch.cat). |
| [`TensorDictBase.pack`](generated/tensordict.TensorDictBase.pack.html#tensordict.TensorDictBase.pack)([dim, layout]) | Packs all the leaves of a tensordict into one tensor of features. |
| [`TensorDictBase.stack_from_tensordict`](generated/tensordict.TensorDictBase.stack_from_tensordict.html#tensordict.TensorDictBase.stack_from_tensordict)([dim, ...]) | Stacks all the leaves of the tensordict into one tensor, with [`torch.stack()`](https://docs.pytorch.org/docs/stable/generated/torch.stack.html#torch.stack). |

Classes and functions

| [`NestedKey`](generated/tensordict.NestedKey.html#tensordict.NestedKey)() | The type of the keys of a tensordict: a string, or a tuple of strings for a nested entry. |
| --- | --- |
| [`PackLayout`](generated/tensordict.PackLayout.html#tensordict.PackLayout)(keys, shapes[, dim, nested, ...]) | The layout of a tensordict packed by [`pack()`](generated/tensordict.TensorDictBase.pack.html#tensordict.TensorDictBase.pack). |

| [`unravel_key`](generated/tensordict.unravel_key.html#tensordict.unravel_key)(key) | Unravels a nested key into a str or a flat tuple of str. |
| --- | --- |
| [`unravel_key_list`](generated/tensordict.unravel_key_list.html#tensordict.unravel_key_list)(keys) | Unravels every key of a list, as [`unravel_key()`](generated/tensordict.unravel_key.html#tensordict.unravel_key) does. |
| [`unravel_key_to_tuple`](generated/tensordict.unravel_key_to_tuple.html#tensordict.unravel_key_to_tuple)(key) | Unravels a nested key into a flat tuple of str. |
| [`merge_tensordicts`](generated/tensordict.merge_tensordicts.html#tensordict.merge_tensordicts)(*tensordicts[, callback_exist]) | Merges the entries of several tensordicts into a new tensordict. |

### Indexing, masking, and splitting

Index every entry along the batch dimensions at once, and split a
tensordict into several tensordicts. `td[0]` and `td[mask]` index the
batch dimensions. `td["key"]` returns an entry: see
Keys, values, and entries. For the rules, see [Batch size, shapes, and indexing](../concepts/batch_size.html#batch-size).

| [`TensorDictBase.__getitem__`](generated/tensordict.TensorDictBase.__getitem__.html#tensordict.TensorDictBase.__getitem__)() | Returns the entry at a key, or the tensordict indexed along its batch dimensions. |
| --- | --- |
| [`TensorDictBase.__iter__`](generated/tensordict.TensorDictBase.__iter__.html#tensordict.TensorDictBase.__iter__)() | Iterates over the first batch dimension of the tensordict. |
| [`TensorDictBase.gather`](generated/tensordict.TensorDictBase.gather.html#tensordict.TensorDictBase.gather)(dim, index[, out, inplace]) | Gathers values along a batch dimension, as [`torch.gather()`](https://docs.pytorch.org/docs/stable/generated/torch.gather.html#torch.gather) does. |
| [`TensorDictBase.masked_fill`](generated/tensordict.TensorDictBase.masked_fill.html#tensordict.TensorDictBase.masked_fill)(mask, value) | Returns a copy of the tensordict with `value` written into every leaf where `mask` is `True`. |
| [`TensorDictBase.masked_fill_`](generated/tensordict.TensorDictBase.masked_fill_.html#tensordict.TensorDictBase.masked_fill_)(mask, value) | Writes `value` into every leaf where `mask` is `True`, in place. |
| [`TensorDictBase.masked_select`](generated/tensordict.TensorDictBase.masked_select.html#tensordict.TensorDictBase.masked_select)(mask, *[, inplace]) | Returns a tensordict with the elements of the batch where `mask` is `True`. |
| [`TensorDictBase.where`](generated/tensordict.TensorDictBase.where.html#tensordict.TensorDictBase.where)(condition, other, *[, ...]) | Returns a tensordict with the leaves where `condition` is `True` and `other` elsewhere. |
| [`TensorDictBase.split`](generated/tensordict.TensorDictBase.split.html#tensordict.TensorDictBase.split)(split_size[, dim]) | Splits the tensordict along a batch dimension, as [`torch.split()`](https://docs.pytorch.org/docs/stable/generated/torch.split.html#torch.split) does. |
| [`TensorDictBase.chunk`](generated/tensordict.TensorDictBase.chunk.html#tensordict.TensorDictBase.chunk)(chunks[, dim]) | Splits the tensordict into `chunks` pieces along a batch dimension, as [`torch.chunk()`](https://docs.pytorch.org/docs/stable/generated/torch.chunk.html#torch.chunk) does. |
| [`TensorDictBase.tensor_split`](generated/tensordict.TensorDictBase.tensor_split.html#tensordict.TensorDictBase.tensor_split)(indices_or_sections) | Splits the tensordict along a batch dimension, as [`torch.tensor_split()`](https://docs.pytorch.org/docs/stable/generated/torch.tensor_split.html#torch.tensor_split) does. |
| [`TensorDictBase.unbind`](generated/tensordict.TensorDictBase.unbind.html#tensordict.TensorDictBase.unbind)(dim) | Returns a tuple of the tensordicts along a batch dimension, as [`torch.Tensor.unbind()`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.unbind.html#torch.Tensor.unbind) does. |
| [`TensorDictBase.narrow`](generated/tensordict.TensorDictBase.narrow.html#tensordict.TensorDictBase.narrow)(dim, start, length) | Returns a tensordict narrowed along a batch dimension, as [`torch.Tensor.narrow()`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.narrow.html#torch.Tensor.narrow) does. |

### Shape, batch dimensions, and stacking

Read and change the batch size and the dimension names, reshape and expand
a tensordict, and stack or concatenate several tensordicts. These methods
change the batch dimensions. The dimensions of each entry after the batch
dimensions do not change. See [Batch size, shapes, and indexing](../concepts/batch_size.html#batch-size) and [Stacking and variable-size data](../concepts/stacking.html#stacking).

Methods

| [`TensorDictBase.batch_size`](generated/tensordict.TensorDictBase.batch_size.html#tensordict.TensorDictBase.batch_size) | The batch size of the tensordict, as a [`torch.Size`](https://docs.pytorch.org/docs/stable/size.html#torch.Size). |
| --- | --- |
| [`TensorDictBase.batch_dims`](generated/tensordict.TensorDictBase.batch_dims.html#tensordict.TensorDictBase.batch_dims) | The number of batch dimensions, that is `len(self.batch_size)`. |
| [`TensorDictBase.shape`](generated/tensordict.TensorDictBase.shape.html#tensordict.TensorDictBase.shape) | the same as [`batch_size`](generated/tensordict.TensorDictBase.batch_size.html#tensordict.TensorDictBase.batch_size). |
| [`TensorDictBase.ndim`](generated/tensordict.TensorDictBase.ndim.html#tensordict.TensorDictBase.ndim) | The number of batch dimensions, the same as [`batch_dims`](generated/tensordict.TensorDictBase.batch_dims.html#tensordict.TensorDictBase.batch_dims). |
| [`TensorDictBase.ndimension`](generated/tensordict.TensorDictBase.ndimension.html#tensordict.TensorDictBase.ndimension)() | Returns the number of batch dimensions, the same as [`batch_dims`](generated/tensordict.TensorDictBase.batch_dims.html#tensordict.TensorDictBase.batch_dims). |
| [`TensorDictBase.dim`](generated/tensordict.TensorDictBase.dim.html#tensordict.TensorDictBase.dim)() | Returns the number of batch dimensions, the same as [`batch_dims`](generated/tensordict.TensorDictBase.batch_dims.html#tensordict.TensorDictBase.batch_dims). |
| [`TensorDictBase.size`](generated/tensordict.TensorDictBase.size.html#tensordict.TensorDictBase.size)([dim]) | Returns the batch size, or the size of one batch dimension. |
| [`TensorDictBase.numel`](generated/tensordict.TensorDictBase.numel.html#tensordict.TensorDictBase.numel)() | Returns the number of elements in the batch, which is the product of `batch_size`. |
| [`TensorDictBase.__len__`](generated/tensordict.TensorDictBase.__len__.html#tensordict.TensorDictBase.__len__)() | Returns the size of the first batch dimension, or 0 if the tensordict has no batch dimensions. |
| [`TensorDictBase.names`](generated/tensordict.TensorDictBase.names.html#tensordict.TensorDictBase.names) | The dimension names of the tensordict, one per batch dimension. |
| [`TensorDictBase.refine_names`](generated/tensordict.TensorDictBase.refine_names.html#tensordict.TensorDictBase.refine_names)(*names) | Names the unnamed batch dimensions, and checks the named ones, in place. |
| [`TensorDictBase.rename`](generated/tensordict.TensorDictBase.rename.html#tensordict.TensorDictBase.rename)(*names, **rename_map) | Returns a shallow copy of the tensordict with new names for its batch dimensions. |
| [`TensorDictBase.rename_`](generated/tensordict.TensorDictBase.rename_.html#tensordict.TensorDictBase.rename_)(*names, **rename_map) | Renames the batch dimensions of the tensordict in place and returns `self`. |
| [`TensorDictBase.auto_batch_size_`](generated/tensordict.TensorDictBase.auto_batch_size_.html#tensordict.TensorDictBase.auto_batch_size_)([...]) | Sets the batch size to the largest leading shape that all the entries share. |
| [`TensorDictBase.reshape`](generated/tensordict.TensorDictBase.reshape.html#tensordict.TensorDictBase.reshape)(*args, **kwargs) | Returns a tensordict whose batch dimensions are reshaped to `shape`. |
| [`TensorDictBase.view`](generated/tensordict.TensorDictBase.view.html#tensordict.TensorDictBase.view)(*shape[, size, batch_size]) | Returns a tensordict with a new batch size, whose leaves are views of the original leaves. |
| [`TensorDictBase.flatten`](generated/tensordict.TensorDictBase.flatten.html#tensordict.TensorDictBase.flatten)([start_dim, end_dim, ...]) | Returns a tensordict whose batch dimensions `start_dim` to `end_dim` are flattened into one. |
| [`TensorDictBase.unflatten`](generated/tensordict.TensorDictBase.unflatten.html#tensordict.TensorDictBase.unflatten)(dim, ...[, inplace]) | Returns a tensordict whose batch dimension `dim` is split into several. |
| [`TensorDictBase.permute`](generated/tensordict.TensorDictBase.permute.html#tensordict.TensorDictBase.permute)(*args, **kwargs) | Returns a tensordict with its batch dimensions permuted, as [`torch.Tensor.permute()`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.permute.html#torch.Tensor.permute) does. |
| [`TensorDictBase.transpose`](generated/tensordict.TensorDictBase.transpose.html#tensordict.TensorDictBase.transpose)(dim0, dim1) | Returns a tensordict with the batch dimensions `dim0` and `dim1` swapped. |
| [`TensorDictBase.swapaxes`](generated/tensordict.TensorDictBase.swapaxes.html#tensordict.TensorDictBase.swapaxes)(axis0, axis1) | Returns a tensordict with two batch dimensions swapped, as [`transpose()`](generated/tensordict.TensorDictBase.transpose.html#tensordict.TensorDictBase.transpose) does. |
| [`TensorDictBase.swapdims`](generated/tensordict.TensorDictBase.swapdims.html#tensordict.TensorDictBase.swapdims)(axis0, axis1) | Returns a tensordict with two batch dimensions swapped, as [`transpose()`](generated/tensordict.TensorDictBase.transpose.html#tensordict.TensorDictBase.transpose) does. |
| [`TensorDictBase.moveaxis`](generated/tensordict.TensorDictBase.moveaxis.html#tensordict.TensorDictBase.moveaxis)(source, destination) | Returns a tensordict with batch dimensions moved to new positions, as [`torch.movedim()`](https://docs.pytorch.org/docs/stable/generated/torch.movedim.html#torch.movedim) does. |
| [`TensorDictBase.movedim`](generated/tensordict.TensorDictBase.movedim.html#tensordict.TensorDictBase.movedim)(source, destination) | Returns a tensordict with batch dimensions moved to new positions, as [`torch.movedim()`](https://docs.pytorch.org/docs/stable/generated/torch.movedim.html#torch.movedim) does. |
| [`TensorDictBase.squeeze`](generated/tensordict.TensorDictBase.squeeze.html#tensordict.TensorDictBase.squeeze)(*args, **kwargs) | Returns a tensordict without the batch dimensions of size 1. |
| [`TensorDictBase.unsqueeze`](generated/tensordict.TensorDictBase.unsqueeze.html#tensordict.TensorDictBase.unsqueeze)(*args, **kwargs) | Returns a tensordict with a new batch dimension of size 1 at position `dim`. |
| [`TensorDictBase.expand`](generated/tensordict.TensorDictBase.expand.html#tensordict.TensorDictBase.expand)(*args, **kwargs) | Returns a tensordict whose batch dimensions are expanded to `shape`. |
| [`TensorDictBase.expand_as`](generated/tensordict.TensorDictBase.expand_as.html#tensordict.TensorDictBase.expand_as)(other) | Returns the tensordict expanded to the shape of `other`. |
| [`TensorDictBase.broadcast_to`](generated/tensordict.TensorDictBase.broadcast_to.html#tensordict.TensorDictBase.broadcast_to)(shape) | Returns the tensordict broadcast to the batch size `shape`, as [`torch.broadcast_to()`](https://docs.pytorch.org/docs/stable/generated/torch.broadcast_to.html#torch.broadcast_to) does. |
| [`TensorDictBase.repeat`](generated/tensordict.TensorDictBase.repeat.html#tensordict.TensorDictBase.repeat)(*repeats[, inplace]) | Repeats the batch along each batch dimension, as [`torch.Tensor.repeat()`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.repeat.html#torch.Tensor.repeat) does. |
| [`TensorDictBase.repeat_interleave`](generated/tensordict.TensorDictBase.repeat_interleave.html#tensordict.TensorDictBase.repeat_interleave)(repeats[, ...]) | Repeats each element of the batch along `dim`, as [`torch.Tensor.repeat_interleave()`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.repeat_interleave.html#torch.Tensor.repeat_interleave) does. |
| [`TensorDictBase.tile`](generated/tensordict.TensorDictBase.tile.html#tensordict.TensorDictBase.tile)(dims) | Returns a tensordict whose batch is repeated, as [`torch.tile()`](https://docs.pytorch.org/docs/stable/generated/torch.tile.html#torch.tile) does. |
| [`TensorDictBase.flip`](generated/tensordict.TensorDictBase.flip.html#tensordict.TensorDictBase.flip)(dims) | Returns a tensordict with the elements reversed along batch dimensions, as [`torch.flip()`](https://docs.pytorch.org/docs/stable/generated/torch.flip.html#torch.flip) does. |
| [`TensorDictBase.fliplr`](generated/tensordict.TensorDictBase.fliplr.html#tensordict.TensorDictBase.fliplr)() | Returns a tensordict with the elements reversed along batch dimension 1. |
| [`TensorDictBase.flipud`](generated/tensordict.TensorDictBase.flipud.html#tensordict.TensorDictBase.flipud)() | Returns a tensordict with the elements reversed along batch dimension 0. |
| [`TensorDictBase.roll`](generated/tensordict.TensorDictBase.roll.html#tensordict.TensorDictBase.roll)(shifts[, dims, inplace]) | Returns a tensordict with the elements rolled along batch dimensions, as [`torch.roll()`](https://docs.pytorch.org/docs/stable/generated/torch.roll.html#torch.roll) does. |
| [`TensorDictBase.rot90`](generated/tensordict.TensorDictBase.rot90.html#tensordict.TensorDictBase.rot90)([k, dims]) | Returns a tensordict rotated by 90 degrees in the plane of two batch dimensions. |
| [`TensorDictBase.pad`](generated/tensordict.TensorDictBase.pad.html#tensordict.TensorDictBase.pad)(pad_size[, value, ...]) | Returns the tensordict padded along its batch dimensions with a constant value. |
| [`TensorDictBase.atleast_1d`](generated/tensordict.TensorDictBase.atleast_1d.html#tensordict.TensorDictBase.atleast_1d)() | Returns the tensordict with at least 1 batch dimension. |
| [`TensorDictBase.atleast_2d`](generated/tensordict.TensorDictBase.atleast_2d.html#tensordict.TensorDictBase.atleast_2d)() | Returns the tensordict with at least 2 batch dimensions. |
| [`TensorDictBase.atleast_3d`](generated/tensordict.TensorDictBase.atleast_3d.html#tensordict.TensorDictBase.atleast_3d)() | Returns the tensordict with at least 3 batch dimensions. |
| [`TensorDictBase.contiguous`](generated/tensordict.TensorDictBase.contiguous.html#tensordict.TensorDictBase.contiguous)(*[, canonical, ...]) | Returns a new tensordict with contiguous leaves. |
| [`TensorDictBase.is_contiguous`](generated/tensordict.TensorDictBase.is_contiguous.html#tensordict.TensorDictBase.is_contiguous)() | `True` if all the leaves of the tensordict, nested ones included, are contiguous. |
| [`TensorDictBase.stack`](generated/tensordict.TensorDictBase.stack.html#tensordict.TensorDictBase.stack)(input[, dim, out]) | Stacks tensordicts into one dense tensordict along a new dimension `dim`. |
| [`TensorDictBase.cat`](generated/tensordict.TensorDictBase.cat.html#tensordict.TensorDictBase.cat)(input[, dim, out]) | Concatenates tensordicts along an existing batch dimension `dim`. |
| [`TensorDictBase.lazy_stack`](generated/tensordict.TensorDictBase.lazy_stack.html#tensordict.TensorDictBase.lazy_stack)(input[, dim, out]) | Stacks tensordicts into a lazy stack. |
| [`TensorDictBase.maybe_dense_stack`](generated/tensordict.TensorDictBase.maybe_dense_stack.html#tensordict.TensorDictBase.maybe_dense_stack)(input[, ...]) | Stacks tensordicts densely if possible, or into a lazy stack otherwise. |
| [`TensorDictBase.to_lazystack`](generated/tensordict.TensorDictBase.to_lazystack.html#tensordict.TensorDictBase.to_lazystack)([dim]) | Returns a lazy stack of the slices of the tensordict along `dim`. |
| [`TensorDictBase.densify`](generated/tensordict.TensorDictBase.densify.html#tensordict.TensorDictBase.densify)([layout]) | Returns the tensordict with its lazy stacks replaced by dense tensordicts. |
| [`TensorDictBase.to_padded_tensor`](generated/tensordict.TensorDictBase.to_padded_tensor.html#tensordict.TensorDictBase.to_padded_tensor)([padding, ...]) | Returns a tensordict whose nested-tensor leaves are converted to padded tensors. |

Functions

| [`stack`](generated/tensordict.stack.html#tensordict.stack)(input[, dim, out]) | Stacks tensordicts into one dense tensordict along a new dimension `dim`. |
| --- | --- |
| [`cat`](generated/tensordict.cat.html#tensordict.cat)(input[, dim, out]) | Concatenates tensordicts along an existing batch dimension `dim`. |
| [`lazy_stack`](generated/tensordict.lazy_stack.html#tensordict.lazy_stack)(input[, dim, out]) | Stacks tensordicts into a lazy stack ([`LazyStackedTensorDict`](generated/tensordict.LazyStackedTensorDict.html#tensordict.LazyStackedTensorDict)). |
| [`maybe_dense_stack`](generated/tensordict.maybe_dense_stack.html#tensordict.maybe_dense_stack)(input[, dim, out]) | Stacks tensordicts densely if possible, or into a lazy stack otherwise. |
| [`dense_stack_tds`](generated/tensordict.dense_stack_tds.html#tensordict.dense_stack_tds)(td_list[, dim]) | Stacks a list of tensordicts, or the tensordicts of a lazy stack, densely if possible. |
| [`pad`](generated/tensordict.pad.html#tensordict.pad)(tensordict, pad_size[, value, inplace, safe]) | Pads all tensors in a tensordict along the batch dimensions with a constant value. |
| [`pad_sequence`](generated/tensordict.pad_sequence.html#tensordict.pad_sequence)(list_of_tensordicts[, pad_dim, ...]) | Pads tensordicts to a common size along one dimension and stacks them densely. |
| [`unpad_sequence`](generated/tensordict.unpad_sequence.html#tensordict.unpad_sequence)(padded[, mask, pad_dim]) | Splits a padded batch back into its variable-length samples. |
| [`collate`](generated/tensordict.collate.html#tensordict.collate)(batch) | Collates a list of samples into a batch, for [`DataLoader`](https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader). |

### Device, dtype, and memory

Move a tensordict to a device, cast its entries to another dtype, and
control the memory that holds them. See [Devices, dtypes, and memory](../concepts/devices.html#devices).

Device and memory

| [`TensorDictBase.to`](generated/tensordict.TensorDictBase.to.html#tensordict.TensorDictBase.to)(*args, **kwargs) | Returns a tensordict cast to another device, dtype or memory format, or with a new batch size. |
| --- | --- |
| [`TensorDictBase.cpu`](generated/tensordict.TensorDictBase.cpu.html#tensordict.TensorDictBase.cpu)(**kwargs) | Returns the tensordict on the CPU, as `to("cpu", **kwargs)` does. |
| [`TensorDictBase.cuda`](generated/tensordict.TensorDictBase.cuda.html#tensordict.TensorDictBase.cuda)([device]) | Returns the tensordict on a CUDA device, as `to()` does. |
| [`TensorDictBase.device`](generated/tensordict.TensorDictBase.device.html#tensordict.TensorDictBase.device) | The device of the tensordict, or `None` if it has none. |
| [`TensorDictBase.auto_device_`](generated/tensordict.TensorDictBase.auto_device_.html#tensordict.TensorDictBase.auto_device_)() | Sets the device of the tensordict to the device of its leaves, if they all share one. |
| [`TensorDictBase.clear_device_`](generated/tensordict.TensorDictBase.clear_device_.html#tensordict.TensorDictBase.clear_device_)() | Sets the device of the tensordict and its sub-tensordicts to `None`, and returns it. |
| [`TensorDictBase.is_cpu`](generated/tensordict.TensorDictBase.is_cpu.html#tensordict.TensorDictBase.is_cpu) | Whether the tensordict has a CPU device (`False` if it has no device). |
| [`TensorDictBase.is_cuda`](generated/tensordict.TensorDictBase.is_cuda.html#tensordict.TensorDictBase.is_cuda) | Whether the tensordict has a CUDA device (`False` if it has no device). |
| [`TensorDictBase.is_meta`](generated/tensordict.TensorDictBase.is_meta.html#tensordict.TensorDictBase.is_meta) | Whether the tensordict has a meta device (`False` if it has no device). |
| [`TensorDictBase.pin_memory`](generated/tensordict.TensorDictBase.pin_memory.html#tensordict.TensorDictBase.pin_memory)([num_threads, inplace]) | Returns a tensordict whose tensor leaves are in pinned (page-locked) memory. |
| [`TensorDictBase.share_memory_`](generated/tensordict.TensorDictBase.share_memory_.html#tensordict.TensorDictBase.share_memory_)() | Moves the leaves to shared memory, locks the tensordict, and returns `self`. |
| [`TensorDictBase.is_shared`](generated/tensordict.TensorDictBase.is_shared.html#tensordict.TensorDictBase.is_shared)() | Whether the tensordict is in shared memory. |
| [`TensorDictBase.record_stream`](generated/tensordict.TensorDictBase.record_stream.html#tensordict.TensorDictBase.record_stream)(stream) | Marks the tensor leaves of the tensordict as used by `stream`. |
| [`TensorDictBase.consolidate`](generated/tensordict.TensorDictBase.consolidate.html#tensordict.TensorDictBase.consolidate)([filename, ...]) | Copies all the leaves into one contiguous storage, and returns a tensordict of views of it. |
| [`TensorDictBase.is_consolidated`](generated/tensordict.TensorDictBase.is_consolidated.html#tensordict.TensorDictBase.is_consolidated)() | Whether the leaves of the tensordict are views into one storage made by [`consolidate()`](generated/tensordict.TensorDictBase.consolidate.html#tensordict.TensorDictBase.consolidate). |
| [`TensorDictBase.from_consolidated`](generated/tensordict.TensorDictBase.from_consolidated.html#tensordict.TensorDictBase.from_consolidated)(filename) | Reconstructs a tensordict from a file written by `consolidate(filename=...)`. |
| [`TensorDictBase.bytes`](generated/tensordict.TensorDictBase.bytes.html#tensordict.TensorDictBase.bytes)(*[, count_duplicates]) | Returns the number of bytes of the tensor leaves. |
| [`TensorDictBase.data_ptr`](generated/tensordict.TensorDictBase.data_ptr.html#tensordict.TensorDictBase.data_ptr)(*[, storage]) | Returns the data pointers of the leaves, as a tensordict of integers. |
| [`TensorDictBase.param_count`](generated/tensordict.TensorDictBase.param_count.html#tensordict.TensorDictBase.param_count)(*[, count_duplicates]) | Returns the total number of elements of the tensor leaves. |
| [`TensorDictBase.attrs`](generated/tensordict.TensorDictBase.attrs.html#tensordict.TensorDictBase.attrs)(*[, fields, num_threads]) | Returns a tensordict that describes each tensor leaf with a [`TensorAttrs`](generated/tensordict.TensorAttrs.html#tensordict.TensorAttrs). |
| [`TensorDictBase.clear_refs_for_compile_`](generated/tensordict.TensorDictBase.clear_refs_for_compile_.html#tensordict.TensorDictBase.clear_refs_for_compile_)() | Clears the record of the last operation of the tensordict and its nested tensordicts. |

Dtype

| [`TensorDictBase.dtype`](generated/tensordict.TensorDictBase.dtype.html#tensordict.TensorDictBase.dtype) | The dtype of the leaves, or `None` if they have different dtypes or there is no leaf. |
| --- | --- |
| [`TensorDictBase.type`](generated/tensordict.TensorDictBase.type.html#tensordict.TensorDictBase.type)(dst_type, *[, cast_filter]) | Returns a tensordict whose tensor leaves are cast to `dst_type`. |
| [`TensorDictBase.is_floating_point`](generated/tensordict.TensorDictBase.is_floating_point.html#tensordict.TensorDictBase.is_floating_point)() | `True` if every tensor leaf of the tensordict, nested ones included, has a floating-point dtype. |
| [`TensorDictBase.bool`](generated/tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)(*[, cast_filter]) | Returns a tensordict whose tensor leaves are cast to `torch.bool`. |
| [`TensorDictBase.half`](generated/tensordict.TensorDictBase.half.html#tensordict.TensorDictBase.half)(*[, cast_filter]) | Returns a tensordict whose tensor leaves are cast to `torch.float16` (`torch.half`). |
| [`TensorDictBase.float`](generated/tensordict.TensorDictBase.float.html#tensordict.TensorDictBase.float)(*[, cast_filter]) | Returns a tensordict whose tensor leaves are cast to `torch.float32` (`torch.float`). |
| [`TensorDictBase.double`](generated/tensordict.TensorDictBase.double.html#tensordict.TensorDictBase.double)(*[, cast_filter]) | Returns a tensordict whose tensor leaves are cast to `torch.float64` (`torch.double`). |
| [`TensorDictBase.int`](generated/tensordict.TensorDictBase.int.html#tensordict.TensorDictBase.int)(*[, cast_filter]) | Returns a tensordict whose tensor leaves are cast to `torch.int32` (`torch.int`). |
| [`TensorDictBase.bfloat16`](generated/tensordict.TensorDictBase.bfloat16.html#tensordict.TensorDictBase.bfloat16)(*[, cast_filter]) | Returns a tensordict whose tensor leaves are cast to `torch.bfloat16`. |
| [`TensorDictBase.float16`](generated/tensordict.TensorDictBase.float16.html#tensordict.TensorDictBase.float16)(*[, cast_filter]) | Returns a tensordict whose tensor leaves are cast to `torch.float16`. |
| [`TensorDictBase.float32`](generated/tensordict.TensorDictBase.float32.html#tensordict.TensorDictBase.float32)(*[, cast_filter]) | Returns a tensordict whose tensor leaves are cast to `torch.float32`. |
| [`TensorDictBase.float64`](generated/tensordict.TensorDictBase.float64.html#tensordict.TensorDictBase.float64)(*[, cast_filter]) | Returns a tensordict whose tensor leaves are cast to `torch.float64`. |
| [`TensorDictBase.int8`](generated/tensordict.TensorDictBase.int8.html#tensordict.TensorDictBase.int8)(*[, cast_filter]) | Returns a tensordict whose tensor leaves are cast to `torch.int8`. |
| [`TensorDictBase.int16`](generated/tensordict.TensorDictBase.int16.html#tensordict.TensorDictBase.int16)(*[, cast_filter]) | Returns a tensordict whose tensor leaves are cast to `torch.int16`. |
| [`TensorDictBase.int32`](generated/tensordict.TensorDictBase.int32.html#tensordict.TensorDictBase.int32)(*[, cast_filter]) | Returns a tensordict whose tensor leaves are cast to `torch.int32`. |
| [`TensorDictBase.int64`](generated/tensordict.TensorDictBase.int64.html#tensordict.TensorDictBase.int64)(*[, cast_filter]) | Returns a tensordict whose tensor leaves are cast to `torch.int64`. |
| [`TensorDictBase.uint8`](generated/tensordict.TensorDictBase.uint8.html#tensordict.TensorDictBase.uint8)(*[, cast_filter]) | Returns a tensordict whose tensor leaves are cast to `torch.uint8`. |
| [`TensorDictBase.uint16`](generated/tensordict.TensorDictBase.uint16.html#tensordict.TensorDictBase.uint16)(*[, cast_filter]) | Returns a tensordict whose tensor leaves are cast to `torch.uint16`. |
| [`TensorDictBase.uint32`](generated/tensordict.TensorDictBase.uint32.html#tensordict.TensorDictBase.uint32)(*[, cast_filter]) | Returns a tensordict whose tensor leaves are cast to `torch.uint32`. |
| [`TensorDictBase.uint64`](generated/tensordict.TensorDictBase.uint64.html#tensordict.TensorDictBase.uint64)(*[, cast_filter]) | Returns a tensordict whose tensor leaves are cast to `torch.uint64`. |
| [`TensorDictBase.complex32`](generated/tensordict.TensorDictBase.complex32.html#tensordict.TensorDictBase.complex32)(*[, cast_filter]) | Returns a tensordict whose tensor leaves are cast to `torch.complex32`. |
| [`TensorDictBase.complex64`](generated/tensordict.TensorDictBase.complex64.html#tensordict.TensorDictBase.complex64)(*[, cast_filter]) | Returns a tensordict whose tensor leaves are cast to `torch.complex64`. |
| [`TensorDictBase.complex128`](generated/tensordict.TensorDictBase.complex128.html#tensordict.TensorDictBase.complex128)(*[, cast_filter]) | Returns a tensordict whose tensor leaves are cast to `torch.complex128`. |
| [`TensorDictBase.qint8`](generated/tensordict.TensorDictBase.qint8.html#tensordict.TensorDictBase.qint8)(*[, cast_filter]) | Returns a tensordict whose tensor leaves are cast to `torch.qint8`. |
| [`TensorDictBase.qint32`](generated/tensordict.TensorDictBase.qint32.html#tensordict.TensorDictBase.qint32)(*[, cast_filter]) | Returns a tensordict whose tensor leaves are cast to `torch.qint32`. |
| [`TensorDictBase.quint8`](generated/tensordict.TensorDictBase.quint8.html#tensordict.TensorDictBase.quint8)(*[, cast_filter]) | Returns a tensordict whose tensor leaves are cast to `torch.quint8`. |
| [`TensorDictBase.quint4x2`](generated/tensordict.TensorDictBase.quint4x2.html#tensordict.TensorDictBase.quint4x2)(*[, cast_filter]) | Returns a tensordict whose tensor leaves are cast to `torch.quint4x2`. |

`to` and the dtype shorthands take a `cast_filter` predicate
`fn(key, value) -> bool` that selects the leaves to cast. The predicates
below test the dtype of their last argument, so they serve as a
`cast_filter` and as a test in [`apply()`](generated/tensordict.TensorDictBase.apply.html#tensordict.TensorDictBase.apply).

Functions

| [`from_consolidated`](generated/tensordict.from_consolidated.html#tensordict.from_consolidated)(filename) | Reconstructs a tensordict from a file written by `consolidate(filename=...)`. |
| --- | --- |
| [`is_inexact`](generated/tensordict.is_inexact.html#tensordict.is_inexact)(*args) | Whether the last argument is a floating-point or complex tensor. |
| [`is_floating`](generated/tensordict.is_floating.html#tensordict.is_floating)(*args) | Whether the last argument is a floating-point tensor. |
| [`is_integer`](generated/tensordict.is_integer.html#tensordict.is_integer)(*args) | Whether the last argument is an integer tensor (signed or unsigned, not bool). |
| [`is_bool`](generated/tensordict.is_bool.html#tensordict.is_bool)(*args) | Whether the last argument is a boolean tensor. |

### Pointwise math

Apply arithmetic and elementwise functions to every entry. The Python
operators (`+`, `-`, `*`, `/`, `**`, `==`, `&`, ...) call
these methods. A method whose name ends with an underscore, such as
`add_`, writes into the existing tensors: see
[In-place operations](../concepts/operations.html#inplace-conventions). For broadcasting and for tensordicts with
different keys, see [Arithmetic, apply, reductions, and gradients](../concepts/operations.html#operations).

Arithmetic

| [`TensorDictBase.add`](generated/tensordict.TensorDictBase.add.html#tensordict.TensorDictBase.add)(other, *[, alpha, default]) | Returns a new tensordict with `other`, scaled by `alpha`, added to every leaf. |
| --- | --- |
| [`TensorDictBase.add_`](generated/tensordict.TensorDictBase.add_.html#tensordict.TensorDictBase.add_)(other, *[, alpha]) | Adds `other`, scaled by `alpha`, to every leaf in place and returns `self`. |
| [`TensorDictBase.sub`](generated/tensordict.TensorDictBase.sub.html#tensordict.TensorDictBase.sub)(other, *[, alpha, default]) | Returns a new tensordict with `other`, scaled by `alpha`, subtracted from every leaf. |
| [`TensorDictBase.sub_`](generated/tensordict.TensorDictBase.sub_.html#tensordict.TensorDictBase.sub_)(other[, alpha]) | Subtracts `other`, scaled by `alpha`, from every leaf in place and returns `self`. |
| [`TensorDictBase.rsub`](generated/tensordict.TensorDictBase.rsub.html#tensordict.TensorDictBase.rsub)(other, *[, alpha, default]) | Returns a new tensordict with every leaf, scaled by `alpha`, subtracted from `other`. |
| [`TensorDictBase.mul`](generated/tensordict.TensorDictBase.mul.html#tensordict.TensorDictBase.mul)(other, *[, default]) | Returns a new tensordict with every leaf multiplied by `other`. |
| [`TensorDictBase.mul_`](generated/tensordict.TensorDictBase.mul_.html#tensordict.TensorDictBase.mul_)(other) | Multiplies every leaf by `other` in place and returns `self`. |
| [`TensorDictBase.div`](generated/tensordict.TensorDictBase.div.html#tensordict.TensorDictBase.div)(other, *[, default]) | Returns a new tensordict with every leaf divided by `other`. |
| [`TensorDictBase.div_`](generated/tensordict.TensorDictBase.div_.html#tensordict.TensorDictBase.div_)(other) | Divides every leaf by `other` in place and returns `self`. |
| [`TensorDictBase.pow`](generated/tensordict.TensorDictBase.pow.html#tensordict.TensorDictBase.pow)(other, *[, default]) | Returns a new tensordict with every leaf raised to the power `other`. |
| [`TensorDictBase.pow_`](generated/tensordict.TensorDictBase.pow_.html#tensordict.TensorDictBase.pow_)(other) | Raises every leaf to the power `other` in place and returns `self`. |
| [`TensorDictBase.mod`](generated/tensordict.TensorDictBase.mod.html#tensordict.TensorDictBase.mod)(other) | Returns a new tensordict with the remainder of every leaf divided by `other`. |
| [`TensorDictBase.lerp`](generated/tensordict.TensorDictBase.lerp.html#tensordict.TensorDictBase.lerp)(end, weight) | Returns a new tensordict that interpolates linearly from every leaf (the start) to `end`. |
| [`TensorDictBase.lerp_`](generated/tensordict.TensorDictBase.lerp_.html#tensordict.TensorDictBase.lerp_)(end, weight) | Interpolates linearly from every leaf to `end` in place and returns `self`. |
| [`TensorDictBase.addcdiv`](generated/tensordict.TensorDictBase.addcdiv.html#tensordict.TensorDictBase.addcdiv)(other1, other2[, value]) | Returns a new tensordict with `value * other1 / other2` added to every leaf. |
| [`TensorDictBase.addcdiv_`](generated/tensordict.TensorDictBase.addcdiv_.html#tensordict.TensorDictBase.addcdiv_)(other1, other2, *[, ...]) | Adds `value * other1 / other2` to every leaf in place and returns `self`. |
| [`TensorDictBase.addcmul`](generated/tensordict.TensorDictBase.addcmul.html#tensordict.TensorDictBase.addcmul)(other1, other2, *[, ...]) | Returns a new tensordict with `value * other1 * other2` added to every leaf. |
| [`TensorDictBase.addcmul_`](generated/tensordict.TensorDictBase.addcmul_.html#tensordict.TensorDictBase.addcmul_)(other1, other2, *[, ...]) | Adds `value * other1 * other2` to every leaf in place and returns `self`. |
| [`TensorDictBase.clamp`](generated/tensordict.TensorDictBase.clamp.html#tensordict.TensorDictBase.clamp)([min, max, out]) | Returns a new tensordict with every leaf clamped into the range `[min, max]`. |
| [`TensorDictBase.clamp_min`](generated/tensordict.TensorDictBase.clamp_min.html#tensordict.TensorDictBase.clamp_min)(other[, default]) | Returns a new tensordict with every leaf clamped to at least `other`. |
| [`TensorDictBase.clamp_min_`](generated/tensordict.TensorDictBase.clamp_min_.html#tensordict.TensorDictBase.clamp_min_)(other) | Clamps every leaf to at least `other` in place and returns `self`. |
| [`TensorDictBase.clamp_max`](generated/tensordict.TensorDictBase.clamp_max.html#tensordict.TensorDictBase.clamp_max)(other, *[, default]) | Returns a new tensordict with every leaf clamped to at most `other`. |
| [`TensorDictBase.clamp_max_`](generated/tensordict.TensorDictBase.clamp_max_.html#tensordict.TensorDictBase.clamp_max_)(other) | Clamps every leaf to at most `other` in place and returns `self`. |
| [`TensorDictBase.maximum`](generated/tensordict.TensorDictBase.maximum.html#tensordict.TensorDictBase.maximum)(other, *[, default]) | Returns a new tensordict with the element-wise maximum of every leaf and `other`. |
| [`TensorDictBase.maximum_`](generated/tensordict.TensorDictBase.maximum_.html#tensordict.TensorDictBase.maximum_)(other) | Replaces every leaf with its element-wise maximum with `other` in place and returns `self`. |
| [`TensorDictBase.minimum`](generated/tensordict.TensorDictBase.minimum.html#tensordict.TensorDictBase.minimum)(other, *[, default]) | Returns a new tensordict with the element-wise minimum of every leaf and `other`. |
| [`TensorDictBase.minimum_`](generated/tensordict.TensorDictBase.minimum_.html#tensordict.TensorDictBase.minimum_)(other) | Replaces every leaf with its element-wise minimum with `other` in place and returns `self`. |
| [`TensorDictBase.zero_`](generated/tensordict.TensorDictBase.zero_.html#tensordict.TensorDictBase.zero_)() | Fills every leaf with zeros in place and returns `self`. |

Elementwise functions

| [`TensorDictBase.abs`](generated/tensordict.TensorDictBase.abs.html#tensordict.TensorDictBase.abs)() | Returns a new tensordict with [`torch.abs()`](https://docs.pytorch.org/docs/stable/generated/torch.abs.html#torch.abs) applied to every leaf. |
| --- | --- |
| [`TensorDictBase.abs_`](generated/tensordict.TensorDictBase.abs_.html#tensordict.TensorDictBase.abs_)() | Applies [`torch.abs()`](https://docs.pytorch.org/docs/stable/generated/torch.abs.html#torch.abs) to every leaf in place and returns `self`. |
| [`TensorDictBase.neg`](generated/tensordict.TensorDictBase.neg.html#tensordict.TensorDictBase.neg)() | Returns a new tensordict with [`torch.neg()`](https://docs.pytorch.org/docs/stable/generated/torch.neg.html#torch.neg) applied to every leaf. |
| [`TensorDictBase.neg_`](generated/tensordict.TensorDictBase.neg_.html#tensordict.TensorDictBase.neg_)() | Applies [`torch.neg()`](https://docs.pytorch.org/docs/stable/generated/torch.neg.html#torch.neg) to every leaf in place and returns `self`. |
| [`TensorDictBase.reciprocal`](generated/tensordict.TensorDictBase.reciprocal.html#tensordict.TensorDictBase.reciprocal)() | Returns a new tensordict with [`torch.reciprocal()`](https://docs.pytorch.org/docs/stable/generated/torch.reciprocal.html#torch.reciprocal) applied to every leaf. |
| [`TensorDictBase.reciprocal_`](generated/tensordict.TensorDictBase.reciprocal_.html#tensordict.TensorDictBase.reciprocal_)() | Applies [`torch.reciprocal()`](https://docs.pytorch.org/docs/stable/generated/torch.reciprocal.html#torch.reciprocal) to every leaf in place and returns `self`. |
| [`TensorDictBase.sqrt`](generated/tensordict.TensorDictBase.sqrt.html#tensordict.TensorDictBase.sqrt)() | Returns a new tensordict with [`torch.sqrt()`](https://docs.pytorch.org/docs/stable/generated/torch.sqrt.html#torch.sqrt) applied to every leaf. |
| [`TensorDictBase.sqrt_`](generated/tensordict.TensorDictBase.sqrt_.html#tensordict.TensorDictBase.sqrt_)() | Applies [`torch.sqrt()`](https://docs.pytorch.org/docs/stable/generated/torch.sqrt.html#torch.sqrt) to every leaf in place and returns `self`. |
| [`TensorDictBase.exp`](generated/tensordict.TensorDictBase.exp.html#tensordict.TensorDictBase.exp)() | Returns a new tensordict with [`torch.exp()`](https://docs.pytorch.org/docs/stable/generated/torch.exp.html#torch.exp) applied to every leaf. |
| [`TensorDictBase.exp_`](generated/tensordict.TensorDictBase.exp_.html#tensordict.TensorDictBase.exp_)() | Applies [`torch.exp()`](https://docs.pytorch.org/docs/stable/generated/torch.exp.html#torch.exp) to every leaf in place and returns `self`. |
| [`TensorDictBase.expm1`](generated/tensordict.TensorDictBase.expm1.html#tensordict.TensorDictBase.expm1)() | Returns a new tensordict with [`torch.expm1()`](https://docs.pytorch.org/docs/stable/generated/torch.expm1.html#torch.expm1) applied to every leaf. |
| [`TensorDictBase.expm1_`](generated/tensordict.TensorDictBase.expm1_.html#tensordict.TensorDictBase.expm1_)() | Applies [`torch.expm1()`](https://docs.pytorch.org/docs/stable/generated/torch.expm1.html#torch.expm1) to every leaf in place and returns `self`. |
| [`TensorDictBase.log`](generated/tensordict.TensorDictBase.log.html#tensordict.TensorDictBase.log)() | Returns a new tensordict with [`torch.log()`](https://docs.pytorch.org/docs/stable/generated/torch.log.html#torch.log) applied to every leaf. |
| [`TensorDictBase.log_`](generated/tensordict.TensorDictBase.log_.html#tensordict.TensorDictBase.log_)() | Applies [`torch.log()`](https://docs.pytorch.org/docs/stable/generated/torch.log.html#torch.log) to every leaf in place and returns `self`. |
| [`TensorDictBase.log10`](generated/tensordict.TensorDictBase.log10.html#tensordict.TensorDictBase.log10)() | Returns a new tensordict with [`torch.log10()`](https://docs.pytorch.org/docs/stable/generated/torch.log10.html#torch.log10) applied to every leaf. |
| [`TensorDictBase.log10_`](generated/tensordict.TensorDictBase.log10_.html#tensordict.TensorDictBase.log10_)() | Applies [`torch.log10()`](https://docs.pytorch.org/docs/stable/generated/torch.log10.html#torch.log10) to every leaf in place and returns `self`. |
| [`TensorDictBase.log1p`](generated/tensordict.TensorDictBase.log1p.html#tensordict.TensorDictBase.log1p)() | Returns a new tensordict with [`torch.log1p()`](https://docs.pytorch.org/docs/stable/generated/torch.log1p.html#torch.log1p) applied to every leaf. |
| [`TensorDictBase.log1p_`](generated/tensordict.TensorDictBase.log1p_.html#tensordict.TensorDictBase.log1p_)() | Applies [`torch.log1p()`](https://docs.pytorch.org/docs/stable/generated/torch.log1p.html#torch.log1p) to every leaf in place and returns `self`. |
| [`TensorDictBase.log2`](generated/tensordict.TensorDictBase.log2.html#tensordict.TensorDictBase.log2)() | Returns a new tensordict with [`torch.log2()`](https://docs.pytorch.org/docs/stable/generated/torch.log2.html#torch.log2) applied to every leaf. |
| [`TensorDictBase.log2_`](generated/tensordict.TensorDictBase.log2_.html#tensordict.TensorDictBase.log2_)() | Applies [`torch.log2()`](https://docs.pytorch.org/docs/stable/generated/torch.log2.html#torch.log2) to every leaf in place and returns `self`. |
| [`TensorDictBase.sin`](generated/tensordict.TensorDictBase.sin.html#tensordict.TensorDictBase.sin)() | Returns a new tensordict with [`torch.sin()`](https://docs.pytorch.org/docs/stable/generated/torch.sin.html#torch.sin) applied to every leaf. |
| [`TensorDictBase.sin_`](generated/tensordict.TensorDictBase.sin_.html#tensordict.TensorDictBase.sin_)() | Applies [`torch.sin()`](https://docs.pytorch.org/docs/stable/generated/torch.sin.html#torch.sin) to every leaf in place and returns `self`. |
| [`TensorDictBase.cos`](generated/tensordict.TensorDictBase.cos.html#tensordict.TensorDictBase.cos)() | Returns a new tensordict with [`torch.cos()`](https://docs.pytorch.org/docs/stable/generated/torch.cos.html#torch.cos) applied to every leaf. |
| [`TensorDictBase.cos_`](generated/tensordict.TensorDictBase.cos_.html#tensordict.TensorDictBase.cos_)() | Applies [`torch.cos()`](https://docs.pytorch.org/docs/stable/generated/torch.cos.html#torch.cos) to every leaf in place and returns `self`. |
| [`TensorDictBase.tan`](generated/tensordict.TensorDictBase.tan.html#tensordict.TensorDictBase.tan)() | Returns a new tensordict with [`torch.tan()`](https://docs.pytorch.org/docs/stable/generated/torch.tan.html#torch.tan) applied to every leaf. |
| [`TensorDictBase.tan_`](generated/tensordict.TensorDictBase.tan_.html#tensordict.TensorDictBase.tan_)() | Applies [`torch.tan()`](https://docs.pytorch.org/docs/stable/generated/torch.tan.html#torch.tan) to every leaf in place and returns `self`. |
| [`TensorDictBase.asin`](generated/tensordict.TensorDictBase.asin.html#tensordict.TensorDictBase.asin)() | Returns a new tensordict with [`torch.asin()`](https://docs.pytorch.org/docs/stable/generated/torch.asin.html#torch.asin) applied to every leaf. |
| [`TensorDictBase.asin_`](generated/tensordict.TensorDictBase.asin_.html#tensordict.TensorDictBase.asin_)() | Applies [`torch.asin()`](https://docs.pytorch.org/docs/stable/generated/torch.asin.html#torch.asin) to every leaf in place and returns `self`. |
| [`TensorDictBase.acos`](generated/tensordict.TensorDictBase.acos.html#tensordict.TensorDictBase.acos)() | Returns a new tensordict with [`torch.acos()`](https://docs.pytorch.org/docs/stable/generated/torch.acos.html#torch.acos) applied to every leaf. |
| [`TensorDictBase.acos_`](generated/tensordict.TensorDictBase.acos_.html#tensordict.TensorDictBase.acos_)() | Applies [`torch.acos()`](https://docs.pytorch.org/docs/stable/generated/torch.acos.html#torch.acos) to every leaf in place and returns `self`. |
| [`TensorDictBase.atan`](generated/tensordict.TensorDictBase.atan.html#tensordict.TensorDictBase.atan)() | Returns a new tensordict with [`torch.atan()`](https://docs.pytorch.org/docs/stable/generated/torch.atan.html#torch.atan) applied to every leaf. |
| [`TensorDictBase.atan_`](generated/tensordict.TensorDictBase.atan_.html#tensordict.TensorDictBase.atan_)() | Applies [`torch.atan()`](https://docs.pytorch.org/docs/stable/generated/torch.atan.html#torch.atan) to every leaf in place and returns `self`. |
| [`TensorDictBase.sinh`](generated/tensordict.TensorDictBase.sinh.html#tensordict.TensorDictBase.sinh)() | Returns a new tensordict with [`torch.sinh()`](https://docs.pytorch.org/docs/stable/generated/torch.sinh.html#torch.sinh) applied to every leaf. |
| [`TensorDictBase.sinh_`](generated/tensordict.TensorDictBase.sinh_.html#tensordict.TensorDictBase.sinh_)() | Applies [`torch.sinh()`](https://docs.pytorch.org/docs/stable/generated/torch.sinh.html#torch.sinh) to every leaf in place and returns `self`. |
| [`TensorDictBase.cosh`](generated/tensordict.TensorDictBase.cosh.html#tensordict.TensorDictBase.cosh)() | Returns a new tensordict with [`torch.cosh()`](https://docs.pytorch.org/docs/stable/generated/torch.cosh.html#torch.cosh) applied to every leaf. |
| [`TensorDictBase.cosh_`](generated/tensordict.TensorDictBase.cosh_.html#tensordict.TensorDictBase.cosh_)() | Applies [`torch.cosh()`](https://docs.pytorch.org/docs/stable/generated/torch.cosh.html#torch.cosh) to every leaf in place and returns `self`. |
| [`TensorDictBase.tanh`](generated/tensordict.TensorDictBase.tanh.html#tensordict.TensorDictBase.tanh)() | Returns a new tensordict with [`torch.tanh()`](https://docs.pytorch.org/docs/stable/generated/torch.tanh.html#torch.tanh) applied to every leaf. |
| [`TensorDictBase.tanh_`](generated/tensordict.TensorDictBase.tanh_.html#tensordict.TensorDictBase.tanh_)() | Applies [`torch.tanh()`](https://docs.pytorch.org/docs/stable/generated/torch.tanh.html#torch.tanh) to every leaf in place and returns `self`. |
| [`TensorDictBase.sigmoid`](generated/tensordict.TensorDictBase.sigmoid.html#tensordict.TensorDictBase.sigmoid)() | Returns a new tensordict with [`torch.sigmoid()`](https://docs.pytorch.org/docs/stable/generated/torch.sigmoid.html#torch.sigmoid) applied to every leaf. |
| [`TensorDictBase.sigmoid_`](generated/tensordict.TensorDictBase.sigmoid_.html#tensordict.TensorDictBase.sigmoid_)() | Applies [`torch.sigmoid()`](https://docs.pytorch.org/docs/stable/generated/torch.sigmoid.html#torch.sigmoid) to every leaf in place and returns `self`. |
| [`TensorDictBase.erf`](generated/tensordict.TensorDictBase.erf.html#tensordict.TensorDictBase.erf)() | Returns a new tensordict with [`torch.erf()`](https://docs.pytorch.org/docs/stable/generated/torch.erf.html#torch.erf) applied to every leaf. |
| [`TensorDictBase.erf_`](generated/tensordict.TensorDictBase.erf_.html#tensordict.TensorDictBase.erf_)() | Applies [`torch.erf()`](https://docs.pytorch.org/docs/stable/generated/torch.erf.html#torch.erf) to every leaf in place and returns `self`. |
| [`TensorDictBase.erfc`](generated/tensordict.TensorDictBase.erfc.html#tensordict.TensorDictBase.erfc)() | Returns a new tensordict with [`torch.erfc()`](https://docs.pytorch.org/docs/stable/generated/torch.erfc.html#torch.erfc) applied to every leaf. |
| [`TensorDictBase.erfc_`](generated/tensordict.TensorDictBase.erfc_.html#tensordict.TensorDictBase.erfc_)() | Applies [`torch.erfc()`](https://docs.pytorch.org/docs/stable/generated/torch.erfc.html#torch.erfc) to every leaf in place and returns `self`. |
| [`TensorDictBase.lgamma`](generated/tensordict.TensorDictBase.lgamma.html#tensordict.TensorDictBase.lgamma)() | Returns a new tensordict with [`torch.lgamma()`](https://docs.pytorch.org/docs/stable/generated/torch.lgamma.html#torch.lgamma) applied to every leaf. |
| [`TensorDictBase.lgamma_`](generated/tensordict.TensorDictBase.lgamma_.html#tensordict.TensorDictBase.lgamma_)() | Applies [`torch.lgamma()`](https://docs.pytorch.org/docs/stable/generated/torch.lgamma.html#torch.lgamma) to every leaf in place and returns `self`. |
| [`TensorDictBase.ceil`](generated/tensordict.TensorDictBase.ceil.html#tensordict.TensorDictBase.ceil)() | Returns a new tensordict with [`torch.ceil()`](https://docs.pytorch.org/docs/stable/generated/torch.ceil.html#torch.ceil) applied to every leaf. |
| [`TensorDictBase.ceil_`](generated/tensordict.TensorDictBase.ceil_.html#tensordict.TensorDictBase.ceil_)() | Applies [`torch.ceil()`](https://docs.pytorch.org/docs/stable/generated/torch.ceil.html#torch.ceil) to every leaf in place and returns `self`. |
| [`TensorDictBase.floor`](generated/tensordict.TensorDictBase.floor.html#tensordict.TensorDictBase.floor)() | Returns a new tensordict with [`torch.floor()`](https://docs.pytorch.org/docs/stable/generated/torch.floor.html#torch.floor) applied to every leaf. |
| [`TensorDictBase.floor_`](generated/tensordict.TensorDictBase.floor_.html#tensordict.TensorDictBase.floor_)() | Applies [`torch.floor()`](https://docs.pytorch.org/docs/stable/generated/torch.floor.html#torch.floor) to every leaf in place and returns `self`. |
| [`TensorDictBase.round`](generated/tensordict.TensorDictBase.round.html#tensordict.TensorDictBase.round)() | Returns a new tensordict with [`torch.round()`](https://docs.pytorch.org/docs/stable/generated/torch.round.html#torch.round) applied to every leaf. |
| [`TensorDictBase.round_`](generated/tensordict.TensorDictBase.round_.html#tensordict.TensorDictBase.round_)() | Applies [`torch.round()`](https://docs.pytorch.org/docs/stable/generated/torch.round.html#torch.round) to every leaf in place and returns `self`. |
| [`TensorDictBase.trunc`](generated/tensordict.TensorDictBase.trunc.html#tensordict.TensorDictBase.trunc)() | Returns a new tensordict with [`torch.trunc()`](https://docs.pytorch.org/docs/stable/generated/torch.trunc.html#torch.trunc) applied to every leaf. |
| [`TensorDictBase.trunc_`](generated/tensordict.TensorDictBase.trunc_.html#tensordict.TensorDictBase.trunc_)() | Applies [`torch.trunc()`](https://docs.pytorch.org/docs/stable/generated/torch.trunc.html#torch.trunc) to every leaf in place and returns `self`. |
| [`TensorDictBase.frac`](generated/tensordict.TensorDictBase.frac.html#tensordict.TensorDictBase.frac)() | Returns a new tensordict with [`torch.frac()`](https://docs.pytorch.org/docs/stable/generated/torch.frac.html#torch.frac) applied to every leaf. |
| [`TensorDictBase.frac_`](generated/tensordict.TensorDictBase.frac_.html#tensordict.TensorDictBase.frac_)() | Applies [`torch.frac()`](https://docs.pytorch.org/docs/stable/generated/torch.frac.html#torch.frac) to every leaf in place and returns `self`. |
| [`TensorDictBase.sign`](generated/tensordict.TensorDictBase.sign.html#tensordict.TensorDictBase.sign)() | Returns a new tensordict with [`torch.sign()`](https://docs.pytorch.org/docs/stable/generated/torch.sign.html#torch.sign) applied to every leaf. |
| [`TensorDictBase.sign_`](generated/tensordict.TensorDictBase.sign_.html#tensordict.TensorDictBase.sign_)() | Applies [`torch.sign()`](https://docs.pytorch.org/docs/stable/generated/torch.sign.html#torch.sign) to every leaf in place and returns `self`. |
| [`TensorDictBase.softmax`](generated/tensordict.TensorDictBase.softmax.html#tensordict.TensorDictBase.softmax)(dim[, dtype]) | Returns a new tensordict with [`torch.softmax()`](https://docs.pytorch.org/docs/stable/generated/torch.softmax.html#torch.softmax) applied to every leaf along a batch dimension. |

Comparisons and logical operations

| [`TensorDictBase.__eq__`](generated/tensordict.TensorDictBase.__eq__.html#tensordict.TensorDictBase.__eq__)(other) | Compares the tensordict with `other` leaf by leaf with `==`. |
| --- | --- |
| [`TensorDictBase.__ne__`](generated/tensordict.TensorDictBase.__ne__.html#tensordict.TensorDictBase.__ne__)(other) | Compares the tensordict with `other` leaf by leaf with `!=`. |
| [`TensorDictBase.__lt__`](generated/tensordict.TensorDictBase.__lt__.html#tensordict.TensorDictBase.__lt__)(other) | Compares the tensordict with `other` leaf by leaf with `<`. |
| [`TensorDictBase.__le__`](generated/tensordict.TensorDictBase.__le__.html#tensordict.TensorDictBase.__le__)(other) | Compares the tensordict with `other` leaf by leaf with `<=`. |
| [`TensorDictBase.__gt__`](generated/tensordict.TensorDictBase.__gt__.html#tensordict.TensorDictBase.__gt__)(other) | Compares the tensordict with `other` leaf by leaf with `>`. |
| [`TensorDictBase.__ge__`](generated/tensordict.TensorDictBase.__ge__.html#tensordict.TensorDictBase.__ge__)(other) | Compares the tensordict with `other` leaf by leaf with `>=`. |
| [`TensorDictBase.__and__`](generated/tensordict.TensorDictBase.__and__.html#tensordict.TensorDictBase.__and__)(other) | Computes `self & other` leaf by leaf and returns a new tensordict. |
| [`TensorDictBase.__or__`](generated/tensordict.TensorDictBase.__or__.html#tensordict.TensorDictBase.__or__)(other) | Computes `self \| other` leaf by leaf and returns a new tensordict. |
| [`TensorDictBase.__xor__`](generated/tensordict.TensorDictBase.__xor__.html#tensordict.TensorDictBase.__xor__)(other) | Computes `self ^ other` leaf by leaf and returns a new tensordict. |
| [`TensorDictBase.__invert__`](generated/tensordict.TensorDictBase.__invert__.html#tensordict.TensorDictBase.__invert__)() | Returns `~td`, a new tensordict with `~` applied to every leaf. |
| [`TensorDictBase.bitwise_and`](generated/tensordict.TensorDictBase.bitwise_and.html#tensordict.TensorDictBase.bitwise_and)(other, *[, default]) | Returns a new tensordict with the bitwise AND of every leaf and `other`. |
| [`TensorDictBase.logical_and`](generated/tensordict.TensorDictBase.logical_and.html#tensordict.TensorDictBase.logical_and)(other, *[, default]) | Returns a new tensordict with the logical AND of every leaf and `other`. |
| [`TensorDictBase.isfinite`](generated/tensordict.TensorDictBase.isfinite.html#tensordict.TensorDictBase.isfinite)() | Returns a new tensordict with [`torch.isfinite()`](https://docs.pytorch.org/docs/stable/generated/torch.isfinite.html#torch.isfinite) applied to every leaf. |
| [`TensorDictBase.isnan`](generated/tensordict.TensorDictBase.isnan.html#tensordict.TensorDictBase.isnan)() | Returns a new tensordict with [`torch.isnan()`](https://docs.pytorch.org/docs/stable/generated/torch.isnan.html#torch.isnan) applied to every leaf. |
| [`TensorDictBase.isneginf`](generated/tensordict.TensorDictBase.isneginf.html#tensordict.TensorDictBase.isneginf)() | Returns a new tensordict with [`torch.isneginf()`](https://docs.pytorch.org/docs/stable/generated/torch.isneginf.html#torch.isneginf) applied to every leaf. |
| [`TensorDictBase.isposinf`](generated/tensordict.TensorDictBase.isposinf.html#tensordict.TensorDictBase.isposinf)() | Returns a new tensordict with [`torch.isposinf()`](https://docs.pytorch.org/docs/stable/generated/torch.isposinf.html#torch.isposinf) applied to every leaf. |
| [`TensorDictBase.isreal`](generated/tensordict.TensorDictBase.isreal.html#tensordict.TensorDictBase.isreal)() | Returns a new tensordict with [`torch.isreal()`](https://docs.pytorch.org/docs/stable/generated/torch.isreal.html#torch.isreal) applied to every leaf. |

### Reductions

Reduce each entry over all its elements, or along batch dimensions with
`dim`. With `reduce=True`, most reductions return one tensor for the
whole tensordict. See [Arithmetic, apply, reductions, and gradients](../concepts/operations.html#operations).

| [`TensorDictBase.all`](generated/tensordict.TensorDictBase.all.html#tensordict.TensorDictBase.all)([dim]) | Returns whether every element of every leaf is `True` (nonzero). |
| --- | --- |
| [`TensorDictBase.any`](generated/tensordict.TensorDictBase.any.html#tensordict.TensorDictBase.any)([dim]) | Returns whether any element of any leaf is `True` (nonzero). |
| [`TensorDictBase.amax`](generated/tensordict.TensorDictBase.amax.html#tensordict.TensorDictBase.amax)([dim, keepdim, reduce]) | Returns the maximum of every leaf, over all its elements or along a batch dimension. |
| [`TensorDictBase.amin`](generated/tensordict.TensorDictBase.amin.html#tensordict.TensorDictBase.amin)([dim, keepdim, reduce]) | Returns the minimum of every leaf, over all its elements or along a batch dimension. |
| [`TensorDictBase.max`](generated/tensordict.TensorDictBase.max.html#tensordict.TensorDictBase.max)([dim, keepdim, reduce, ...]) | Returns the maximum of every leaf, over all its elements or along a batch dimension. |
| [`TensorDictBase.min`](generated/tensordict.TensorDictBase.min.html#tensordict.TensorDictBase.min)([dim, keepdim, reduce, ...]) | Returns the minimum of every leaf, over all its elements or along a batch dimension. |
| [`TensorDictBase.cummax`](generated/tensordict.TensorDictBase.cummax.html#tensordict.TensorDictBase.cummax)(dim, *[, reduce, ...]) | Returns the cumulative maximum of every leaf along a batch dimension. |
| [`TensorDictBase.cummin`](generated/tensordict.TensorDictBase.cummin.html#tensordict.TensorDictBase.cummin)(dim, *[, reduce, ...]) | Returns the cumulative minimum of every leaf along a batch dimension. |
| [`TensorDictBase.mean`](generated/tensordict.TensorDictBase.mean.html#tensordict.TensorDictBase.mean)([dim, keepdim, dtype, ...]) | Returns the mean of every leaf, over all its elements or along batch dimensions. |
| [`TensorDictBase.nanmean`](generated/tensordict.TensorDictBase.nanmean.html#tensordict.TensorDictBase.nanmean)([dim, keepdim, ...]) | Returns the mean of the non-NaN elements of every leaf. |
| [`TensorDictBase.sum`](generated/tensordict.TensorDictBase.sum.html#tensordict.TensorDictBase.sum)([dim, keepdim, dtype, ...]) | Returns the sum of every leaf, over all its elements or along batch dimensions. |
| [`TensorDictBase.nansum`](generated/tensordict.TensorDictBase.nansum.html#tensordict.TensorDictBase.nansum)([dim, keepdim, dtype, ...]) | Returns the sum of the non-NaN elements of every leaf. |
| [`TensorDictBase.prod`](generated/tensordict.TensorDictBase.prod.html#tensordict.TensorDictBase.prod)([dim, keepdim, dtype, ...]) | Returns the product of every leaf, over all its elements or along a batch dimension. |
| [`TensorDictBase.std`](generated/tensordict.TensorDictBase.std.html#tensordict.TensorDictBase.std)([dim, keepdim, ...]) | Returns the standard deviation of every leaf, over all its elements or along batch dimensions. |
| [`TensorDictBase.var`](generated/tensordict.TensorDictBase.var.html#tensordict.TensorDictBase.var)([dim, keepdim, ...]) | Returns the variance of every leaf, over all its elements or along batch dimensions. |
| [`TensorDictBase.quantile`](generated/tensordict.TensorDictBase.quantile.html#tensordict.TensorDictBase.quantile)(q[, dim, keepdim, ...]) | Returns the `q`-th quantile of every leaf, over all its elements or along a batch dimension. |
| [`TensorDictBase.logsumexp`](generated/tensordict.TensorDictBase.logsumexp.html#tensordict.TensorDictBase.logsumexp)([dim, keepdim, out]) | Returns the log of the summed exponentials of every leaf along batch dimensions. |
| [`TensorDictBase.norm`](generated/tensordict.TensorDictBase.norm.html#tensordict.TensorDictBase.norm)(*[, out, dtype, ...]) | Returns a tensordict with the norm of each leaf. |

### Apply, modules, and gradients

Apply a function to every entry, and map a function over the batch. Put
the parameters of a module in a tensordict and back, and compute
gradients.
See [Arithmetic, apply, reductions, and gradients](../concepts/operations.html#operations), [Modules: TensorDictModule and tensordict.nn](../modules.html#modules-guide), and [torch.compile, torch.func, torch.export, and CUDA graphs](../compile.html#compile).

Methods

| [`TensorDictBase.apply`](generated/tensordict.TensorDictBase.apply.html#tensordict.TensorDictBase.apply)(fn, *others[, ...]) | Applies a callable to every leaf and returns a new tensordict with the results. |
| --- | --- |
| [`TensorDictBase.apply_`](generated/tensordict.TensorDictBase.apply_.html#tensordict.TensorDictBase.apply_)(fn, *others, **kwargs) | Applies a callable to every leaf and writes the results into the existing leaves. |
| [`TensorDictBase.named_apply`](generated/tensordict.TensorDictBase.named_apply.html#tensordict.TensorDictBase.named_apply)(fn, *others[, ...]) | Applies a callable to every leaf and its key, and returns a new tensordict with the results. |
| [`TensorDictBase.map`](generated/tensordict.TensorDictBase.map.html#tensordict.TensorDictBase.map)(fn[, dim, num_workers, ...]) | Applies a function to chunks of the tensordict in worker processes, and gathers the results. |
| [`TensorDictBase.map_iter`](generated/tensordict.TensorDictBase.map_iter.html#tensordict.TensorDictBase.map_iter)(fn[, dim, ...]) | Applies a function to chunks of the tensordict in worker processes, and yields the results. |
| [`TensorDictBase.from_module`](generated/tensordict.TensorDictBase.from_module.html#tensordict.TensorDictBase.from_module)(module[, ...]) | Returns a tensordict of the parameters and buffers of a module. |
| [`TensorDictBase.from_modules`](generated/tensordict.TensorDictBase.from_modules.html#tensordict.TensorDictBase.from_modules)(*modules[, ...]) | Stacks the parameters and buffers of several modules in one tensordict. |
| [`TensorDictBase.to_module`](generated/tensordict.TensorDictBase.to_module.html#tensordict.TensorDictBase.to_module)(module, *[, ...]) | Writes the content of the tensordict into the parameters and buffers of a module, recursively. |
| [`TensorDictBase.backward`](generated/tensordict.TensorDictBase.backward.html#tensordict.TensorDictBase.backward)([gradient, ...]) | Computes the gradients of the leaves of the tensordict that require gradients. |
| [`TensorDictBase.requires_grad`](generated/tensordict.TensorDictBase.requires_grad.html#tensordict.TensorDictBase.requires_grad) | `True` if any leaf of the tensordict, nested ones included, requires gradients. |
| [`TensorDictBase.requires_grad_`](generated/tensordict.TensorDictBase.requires_grad_.html#tensordict.TensorDictBase.requires_grad_)([requires_grad]) | Sets `requires_grad` on every tensor leaf in place, and returns the tensordict. |
| [`TensorDictBase.detach`](generated/tensordict.TensorDictBase.detach.html#tensordict.TensorDictBase.detach)() | Returns a new tensordict with every tensor leaf detached from the autograd graph. |
| [`TensorDictBase.detach_`](generated/tensordict.TensorDictBase.detach_.html#tensordict.TensorDictBase.detach_)() | Detaches every tensor leaf from the autograd graph in place, and returns the tensordict. |
| [`TensorDictBase.grad`](generated/tensordict.TensorDictBase.grad.html#tensordict.TensorDictBase.grad) | A tensordict of the `.grad` of every leaf, or `None` if no leaf has one. |
| [`TensorDictBase.data`](generated/tensordict.TensorDictBase.data.html#tensordict.TensorDictBase.data) | the same storage, without autograd. |
| [`TensorDictBase.zero_grad`](generated/tensordict.TensorDictBase.zero_grad.html#tensordict.TensorDictBase.zero_grad)([set_to_none]) | Resets the `.grad` of every leaf, nested ones included, and returns `self`. |

Functions

| [`from_module`](generated/tensordict.from_module.html#tensordict.from_module)(module[, as_module, lock, ...]) | Returns a locked tensordict that holds the parameters and buffers of a module. |
| --- | --- |
| [`from_modules`](generated/tensordict.from_modules.html#tensordict.from_modules)(*modules[, as_module, lock, ...]) | Stacks the parameters and buffers of several modules in one tensordict. |
| [`default_is_leaf`](generated/tensordict.default_is_leaf.html#tensordict.default_is_leaf)(cls) | Returns `True` if a type is a leaf: anything but a tensordict or a tensorclass. |
| [`is_leaf_nontensor`](generated/tensordict.is_leaf_nontensor.html#tensordict.is_leaf_nontensor)(cls) | Returns `True` if a type is a tensor or non-tensor data. |

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
| [`TensorDictBase.memmap`](generated/tensordict.TensorDictBase.memmap.html#tensordict.TensorDictBase.memmap)([prefix, ...]) | Writes every leaf to a memory-mapped file and returns a new tensordict of them. |
| [`TensorDictBase.memmap_`](generated/tensordict.TensorDictBase.memmap_.html#tensordict.TensorDictBase.memmap_)([prefix, ...]) | Writes every leaf to a memory-mapped file in place, and returns `self`. |
| [`TensorDictBase.memmap_like`](generated/tensordict.TensorDictBase.memmap_like.html#tensordict.TensorDictBase.memmap_like)([prefix, ...]) | Creates a memory-mapped tensordict with the same keys, shapes and dtypes, without copying the values. |
| [`TensorDictBase.memmap_refresh_`](generated/tensordict.TensorDictBase.memmap_refresh_.html#tensordict.TensorDictBase.memmap_refresh_)(*[, allow_pickle]) | Reloads the tensordict from its [`saved_path`](generated/tensordict.TensorDictBase.saved_path.html#tensordict.TensorDictBase.saved_path), and returns `self`. |
| [`TensorDictBase.load_memmap`](generated/tensordict.TensorDictBase.load_memmap.html#tensordict.TensorDictBase.load_memmap)(prefix[, device, ...]) | Loads a tensordict that [`memmap()`](generated/tensordict.TensorDictBase.memmap.html#tensordict.TensorDictBase.memmap) or [`save()`](generated/tensordict.TensorDictBase.save.html#tensordict.TensorDictBase.save) wrote, from a directory or an archive. |
| [`TensorDictBase.load_memmap_`](generated/tensordict.TensorDictBase.load_memmap_.html#tensordict.TensorDictBase.load_memmap_)(prefix[, ...]) | Loads a saved tensordict into the tensordict, and returns `self`. |
| [`TensorDictBase.make_memmap`](generated/tensordict.TensorDictBase.make_memmap.html#tensordict.TensorDictBase.make_memmap)(key, shape, *[, ...]) | Adds an uninitialized memory-mapped tensor of shape `shape` at `key`, and returns it. |
| [`TensorDictBase.make_memmap_from_storage`](generated/tensordict.TensorDictBase.make_memmap_from_storage.html#tensordict.TensorDictBase.make_memmap_from_storage)(key, ...) | Adds a memory-mapped tensor at `key` that reads its data from `storage`, and returns it. |
| [`TensorDictBase.make_memmap_from_tensor`](generated/tensordict.TensorDictBase.make_memmap_from_tensor.html#tensordict.TensorDictBase.make_memmap_from_tensor)(key, ...) | Adds a memory-mapped copy of `tensor` at `key`, and returns it. |
| [`TensorDictBase.is_memmap`](generated/tensordict.TensorDictBase.is_memmap.html#tensordict.TensorDictBase.is_memmap)() | Whether the tensordict is memory-mapped. |
| [`TensorDictBase.saved_path`](generated/tensordict.TensorDictBase.saved_path.html#tensordict.TensorDictBase.saved_path) | The directory where a memory-mapped tensordict stores its files, as a [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path). |
| [`TensorDictBase.state_dict`](generated/tensordict.TensorDictBase.state_dict.html#tensordict.TensorDictBase.state_dict)([destination, ...]) | Returns the leaves of the tensordict in an `OrderedDict`, as [`torch.nn.Module.state_dict()`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.state_dict) does. |
| [`TensorDictBase.load_state_dict`](generated/tensordict.TensorDictBase.load_state_dict.html#tensordict.TensorDictBase.load_state_dict)(state_dict[, ...]) | Loads a state dict, as [`state_dict()`](generated/tensordict.TensorClass.html#tensordict.TensorClass.state_dict) returns it, into the tensordict, and returns `self`. |
| [`TensorDictBase.from_h5`](generated/tensordict.TensorDictBase.from_h5.html#tensordict.TensorDictBase.from_h5)(filename, *[, mode, ...]) | Opens an HDF5 file as a [`PersistentTensorDict`](generated/tensordict.PersistentTensorDict.html#tensordict.PersistentTensorDict). |
| [`TensorDictBase.to_h5`](generated/tensordict.TensorDictBase.to_h5.html#tensordict.TensorDictBase.to_h5)(filename, **kwargs) | Writes the tensordict to a new HDF5 file and returns a tensordict backed by it. |
| [`TensorDictBase.from_zarr`](generated/tensordict.TensorDictBase.from_zarr.html#tensordict.TensorDictBase.from_zarr)(filename, *[, ...]) | Opens a zarr store as a [`PersistentTensorDict`](generated/tensordict.PersistentTensorDict.html#tensordict.PersistentTensorDict). |
| [`TensorDictBase.to_zarr`](generated/tensordict.TensorDictBase.to_zarr.html#tensordict.TensorDictBase.to_zarr)(filename, *[, ...]) | Writes the tensordict to a new zarr store and returns a tensordict backed by it. |
| [`TensorDictBase.from_csv`](generated/tensordict.TensorDictBase.from_csv.html#tensordict.TensorDictBase.from_csv)(path, *[, ...]) | Reads a CSV file into a tensordict, with one entry per column. |
| [`TensorDictBase.to_csv`](generated/tensordict.TensorDictBase.to_csv.html#tensordict.TensorDictBase.to_csv)(path, *[, separator]) | Writes the tensordict to a CSV file, with one column per leaf. |
| [`TensorDictBase.from_json`](generated/tensordict.TensorDictBase.from_json.html#tensordict.TensorDictBase.from_json)(path, *[, ...]) | Reads a JSON file of records into a tensordict, with one entry per field. |
| [`TensorDictBase.to_json`](generated/tensordict.TensorDictBase.to_json.html#tensordict.TensorDictBase.to_json)(path, *[, separator, ...]) | Writes the tensordict to a JSON file, as an array of records. |
| [`TensorDictBase.from_parquet`](generated/tensordict.TensorDictBase.from_parquet.html#tensordict.TensorDictBase.from_parquet)(path, *[, ...]) | Reads a Parquet file into a tensordict, with one entry per column. |
| [`TensorDictBase.to_parquet`](generated/tensordict.TensorDictBase.to_parquet.html#tensordict.TensorDictBase.to_parquet)(path, *[, separator]) | Writes the tensordict to a Parquet file, with one column per leaf. |
| [`TensorDictBase.to_mds`](generated/tensordict.TensorDictBase.to_mds.html#tensordict.TensorDictBase.to_mds)(*[, columns, ...]) | Writes the tensordict to a MosaicML Streaming (MDS) dataset. |
| [`TensorDictBase.to_store`](generated/tensordict.TensorDictBase.to_store.html#tensordict.TensorDictBase.to_store)(*[, backend, host, ...]) | Uploads the tensordict to a key-value server and returns a tensordict backed by it. |

Functions

| [`save`](generated/tensordict.save.html#tensordict.save)(data[, prefix, copy_existing, ...]) | Saves a tensordict to disk, and returns a memory-mapped copy of it. |
| --- | --- |
| [`load`](generated/tensordict.load.html#tensordict.load)(prefix[, device, non_blocking, out, ...]) | Loads a tensordict that [`save()`](generated/tensordict.save.html#tensordict.save) wrote. |
| [`memmap`](generated/tensordict.memmap.html#tensordict.memmap)(data[, prefix, copy_existing, ...]) | Writes every leaf to a memory-mapped file and returns a new tensordict of them. |
| [`load_memmap`](generated/tensordict.load_memmap.html#tensordict.load_memmap)(prefix[, device, non_blocking, ...]) | Loads a tensordict that [`memmap()`](generated/tensordict.memmap.html#tensordict.memmap) or [`save()`](generated/tensordict.save.html#tensordict.save) wrote. |
| [`pack_memmap`](generated/tensordict.pack_memmap.html#tensordict.pack_memmap)(prefix, archive_path, *[, ...]) | Packs a memory-mapped tensordict directory into a single-file zip archive. |
| [`unpack_memmap`](generated/tensordict.unpack_memmap.html#tensordict.unpack_memmap)(archive_path, prefix) | Extracts a memory-mapped tensordict archive into a directory. |
| [`is_memmap_archive`](generated/tensordict.is_memmap_archive.html#tensordict.is_memmap_archive)(path) | Whether `path` is a memory-mapped tensordict archive. |
| [`refresh_archive_checksums`](generated/tensordict.refresh_archive_checksums.html#tensordict.refresh_archive_checksums)(archive_path) | Recomputes the CRC-32 checksums of a memory-mapped archive after in-place writes. |
| [`from_h5`](generated/tensordict.from_h5.html#tensordict.from_h5)(h5_file, *[, mode, auto_batch_size, ...]) | Opens an HDF5 file as a [`PersistentTensorDict`](generated/tensordict.PersistentTensorDict.html#tensordict.PersistentTensorDict). |
| [`from_zarr`](generated/tensordict.from_zarr.html#tensordict.from_zarr)(zarr_file, *[, mode, ...]) | Opens a zarr store as a [`PersistentTensorDict`](generated/tensordict.PersistentTensorDict.html#tensordict.PersistentTensorDict). |
| [`from_csv`](generated/tensordict.from_csv.html#tensordict.from_csv)(path, *[, auto_batch_size, ...]) | Reads a CSV file into a tensordict, with one entry per column. |
| [`from_json`](generated/tensordict.from_json.html#tensordict.from_json)(path, *[, auto_batch_size, ...]) | Reads a JSON file of records into a tensordict, with one entry per field. |
| [`from_parquet`](generated/tensordict.from_parquet.html#tensordict.from_parquet)(path, *[, auto_batch_size, ...]) | Reads a Parquet file into a tensordict, with one entry per column. |

### Locking

A locked tensordict raises an error if you add, replace, rename, or delete
an entry. In-place writes with `set_` and `update_` still work, also for
non-tensor entries. `unlock_` also works in a `with` block: the
tensordict is locked again at the end of the block. See [Keys, nested keys, and locking](../concepts/keys.html#keys).

| [`TensorDictBase.lock_`](generated/tensordict.TensorDictBase.lock_.html#tensordict.TensorDictBase.lock_)() | Locks the tensordict and its nested tensordicts, and returns it. |
| --- | --- |
| [`TensorDictBase.unlock_`](generated/tensordict.TensorDictBase.unlock_.html#tensordict.TensorDictBase.unlock_)() | Unlocks the tensordict and its nested tensordicts, and returns it. |
| [`TensorDictBase.is_locked`](generated/tensordict.TensorDictBase.is_locked.html#tensordict.TensorDictBase.is_locked) | `True` if the tensordict is locked. |

### Distributed and remote

Send and receive tensordicts between processes with
[`torch.distributed`](https://docs.pytorch.org/docs/stable/distributed.html#module-torch.distributed), run collective operations on them, and stream
them over UCXX (see [Key-value stores and UCXX](storage.html#api-storage)). See [Distributed tensordicts](../distributed.html#distributed).

| [`TensorDictBase.send`](generated/tensordict.TensorDictBase.send.html#tensordict.TensorDictBase.send)([dst, group, group_dst, ...]) | Sends the leaves of the tensordict to another rank, with [`torch.distributed`](https://docs.pytorch.org/docs/stable/distributed.html#module-torch.distributed). |
| --- | --- |
| [`TensorDictBase.recv`](generated/tensordict.TensorDictBase.recv.html#tensordict.TensorDictBase.recv)([src, group, group_src, ...]) | Receives the leaves that [`send()`](generated/tensordict.TensorDictBase.send.html#tensordict.TensorDictBase.send) sends, and writes them into the tensordict in place. |
| [`TensorDictBase.isend`](generated/tensordict.TensorDictBase.isend.html#tensordict.TensorDictBase.isend)([dst, group, ...]) | Sends the leaves of the tensordict to another rank, one asynchronous transfer per leaf. |
| [`TensorDictBase.irecv`](generated/tensordict.TensorDictBase.irecv.html#tensordict.TensorDictBase.irecv)([src, group, ...]) | Receives the leaves that [`isend()`](generated/tensordict.TensorDictBase.isend.html#tensordict.TensorDictBase.isend) or [`send()`](generated/tensordict.TensorDictBase.send.html#tensordict.TensorDictBase.send) sends, one asynchronous transfer per leaf. |
| [`TensorDictBase.all_reduce`](generated/tensordict.TensorDictBase.all_reduce.html#tensordict.TensorDictBase.all_reduce)([op, group, async_op]) | Reduces each leaf over all the ranks of the group, in place, so that every rank holds the result. |
| [`TensorDictBase.reduce`](generated/tensordict.TensorDictBase.reduce.html#tensordict.TensorDictBase.reduce)(dst[, op, async_op, ...]) | Reduces each leaf over all the ranks of the group, into the tensordict of rank `dst`. |
| [`TensorDictBase.broadcast`](generated/tensordict.TensorDictBase.broadcast.html#tensordict.TensorDictBase.broadcast)(src, *[, group, ...]) | Sends the tensordict of rank `src` to every rank of the group, and returns it on each rank. |
| [`TensorDictBase.scatter`](generated/tensordict.TensorDictBase.scatter.html#tensordict.TensorDictBase.scatter)(src[, tensordicts, ...]) | Scatters a list of tensordicts from `src` to all ranks. |
| [`TensorDictBase.all_gather`](generated/tensordict.TensorDictBase.all_gather.html#tensordict.TensorDictBase.all_gather)(*[, group]) | Returns a list with the tensordict of each rank of the group, on every rank. |
| [`TensorDictBase.gather_and_stack`](generated/tensordict.TensorDictBase.gather_and_stack.html#tensordict.TensorDictBase.gather_and_stack)(dst[, group]) | Stacks the tensordicts of the other ranks into the tensordict of rank `dst`, in place. |
| [`TensorDictBase.init_remote`](generated/tensordict.TensorDictBase.init_remote.html#tensordict.TensorDictBase.init_remote)([dst, group, ...]) | Sends the structure and the content of the tensordict to a rank that does not know them. |
| [`TensorDictBase.from_remote_init`](generated/tensordict.TensorDictBase.from_remote_init.html#tensordict.TensorDictBase.from_remote_init)(src[, ...]) | Receives a tensordict that [`init_remote()`](generated/tensordict.TensorDictBase.init_remote.html#tensordict.TensorDictBase.init_remote) sends, and returns it. |
| [`TensorDictBase.asend`](generated/tensordict.TensorDictBase.asend.html#tensordict.TensorDictBase.asend)(dst) | Sends the tensordict through a UCXX pipe, asynchronously. |
| [`TensorDictBase.arecv`](generated/tensordict.TensorDictBase.arecv.html#tensordict.TensorDictBase.arecv)(src, *[, device]) | Receives a tensordict from a UCXX pipe, into the tensordict if possible, asynchronously. |

### Deprecated methods

These methods still work, and each call raises a `DeprecationWarning`.
Their pages name the replacement and the release that removes them.

| [`TensorDictBase.load_`](generated/tensordict.TensorDictBase.load_.html#tensordict.TensorDictBase.load_)(prefix, *args, **kwargs) | Loads a saved tensordict into the tensordict. |
| --- | --- |
| [`TensorDictBase.pin_memory_`](generated/tensordict.TensorDictBase.pin_memory_.html#tensordict.TensorDictBase.pin_memory_)([num_threads]) | Pins the tensor leaves of the tensordict and returns `self`. |

## Other functions

### Type checks and testing

Test whether an object is a tensordict, a tensorclass, or non-tensor data,
and compare two tensordicts in tests.

| [`is_tensor_collection`](generated/tensordict.is_tensor_collection.html#tensordict.is_tensor_collection)(datatype) | Returns `True` if an object or a type is a tensordict or a tensorclass. |
| --- | --- |
| [`is_tensorclass`](generated/tensordict.is_tensorclass.html#tensordict.is_tensorclass)(obj) | Returns `True` if `obj` is a tensorclass type or an instance of one. |
| [`is_non_tensor`](generated/tensordict.is_non_tensor.html#tensordict.is_non_tensor)(data) | Returns `True` if `data` is non-tensor data or a stack of non-tensor data. |
| [`assert_close`](generated/tensordict.assert_close.html#tensordict.assert_close)(actual, expected[, rtol, atol, ...]) | Asserts that two tensordicts have the same keys and close values. |
| [`assert_allclose_td`](generated/tensordict.assert_allclose_td.html#tensordict.assert_allclose_td)(actual, expected[, rtol, ...]) | Asserts that two tensordicts have the same keys and close values. |

### Printing

Choose what the representation of a tensordict shows, and rebuild a
tensordict from its printed form. See [Printing and display](../printing.html#printing).

| [`set_printoptions`](generated/tensordict.set_printoptions.html#tensordict.set_printoptions)(*[, verbose]) | Sets the options that control what the repr of a tensordict shows. |
| --- | --- |
| [`get_printoptions`](generated/tensordict.get_printoptions.html#tensordict.get_printoptions)() | Returns the current print options of tensordicts as a dict. |
| [`parse_tensor_dict_string`](generated/tensordict.parse_tensor_dict_string.html#tensordict.parse_tensor_dict_string)(s) | Builds a [`TensorDict`](generated/tensordict.TensorDict.html#tensordict.TensorDict) from its printed form. |

### Global settings

These functions read or change a behavior of tensordict, for the whole
process or inside a `with` block. `set_list_to_stack(False)` and
`set_capture_non_tensor_stack(True)` are deprecated, and their pages say
what to use instead.

| [`list_to_stack`](generated/tensordict.list_to_stack.html#tensordict.list_to_stack)([allow_none]) | Returns whether assigning a list to a key stores a [`NonTensorStack`](generated/tensordict.NonTensorStack.html#tensordict.NonTensorStack). |
| --- | --- |
| [`set_list_to_stack`](generated/tensordict.set_list_to_stack.html#tensordict.set_list_to_stack)(mode) | Sets whether assigning a list to a key stores a stack of non-tensor data. |
| [`capture_non_tensor_stack`](generated/tensordict.capture_non_tensor_stack.html#tensordict.capture_non_tensor_stack)([allow_none]) | Returns whether stacking identical non-tensor data gives one value instead of a stack. |
| [`set_capture_non_tensor_stack`](generated/tensordict.set_capture_non_tensor_stack.html#tensordict.set_capture_non_tensor_stack)(mode) | Sets whether stacking identical non-tensor data gives one value instead of a stack. |

### Deprecated functions

These functions still work. Their pages name the replacement and the
release that removes them.

| [`get_defaults_to_none`](generated/tensordict.get_defaults_to_none.html#tensordict.get_defaults_to_none)([set_to_none]) | Returns whether [`get()`](generated/tensordict.TensorDictBase.get.html#tensordict.TensorDictBase.get) returns `None` for a missing key. |
| --- | --- |
| [`set_get_defaults_to_none`](generated/tensordict.set_get_defaults_to_none.html#tensordict.set_get_defaults_to_none)([set_to_none]) | Sets whether [`get()`](generated/tensordict.TensorDictBase.get.html#tensordict.TensorDictBase.get) returns `None` or raises for a missing key. |
| [`lazy_legacy`](generated/tensordict.lazy_legacy.html#tensordict.lazy_legacy)([allow_none]) | Returns whether the legacy lazy mode is on. |
| [`set_lazy_legacy`](generated/tensordict.set_lazy_legacy.html#tensordict.set_lazy_legacy)(mode) | Turns the legacy lazy mode on or off. |

## Package constants

tensordict.KEY_SEPARATOR

`"."`, the default `separator` of
[`flatten_keys()`](generated/tensordict.TensorDictBase.flatten_keys.html#tensordict.TensorDictBase.flatten_keys) and
[`unflatten_keys()`](generated/tensordict.TensorDictBase.unflatten_keys.html#tensordict.TensorDictBase.unflatten_keys), for code that joins or
splits nested keys the same way, such as
`tuple(name.split(KEY_SEPARATOR))`. The names of
[`state_dict()`](generated/tensordict.TensorDictBase.state_dict.html#tensordict.TensorDictBase.state_dict) also use `"."`, the
`torch.nn.Module` convention, independently of this constant.
Reassigning it has no effect: the defaults are bound when the methods are
defined. It can also be imported from `tensordict.utils`.

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