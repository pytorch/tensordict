# Architecture

A map of the `tensordict` package for contributors and agents. The house
rules are in [`AGENTS.md`](AGENTS.md), and setup and the PR process are in
[`CONTRIBUTING.md`](CONTRIBUTING.md).

## Modules

Paths in this section and the next are relative to `tensordict/`. Core
containers:

- `_tensorcollection.py`: `TensorCollection`, the common base of
  `TensorDictBase` and `TensorClass`. Its method signatures are in
  `_tensorcollection.pyi`.
- `base.py`: `TensorDictBase`: its abstract methods, which each backend
  implements, its core (keys, `get`, `set`, `update`, `apply`, locking,
  names and its properties), the methods that cannot move to a mixin (see
  "Adding an operation"), and the helpers that the rest of the package
  imports from `tensordict.base`.
- `_base/`: most of the other methods of `TensorDictBase`, in mixins that
  `TensorDictBase` inherits from, one module per area of the API:
  `pointwise.py` (arithmetic, comparisons and elementwise math),
  `reductions.py`, `shape.py` (views, reshaping, stacking, splitting and
  padding), `device.py` (device and dtype conversions, pinning, sharing and
  memory size), `serialization.py` (tensordict's own storage: memmap, saving
  and loading, consolidation and state dicts), `convert.py` (conversions to
  and from other containers and file formats, such as dicts, pytrees, H5,
  Zarr, pandas and `nn.Module` parameters) and `distributed.py`. These
  modules import `base.py`'s helpers, so `base.py` imports them after its
  helpers and before the class. `unsupported.py` holds the mixins of the
  operations that some backends cannot support (see "The TensorDictBase
  contract"). `_base/factories.py` holds the module-level
  constructors (`from_dict`, `from_h5`, ...) that `base.py` re-exports.
- `_td.py`: `TensorDict`, the dense implementation, `_SubTensorDict`, a
  view on an index of another tensordict, and `_StateFromSource`, the mixin
  through which a view reports the state of its source.
- `_td_functions.py`: the module-level functions `stack`, `lazy_stack`,
  `cat`, `save`, `load`, `memmap`, `load_memmap`, `from_module`,
  `from_pytree` and others. `_td.py` re-exports them.
- `_base/factories.py`: the `from_*` constructors (`from_any`, `from_dict`,
  `from_h5`, `from_zarr`, `from_pandas`, ...). `base.py` re-exports them.
- `functional.py`: `pad`, `pad_sequence`, `merge_tensordicts`,
  `dense_stack_tds` and `make_tensordict`.
- `_indexing.py`: reads an index as torch does, so that the batch size of
  an indexing result follows torch's rules. Its docstring has the model.
- `_nestedkey.py`: `NestedKey`, a string or a tuple of nested keys.
- `_unbatched.py`: `UnbatchedTensor`, a tensor that the shape operations of
  its parent leave unchanged.

Lazy stacks and views:

- `_lazy.py`: `LazyStackedTensorDict`, which stacks tensordicts without
  copying them. `_split_index` reads an index, with the model of
  `_indexing.py`, as one index per member that it reaches: an int along the
  stack dim selects one member, a slice selects members whose results are
  stacked, and an advanced index gathers the elements of its block from the
  members. Reads and writes use this split. The file also holds
  `_CustomOpTensorDict` and its subclasses (`_UnsqueezedTensorDict`,
  `_SqueezedTensorDict`, `_ViewedTensorDict`, `_TransposedTensorDict`,
  `_PermutedTensorDict`), which belong to the legacy lazy mode
  (`set_lazy_legacy(True)`).

Persistence and serialization:

- `memmap.py`: `MemoryMappedTensor`, a tensor in a memory-mapped file.
- `_td_memmap.py`: the memmap directory layout (a `meta.json` per level).
- `_archive.py`: single-file zip archives of a memmap directory
  (`pack_memmap`, `unpack_memmap`, `is_memmap_archive`).
- `persistent.py`: `PersistentTensorDict`, with an H5 backend (`_H5Backend`,
  h5py) and a zarr backend (`_ZarrBackend`).
- `_utils_key_json.py`: encodes keys into file names, and selects the JSON
  backend (`json` or `orjson`).
- `_tabular.py`: pandas, CSV, Parquet and JSON import and export.
- `_arrow.py`: `from_arrow` and `to_arrow`, which convert between Arrow
  tables and tensordicts, viewing the Arrow buffers where they can.
- `_datasets.py`: `to_mds`, which writes a MosaicML streaming dataset.

Typed containers:

- `tensorclass.py`: `TensorClass`, the `@tensorclass` decorator,
  `NonTensorData`, `MetaData` and `NonTensorStack`. A tensorclass instance
  keeps its tensors in a `TensorDict` (`_tensordict`) and its other fields in
  `_non_tensordict`. Most of its methods are TensorDict methods, installed
  from the `_METHOD_FROM_TD` and `_FALLBACK_METHOD_FROM_TD*` lists.
- `typedtensordict.py`: `TypedTensorDict`, a `TensorDictBase` with typed
  fields. It stores its data in `_source`, which can be any tensordict.

Modules (`nn/`):

- `nn/common.py`: `TensorDictModuleBase`, `TensorDictModule`, `WrapModule`
  and the `dispatch` decorator.
- `nn/sequence.py`: `TensorDictSequential`.
- `nn/probabilistic.py`: `ProbabilisticTensorDictModule`,
  `ProbabilisticTensorDictSequential` and the interaction types.
- `nn/params.py`: `TensorDictParams`, a tensordict that registers its
  entries as parameters and buffers of an `nn.Module`.
- `nn/distributions/`: `CompositeDistribution`, `TruncatedNormal` and
  other distributions. Also `nn/cudagraphs.py` (`CudaGraphModule`),
  `nn/ensemble.py`, `nn/functional_modules.py`, `nn/tensorclass_module.py`
  and `nn/utils.py`.

Stores (`store/`):

- `store/_store.py`: `TensorDictStore`, which keeps its tensors in a
  key-value server that speaks the Redis protocol, such as Redis or
  Dragonfly.
- `store/_lazy.py`: `LazyStackedTensorDictStore`, a lazy stack of stores,
  and `_StoreStackElementView`, its write-through view of one member.
  `store/_store.py` re-exports them.
- `store/_connection.py`: `_StoreConnection`, the mixin of both store
  classes that connects to the server, names the server keys, pickles and
  cleans up.
- `store/_utils.py`: byte-range helpers and the Lua scripts that the stores
  run on the server.

Dispatch and pickling: `_torch_func.py`, `_pytree.py`, `_reductions.py`.
See "How calls reach a tensordict" below.

Options and utilities:

- `_utils_options.py`: `set_printoptions`, `set_lazy_legacy`,
  `set_capture_non_tensor_stack`, `set_list_to_stack` and their getters.
  `utils.py` re-exports them.
- `utils.py`: shared helpers, including `logger` and `timeit`.
- `_contextlib.py`: decorator context managers, and `LAST_OP_MAPS`, the
  functions that undo an operation at the end of a `with` block, as in
  `with td.permute(1, 0) as tdp:`.
- `_deprecation.py`: the helpers that deprecate a function, a method, a
  property or a module attribute. Each takes the release that removes the
  name, and `test_deprecation_deadlines` fails once `version.txt` reaches it.
- `_ucxx.py`: `TensorDictPipe` and `TensorDictServer`, transport over UCXX.
- `prototype/fx.py`: `symbolic_trace` for tensordict modules.
- `_testing.py`: tensorclasses that the distributed tests import by name.

`tensordict` is pure Python. `_C/` keeps the module path `tensordict._C`,
which was a C++ extension, as a deprecated re-export of the nested-key
helpers of `utils.py` (`unravel_key`, `unravel_key_list`,
`_unravel_key_to_tuple`).

Compatibility: `tensordict.py` keeps the old import path
`tensordict.tensordict`, which TorchRL's tests still import.

Stubs: the `.pyi` files give type checkers and editors the signatures.
`tensorclass.pyi` gives `TensorClass` its completions.

## Class hierarchy

```
TensorCollection                       _tensorcollection.py
├── TensorDictBase                     base.py, with the mixins in _base/
│   ├── TensorDict                     _td.py
│   ├── _SubTensorDict                 _td.py
│   ├── LazyStackedTensorDict          _lazy.py
│   │   └── NonTensorStack             tensorclass.py
│   ├── _CustomOpTensorDict            _lazy.py (legacy lazy mode)
│   ├── PersistentTensorDict           persistent.py
│   ├── TensorDictStore                store/_store.py
│   ├── LazyStackedTensorDictStore     store/_lazy.py
│   ├── _StoreStackElementView         store/_lazy.py
│   ├── TensorDictParams               nn/params.py (also an nn.Module)
│   └── TypedTensorDict                typedtensordict.py
└── TensorClass                        tensorclass.py
    ├── NonTensorDataBase
    │   ├── NonTensorData
    │   └── MetaData
    └── user subclasses of TensorClass
```

A class decorated with `@tensorclass` is not a subclass of `TensorClass` or
`TensorCollection`: the decorator installs the same methods on the class
itself. `is_tensorclass()` recognizes both forms.

`MemoryMappedTensor` (`memmap.py`) and `UnbatchedTensor` (`_unbatched.py`)
are `torch.Tensor` subclasses that a tensordict can store as entries.

## The TensorDictBase contract

`TensorDictBase` marks its backend-specific methods with
`@abc.abstractmethod`: the storage hooks (`_get_str`, `_set_str`,
`_set_at_str`, ...), the shape operations (`_view`, `_permute`,
`_unsqueeze`, ...), `keys`, `entry_class`, `from_dict`, `_clone` and others.
A class that lacks one cannot be instantiated. The rest of `TensorDictBase`
(`base.py` and the mixins in `_base/`) is written on top of these methods,
and all the implementations share it.

Each public method is written once, on `TensorDictBase`, with its signature,
docstring, argument normalization, validation, lock checks, error types and
messages, and the bookkeeping of its result (batch size, names, device,
lock). When part of an operation depends on the storage, the public method
calls a private hook for that part, or reads a class attribute that selects
one of its code paths. A class puts what it does differently into a hook
override or a class attribute, not into an override of the public method.
The docstring of a hook on `TensorDictBase` states its contract: what the
public method has done before the call, and what the hook must do and
return. `test_conformance.py::TestOneImplementation` checks that each hook
a subclass overrides has a docstring, apart from a listed set of older
hooks. The base implementation of a hook is the generic
algorithm, so only the classes that can do better, or must do something
else, override it. A new backend therefore implements the abstract
methods, and overrides a hook or sets a class attribute where its storage
can do better; it inherits the public methods.

In the tables below, *lazy* is `LazyStackedTensorDict` and so also its
subclass `NonTensorStack`, *NTS* is `NonTensorStack` alone, *sub* is
`_SubTensorDict`, *file* is `PersistentTensorDict` (h5 and zarr), *store*
is `TensorDictStore`, *lazy store* is `LazyStackedTensorDictStore`,
*element* is `_StoreStackElementView` (`store[i]`), *typed* is
`TypedTensorDict` and *params* is `TensorDictParams`.

| hook | called by | what it must do | overridden by |
| --- | --- | --- | --- |
| `_chunk(chunks, dim)` | `chunk` | return the chunks; the default calls `split` | `TensorDict`, element, params, typed |
| `_split(split_sizes, dim)` | `split`, the default `_chunk` | return the pieces, each with the names, device and lock of `self` | lazy, params, typed |
| `_reshape(shape, names, leaf_fn)` | `reshape`, `flatten`, `unflatten` without `inplace` | return a tensordict of batch size `shape`, with views of the leaves where the layout allows | lazy, NTS, typed |
| `_expand(shape, names)` | `expand` | return expanded views | lazy, params, typed |
| `_broadcast_to(shape)` | `broadcast_to` | return broadcast views, without names | NTS |
| `_repeat_interleave(repeats, dim, output_size)` | `repeat_interleave` without `inplace` | repeat the elements along `dim` | lazy |
| `_batch_op(fn, batch_size, names, *, index)` | `flip`, `roll`, `rot90`, `tile`, `narrow`, the default `_broadcast_to` | apply `fn` to each leaf, and set the lock, shared and memmap flags | NTS |
| `_densify(layout)` | `densify` | replace the lazy stacks with dense entries | lazy, NTS |
| `_compare(other, op, reflected_op, default)` | `==`, `!=`, `<`, `<=`, `>`, `>=`, `\|`, `^` | broadcast, pair the leaves by key, raise `KeyError` on mismatched keys | lazy, params, typed |
| `_where(condition, other, *, out, pad, update_batch_size)` | `where`, without `_masking_on_dense_copy` | return the result, written into `out` if given | lazy, params, typed |
| `_reduction_parts()` | `all()`, `any()` without a dim | return the parts to reduce: the values, or the members of a lazy stack | lazy |
| `_reduced_names(dim)` | `all(dim)`, `any(dim)` | return the names of the result | lazy |
| `_softmax(dim, dtype)` | `softmax` | return the softmax along the batch dim `dim`; the default calls `softmax` on the nested lazy stacks, whose members lack the stack dim | lazy, typed |
| `_masked_fill_(mask, value)` | `masked_fill_` | fill in place where `mask` is True | sub, lazy, file, lazy store, params, typed |
| `_masked_fill(mask, value)` | `masked_fill`, without `_masking_on_dense_copy` | return a filled copy; the default fills a clone in place | lazy |
| `_fill_(key, value)` | `fill_`, and `zero_` with `_fills_without_reading` | fill the entry at the tuple `key`; raise `KeyError` if it is missing | file |
| `_detach_()` | `detach_`, without `_storage_holds_no_gradients` | detach the leaves in place | sub (raises), lazy, params, typed |
| `_apply_(fn, *others, **kwargs)` | `apply_` | write the results into the leaves; the default calls `apply(..., inplace=True)` | lazy |
| `_update_fast(source)` | `update` with default arguments, outside of `torch.compile` | write without the checks and return `True`, or write nothing and return `False` | `TensorDict` |
| `_update_entries(source, *, clone, inplace, ...)` | `update` | write the entries, after the lock check and the handling of `None`, `self` and `keys_to_update` | sub, lazy, lazy store |
| `_update_(source, *, clone, non_blocking, keys_to_update)` | `update_` | copy into the existing leaves | sub, lazy, store, lazy store, element |
| `_update_plan(source, *, non_blocking)` | the base `_update_` without `keys_to_update`, and the lazy `_update_` for each member | pair the leaves that `update_` copies, write nothing, and return `(dests, sources, finish)`, or `None` where no leaf matches | lazy |
| `_update_at_(source, idx, *, ...)` | `update_at_` with a non-empty index | copy into the leaves at `idx` | lazy, NTS |
| `_setitem_index(index, value)` | `td[index] = value` | write `value` at the batch `index`; a str or tuple key and an `Ellipsis` are handled before the call | lazy, file, store, lazy store, element |
| `_empty(recurse, batch_size, device, names)` | `empty` | return a new, unlocked tensordict without tensor leaves | lazy, file, store; one body for lazy store and element |
| `_contains_str(key)` | `_convert_inplace`, which `_set_str` calls | say whether `key` is a key, without reading data | `TensorDict`, file, store |
| `_get_tuple(key, default)` | `get` | return the entry, or `default`; raise `ValueError` for a path through a tensor (files and the store return `default`) | every class but element |
| `_get_at_tuple(key, idx, default)` | `get_at` | read the entry at `idx` | file, store, lazy store, element, params |
| `_create_nested_tuple(key)` | `create_nested` | add an empty tensordict at the tuple `key`; a plain `TensorDict` keeps the plain, unlocked `TensorDict`s already on the path, and every other class replaces each level | lazy (a stack of plain `TensorDict`s, or of such stacks, hands the key to each member), sub and typed (follow the rule of a plain source), file (keeps the groups on the path) |
| `_del(key)` | `del_` | delete the entry at the tuple `key`; raise `KeyError` for a missing key and `ValueError` for a path through a tensor (for `()`, the file and the stores keep their own behaviour until 0.17) | every class but params and typed (sub forwards it) |
| `_members_as_lists()` | `items`, `values` with `_NESTED_TENSORS_AS_LISTS` | return the members to walk one by one, or `None` | lazy |
| `_contains_key(key)` | `key in td`, for a str key and an unravelled, non-empty tuple key | say whether `key` is in `keys(True, is_leaf=_is_leaf_nontensor)`, without reading the leaves | `TensorDict` (not its subclasses) |
| `_contains_tensordict(item)` | `item in td`, when `item` is not a key | say whether `item` is in `self`; the default raises | lazy |
| `_flatten_keys_outplace(separator, is_leaf)` | `flatten_keys` without `inplace` | return a new tensordict with the leaves under flat keys; raise `KeyError` when two keys collide | lazy (flattens each member) |
| `_rename_key(old_key, new_key)` | `rename_key_` | move the entry | file, store, lazy store, element (raises) |
| `_popped_value(value)` | `pop`, `popitem` | return a copy of a nested view that `del_` is about to delete | file, store |
| `_lock_sub_tensordicts()` | `lock_`, `unlock_`, the default `_rename_subtds` | return the sub-tensordicts to reach, without reading data | file, store, lazy store, element |
| `_rename_subtds(value)` | each change of names | rename the sub-tensordicts; the default renames those of `_lock_sub_tensordicts` | `TensorDict`, sub, lazy, file, params, typed |
| `_share_memory_()` | `share_memory_` | place the leaves in shared memory and lock; the default raises | `TensorDict`, sub (raises), lazy, params, typed |
| `_contiguous_inplace(canonical)` | `contiguous(inplace=True)` | make the leaves contiguous; the default raises | `TensorDict` |
| `_to(device, dtype, ...)` | `to`, after it parses the arguments | return the cast tensordict, or `self` | sub, lazy, file |
| `_write_back_to(result)` | the exit of `with td.to(...)`, with `_to_writes_back_in_place` | write the copy back into the storage | file (no default) |
| `_reduce(dst, op, async_op, return_premature, _future_list, group)` | `reduce` | issue one `dist.reduce` per leaf, in `sorted_keys` order, as a dense tensordict of the same keys and shapes does; the root waits for every entry with `_finish_reduce`, then raises the first error | lazy (reduces each stacked leaf and writes the result back into the members on rank `dst`) |
| `_to_tensordict(*, retain_none)` | `to_tensordict` | return the dense copy | store, element, NTS (returns `self`) |
| `_repr_parts()` | `__repr__` | return the lines inside `ClassName(...)`, without reading data | lazy, NTS, store, lazy store, element |
| `_wrap_archive_result(result)` | `memmap`, `memmap_like` after an archive write | return `result`, the archive reloaded with `load_memmap`, in the classes that a directory write returns, without copying data; the default returns it unchanged | typed (wraps it and its nested `TypedTensorDict` fields) |
| `_new_unsafe(source, *, batch_size, ...)` | `split`, `contiguous`, `empty`, `_unbind`, `_index_tensordict` | build an instance without validation; accept every keyword of `TensorDict._new_unsafe` | `TensorDict`, params, typed |

`from_h5` and `from_zarr` open the file with `PersistentTensorDict._open_h5`
and `_open_zarr`. The `batch_size` setter checks a new batch size with
`_check_new_batch_size` (in `_batch_size_setter`), which the stores override
to compare it with the shapes in their metadata.

| class attribute | effect | set by |
| --- | --- | --- |
| `_lazy` | the base treats the class as a lazy view: for example, a new batch size and `masked_select(inplace=True)` raise | lazy, sub, the legacy lazy views |
| `_reads_return_copies` | the in-place methods run on copies of the leaves and write them back (see below) | file, store, lazy store, element; a property on sub, lazy and typed |
| `_nested_reads_return_copies` | `update` and `_fill_` write the leaves of a nested tensordict back under their full keys | lazy store, element |
| `_update_stores_copy_of_self` | `update({key: self})` writes a copy of the data instead of raising that a tensordict cannot contain itself | store, lazy store, element |
| `_masking_on_dense_copy` | `where`, `masked_fill`, and `to` with a new device, dtype or memory format, work on a `to_tensordict()` copy | sub, file, store, lazy store, element, the legacy lazy views |
| `_masked_select_on_dense_copy` | `masked_select` masks a `to_tensordict()` copy | store, lazy store, element |
| `_masked_select_inplace_unsupported` | `masked_select(inplace=True)` raises, as on a lazy stack | lazy store, element |
| `_storage_holds_no_gradients` | `detach_` returns without reading | file, store, lazy store, element |
| `_fills_without_reading` | `zero_` calls `_fill_` for each key | file |
| `_leaves_built_on_read` | `is_contiguous()` returns `False` without reading | lazy, file, store, lazy store, element |
| `_contiguous_via_to_tensordict` | `contiguous()` returns `to_tensordict()` | store, lazy store, element |
| `_pin_memory_unsupported` | `pin_memory` raises | sub, file, store, lazy store, element |
| `_memmap_inplace_unsupported` | `memmap_` raises | sub |
| `_map_on_dense_copies` | `map` sends a `to_tensordict()` copy of each chunk to the workers | file |
| `_to_writes_back_in_place` | the exit of `with td.to(...)` calls `_write_back_to` | file |
| `_has_exclusive_keys` | `to_dict`, `state_dict`, `flatten_keys` and `unflatten_keys` raise | a property on lazy |
| `_wrapped_attr` | the name of the attribute that holds the wrapped tensordict, which the pointwise ops read to find a lazy stack behind an operand | typed (`"_source"`) |
| `_rename_key_lenient` | `rename_key_` renames in a locked tensordict with a warning, until 0.17 | file, store, lazy store |
| `_inplace_shape_ops_unsupported` | `reshape` and `repeat_interleave` with `inplace=True` raise before they write | lazy |
| `_inplace_batch_size_change` | `False`: the in-place shape ops that change the batch size raise before they write | file (`False`); typed (a property: `False` over a lazy stack, a sub-tensordict or a file) |
| `_flatten_lenient` | `flatten(d, d)` returns `self`; with `start_dim > end_dim`, `flatten` inserts a dim of size 1 with a `FutureWarning` until 0.17, where the others raise | lazy |
| `_gather_inplace_warns` | `gather(inplace=True)` with an index of the batch shape writes, and warns that it raises from 0.17 | typed (a property: over a lazy stack) |
| `_get_ignores_kwargs` | `False`: `get` honors `as_list` and `as_nested_tensor` instead of warning | lazy (`False`) |
| `_setitem_key_inplace` | the `inplace` argument with which `td[key] = value` calls `_set_tuple` | `BEST_ATTEMPT_INPLACE` on sub, file, store, lazy store, element |

A view reports the state of the tensordict that it reads, `_source`:
`_SubTensorDict` and `TypedTensorDict` list the mixin `_StateFromSource`
(`_td.py`) before `TensorDictBase`. The mixin reads `is_locked`, `device`,
the shared and memmap flags, `_has_names()`, `_storage_is_read_only()` and
`_has_non_tensor` from the source.

The `batch_size` and `device` properties read the `_batch_size` and `_device`
attributes, which each class sets, and `_change_batch_size` writes
`_batch_size`. A class that derives one of them (a lazy stack, a store that
persists it) replaces only the accessor it needs, with
`@TensorDictBase.batch_size.getter` or `@TensorDictBase.device.setter`.

The generic `_select` and `_exclude` return a `TensorDict` copy of the entries
they keep. `TensorDict` and `LazyStackedTensorDict` override them to share the
values instead.

Each public override that a class keeps, and each hook override, says why in
a comment that starts with `# Override:`. Some classes keep public
overrides: the constructors and the pickling methods, the value API of
`NonTensorStack`, dense fast paths on `TensorDict` (`del_`, `popitem`,
`empty`, ...), and the forwarders of `TensorDictParams` and
`TypedTensorDict`. `test_conformance.py::TestOneImplementation` lists the
public overrides that each class may keep, so that a new one fails the test
until it is replaced by a hook or added to the list with its reason.

The implementations also reuse code in three ways:

- A few class-level aliases still borrow TensorDict's code where it is
  specific to dense storage: `_SubTensorDict._add_batch_dim`, the vmap hooks
  of the legacy lazy views and `TensorDictParams._memmap_`. The default
  `_load_memmap` of `TensorDictBase` calls `TensorDict._load_memmap`, so that
  a store, a `PersistentTensorDict` or a `TensorDictParams` loads a memmap as
  a `TensorDict`.
- `TensorDictParams` and `TypedTensorDict` forward calls to the tensordict
  they wrap. Where they can, they forward the hook and inherit the public
  method; a comment gives the reason for each public forwarder they keep.
  `TensorDictParams`
  forwards to `_param_td` through the `_fallback` decorators in
  `nn/params.py`, so that the results keep their parameters.
  `TypedTensorDict` forwards to `_source` through the `_*_DELEGATES` lists
  at the end of `typedtensordict.py`, and derives `_reads_return_copies`,
  `_inplace_batch_size_change` and `_gather_inplace_warns` from it, so that
  an in-place operation over a lazy stack, a sub-tensordict or a file takes
  the same path as on the source.
- The two store classes share their connection, server keys, pickling and
  cleanup through the mixin `_StoreConnection` (`store/_connection.py`),
  which they configure with class attributes (`_global_key_suffixes`,
  `_pickled_attrs`, `_caches_keys`) and two hooks (`_entry_server_keys`,
  `_init_local_state`).

Some classes read copies of their data: each read of a
`PersistentTensorDict` or a store returns a new tensor, and so does a read of
a `TypedTensorDict` over a lazy stack or of a `_SubTensorDict` with an index
tensor. They set `_reads_return_copies`. The in-place methods (`add_`,
`abs_`, `+=`, ...) change the tensors that `_items_list` returns. On these
classes, the `@_write_back_inplace` decorator runs the method on a
`TensorDict` of the leaves that it reads (`_inplace_on_copies`, which a lazy
stack, a lazy store and `TypedTensorDict` override), and `_write_back`
writes each leaf back with `_set_tuple(..., inplace=True)`; `update_` calls
`_write_back` itself. A new in-place method that changes the leaves takes
the decorator. A file opened read-only stays unchanged, with a warning
(`_storage_is_read_only`).

An operation that a backend cannot support raises. The raising bodies are
written once: private hooks in the mixins of `_base/unsupported.py`
(`_NoShapeOps` for the shape hooks, `_NoVmap` for the vmap hooks,
`_NoSetNonTensor`, `_NoSelectExclude`, and `_StoreUnsupported`, which
combines them for the stores), and public methods behind a class attribute
(`_pin_memory_unsupported`, `_memmap_inplace_unsupported`, ...). A class
lists the mixins before `TensorDictBase` in its bases, so the hooks stay
abstract on `TensorDictBase`. For example, `TensorDictStore._view` raises a
`RuntimeError` that asks the user to call `to_tensordict()` first. The
`make_memmap*` methods, which only `TensorDict` implements, raise by
default on `TensorDictBase` instead.

## How calls reach a tensordict

- Methods: a public method of `TensorDictBase` (in `base.py` or a mixin of
  `_base/`) handles the arguments, the checks and the bookkeeping of the
  result, and calls the abstract methods and the private hooks of "The
  TensorDictBase contract", which the classes implement or override.
  `TensorDictParams` and `TypedTensorDict` mostly inherit the public method
  and forward the hook to the tensordict they wrap.
- Torch functions: `TensorDictBase.__torch_function__` looks up the function
  in `TD_HANDLED_FUNCTIONS`, which the `@implements_for_td(torch.<name>)`
  decorators in `_torch_func.py` fill. A function that is not in the table
  returns `NotImplemented`. `LazyStackedTensorDict` first checks
  `LAZY_TD_HANDLED_FUNCTIONS` (`@implements_for_lazy_td`).
  `TensorDictParams` uses a copy of the table, `_TDPARAM_HANDLED_FUNCTIONS`.
  A tensorclass passes the torch functions listed in `_TD_PASS_THROUGH`
  (`tensorclass.py`) to its tensordict.
- Pytree: `_pytree.py` registers `TensorDict`, `_SubTensorDict`,
  `PersistentTensorDict` and `LazyStackedTensorDict` with
  `torch.utils._pytree`. Each tensorclass and each `TypedTensorDict`
  subclass is registered when it is created.
- Pickling: `_reductions.py` registers `TensorDict` and
  `LazyStackedTensorDict` with `copyreg` and `multiprocessing.reduction`. A
  consolidated tensordict is pickled as its single storage plus metadata.
  The other classes use `__getstate__` and `__setstate__`.
- `torch.compile`: Dynamo traces the Python code of the library, and uses
  the pytree registration to flatten tensordicts. Some code paths branch
  on `is_compiling()` to take a path that Dynamo can trace. See the
  `torch.compile` section of `AGENTS.md`.

## Adding an operation

Read a recent PR that added a similar operation with `git show --stat`, such
as #1733 (`backward`) or `ec8d0082e` (`roll`). A complete change usually
touches these places:

1. The method and its docstring, on `TensorDictBase`: in the mixin of
   `_base/` for its area, or in `base.py` for core operations. If it can be
   written with other methods, write it once there. Make it abstract only if
   each backend needs its own code. If only part of it depends on the
   storage, write the public method once, with the argument checks, the
   error messages and the bookkeeping of the result, and let it call a
   private hook named after it (`chunk` calls `_chunk`), as in "The
   TensorDictBase contract". The hook's base implementation is the generic
   algorithm, and its docstring states its contract. When the classes differ
   only in whether they take one path or another, read a class attribute
   with a default on `TensorDictBase`, rather than check the class
   with `isinstance`. Abstract methods, properties,
   `implement_for` overloads and methods that read a global that `base.py`
   rebinds stay in `base.py`. The comment above the mixin imports in
   `base.py` gives the rules for code in a mixin. In particular, a mixin
   imports `TensorDictBase` under `TYPE_CHECKING` only, and `base.py` binds
   it into the mixin modules once the class exists, so that annotations
   that name it resolve. Code that runs at import time, such as decorators
   and default values, cannot use it, and a method that uses it at run time
   imports it locally (ruff's TC004 flags a run-time use of the
   `TYPE_CHECKING` import).
2. If it is abstract: an implementation in each class of the hierarchy
   above (or a raise, in a mixin of `_base/unsupported.py`). If it is a
   hook: an override in each class that can do better. `TypedTensorDict`
   and `TensorDictParams` inherit the public method. When the hook must run
   on the tensordict they wrap, to keep a lazy source lazy or the
   parameters in the result, add its name to one of the `_*_DELEGATES`
   lists of `typedtensordict.py` and a `_fallback` forwarder to
   `nn/params.py`.
3. `tensorclass.py`: the name in `_FALLBACK_METHOD_FROM_TD`, or in another
   `_METHOD_FROM_TD*` list. `test_sorted_methods` checks that the lists are
   sorted.
4. `_torch_func.py`: an `@implements_for_td` handler if the operation is
   also a torch function, and that function in `_TD_PASS_THROUGH` so that
   tensorclasses accept it.
5. `_contextlib.py`: an entry in `LAST_OP_MAPS` if the operation can be
   undone at the end of a `with` block.
6. Stubs: the signature in `tensorclass.pyi` and `_tensorcollection.pyi`.
   `test_tensorclass_stub_methods` fails if a public `TensorDict` method is
   missing from `tensorclass.pyi`. No test checks `_tensorcollection.pyi`.
7. Docs: a new public class or function goes in
   `docs/source/reference/{td,tc,nn,ttd}.rst`. Methods appear on their
   class's page, and a notable one can get a section, as `backward` did.
8. Tests:
   - `test/tensordict/test_methods.py`: `TestTensorDicts` runs each test on
     each tensordict type that the fixtures of `TestTensorDictsBase` in
     `test/_utils_internal.py` build (dense, nested, stacked, sub, memmap,
     h5, params, typed, legacy views, ...).
   - The stores are not among these fixtures: see `test/store/test_store.py`.
   - Tensorclasses: `test/tensorclass/test_tensorclass.py`.
   - `test/compile/test_compile.py` for hot paths.
   - Indexing: `test/tensordict/test_indexing.py` compares each type with
     torch. Its known failures are listed in
     `test/tensordict/indexing_known_failures.json`. When a change fixes one,
     set `TENSORDICT_UPDATE_KNOWN_FAILURES=1` to rewrite that file.
   - Other operations: `test/tensordict/test_conformance.py` runs a set of
     operations on each type and on its dense copy and compares the
     results. Its known failures are in
     `test/tensordict/conformance_known_failures.json`, rewritten the same
     way. `TestContainersCompile`, in the same file, runs these operations
     under `torch.compile(fullgraph=True)` and compares them with eager
     mode. Its known failures change with the torch version: they are in
     `conformance_compile_known_failures.json`, and the tests run only
     with `TENSORDICT_COMPILE_CONFORMANCE=1`, which the CI job with the
     newest stable torch sets. `TestOneImplementation` fails when a class
     overrides a public method of `TensorDictBase` that is not in its list
     of allowed overrides: write a hook instead, or add the override to the
     list and give the reason in its `# Override:` comment.
9. `benchmarks/` when the operation is on a hot path.

## Where things live

- Tests: `test/<area>/test_*.py`, with the areas `compile`, `distributed`,
  `memmap`, `nn`, `store`, `tensorclass`, `tensordict` and `utils`. Shared
  fixtures and helpers are in `test/_utils_internal.py` and
  `test/conftest.py`.
- Docs: Sphinx sources in `docs/source/`. The user guides are the `.rst`
  files at that level (`overview.rst`, `saving.rst`, `storage.rst`, ...),
  and the API reference is in `docs/source/reference/`.
  `docs/requirements.txt` lists the build dependencies.
- Tutorials: `tutorials/sphinx_tuto/*.py`, rendered with sphinx-gallery.
  `docs/source/conf.py` copies them, and `tutorials/media/`, into the docs
  tree when the docs build.
- Benchmarks: `benchmarks/`, in the folders `common`, `compile`, `nn`,
  `storage`, `tensorclass` and `distributed`. They are pytest-benchmark
  tests, and `run_test.sh` also runs them in the test jobs.
- CI: workflows in `.github/workflows/`. The Linux and macOS test jobs run
  the scripts in `.github/unittest/linux/scripts/`, and `run_test.sh` there
  sets the test environment variables. `test-rl-gpu.yml` runs TorchRL's
  tests with `.github/unittest/rl_linux_optdeps/scripts/run_all.sh`. Other
  CI helpers are in `.github/scripts/`.
- Lint: `.pre-commit-config.yaml`, which runs `scripts/check_rst_titles.py`.
- Packaging: `setup.py`, `pyproject.toml`, `version.txt` and `packaging/`.
