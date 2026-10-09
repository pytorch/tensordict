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
- `base.py`: `TensorDictBase`, with most of the shared method code and
  docstrings. At about 18,000 lines, it is the largest file.
- `_td.py`: `TensorDict`, the dense implementation, and `_SubTensorDict`, a
  view on an index of another tensordict.
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
  copying them. An index selects members along the stack dim and indexes
  each member along the other dims. The file also holds
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
- `tabular.py`: pandas, CSV, Parquet and JSON import and export.
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

- `store/_store.py`: `TensorDictStore`, `LazyStackedTensorDictStore` and
  `_StoreStackElementView`. They keep their tensors in a key-value server
  that speaks the Redis protocol, such as Redis or Dragonfly.
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
- `_ucxx.py`: `TensorDictPipe` and `TensorDictServer`, transport over UCXX.
- `prototype/fx.py`: `symbolic_trace` for tensordict modules.
- `testing.py`: tensorclasses that the distributed tests import by name.

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
├── TensorDictBase                     base.py
│   ├── TensorDict                     _td.py
│   ├── _SubTensorDict                 _td.py
│   ├── LazyStackedTensorDict          _lazy.py
│   │   └── NonTensorStack             tensorclass.py
│   ├── _CustomOpTensorDict            _lazy.py (legacy lazy mode)
│   ├── PersistentTensorDict           persistent.py
│   ├── TensorDictStore                store/_store.py
│   ├── LazyStackedTensorDictStore     store/_store.py
│   ├── _StoreStackElementView         store/_store.py
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
`@abc.abstractmethod`: the storage hooks (`_get_str`, `_set_str`, ...),
the shape operations (`_view`, `_permute`, `_unsqueeze`, ...), `keys`,
`_index_tensordict`, `_clone` and others, 40 in all. A class that lacks one
cannot be instantiated. The rest of `TensorDictBase` is written on top of
these methods, and all the implementations share it, including generic
implementations of operations such as `reshape`, `split`, `_apply_nest` and
the comparison operators, which a class overrides only when it can do better.

The implementations also reuse code in two ways:

- A few class-level aliases still borrow TensorDict's code, where it is
  specific to dense storage or has not been moved to `TensorDictBase` yet. For
  example, `_load_memmap = TensorDict._load_memmap` binds the classmethod to
  `TensorDict`, so that a store or a `PersistentTensorDict` loads a memmap as
  a `TensorDict`.
- `TensorDictParams` forwards most calls to the tensordict it wraps
  (`_param_td`), through the `_fallback` decorators in `nn/params.py`.
  `TypedTensorDict` forwards to `_source` through the `_*_DELEGATES` lists at
  the end of `typedtensordict.py`.

An operation that a backend cannot support raises. For example,
`TensorDictStore._view` raises a `RuntimeError` that asks the user to call
`to_tensordict()` first.

## How calls reach a tensordict

- Methods: a public method in `base.py` calls the abstract hooks, which each
  class implements.
- Torch functions: `TensorDictBase.__torch_function__` looks up the function
  in `TD_HANDLED_FUNCTIONS`, which the `@implements_for_td(torch.<name>)`
  decorators in `_torch_func.py` fill. A function that is not in the table
  returns `NotImplemented`. `LazyStackedTensorDict` first checks
  `LAZY_TD_HANDLED_FUNCTIONS` (`@implements_for_lazy_td`).
  `TensorDictParams` uses a copy of the table, `TDPARAM_HANDLED_FUNCTIONS`.
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

1. `base.py`: the method and its docstring, on `TensorDictBase`. If it can
   be written with other methods, write it once there. Make it abstract
   only if each backend needs its own code.
2. If it is abstract: an implementation in each class of the hierarchy
   above (or a raise), and the name in one of the `_*_DELEGATES` lists of
   `typedtensordict.py`.
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
