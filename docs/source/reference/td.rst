.. currentmodule:: tensordict

.. _api-tensordict:

tensordict: containers and functions
====================================

.. module:: tensordict

This page lists the tensordict containers, and the methods that they all
share, grouped by task. Each group also lists the functions of the
``tensordict`` module for its task. A :class:`~tensordict.TensorDict` is a
``dict[str, torch.Tensor]`` that behaves like one tensor. Its entries share
their leading dimensions, the *batch size*, and an operation such as
indexing, reshaping, or :meth:`~tensordict.TensorDictBase.to` applies to every
entry. For an introduction, see :ref:`overview`.

Containers
----------

:class:`TensorDictBase` is the abstract base class of all tensordicts.
Its subclasses keep their entries in different places:

- :class:`TensorDict` keeps its tensors in memory. It is the default
  container.
- :class:`LazyStackedTensorDict` is a stack of tensordicts that keeps each
  tensordict separate, so that their entries can have different shapes.
  See :ref:`stacking`.
- :class:`PersistentTensorDict` reads and writes its entries in an HDF5 or
  zarr file. See :ref:`storage`.
- :class:`TensorDictParams` is a tensordict and a :class:`torch.nn.Module`.
  If a module holds one, the parameters and buffers of the module include
  its tensors.

:class:`MemoryMappedTensor` is not a tensordict. It is a tensor whose data
is in a file, and the type of the entries of a memory-mapped tensordict.

:class:`TensorCollection` is the common base class of the tensordicts and
the subclasses of :class:`TensorClass`: the type hint for "a tensordict or
a tensorclass".

Other containers have their own pages: :class:`TensorClass`, the class
form with typed fields and methods, and non-tensor data
(:ref:`api-tensorclass`), :class:`TypedTensorDict`
(:ref:`api-typed-tensordict`), and the key-value stores
(:ref:`api-storage`). To choose one, see :ref:`choosing-a-container`.

.. autosummary::
    :toctree: generated/
    :template: tensordict_class.rst

    TensorDictBase
    TensorDict
    LazyStackedTensorDict
    PersistentTensorDict
    TensorDictParams

.. autosummary::
    :toctree: generated/
    :template: class.rst

    MemoryMappedTensor
    TensorCollection

.. _api-tensordict-methods:

Methods by task
---------------

The tables below list the methods of :class:`TensorDictBase`, with one
page for each method. Every tensordict class and every tensorclass has
these methods. A class page lists only the methods that the class adds,
and the overrides that have their own documentation. When the
``tensordict`` module has a function for the same task, the function is
in the same group.

.. _api-methods-construction:

Construction and conversion
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a tensordict from Python, numpy, or pandas data, and convert a
tensordict to other types. Most ``from_*`` class methods have a function
of the same name in the ``tensordict`` module. To read a tensordict from a
file, see :ref:`api-methods-saving`. For the arguments of the constructor,
see :class:`TensorDict`.

.. rubric:: Methods

.. autosummary::
    :toctree: generated/
    :template: method.rst

    TensorDictBase.from_any
    TensorDictBase.from_dict
    TensorDictBase.from_dict_instance
    TensorDictBase.from_namedtuple
    TensorDictBase.from_pandas
    TensorDictBase.from_struct_array
    TensorDictBase.from_tuple
    TensorDictBase.from_pytree
    TensorDictBase.from_dataclass
    TensorDictBase.from_list
    TensorDictBase.from_schema
    TensorDictBase.fromkeys
    TensorDictBase.empty
    TensorDictBase.new_empty
    TensorDictBase.new_full
    TensorDictBase.new_ones
    TensorDictBase.new_tensor
    TensorDictBase.new_zeros
    TensorDictBase.copy
    TensorDictBase.clone
    TensorDictBase.to_dict
    TensorDictBase.to_namedtuple
    TensorDictBase.to_pandas
    TensorDictBase.to_pytree
    TensorDictBase.to_struct_array
    TensorDictBase.to_tensordict
    TensorDictBase.tolist
    TensorDictBase.numpy
    TensorDictBase.as_tensor

.. rubric:: Functions

.. autosummary::
    :toctree: generated/
    :template: function.rst

    from_any
    from_dict
    from_namedtuple
    from_pandas
    from_struct_array
    from_tuple
    from_pytree
    fromkeys
    make_tensordict

.. _api-methods-keys:

Keys, values, and entries
~~~~~~~~~~~~~~~~~~~~~~~~~

Read, write, select, and rename the entries of a tensordict. A key is a
string, or a tuple of strings that points to an entry of a nested
tensordict. :class:`NestedKey` is the type of these keys. For the rules,
see :ref:`keys`. For values that are not tensors, see
:ref:`non-tensor-data`.

.. rubric:: Methods

.. autosummary::
    :toctree: generated/
    :template: method.rst

    TensorDictBase.keys
    TensorDictBase.values
    TensorDictBase.items
    TensorDictBase.sorted_keys
    TensorDictBase.get
    TensorDictBase.get_at
    TensorDictBase.set
    TensorDictBase.set_
    TensorDictBase.set_at_
    TensorDictBase.setdefault
    TensorDictBase.update
    TensorDictBase.update_
    TensorDictBase.update_at_
    TensorDictBase.copy_
    TensorDictBase.copy_at_
    TensorDictBase.replace
    TensorDictBase.pop
    TensorDictBase.popitem
    TensorDictBase.del_
    TensorDictBase.clear
    TensorDictBase.select
    TensorDictBase.exclude
    TensorDictBase.split_keys
    TensorDictBase.separates
    TensorDictBase.rename_key_
    TensorDictBase.rename_keys
    TensorDictBase.flatten_keys
    TensorDictBase.unflatten_keys
    TensorDictBase.create_nested
    TensorDictBase.depth
    TensorDictBase.entry_class
    TensorDictBase.get_item_shape
    TensorDictBase.is_empty
    TensorDictBase.fill_
    TensorDictBase.set_non_tensor
    TensorDictBase.get_non_tensor
    TensorDictBase.is_non_tensor
    TensorDictBase.non_tensor_items
    TensorDictBase.filter_non_tensor_data
    TensorDictBase.filter_empty_
    TensorDictBase.cat_tensors
    TensorDictBase.stack_tensors
    TensorDictBase.cat_from_tensordict
    TensorDictBase.pack
    TensorDictBase.stack_from_tensordict

.. rubric:: Classes and functions

.. autosummary::
    :toctree: generated/
    :template: class.rst

    NestedKey
    PackLayout

.. autosummary::
    :toctree: generated/
    :template: function.rst

    unravel_key
    unravel_key_list
    unravel_key_to_tuple
    merge_tensordicts

.. _api-methods-indexing:

Indexing, masking, and splitting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Index every entry along the batch dimensions at once, and split a
tensordict into several tensordicts. ``td[0]`` and ``td[mask]`` index the
batch dimensions. ``td["key"]`` returns an entry: see
:ref:`api-methods-keys`. For the rules, see :ref:`batch-size`.

.. autosummary::
    :toctree: generated/
    :template: method.rst

    TensorDictBase.__getitem__
    TensorDictBase.__iter__
    TensorDictBase.gather
    TensorDictBase.masked_fill
    TensorDictBase.masked_fill_
    TensorDictBase.masked_select
    TensorDictBase.where
    TensorDictBase.split
    TensorDictBase.chunk
    TensorDictBase.tensor_split
    TensorDictBase.unbind
    TensorDictBase.narrow

.. _api-methods-shape:

Shape, batch dimensions, and stacking
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Read and change the batch size and the dimension names, reshape and expand
a tensordict, and stack or concatenate several tensordicts. These methods
change the batch dimensions. The dimensions of each entry after the batch
dimensions do not change. See :ref:`batch-size` and :ref:`stacking`.

.. rubric:: Methods

.. autosummary::
    :toctree: generated/
    :template: method.rst

    TensorDictBase.batch_size
    TensorDictBase.batch_dims
    TensorDictBase.shape
    TensorDictBase.ndim
    TensorDictBase.ndimension
    TensorDictBase.dim
    TensorDictBase.size
    TensorDictBase.numel
    TensorDictBase.__len__
    TensorDictBase.names
    TensorDictBase.refine_names
    TensorDictBase.rename
    TensorDictBase.rename_
    TensorDictBase.auto_batch_size_
    TensorDictBase.reshape
    TensorDictBase.view
    TensorDictBase.flatten
    TensorDictBase.unflatten
    TensorDictBase.permute
    TensorDictBase.transpose
    TensorDictBase.swapaxes
    TensorDictBase.swapdims
    TensorDictBase.moveaxis
    TensorDictBase.movedim
    TensorDictBase.squeeze
    TensorDictBase.unsqueeze
    TensorDictBase.expand
    TensorDictBase.expand_as
    TensorDictBase.broadcast_to
    TensorDictBase.repeat
    TensorDictBase.repeat_interleave
    TensorDictBase.tile
    TensorDictBase.flip
    TensorDictBase.fliplr
    TensorDictBase.flipud
    TensorDictBase.roll
    TensorDictBase.rot90
    TensorDictBase.pad
    TensorDictBase.atleast_1d
    TensorDictBase.atleast_2d
    TensorDictBase.atleast_3d
    TensorDictBase.contiguous
    TensorDictBase.is_contiguous
    TensorDictBase.stack
    TensorDictBase.cat
    TensorDictBase.lazy_stack
    TensorDictBase.maybe_dense_stack
    TensorDictBase.to_lazystack
    TensorDictBase.densify
    TensorDictBase.to_padded_tensor

.. rubric:: Functions

.. autosummary::
    :toctree: generated/
    :template: function.rst

    stack
    cat
    lazy_stack
    maybe_dense_stack
    dense_stack_tds
    pad
    pad_sequence
    unpad_sequence
    collate

.. _api-methods-device:

Device, dtype, and memory
~~~~~~~~~~~~~~~~~~~~~~~~~

Move a tensordict to a device, cast its entries to another dtype, and
control the memory that holds them. See :ref:`devices`.

.. rubric:: Device and memory

.. autosummary::
    :toctree: generated/
    :template: method.rst

    TensorDictBase.to
    TensorDictBase.cpu
    TensorDictBase.cuda
    TensorDictBase.device
    TensorDictBase.auto_device_
    TensorDictBase.clear_device_
    TensorDictBase.is_cpu
    TensorDictBase.is_cuda
    TensorDictBase.is_meta
    TensorDictBase.pin_memory
    TensorDictBase.share_memory_
    TensorDictBase.is_shared
    TensorDictBase.record_stream
    TensorDictBase.consolidate
    TensorDictBase.is_consolidated
    TensorDictBase.from_consolidated
    TensorDictBase.bytes
    TensorDictBase.data_ptr
    TensorDictBase.param_count
    TensorDictBase.attrs
    TensorDictBase.clear_refs_for_compile_

.. rubric:: Dtype

.. autosummary::
    :toctree: generated/
    :template: method.rst

    TensorDictBase.dtype
    TensorDictBase.type
    TensorDictBase.is_floating_point
    TensorDictBase.bool
    TensorDictBase.half
    TensorDictBase.float
    TensorDictBase.double
    TensorDictBase.int
    TensorDictBase.bfloat16
    TensorDictBase.float16
    TensorDictBase.float32
    TensorDictBase.float64
    TensorDictBase.int8
    TensorDictBase.int16
    TensorDictBase.int32
    TensorDictBase.int64
    TensorDictBase.uint8
    TensorDictBase.uint16
    TensorDictBase.uint32
    TensorDictBase.uint64
    TensorDictBase.complex32
    TensorDictBase.complex64
    TensorDictBase.complex128
    TensorDictBase.qint8
    TensorDictBase.qint32
    TensorDictBase.quint8
    TensorDictBase.quint4x2

``to`` and the dtype shorthands take a ``cast_filter`` predicate
``fn(key, value) -> bool`` that selects the leaves to cast. The predicates
below test the dtype of their last argument, so they serve as a
``cast_filter`` and as a test in :meth:`~tensordict.TensorDictBase.apply`.

.. rubric:: Functions

.. autosummary::
    :toctree: generated/
    :template: function.rst

    from_consolidated
    is_inexact
    is_floating
    is_integer
    is_bool

.. _api-methods-pointwise:

Pointwise math
~~~~~~~~~~~~~~

Apply arithmetic and elementwise functions to every entry. The Python
operators (``+``, ``-``, ``*``, ``/``, ``**``, ``==``, ``&``, ...) call
these methods. A method whose name ends with an underscore, such as
``add_``, writes into the existing tensors: see
:ref:`inplace-conventions`. For broadcasting and for tensordicts with
different keys, see :ref:`operations`.

.. rubric:: Arithmetic

.. autosummary::
    :toctree: generated/
    :template: method.rst

    TensorDictBase.add
    TensorDictBase.add_
    TensorDictBase.sub
    TensorDictBase.sub_
    TensorDictBase.rsub
    TensorDictBase.mul
    TensorDictBase.mul_
    TensorDictBase.div
    TensorDictBase.div_
    TensorDictBase.pow
    TensorDictBase.pow_
    TensorDictBase.mod
    TensorDictBase.lerp
    TensorDictBase.lerp_
    TensorDictBase.addcdiv
    TensorDictBase.addcdiv_
    TensorDictBase.addcmul
    TensorDictBase.addcmul_
    TensorDictBase.clamp
    TensorDictBase.clamp_min
    TensorDictBase.clamp_min_
    TensorDictBase.clamp_max
    TensorDictBase.clamp_max_
    TensorDictBase.maximum
    TensorDictBase.maximum_
    TensorDictBase.minimum
    TensorDictBase.minimum_
    TensorDictBase.zero_

.. rubric:: Elementwise functions

.. autosummary::
    :toctree: generated/
    :template: method.rst

    TensorDictBase.abs
    TensorDictBase.abs_
    TensorDictBase.neg
    TensorDictBase.neg_
    TensorDictBase.reciprocal
    TensorDictBase.reciprocal_
    TensorDictBase.sqrt
    TensorDictBase.sqrt_
    TensorDictBase.exp
    TensorDictBase.exp_
    TensorDictBase.expm1
    TensorDictBase.expm1_
    TensorDictBase.log
    TensorDictBase.log_
    TensorDictBase.log10
    TensorDictBase.log10_
    TensorDictBase.log1p
    TensorDictBase.log1p_
    TensorDictBase.log2
    TensorDictBase.log2_
    TensorDictBase.sin
    TensorDictBase.sin_
    TensorDictBase.cos
    TensorDictBase.cos_
    TensorDictBase.tan
    TensorDictBase.tan_
    TensorDictBase.asin
    TensorDictBase.asin_
    TensorDictBase.acos
    TensorDictBase.acos_
    TensorDictBase.atan
    TensorDictBase.atan_
    TensorDictBase.sinh
    TensorDictBase.sinh_
    TensorDictBase.cosh
    TensorDictBase.cosh_
    TensorDictBase.tanh
    TensorDictBase.tanh_
    TensorDictBase.sigmoid
    TensorDictBase.sigmoid_
    TensorDictBase.erf
    TensorDictBase.erf_
    TensorDictBase.erfc
    TensorDictBase.erfc_
    TensorDictBase.lgamma
    TensorDictBase.lgamma_
    TensorDictBase.ceil
    TensorDictBase.ceil_
    TensorDictBase.floor
    TensorDictBase.floor_
    TensorDictBase.round
    TensorDictBase.round_
    TensorDictBase.trunc
    TensorDictBase.trunc_
    TensorDictBase.frac
    TensorDictBase.frac_
    TensorDictBase.sign
    TensorDictBase.sign_
    TensorDictBase.softmax

.. rubric:: Comparisons and logical operations

.. autosummary::
    :toctree: generated/
    :template: method.rst

    TensorDictBase.__eq__
    TensorDictBase.__ne__
    TensorDictBase.__lt__
    TensorDictBase.__le__
    TensorDictBase.__gt__
    TensorDictBase.__ge__
    TensorDictBase.__and__
    TensorDictBase.__or__
    TensorDictBase.__xor__
    TensorDictBase.__invert__
    TensorDictBase.bitwise_and
    TensorDictBase.logical_and
    TensorDictBase.isfinite
    TensorDictBase.isnan
    TensorDictBase.isneginf
    TensorDictBase.isposinf
    TensorDictBase.isreal

.. _api-methods-reductions:

Reductions
~~~~~~~~~~

Reduce each entry over all its elements, or along batch dimensions with
``dim``. With ``reduce=True``, most reductions return one tensor for the
whole tensordict. See :ref:`operations`.

.. autosummary::
    :toctree: generated/
    :template: method.rst

    TensorDictBase.all
    TensorDictBase.any
    TensorDictBase.amax
    TensorDictBase.amin
    TensorDictBase.max
    TensorDictBase.min
    TensorDictBase.cummax
    TensorDictBase.cummin
    TensorDictBase.mean
    TensorDictBase.nanmean
    TensorDictBase.sum
    TensorDictBase.nansum
    TensorDictBase.prod
    TensorDictBase.std
    TensorDictBase.var
    TensorDictBase.quantile
    TensorDictBase.logsumexp
    TensorDictBase.norm

.. _api-methods-functional:

Apply, modules, and gradients
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Apply a function to every entry, and map a function over the batch. Put
the parameters of a module in a tensordict and back, and compute
gradients.
See :ref:`operations`, :ref:`modules-guide`, and :ref:`compile`.

.. rubric:: Methods

.. autosummary::
    :toctree: generated/
    :template: method.rst

    TensorDictBase.apply
    TensorDictBase.apply_
    TensorDictBase.named_apply
    TensorDictBase.map
    TensorDictBase.map_iter
    TensorDictBase.from_module
    TensorDictBase.from_modules
    TensorDictBase.to_module
    TensorDictBase.backward
    TensorDictBase.requires_grad
    TensorDictBase.requires_grad_
    TensorDictBase.detach
    TensorDictBase.detach_
    TensorDictBase.grad
    TensorDictBase.data
    TensorDictBase.zero_grad

.. rubric:: Functions

.. autosummary::
    :toctree: generated/
    :template: function.rst

    from_module
    from_modules
    default_is_leaf
    is_leaf_nontensor

.. _api-methods-saving:

Saving, loading, and file formats
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Save a tensordict to disk and load it again, and write its entries to
memory-mapped files. Other methods read and write key-value stores, and HDF5,
zarr, CSV, JSON, Parquet, and MDS files.
``memmap(return_early=True)`` returns a
:class:`~tensordict.utils.TensorDictFuture`. See :ref:`saving` and
:ref:`storage`.

.. rubric:: Methods

.. autosummary::
    :toctree: generated/
    :template: method.rst

    TensorDictBase.save
    TensorDictBase.load
    TensorDictBase.dumps
    TensorDictBase.memmap
    TensorDictBase.memmap_
    TensorDictBase.memmap_like
    TensorDictBase.memmap_refresh_
    TensorDictBase.load_memmap
    TensorDictBase.load_memmap_
    TensorDictBase.make_memmap
    TensorDictBase.make_memmap_from_storage
    TensorDictBase.make_memmap_from_tensor
    TensorDictBase.is_memmap
    TensorDictBase.saved_path
    TensorDictBase.state_dict
    TensorDictBase.load_state_dict
    TensorDictBase.from_h5
    TensorDictBase.to_h5
    TensorDictBase.from_zarr
    TensorDictBase.to_zarr
    TensorDictBase.from_csv
    TensorDictBase.to_csv
    TensorDictBase.from_json
    TensorDictBase.to_json
    TensorDictBase.from_parquet
    TensorDictBase.to_parquet
    TensorDictBase.to_mds
    TensorDictBase.to_store

.. rubric:: Functions

.. autosummary::
    :toctree: generated/
    :template: function.rst

    save
    load
    memmap
    load_memmap
    append_memmap
    pack_memmap
    unpack_memmap
    is_memmap_archive
    refresh_archive_checksums
    from_h5
    from_zarr
    from_csv
    from_json
    from_parquet

.. _api-methods-locking:

Locking
~~~~~~~

A locked tensordict raises an error if you add, replace, rename, or delete
an entry. In-place writes with ``set_`` and ``update_`` still work, also for
non-tensor entries. ``unlock_`` also works in a ``with`` block: the
tensordict is locked again at the end of the block. See :ref:`keys`.

.. autosummary::
    :toctree: generated/
    :template: method.rst

    TensorDictBase.lock_
    TensorDictBase.unlock_
    TensorDictBase.is_locked

.. _api-methods-distributed:

Distributed and remote
~~~~~~~~~~~~~~~~~~~~~~

Send and receive tensordicts between processes with
:mod:`torch.distributed`, run collective operations on them, and stream
them over UCXX (see :ref:`api-storage`). See :ref:`distributed`.

.. autosummary::
    :toctree: generated/
    :template: method.rst

    TensorDictBase.send
    TensorDictBase.recv
    TensorDictBase.isend
    TensorDictBase.irecv
    TensorDictBase.all_reduce
    TensorDictBase.reduce
    TensorDictBase.broadcast
    TensorDictBase.scatter
    TensorDictBase.all_gather
    TensorDictBase.gather_and_stack
    TensorDictBase.init_remote
    TensorDictBase.from_remote_init
    TensorDictBase.asend
    TensorDictBase.arecv

.. _api-methods-deprecated:

Deprecated methods
~~~~~~~~~~~~~~~~~~

These methods still work, and each call raises a ``DeprecationWarning``.
Their pages name the replacement and the release that removes them.

.. autosummary::
    :toctree: generated/
    :template: method.rst

    TensorDictBase.load_
    TensorDictBase.pin_memory_

Other functions
---------------

.. _api-functions-checks:

Type checks and testing
~~~~~~~~~~~~~~~~~~~~~~~

Test whether an object is a tensordict, a tensorclass, or non-tensor data,
and compare two tensordicts in tests.

.. autosummary::
    :toctree: generated/
    :template: function.rst

    is_tensor_collection
    is_tensorclass
    is_non_tensor
    assert_close
    assert_allclose_td

.. _api-functions-printing:

Printing
~~~~~~~~

Choose what the representation of a tensordict shows, and rebuild a
tensordict from its printed form. See :ref:`printing`.

.. autosummary::
    :toctree: generated/
    :template: function.rst

    set_printoptions
    get_printoptions
    parse_tensor_dict_string

.. _api-functions-settings:

Global settings
~~~~~~~~~~~~~~~

These functions read or change a behavior of tensordict, for the whole
process or inside a ``with`` block. ``set_list_to_stack(False)`` and
``set_capture_non_tensor_stack(True)`` are deprecated, and their pages say
what to use instead.

.. autosummary::
    :toctree: generated/
    :template: function.rst

    list_to_stack
    set_list_to_stack
    capture_non_tensor_stack
    set_capture_non_tensor_stack

.. _api-functions-deprecated:

Deprecated functions
~~~~~~~~~~~~~~~~~~~~

These functions still work. Their pages name the replacement and the
release that removes them.

.. autosummary::
    :toctree: generated/
    :template: function.rst

    get_defaults_to_none
    set_get_defaults_to_none
    lazy_legacy
    set_lazy_legacy

Package constants
-----------------

.. py:data:: KEY_SEPARATOR
    :module: tensordict

    ``"."``, the default ``separator`` of
    :meth:`~tensordict.TensorDictBase.flatten_keys` and
    :meth:`~tensordict.TensorDictBase.unflatten_keys`, for code that joins or
    splits nested keys the same way, such as
    ``tuple(name.split(KEY_SEPARATOR))``. The names of
    :meth:`~tensordict.TensorDictBase.state_dict` also use ``"."``, the
    ``torch.nn.Module`` convention, independently of this constant.
    Reassigning it has no effect: the defaults are bound when the methods are
    defined. It can also be imported from ``tensordict.utils``.

.. py:data:: NO_DEFAULT
    :module: tensordict.base

    The sentinel that marks an argument as not given where ``None`` is a valid
    value, such as ``default`` in :meth:`~tensordict.TensorDictBase.get_non_tensor`
    or ``device`` in :meth:`~tensordict.TensorDictBase.new_zeros`. It can also
    be imported from ``tensordict.tensordict``.

.. py:data:: __version__
    :module: tensordict

    The version string of the installed tensordict package, or ``None`` if it
    cannot be determined, for example when tensordict is imported from a source
    tree that is not installed.

See also
--------

- :ref:`overview`, :ref:`batch-size`, :ref:`keys`, :ref:`stacking`,
  :ref:`operations`, and :ref:`devices`: the user guide pages that explain
  these methods.
- :ref:`saving`, :ref:`storage`, :ref:`data-loading`, and
  :ref:`distributed`: the how-to guides.
- :ref:`api-utils`: the helpers of ``tensordict.utils``.
