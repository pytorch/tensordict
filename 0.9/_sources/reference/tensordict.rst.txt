.. currentmodule:: tensordict

tensordict package
==================

The :class:`~tensordict.TensorDict` class simplifies the process of passing multiple tensors
from module to module by packing them in a dictionary-like object that inherits features from
regular pytorch tensors.


.. autosummary::
    :toctree: generated/
    :template: td_template.rst

    TensorDictBase
    TensorDict
    LazyStackedTensorDict
    PersistentTensorDict
    TensorDictParams
    get_defaults_to_none

Constructors and handlers
-------------------------

The library offers a few method to interact with other data structures such as numpy structured arrays, namedtuples or
h5 files. The library also exposes dedicated functions to manipulate tensordicts such as ``save``, ``load``, ``stack``
or ``cat``.

.. autosummary::
    :toctree: generated/
    :template: td_template.rst

    cat
    default_is_leaf
    from_any
    from_consolidated
    from_dict
    from_h5
    from_module
    from_modules
    from_namedtuple
    from_pytree
    from_struct_array
    from_tuple
    fromkeys
    is_batchedtensor
    is_leaf_nontensor
    lazy_stack
    load
    load_memmap
    maybe_dense_stack
    memmap
    save
    stack

TensorDict as a context manager
-------------------------------

:class:`~tensordict.TensorDict` can be used as a context manager in situations
where an action has to be done and then undone. This include temporarily
locking/unlocking a tensordict

    >>> data.lock_()  # data.set will result in an exception
    >>> with data.unlock_():
    ...     data.set("key", value)
    >>> assert data.is_locked()

or to execute functional calls with a TensorDict instance containing the
parameters and buffers of a model:

    >>> params = TensorDict.from_module(module).clone()
    >>> params.zero_()
    >>> with params.to_module(module):
    ...     y = module(x)

In the first example, we can modify the tensordict `data` because we have
temporarily unlocked it. In the second example, we populate the module with the
parameters and buffers contained in the `params` tensordict instance, and reset
the original parameters after this call is completed.

Memory-mapped tensors
---------------------

`tensordict` offers the :class:`~tensordict.MemoryMappedTensor` primitive which
allows you to work with tensors stored in physical memory in a handy way.
The main advantages of :class:`~tensordict.MemoryMappedTensor`
are its ease of construction (no need to handle the storage of a tensor),
the possibility to work with big contiguous data that would not fit in memory,
an efficient (de)serialization across processes and efficient indexing of
stored tensors.

If all workers have access to the same storage (both in multiprocess and distributed
settings), passing a :class:`~tensordict.MemoryMappedTensor`
will just consist in passing a reference to a file on disk plus a bunch of
extra meta-data for reconstructing it. The same goes with indexed memory-mapped
tensors as long as the data-pointer of their storage is the same as the original
one.

Indexing memory-mapped tensors is much faster than loading several independent files from
the disk and does not require to load the full content of the array in memory.
However, physical storage of PyTorch tensors should not be any different:

.. code-block:: Python

  >>> my_images = MemoryMappedTensor.empty((1_000_000, 3, 480, 480), dtype=torch.unint8)
  >>> mini_batch = my_images[:10]  # just reads the first 10 images of the dataset

.. autosummary::
    :toctree: generated/
    :template: td_template_noinherit.rst

    MemoryMappedTensor

Pointwise Operations
--------------------

Tensordict supports various pointwise operations, allowing you to perform element-wise computations on the tensors
stored within it. These operations are similar to those performed on regular PyTorch tensors.

Supported Operations
~~~~~~~~~~~~~~~~~~~~

The following pointwise operations are currently supported:

- Left and right addition (`+`)
- Left and right subtraction (`-`)
- Left and right multiplication (`*`)
- Left and right division (`/`)
- Left power (`**`)

Many other ops, like :meth:`~tensordict.TensorDict.clamp`, :meth:`~tensordict.TensorDict.sqrt` etc. are supported.

Performing Pointwise Operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can perform pointwise operations between two Tensordicts or between a Tensordict and a tensor/scalar value.

Example 1: Tensordict-Tensordict Operation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    >>> import torch
    >>> from tensordict import TensorDict
    >>> td1 = TensorDict(
    ...     a=torch.randn(3, 4),
    ...     b=torch.zeros(3, 4, 5),
    ...     c=torch.ones(3, 4, 5, 6),
    ...     batch_size=(3, 4),
    ... )
    >>> td2 = TensorDict(
    ...     a=torch.randn(3, 4),
    ...     b=torch.zeros(3, 4, 5),
    ...     c=torch.ones(3, 4, 5, 6),
    ...     batch_size=(3, 4),
    ... )
    >>> result = td1 * td2

In this example, the * operator is applied element-wise to the corresponding tensors in td1 and td2.

Example 2: Tensordict-Tensor Operation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    >>> import torch
    >>> from tensordict import TensorDict
    >>> td = TensorDict(
    ...     a=torch.randn(3, 4),
    ...     b=torch.zeros(3, 4, 5),
    ...     c=torch.ones(3, 4, 5, 6),
    ...     batch_size=(3, 4),
    ... )
    >>> tensor = torch.randn(4)
    >>> result = td * tensor

ere, the * operator is applied element-wise to each tensor in td and the provided tensor. The tensor is broadcasted to match the shape of each tensor in the Tensordict.

Example 3: Tensordict-Scalar Operation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    >>> import torch
    >>> from tensordict import TensorDict
    >>> td = TensorDict(
    ...     a=torch.randn(3, 4),
    ...     b=torch.zeros(3, 4, 5),
    ...     c=torch.ones(3, 4, 5, 6),
    ...     batch_size=(3, 4),
    ... )
    >>> scalar = 2.0
    >>> result = td * scalar

In this case, the * operator is applied element-wise to each tensor in td and the provided scalar.

Broadcasting Rules
~~~~~~~~~~~~~~~~~~

When performing pointwise operations between a Tensordict and a tensor/scalar, the tensor/scalar is broadcasted to match
the shape of each tensor in the Tensordict: the tensor is broadcast on the left to match the tensordict shape, then
individually broadcast on the right to match the tensors shapes. This follows the standard broadcasting rules used in
PyTorch if one thinks of the ``TensorDict`` as a single tensor instance.

For example, if you have a Tensordict with tensors of shape ``(3, 4)`` and you multiply it by a tensor of shape ``(4,)``,
the tensor will be broadcasted to shape (3, 4) before the operation is applied. If the tensordict contains a tensor of
shape ``(3, 4, 5)``, the tensor used for the multiplication will be broadcast to ``(3, 4, 5)`` on the right for that
multiplication.

If the pointwise operation is executed across multiple tensordicts and their batch-size differ, they will be
broadcasted to a common shape.

Efficiency of pointwise operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When possible, ``torch._foreach_<op>`` fused kernels will be used to speed up the computation of the pointwise
operation.

Handling Missing Entries
~~~~~~~~~~~~~~~~~~~~~~~~

When performing pointwise operations between two Tensordicts, they must have the same keys.
Some operations, like :meth:`~tensordict.TensorDict.add`, have a ``default`` keyword argument that can be used
to operate with tensordict with exclusive entries.
If ``default=None`` (the default), the two Tensordicts must have exactly matching key sets.
If ``default="intersection"``, only the intersecting key sets will be considered, and other keys will be ignored.
In all other cases, ``default`` will be used for all missing entries on both sides of the operation.

Utils
-----

.. autosummary::
    :toctree: generated/
    :template: td_template.rst

    utils.expand_as_right
    utils.expand_right
    utils.isin
    utils.remove_duplicates
    capture_non_tensor_stack
    dense_stack_tds
    is_batchedtensor
    is_tensor_collection
    lazy_legacy
    make_tensordict
    merge_tensordicts
    pad
    pad_sequence
    parse_tensor_dict_string
    set_capture_non_tensor_stack
    set_lazy_legacy
    set_list_to_stack
    list_to_stack
