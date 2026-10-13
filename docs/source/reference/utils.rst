.. _api-utils:

Utilities
=========

.. module:: tensordict.utils

This page lists the names of ``tensordict.utils`` that the ``tensordict``
module does not export: helpers for tensors and tensordicts, the types
that some methods return, and type aliases. Import them from
``tensordict.utils``. The names that ``tensordict`` also exports, such as
:func:`~tensordict.unravel_key` or :func:`~tensordict.assert_close`, are
on :ref:`api-tensordict`.

Tensor and tensordict helpers
-----------------------------

``expand_right`` and ``expand_as_right`` add dimensions on the right of a
tensor, where broadcasting adds them on the left. ``isin`` and
``remove_duplicates`` compare the values of one entry along a batch
dimension.

.. autosummary::
    :toctree: generated/
    :template: function.rst

    expand_right
    expand_as_right
    isin
    remove_duplicates

Returned types
--------------

The methods that have a ``return_early`` argument, such as
:meth:`~tensordict.TensorDictBase.memmap`, return a
:class:`TensorDictFuture` if you pass ``return_early=True``.
:meth:`~tensordict.TensorDictBase.tolist` returns
a :class:`LinkedList` if you pass ``as_linked_list=True``.

.. autosummary::
    :toctree: generated/
    :template: class.rst

    TensorDictFuture
    LinkedList

Type aliases
------------

``DeviceType`` is the type of the ``device`` arguments, and ``IndexType``
is the type of the indices that a tensordict accepts.

.. autosummary::
    :toctree: generated/
    :template: function.rst

    DeviceType
    IndexType

Debugging and profiling
-----------------------

``print_directory_tree`` prints the files of a directory, for example
the directory of a saved tensordict. ``timeit`` and ``logger`` are the
timer and the logger that tensordict uses in its own code. ``Buffer`` is
:class:`torch.nn.Buffer <torch.nn.parameter.Buffer>`, kept here so that old
imports of ``tensordict.utils.Buffer`` still work.

.. autosummary::
    :toctree: generated/
    :template: function.rst

    print_directory_tree
    timeit
    Buffer

.. py:data:: logger

    The :class:`logging.Logger` named ``"tensordict"``. To show fewer
    messages from tensordict, raise the level of this logger, for example
    ``logging.getLogger("tensordict").setLevel(logging.WARNING)``.

See also
--------

- :ref:`api-tensordict`: the functions of the ``tensordict`` module.
- :ref:`saving`: ``memmap(return_early=True)`` and the files of a saved
  tensordict.
