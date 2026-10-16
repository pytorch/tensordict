.. currentmodule:: tensordict

.. _api-tensorclass:

tensorclass and non-tensor data
===============================

This page lists :class:`~tensordict.TensorClass`, the class form of a
tensordict, and the classes for non-tensor data and unbatched entries. A
*tensorclass* is a class that you declare like a dataclass, with typed fields
and methods, and that has the tensor operations of a
:class:`~tensordict.TensorDict`. For a guide to fields, auto-casting, the
class options, and static type checking, see :ref:`tensorclass-guide`.

To declare a tensorclass, subclass :class:`~tensordict.TensorClass`. The
:func:`~tensordict.tensorclass` decorator is the original way, kept for
existing code (see :ref:`tensorclass-legacy-decorator`).

A tensorclass has the methods of :class:`~tensordict.TensorDictBase`,
listed by task in :ref:`api-tensordict-methods`. On a tensorclass, the
methods that return a ``TensorDict`` on a ``TensorDict`` return an
instance of the tensorclass. Examples are ``select``, ``clone``, ``to``,
``apply``, indexing, and reshaping.
On a tensorclass, :meth:`~tensordict.TensorDictBase.rename_keys` raises
``AttributeError`` for a destination that is not a field, and keeps the
sub-tensordict fields that it empties.

Tensorclasses
-------------

.. autosummary::
    :toctree: generated/
    :template: tensordict_class.rst

    TensorClass

.. autosummary::
    :toctree: generated/
    :template: function.rst

    tensorclass
    from_dataclass

Non-tensor data
---------------

A tensordict or a tensorclass can hold values that are not tensors, such
as strings or Python objects. It wraps each one in a
:class:`~tensordict.NonTensorData`, and a stack of them in a
:class:`~tensordict.NonTensorStack`. If you stack
:class:`~tensordict.MetaData` entries that hold the same object, the
result is one ``MetaData``, not a ``NonTensorStack``.
:class:`~tensordict.TensorAttrs` describes the device, dtype, and shape of
an entry: :meth:`~tensordict.TensorDictBase.attrs` returns them. See
:ref:`non-tensor-data`.

.. autosummary::
    :toctree: generated/
    :template: tensordict_class.rst

    NonTensorDataBase
    NonTensorData
    NonTensorStack
    MetaData
    TensorAttrs

Entries that ignore the batch shape
-----------------------------------

.. warning::

    ``UnbatchedTensor`` is experimental: its API can change in a minor release
    without a deprecation cycle.

An :class:`~tensordict.UnbatchedTensor` is a tensor that shape operations
on its tensordict (indexing, reshaping, stacking) leave unchanged.
Operations on its data, such as arithmetic or
:meth:`~tensordict.TensorDictBase.to`, still apply to it. See
:ref:`unbatched-tensor`.

.. autosummary::
    :toctree: generated/
    :template: class.rst

    UnbatchedTensor

See also
--------

- :ref:`tensorclass-guide`: declare and use a tensorclass.
- :ref:`choosing-a-container`: tensorclass, ``TypedTensorDict``, or
  ``TensorDict``.
- :ref:`api-typed-tensordict`: the other container with typed fields.
- The tutorials :doc:`/tutorials/tensorclass_fashion` and
  :doc:`/tutorials/tensorclass_imagenet`.
