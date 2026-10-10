.. currentmodule:: tensordict

.. _api-tensorclass:

tensorclass and non-tensor data
===============================

A *tensorclass* is a class that you declare like a dataclass and that has
the operations of a :class:`~tensordict.TensorDict`. Its fields are typed
attributes, not string keys. To declare one, subclass
:class:`~tensordict.TensorClass`, or decorate a class with
:func:`~tensordict.tensorclass`. For how to declare fields, auto-casting,
the class options and static type checking, see :ref:`tensorclass-guide`.

A tensorclass has the methods of :class:`~tensordict.TensorDictBase`,
listed by task in :ref:`api-tensordict-methods`. A method that returns a
``TensorDict`` for a ``TensorDict`` returns an instance of the tensorclass
for a tensorclass: ``select``, ``clone``, ``to``, ``apply``, indexing and
reshaping are examples.

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
:class:`~tensordict.TensorAttrs` describes the device, dtype and shape of
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
- :ref:`choosing-a-container`: tensorclass, ``TypedTensorDict`` or
  ``TensorDict``.
- :ref:`api-typed-tensordict`: the other container with typed fields.
- The tutorials :doc:`/tutorials/tensorclass_fashion` and
  :doc:`/tutorials/tensorclass_imagenet`.
