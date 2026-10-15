.. currentmodule:: tensordict

.. _api-typed-tensordict:

TypedTensorDict
===============

.. warning::

    ``TypedTensorDict`` is experimental: its API can change in a minor release
    without a deprecation cycle.

This page lists :class:`~tensordict.TypedTensorDict`, a
:class:`~tensordict.TensorDictBase` subclass whose fields you declare as
class annotations, as in a ``TypedDict``. A subclass adds fields to the
fields of its parent, and :meth:`~tensordict.TypedTensorDict.from_tensordict`
wraps any tensordict (H5, Redis, lazy stacks) without a copy. For a guide
with examples, see :ref:`typed-tensordict-guide`.

``TypedTensorDict`` has the methods of :class:`~tensordict.TensorDictBase`,
listed by task in :ref:`api-tensordict-methods`.

.. autosummary::
    :toctree: generated/
    :template: tensordict_class.rst

    TypedTensorDict

See also
--------

- :ref:`typed-tensordict-guide`: declare and use a ``TypedTensorDict``.
- :ref:`choosing-a-container`: tensorclass, ``TypedTensorDict``, or
  ``TensorDict``.
- :ref:`compatibility`: which containers work on which storage backends.
