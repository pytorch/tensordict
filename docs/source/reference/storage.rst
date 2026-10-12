.. _api-storage:

Key-value stores and UCXX
=========================

This page lists the tensordicts that keep their entries on a key-value
server, and the classes and functions that stream tensordicts between
processes over UCXX. For how to use them, see :ref:`storage` and
:ref:`distributed`.

Key-value stores
----------------

.. module:: tensordict.store

A :class:`~tensordict.store.TensorDictStore` keeps its entries on a server
that speaks the Redis protocol, such as Redis or Dragonfly. The data is on
the server, and each read gets it from the server. A
:class:`~tensordict.store.LazyStackedTensorDictStore` is a lazy stack of
tensordicts on such a server, so that its entries can have different
shapes. You need the ``redis`` package and a running server.

.. warning::

    ``LazyStackedTensorDictStore`` is experimental: its API can change in a
    minor release without a deprecation cycle.

Both classes have the methods of :class:`~tensordict.TensorDictBase`,
listed by task in :ref:`api-tensordict-methods`. Their pages list the
methods that they add, such as ``from_store``, ``close`` and
``clear_redis``. ``STORE_BACKENDS`` is the type of their ``backend``
argument.

.. autosummary::
    :toctree: generated/
    :template: tensordict_class.rst

    TensorDictStore
    LazyStackedTensorDictStore

.. autosummary::
    :toctree: generated/
    :template: function.rst

    STORE_BACKENDS

UCXX transport
--------------

.. currentmodule:: tensordict._ucxx

.. warning::

    ``TensorDictPipe``, ``TensorDictServer``, ``send_tensordict`` and
    ``recv_tensordict`` are experimental: their API can change in a minor
    release without a deprecation cycle.

UCXX is the Python interface of UCX, a communication library that
chooses the transport between two processes, such as shared memory,
InfiniBand or TCP. These classes and functions need the optional ``ucxx``
package.

- A :class:`~tensordict._ucxx.TensorDictPipe` is a channel between two
  processes. After the first tensordict, it sends only the data of the
  next tensordicts with the same structure. The receiver writes each
  tensordict into the same buffer, so clone a received tensordict if you
  need it after the next receive.
- A :class:`~tensordict._ucxx.TensorDictServer` accepts connections and
  returns one pipe for each client.
- :func:`~tensordict._ucxx.send_tensordict` and
  :func:`~tensordict._ucxx.recv_tensordict` send or receive one
  tensordict over a UCXX endpoint, with its metadata each time.

:meth:`TensorDictBase.asend() <tensordict.TensorDictBase.asend>` and
:meth:`TensorDictBase.arecv() <tensordict.TensorDictBase.arecv>` send
and receive through a pipe.

Import these names from ``tensordict._ucxx``. It is their only import
path for now, although the module name starts with an underscore.

.. autosummary::
    :toctree: generated/
    :template: class.rst

    TensorDictPipe
    TensorDictServer

.. autosummary::
    :toctree: generated/
    :template: function.rst

    send_tensordict
    recv_tensordict

See also
--------

- :ref:`storage`: start a server, write a tensordict to it and read it
  back.
- :ref:`compatibility`: which containers work on a store.
- :ref:`distributed`: send tensordicts between processes.
- :ref:`api-methods-distributed`: the send and receive methods of every
  tensordict.
