.. _api-reference:

API reference
=============

Each page of the API reference lists the public classes and functions of
one part of tensordict, and each name has its own page. To learn how the
parts fit together, start with :ref:`overview` and the user guide.

- :ref:`api-tensordict`: the container classes (``TensorDict``,
  ``LazyStackedTensorDict``, ``PersistentTensorDict``, ...), the methods
  that all tensordicts share, grouped by task, and the functions of the
  ``tensordict`` module.
- :ref:`api-tensorclass`: ``TensorClass``, the ``@tensorclass``
  decorator, and the classes that hold non-tensor data.
- :ref:`api-typed-tensordict`: ``TypedTensorDict``, a tensordict with
  declared fields.
- :ref:`api-nn`: the modules of ``tensordict.nn``, the probabilistic
  modules and distributions, and FX tracing.
- :ref:`api-storage`: the tensordicts on key-value servers
  (``tensordict.store``), and UCXX transport.
- :ref:`api-utils`: the helpers and types of ``tensordict.utils``.

.. toctree::
    :hidden:
    :maxdepth: 1

    td
    tc
    ttd
    nn
    storage
    utils
