.. _printing:

Printing and display
====================

This page shows what ``print(td)`` displays for a
:class:`~tensordict.TensorDict`, how to choose the attributes that it shows
with :class:`~tensordict.set_printoptions`, and how to rebuild a tensordict
from a printed one. The full list of options is in `Available options`_.

What ``print`` shows
--------------------

``print(td)`` shows the structure of a tensordict, not the values of its
tensors: the key, shape, device and dtype of each field, and the batch size
and device of the tensordict:

    >>> import torch
    >>> from tensordict import TensorDict
    >>> td = TensorDict(
    ...     image=torch.randn(32, 3, 64, 64),
    ...     label=torch.randint(10, (32,)),
    ...     batch_size=[32],
    ... )
    >>> print(td)
    TensorDict(
        fields={
            image: Tensor(shape=torch.Size([32, 3, 64, 64]), device=cpu, dtype=torch.float32, is_shared=False),
            label: Tensor(shape=torch.Size([32]), device=cpu, dtype=torch.int64, is_shared=False)},
        batch_size=torch.Size([32]),
        device=None,
        is_shared=False)

Configuring the display with ``set_printoptions``
-------------------------------------------------

By default, the repr shows the shape, device, dtype and ``is_shared`` of each
field, and the batch size, device and ``is_shared`` of the tensordict. Some
of these attributes can be noise. For instance, if all your work is on the
CPU and nothing is shared, ``device=cpu`` and ``is_shared=False`` are
repeated on every line without adding information.

:class:`~tensordict.set_printoptions` selects the attributes that appear. It
works as a global setter, a context manager or a decorator, as
:class:`torch.no_grad` does for the last two. Its role is similar to
:func:`torch.set_printoptions`, which is a plain function and not a context
manager or a decorator.

Global configuration
~~~~~~~~~~~~~~~~~~~~

Call ``set_printoptions(...).set()`` to change the defaults for the rest of
the process. ``show_device`` and ``show_is_shared`` control the lines of the
tensordict, and ``show_field_device`` and ``show_field_is_shared`` the
attributes of each field:

    >>> from tensordict import set_printoptions
    >>> set_printoptions(
    ...     show_device=False,
    ...     show_is_shared=False,
    ...     show_field_device=False,
    ...     show_field_is_shared=False,
    ... ).set()
    >>> print(td)
    TensorDict(
        fields={
            image: Tensor(shape=torch.Size([32, 3, 64, 64]), dtype=torch.float32),
            label: Tensor(shape=torch.Size([32]), dtype=torch.int64)},
        batch_size=torch.Size([32]))

``set()`` changes only the options that it gets, so
``set_printoptions().set()`` without arguments does not reset anything. To go
back to the defaults, set the changed options again:

    >>> set_printoptions(
    ...     show_device=True,
    ...     show_is_shared=True,
    ...     show_field_device=True,
    ...     show_field_is_shared=True,
    ... ).set()

Scoped configuration (context manager)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use the context manager to change the options for one block of code. At the
end of the block, the previous options come back:

    >>> with set_printoptions(show_dtype=False, show_field_is_shared=False):
    ...     print(td)  # dtype and per-field is_shared hidden
    TensorDict(
        fields={
            image: Tensor(shape=torch.Size([32, 3, 64, 64]), device=cpu),
            label: Tensor(shape=torch.Size([32]), device=cpu)},
        batch_size=torch.Size([32]),
        device=None,
        is_shared=False)
    >>> print(td)  # back to defaults
    TensorDict(
        fields={
            image: Tensor(shape=torch.Size([32, 3, 64, 64]), device=cpu, dtype=torch.float32, is_shared=False),
            label: Tensor(shape=torch.Size([32]), device=cpu, dtype=torch.int64, is_shared=False)},
        batch_size=torch.Size([32]),
        device=None,
        is_shared=False)

Decorator
~~~~~~~~~

Decorate a function to use the options for every ``repr`` call in it. Here,
``show_is_shared=False`` hides the ``is_shared=`` line of the tensordict:

    >>> @set_printoptions(show_is_shared=False)
    ... def summarize(td):
    ...     print(td)
    >>> summarize(td)
    TensorDict(
        fields={
            image: Tensor(shape=torch.Size([32, 3, 64, 64]), device=cpu, dtype=torch.float32, is_shared=False),
            label: Tensor(shape=torch.Size([32]), device=cpu, dtype=torch.int64, is_shared=False)},
        batch_size=torch.Size([32]),
        device=None)

Compact shapes-only display (``verbose=False``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To see only the shapes, pass ``verbose=False``. It hides the device, dtype
and ``is_shared`` information in one argument:

    >>> with set_printoptions(verbose=False):
    ...     print(td)
    TensorDict(
        fields={
            image: Tensor(shape=torch.Size([32, 3, 64, 64])),
            label: Tensor(shape=torch.Size([32]))},
        batch_size=torch.Size([32]))

``verbose=True`` (the default) changes nothing. Explicit keyword arguments
take precedence over ``verbose``, so you can mix the two:

    >>> with set_printoptions(verbose=False, show_dtype=True):
    ...     print(td)
    TensorDict(
        fields={
            image: Tensor(shape=torch.Size([32, 3, 64, 64]), dtype=torch.float32),
            label: Tensor(shape=torch.Size([32]), dtype=torch.int64)},
        batch_size=torch.Size([32]))

Available options
~~~~~~~~~~~~~~~~~

These options control the lines of the outer ``TensorDict(...)`` block:

====================  ===========  ===================================================
Option                Default      Description
====================  ===========  ===================================================
``show_batch_size``   ``True``     Show the ``batch_size=`` line.
``show_device``       ``True``     Show the ``device=`` line.
``show_is_shared``    ``True``     Show the ``is_shared=`` line.
====================  ===========  ===================================================

These options control the attributes in the ``Tensor(...)`` description of
each field:

========================  ===========  ============================================
Option                    Default      Description
========================  ===========  ============================================
``show_shape``            ``True``     Show the ``shape=`` attribute.
``show_field_device``     ``True``     Show the ``device=`` attribute.
``show_dtype``            ``True``     Show the ``dtype=`` attribute.
``show_field_is_shared``  ``True``     Show the ``is_shared=`` attribute.
========================  ===========  ============================================

This option is a shortcut for several of the options above:

====================  ===========  ===================================================
Option                Default      Description
====================  ===========  ===================================================
``verbose``           ``True``     When ``False``, hides ``show_device``,
                                   ``show_is_shared``, ``show_field_device``,
                                   ``show_dtype``, and ``show_field_is_shared``.
                                   Explicit keyword arguments override ``verbose``.
====================  ===========  ===================================================

These options add attributes that are hidden by default, and change the order
of the keys:

======================  ==================  =============================================
Option                  Default             Description
======================  ==================  =============================================
``show_grad``           ``False``            Show ``requires_grad=``.
``show_is_contiguous``  ``False``            Show ``is_contiguous=``.
``show_is_view``        ``False``            Show ``is_view=`` (whether ``._base`` is set).
``show_storage_size``   ``False``            Show ``storage_size=`` (bytes).
``plain``               ``False``            Append a short value summary
                                             (mean/std for floats, min/max for ints).
``sort_keys``           ``"alphabetical"``   Key ordering: ``"alphabetical"`` (default),
                                             ``"insertion"`` (dict order), or a callable
                                             passed as ``key`` to :func:`sorted`.
======================  ==================  =============================================

Querying the current settings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:func:`~tensordict.get_printoptions` returns a dict with the current values:

    >>> from tensordict import get_printoptions
    >>> get_printoptions()
    {'show_batch_size': True, 'show_device': True, ...}

Reconstructing a TensorDict from its printed representation
-----------------------------------------------------------

When you debug, you can get a tensordict repr as a string, for example from a
log file or from the terminal of a colleague.
:func:`~tensordict.parse_tensor_dict_string` builds a
:class:`~tensordict.TensorDict` from that string. The result has the keys,
shapes, dtypes and batch size of the printed tensordict, and its device if
the device has no index. All the values are zeros, because the repr does not
contain the data:

    >>> from tensordict import parse_tensor_dict_string
    >>> s = """TensorDict(
    ...     fields={
    ...         image: Tensor(shape=torch.Size([32, 3, 64, 64]), device=cpu, dtype=torch.float32, is_shared=False),
    ...         label: Tensor(shape=torch.Size([32]), device=cpu, dtype=torch.int64, is_shared=False)},
    ...     batch_size=torch.Size([32]),
    ...     device=cpu,
    ...     is_shared=False)"""
    >>> td = parse_tensor_dict_string(s)
    >>> td.batch_size
    torch.Size([32])
    >>> td["image"].shape
    torch.Size([32, 3, 64, 64])

:func:`~tensordict.parse_tensor_dict_string` is a debugging aid. It reads only
the default print format, with ``shape=``, ``device=``, ``dtype=`` and
``is_shared=`` for each field, and it has these limits:

- If :class:`~tensordict.set_printoptions` hid a field attribute or added one
  (for example ``plain=True``), the parser finds no fields. It returns a
  tensordict with the batch size and no tensor entries, and it does not
  raise an error.
- It drops non-tensor fields
  (`#1956 <https://github.com/pytorch/tensordict/issues/1956>`_), and fields
  on a device with an index, such as ``cuda:0``, without an error. The repr
  of a tensordict with ``device=cuda:0`` raises
  ``ValueError: Device not found in the string``.
- It reads one level of nesting. With two levels or more, the result also
  has a wrong copy of each deeper tensordict at the top level.
- It reads keys that contain only letters, digits and ``_``. A key such as
  ``my-key`` comes back as ``key``.
- It does not keep ``is_shared``, and it returns a
  :class:`~tensordict.TensorDict` for the repr of a lazy stack.

See also
--------

- The reference entries of :class:`~tensordict.set_printoptions`,
  :func:`~tensordict.get_printoptions` and
  :func:`~tensordict.parse_tensor_dict_string`.
- :ref:`devices`: what ``device=None`` in the repr means.
- :ref:`non-tensor-data`: how non-tensor entries appear in the repr.
