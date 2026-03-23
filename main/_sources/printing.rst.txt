Printing and Display
====================

Why printing a TensorDict is more useful than printing a tensor
---------------------------------------------------------------

When working with PyTorch tensors, calling ``print(tensor)`` dumps the raw
numerical content.  In practice, however, most ``print`` calls during debugging
are motivated by a single question: *what does this tensor look like?*  You want
its shape, dtype, device, maybe whether it requires a gradient -- not a wall of
floating-point numbers.

Because a :class:`~tensordict.TensorDict` groups multiple tensors under named
keys, its ``__repr__`` gives you exactly that -- a structured, at-a-glance
summary of every tensor it contains:

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

No data is printed, no truncation ellipses, no guessing at dimensionality.
One glance tells you the names, shapes, dtypes and devices of everything in
the batch.

Configuring the display with ``set_printoptions``
-------------------------------------------------

By default, every attribute is shown for backward compatibility.  In many
situations, though, some of those attributes are noise.  For instance, if all
your work is on CPU and nothing is shared, ``device=cpu`` and
``is_shared=False`` are repeated on every line without adding information.

:class:`~tensordict.set_printoptions` lets you control exactly which attributes
appear.  It works as a **global setter**, a **context manager** or a
**decorator**, following the same pattern as :class:`~tensordict.set_lazy_legacy`
and :func:`torch.set_printoptions`.

Global configuration
~~~~~~~~~~~~~~~~~~~~

Call :meth:`~tensordict.set_printoptions.set` to change the defaults for the
rest of the process:

    >>> from tensordict import set_printoptions
    >>> set_printoptions(show_device=False, show_is_shared=False).set()
    >>> print(td)
    TensorDict(
        fields={
            image: Tensor(shape=torch.Size([32, 3, 64, 64]), dtype=torch.float32),
            label: Tensor(shape=torch.Size([32]), dtype=torch.int64)},
        batch_size=torch.Size([32]))

Scoped configuration (context manager)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use the context-manager form when you only want the change for a specific
block of code.  The previous settings are automatically restored on exit:

    >>> from tensordict import set_printoptions
    >>> with set_printoptions(show_dtype=False, show_is_shared=False):
    ...     print(td)  # dtype and is_shared hidden
    >>> print(td)  # back to defaults

Decorator
~~~~~~~~~

You can also decorate a function so that every ``repr`` call inside it uses
the specified options:

    >>> @set_printoptions(show_is_shared=False)
    ... def summarise(td):
    ...     print(td)

Compact shapes-only display (``verbose=False``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you mainly care about shapes and find the other attributes noisy, pass
``verbose=False`` to hide device, dtype, and is_shared information in one shot:

    >>> with set_printoptions(verbose=False):
    ...     print(td)
    TensorDict(
        fields={
            image: Tensor(shape=torch.Size([32, 3, 64, 64])),
            label: Tensor(shape=torch.Size([32]))},
        batch_size=torch.Size([32]))

``verbose=True`` (the default) is a no-op and preserves backward
compatibility.  Explicit keyword arguments always take precedence over
``verbose``, so you can mix the two:

    >>> with set_printoptions(verbose=False, show_dtype=True):
    ...     print(td)
    TensorDict(
        fields={
            image: Tensor(shape=torch.Size([32, 3, 64, 64]), dtype=torch.float32),
            label: Tensor(shape=torch.Size([32]), dtype=torch.int64)},
        batch_size=torch.Size([32]))

Available options
~~~~~~~~~~~~~~~~~

**TensorDict-level** (these control lines in the outer ``TensorDict(...)``
block):

====================  ===========  ===================================================
Option                Default      Description
====================  ===========  ===================================================
``show_batch_size``   ``True``     Show the ``batch_size=`` line.
``show_device``       ``True``     Show the ``device=`` line.
``show_is_shared``    ``True``     Show the ``is_shared=`` line.
====================  ===========  ===================================================

**Tensor-level** (these control what appears inside each
``Tensor(...)`` field descriptor):

========================  ===========  ============================================
Option                    Default      Description
========================  ===========  ============================================
``show_shape``            ``True``     Show the ``shape=`` attribute.
``show_field_device``     ``True``     Show the ``device=`` attribute.
``show_dtype``            ``True``     Show the ``dtype=`` attribute.
``show_field_is_shared``  ``True``     Show the ``is_shared=`` attribute.
========================  ===========  ============================================

**Shortcut**:

====================  ===========  ===================================================
Option                Default      Description
====================  ===========  ===================================================
``verbose``           ``True``     When ``False``, hides ``show_device``,
                                   ``show_is_shared``, ``show_field_device``,
                                   ``show_dtype``, and ``show_field_is_shared``.
                                   Explicit keyword arguments override ``verbose``.
====================  ===========  ===================================================

**Extended attributes** (off by default -- opt-in for deeper debugging):

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

When debugging, it is common to receive a TensorDict repr as a string -- for
example, pasted from a log file or a colleague's terminal.
:func:`~tensordict.parse_tensor_dict_string` can reconstruct a dummy
:class:`~tensordict.TensorDict` from that string.  The resulting object has the
correct structure, batch size, device and dtypes, but all tensor values are
replaced by zeros (since the repr does not contain actual data):

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

.. note::

    :func:`~tensordict.parse_tensor_dict_string` currently only works with the
    default (``plain``) print format -- the one that includes ``shape=``,
    ``device=``, ``dtype=`` and ``is_shared=`` for every field.
    If attributes have been hidden via :class:`~tensordict.set_printoptions`,
    the regex parser will not find the expected fields and reconstruction will
    fail.  Support for non-default formats will be added in a follow-up PR.
