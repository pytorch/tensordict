.. _non-tensor-data:

Non-tensor data and unbatched entries
=====================================

A tensordict can store strings, labels, and other Python objects next to its
tensors. Indexing and stacking keep them aligned with the batch. A tensordict
can also hold a tensor that belongs to the whole batch, such as a parameter
that all the elements share. Shape operations leave that tensor as it is.

Non-tensor data
---------------

In this example, each DNA read has a sequence and its GC content, the
fraction of G and C bases. A mask on the GC content selects the matching
sequences too:

    >>> import torch
    >>> from tensordict import TensorDict
    >>> reads = TensorDict(
    ...     {
    ...         "seq": ["ACGT", "GGCC", "TTAG"],
    ...         "gc": torch.tensor([0.50, 1.00, 0.25]),
    ...     },
    ...     batch_size=[3],
    ... )
    >>> gc_rich = reads[reads["gc"] > 0.4]
    >>> gc_rich.batch_size
    torch.Size([2])
    >>> gc_rich.get("seq").tolist()
    ['ACGT', 'GGCC']

Three classes hold values that are not tensors:

- :class:`~tensordict.NonTensorData` holds one value for the whole batch.
  Indexing returns the same value.
- :class:`~tensordict.NonTensorStack` holds one value for each element of
  the batch. The list of sequences above becomes one.
- :class:`~tensordict.MetaData` holds one value for the whole batch, like
  :class:`~tensordict.NonTensorData`. Stacking gives one
  :class:`~tensordict.MetaData` if all the values are the same Python object.

What becomes a tensor
~~~~~~~~~~~~~~~~~~~~~

When you set a value that is not a tensor, tensordict converts it:

- Python numbers, booleans, NumPy arrays, and lists of numbers become
  tensors. ``True`` becomes ``tensor(True)``.
- A dict becomes a nested tensordict.
- A list of other values, such as strings, becomes a
  :class:`~tensordict.NonTensorStack` with one value for each item of the
  list. If the tensordict has a batch size, the length of the list must be
  the first batch dimension, or the tensordict raises a ``RuntimeError``.
- Any other value, such as a string or ``None``, becomes a
  :class:`~tensordict.NonTensorData`.

This example shows the type of each entry:

    >>> import numpy as np
    >>> from tensordict import MetaData, NonTensorData, NonTensorStack
    >>> td = TensorDict(
    ...     {"flag": True, "n": 3, "arr": np.zeros(2), "nums": [1, 2], "text": "hello", "words": ["x", "y"]}
    ... )
    >>> for key in ["flag", "n", "arr", "nums", "text", "words"]:
    ...     print(key, type(td.get(key)).__name__)
    flag Tensor
    n Tensor
    arr Tensor
    nums Tensor
    text NonTensorData
    words NonTensorStack

To keep a number or a boolean as a Python value, wrap it in
:class:`~tensordict.NonTensorData`. To store a list as one value, and not as
one value for each element, use :meth:`~tensordict.TensorDict.set_non_tensor`
or wrap the list in :class:`~tensordict.NonTensorData`:

    >>> td = TensorDict({"flag": NonTensorData(True)}, batch_size=[2])
    >>> td["flag"]
    True
    >>> _ = td.set_non_tensor("labels", ["cat", "dog", "bird"])
    >>> td[0]["labels"]
    ['cat', 'dog', 'bird']

In a tensordict without batch dimensions, a number or a boolean, as a Python or
a NumPy scalar, written over an existing :class:`~tensordict.NonTensorData`
entry also becomes a tensor, and the write emits a :class:`FutureWarning`.
Starting with TensorDict 0.17, the default will change to keeping it as
:class:`~tensordict.NonTensorData`, like the entry it replaces
(`#1199 <https://github.com/pytorch/tensordict/issues/1199>`_).
In the same way, :meth:`~tensordict.TensorDict.set_at_` casts a list or a
number written at an index of a non-tensor entry to a tensor, and emits a
:class:`FutureWarning`. Starting with TensorDict 0.17, the default will change
to storing it as non-tensor data
(`#2522 <https://github.com/pytorch/tensordict/issues/2522>`_).

``td[index] = value``, with a list or a number, writes the value into every
entry, and casts it to a tensor in the non-tensor entries too. The write emits a
:class:`FutureWarning`. Starting with TensorDict 0.17, the default will change
to leaving the non-tensor entries unchanged, as a
:class:`~tensordict.store.TensorDictStore` does for a number. To reset the
tensor entries of a tensordict with a string entry, for example:

.. code-block:: python

    td = TensorDict(x=torch.ones(3), name=NonTensorStack("a", "b", "c"), batch_size=[3])
    td[mask] = 0
    # today: td["x"][mask] is 0 and td["name"][mask] holds tensor(0), with a FutureWarning
    # from 0.17: td["x"][mask] is 0 and td["name"] is still ["a", "b", "c"]

To leave the non-tensor entries unchanged now, write into
``td.exclude("name")[mask]``, which shares the tensors of ``td``.

Read a non-tensor entry
~~~~~~~~~~~~~~~~~~~~~~~

You can also create the :class:`~tensordict.NonTensorData` yourself. The
tensordict stores it like a tensor. Here, the entry records the reference
genome of the reads:

    >>> td = TensorDict(
    ...     {
    ...         "genome": NonTensorData("hg38"),
    ...         "score": torch.zeros(()),
    ...     },
    ... )
    >>> print(td)
    TensorDict(batch_size=[], fields={genome: str, score: f32[]})

Indexing with a key (``td["genome"]``) returns the content of the
:class:`~tensordict.NonTensorData`. :meth:`~tensordict.TensorDict.get`
returns the :class:`~tensordict.NonTensorData` object itself.
:meth:`~tensordict.TensorDict.get_non_tensor` returns the content, as
``td["genome"]`` does:

    >>> print(td["genome"])
    hg38
    >>> print(td.get("genome"))
    NonTensorData(data=hg38, batch_size=torch.Size([]), device=None)
    >>> td.get_non_tensor("genome")
    'hg38'

One value for the batch, or one for each element
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In a tensordict with a batch size, a :class:`~tensordict.NonTensorData`
gets the batch size of the tensordict:

    >>> td = TensorDict(
    ...     {
    ...         "genome": NonTensorData("hg38"),
    ...         "score": torch.zeros(3),
    ...     },
    ...     batch_size=[3]
    ... )
    >>> print(td)
    TensorDict(batch_size=[3], fields={genome: str, score: f32[]})

All the elements of the batch have the same value:

    >>> print(td[0])
    TensorDict(batch_size=[], fields={genome: str, score: f32[]})

If you write a different value into one element, the entry becomes a
:class:`~tensordict.NonTensorStack`:

    >>> td[0] = TensorDict({"genome": "mm10", "score": torch.ones(())})
    >>> type(td.get("genome")).__name__
    'NonTensorStack'

When ``td[index] = src`` brings a non-tensor key that a dense ``td`` does not
have, a :class:`~tensordict.NonTensorData` in ``src`` (one value for the whole
batch) fills every row of ``td``, written or not, while a
:class:`~tensordict.NonTensorStack` (one value for each row) leaves ``None`` in
the rows that ``index`` does not select:

    >>> td = TensorDict({"score": torch.zeros(4)}, batch_size=[4])
    >>> td[1:3] = TensorDict({"score": torch.ones(2), "genome": "hg38"}, batch_size=[2])
    >>> td["genome"]
    'hg38'
    >>> td = TensorDict({"score": torch.zeros(4)}, batch_size=[4])
    >>> td[1:3] = TensorDict(
    ...     {"score": torch.ones(2), "seq": NonTensorStack("ACGT", "GGCC")}, batch_size=[2]
    ... )
    >>> td.get("seq").tolist()
    [None, 'ACGT', 'GGCC', None]

To give each element of the batch its own value, use a
:class:`~tensordict.NonTensorStack`:

    >>> td = TensorDict(
    ...     {
    ...         "seq": NonTensorStack("ACGT", "GGCC", "TTAG"),
    ...         "score": torch.zeros(3),
    ...     },
    ...     batch_size=[3]
    ... )
    >>> print(td)
    TensorDict(batch_size=[3], fields={score: f32[], seq: NonTensorStack})

Index the tensordict to get one element. The first element has the first
sequence:

    >>> print(td[0])
    TensorDict(batch_size=[], fields={score: f32[], seq: str})

For a stack, ``td["seq"]`` returns a ``tensordict.utils.LinkedList``. It is a
subclass of :class:`list`, and an assignment to one of its items writes into
the tensordict. :meth:`~tensordict.NonTensorStack.tolist` and
:meth:`~tensordict.utils.LinkedList.tolist` return a plain :class:`list`:

    >>> values = td["seq"]
    >>> isinstance(values, list)
    True
    >>> values[1] = "CCGG"
    >>> td[1]["seq"]
    'CCGG'
    >>> td.get("seq").tolist()
    ['ACGT', 'CCGG', 'TTAG']
    >>> td["seq"].tolist()
    ['ACGT', 'CCGG', 'TTAG']

A :class:`~tensordict.NonTensorData` that holds a list does not give one
value for each element. A non-tensor value can be any Python object, so the
list is one value for the whole batch:

    >>> td = TensorDict(
    ...     {
    ...         "seq": NonTensorData(["ACGT", "GGCC", "TTAG"]),
    ...         "score": torch.zeros(3),
    ...     },
    ...     batch_size=[3]
    ... )
    >>> print(td[0])
    TensorDict(batch_size=[], fields={score: f32[], seq: list})

Stacking tensordicts with non-tensor data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:func:`~torch.stack` makes a :class:`~tensordict.NonTensorStack` from
:class:`~tensordict.NonTensorData` entries, also if their values are the
same:

    >>> td = TensorDict(
    ...     {
    ...         "genome": NonTensorData("hg38"),
    ...         "score": torch.zeros(()),
    ...     },
    ... )
    >>> print(torch.stack([td, td]))
    TensorDict(batch_size=[2], fields={genome: NonTensorStack, score: f32[]})

For :class:`~tensordict.MetaData` entries, :func:`~torch.stack` returns one
:class:`~tensordict.MetaData` if all the stacked values are the same Python
object. It compares them with ``is``, not with ``==``. Otherwise, it returns
a :class:`~tensordict.NonTensorStack`, as for
:class:`~tensordict.NonTensorData`:

    >>> td = TensorDict(
    ...     {
    ...         "genome": MetaData("hg38"),
    ...         "score": torch.zeros(()),
    ...     },
    ... )
    >>> print(torch.stack([td, td]))
    TensorDict(batch_size=[2], fields={genome: MetaData, score: f32[]})
    >>> equal_lists = [TensorDict({"a": MetaData([1, 2])}), TensorDict({"a": MetaData([1, 2])})]
    >>> type(torch.stack(equal_lists).get("a")).__name__
    'NonTensorStack'

:func:`~torch.unsqueeze` also keeps one :class:`~tensordict.MetaData`, and
makes a :class:`~tensordict.NonTensorStack` from a
:class:`~tensordict.NonTensorData`. Use :class:`~tensordict.MetaData` for a
value that must not become a stack, such as a label or a configuration dict
that is the same for every element.

Non-tensor data in other operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Arithmetic, :meth:`~tensordict.TensorDict.apply`, and
  :meth:`~tensordict.TensorDict.to` pass non-tensor entries to the result as
  they are. ``==`` and ``!=`` between two tensordicts compare them (see
  :ref:`operations`).
- ``keys(leaves_only=True)``, ``values(leaves_only=True)``, and
  ``items(leaves_only=True)`` skip the non-tensor entries. To include them,
  pass ``is_leaf=tensordict.is_leaf_nontensor``:

    >>> from tensordict import is_leaf_nontensor
    >>> td = TensorDict({"text": "x", "t": torch.zeros(())})
    >>> list(td.keys(leaves_only=True))
    ['t']
    >>> list(td.keys(leaves_only=True, is_leaf=is_leaf_nontensor))
    ['text', 't']

- Under :func:`torch.compile`, a function that returns a string entry
  compiles again for each new string, and non-strict
  :func:`torch.export.export` does not accept non-tensor entries. See
  :ref:`compile`.
- :meth:`~tensordict.TensorDict.share_memory_` can change non-tensor values
  (see :ref:`devices`), and :func:`~tensordict.parse_tensor_dict_string`
  raises a ``ValueError`` for them (see :ref:`printing`).

.. _unbatched-tensor:

UnbatchedTensor
---------------

.. warning::

    ``UnbatchedTensor`` is experimental: its API can change in a minor release
    without a deprecation cycle.

Some tensors belong to the whole batch, not to each element: a configuration
tensor, a mask, or parameters that all the elements share. The shape of such
a tensor does not start with the batch size. To store it in a tensordict, wrap
it in an :class:`~tensordict.UnbatchedTensor`. In this example, 2 systems of 3
particles share one gravity vector:

    >>> from tensordict import UnbatchedTensor
    >>> td = TensorDict(
    ...     {
    ...         "height": torch.randn(2, 3),
    ...         "gravity": UnbatchedTensor(torch.tensor([0.0, 0.0, -9.8])),
    ...     },
    ...     batch_size=[2, 3],
    ... )

Shape operations and indexing on the tensordict keep the storage of the
:class:`~tensordict.UnbatchedTensor`:

    >>> reshaped = td.reshape(6)
    >>> reshaped["gravity"].data_ptr() == td["gravity"].data_ptr()
    True
    >>> parts = td.unbind(0)
    >>> parts[0]["gravity"].data_ptr() == parts[1]["gravity"].data_ptr()
    True
    >>> td[0]["gravity"].shape
    torch.Size([3])

Pointwise arithmetic applies to the wrapped data:

    >>> td2 = td * 2
    >>> torch.equal(td2["gravity"], torch.tensor([0.0, 0.0, -19.6]))
    True

Rules
~~~~~

:class:`~tensordict.UnbatchedTensor` follows these rules:

- It is a tensor. The wrapped data is a real ``torch.Tensor`` and can be used
  as one.
- Operations that change only the batch shape do not change it.
  ``reshape``, ``view``, ``unbind``, ``split``, ``squeeze``, ``unsqueeze``,
  ``permute``, ``transpose``, ``flatten``, ``unflatten``, ``chunk``,
  ``expand``, ``cat``, ``stack``, and indexing return the same
  :class:`~tensordict.UnbatchedTensor`, with the same storage.
- :meth:`~tensordict.TensorDictBase.apply` and
  :meth:`~tensordict.TensorDictBase.named_apply` with a ``batch_size`` (also
  with ``out=``) count as shape operations: the function is not called on an
  :class:`~tensordict.UnbatchedTensor`, which keeps its data and takes the new
  batch size. Without a ``batch_size``, the function is called on it.
- Other operations work as for a tensor: device and dtype casts (``to``),
  pointwise arithmetic (``+``, ``-``, ``*``, ``/``), ``clone``, gradient
  computation, and access by key.
- Python conversions follow PyTorch. ``str()``, ``bool()``, ``int()``,
  ``float()``, ``complex()``, and conversions to an integer index go to the
  wrapped tensor.

For example, an :class:`~tensordict.UnbatchedTensor` with one element
converts like a regular PyTorch tensor:

    >>> int(UnbatchedTensor(torch.tensor([3])))
    3
    >>> f"{UnbatchedTensor(torch.tensor(1.25)):.1f}"
    '1.2'

Read an unbatched entry
~~~~~~~~~~~~~~~~~~~~~~~

For a plain tensordict, both :meth:`~tensordict.TensorDict.get` and
``td[key]`` return the :class:`~tensordict.UnbatchedTensor` wrapper:

    >>> type(td.get("gravity"))
    <class 'tensordict._unbatched.UnbatchedTensor'>
    >>> type(td["gravity"])
    <class 'tensordict._unbatched.UnbatchedTensor'>

For a tensorclass, both attribute access and ``get()`` also return the
:class:`~tensordict.UnbatchedTensor` wrapper (see :ref:`tensorclass-guide`):

    >>> from tensordict import TensorClass
    >>> class Bodies(TensorClass):
    ...     gravity: torch.Tensor
    ...     height: torch.Tensor
    >>> bodies = Bodies(
    ...     gravity=UnbatchedTensor(torch.tensor([0.0, 0.0, -9.8])),
    ...     height=torch.randn(3),
    ...     batch_size=[3],
    ... )
    >>> type(bodies.gravity)
    <class 'tensordict._unbatched.UnbatchedTensor'>
    >>> type(bodies.get("gravity"))
    <class 'tensordict._unbatched.UnbatchedTensor'>

Limitations
~~~~~~~~~~~

- Memory-mapped serialization to a directory and consolidation do not
  support :class:`~tensordict.UnbatchedTensor` yet.
  :meth:`~tensordict.TensorDict.memmap` with a directory prefix and
  :meth:`~tensordict.TensorDict.consolidate` raise ``NotImplementedError``
  when the tensordict contains one. :func:`torch.save` keeps them.
- :meth:`~tensordict.TensorDict.to_h5`,
  :meth:`~tensordict.TensorDict.to_zarr` and ``.tdz`` archives (written by
  :meth:`~tensordict.TensorDict.memmap` or
  :meth:`~tensordict.TensorDict.save` with a ``.tdz`` path) save
  :class:`~tensordict.UnbatchedTensor` entries as plain tensors, and they
  are read back as plain tensors. Such a save emits a ``UserWarning``.
- If you stack tensordicts whose :class:`~tensordict.UnbatchedTensor`
  entries have different data, the result keeps only the data of the first
  one, and it warns.

See also
--------

- :ref:`api-tensorclass`: the reference entries of
  :class:`~tensordict.NonTensorData`, :class:`~tensordict.NonTensorStack`,
  :class:`~tensordict.MetaData`, and :class:`~tensordict.UnbatchedTensor`.
- :ref:`stacking`: stacks of tensordicts, and variable-size data.
- :ref:`tensorclass-guide`: tensorclass fields that hold non-tensor values.
- :ref:`saving`: how non-tensor entries are saved.
