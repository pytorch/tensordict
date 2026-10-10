.. _non-tensor-data:

Non-tensor data and unbatched entries
=====================================

Most entries of a tensordict are tensors whose first dimensions are the
batch size (see :ref:`batch-size`). Indexing, reshaping and stacking change
these dimensions in all the entries at once. This page covers two kinds of
entries that do not follow these batch rules: values that are not tensors,
such as strings and other Python objects, and tensors that belong to the
whole batch, wrapped in :class:`~tensordict.UnbatchedTensor`.

Non-tensor data
---------------

A tensordict can store values that are not tensors. Three classes hold them:

- :class:`~tensordict.NonTensorData` holds one value for the whole batch.
  Indexing returns the same value.
- :class:`~tensordict.NonTensorStack` holds one value for each element of
  the batch.
- :class:`~tensordict.MetaData` holds one value for the whole batch, like
  :class:`~tensordict.NonTensorData`. Stacking gives one
  :class:`~tensordict.MetaData` if all the values are the same Python object.

What becomes a tensor
~~~~~~~~~~~~~~~~~~~~~

When you set a value that is not a tensor, tensordict converts it:

- Python numbers, booleans, NumPy arrays and lists of numbers become
  tensors. ``True`` becomes ``tensor(True)``.
- A dict becomes a nested tensordict.
- A list of other values, such as strings, becomes a
  :class:`~tensordict.NonTensorStack` with one value for each item of the
  list. If the tensordict has a batch size, the length of the list must be
  the first batch dimension. Otherwise, the tensordict raises a
  ``RuntimeError``.
- Any other value, such as a string or ``None``, becomes a
  :class:`~tensordict.NonTensorData`.

    >>> import torch
    >>> import numpy as np
    >>> from tensordict import MetaData, NonTensorData, NonTensorStack, TensorDict
    >>> td = TensorDict(
    ...     flag=True, n=3, arr=np.zeros(2), nums=[1, 2], text="hello", words=["x", "y"]
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

    >>> td = TensorDict(flag=NonTensorData(True), batch_size=[2])
    >>> td["flag"]
    True
    >>> _ = td.set_non_tensor("labels", ["cat", "dog", "bird"])
    >>> td[0]["labels"]
    ['cat', 'dog', 'bird']

That booleans become tensors is tracked in
`#1199 <https://github.com/pytorch/tensordict/issues/1199>`_.

Creating a TensorDict with non-tensor data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can also create the :class:`~tensordict.NonTensorData` yourself. It is
stored in the tensordict like a tensor:

    >>> td = TensorDict(
    ...     a=NonTensorData("a string!"),
    ...     b=torch.zeros(()),
    ... )
    >>> print(td)
    TensorDict(
        fields={
            a: NonTensorData(data=a string!, batch_size=torch.Size([]), device=None),
            b: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False)},
        batch_size=torch.Size([]),
        device=None,
        is_shared=False)

Accessing non-tensor data
~~~~~~~~~~~~~~~~~~~~~~~~~

Indexing with a key (``td["a"]``) returns the content of the
:class:`~tensordict.NonTensorData`. :meth:`~tensordict.TensorDict.get`
returns the :class:`~tensordict.NonTensorData` object itself.
:meth:`~tensordict.TensorDict.get_non_tensor` returns the content, as
``td["a"]`` does:

    >>> print(td["a"])
    a string!
    >>> print(td.get("a"))
    NonTensorData(data=a string!, batch_size=torch.Size([]), device=None)
    >>> td.get_non_tensor("a")
    'a string!'

Batched non-tensor data
~~~~~~~~~~~~~~~~~~~~~~~

In a tensordict with a batch size, a :class:`~tensordict.NonTensorData`
gets the batch size of the tensordict:

    >>> td = TensorDict(
    ...     a=NonTensorData("a string!"),
    ...     b=torch.zeros(3),
    ...     batch_size=[3]
    ... )
    >>> print(td)
    TensorDict(
        fields={
            a: NonTensorData(data=a string!, batch_size=torch.Size([3]), device=None),
            b: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False)},
        batch_size=torch.Size([3]),
        device=None,
        is_shared=False)

All the elements of the batch have the same value:

    >>> print(td[0])
    TensorDict(
        fields={
            a: NonTensorData(data=a string!, batch_size=torch.Size([]), device=None),
            b: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False)},
        batch_size=torch.Size([]),
        device=None,
        is_shared=False)

If you write a different value into one element, the entry becomes a
:class:`~tensordict.NonTensorStack`:

    >>> td[0] = TensorDict(a="another string!", b=torch.ones(()))
    >>> type(td.get("a")).__name__
    'NonTensorStack'

Stacked non-tensor data
~~~~~~~~~~~~~~~~~~~~~~~

To give each element of the batch its own value, use a
:class:`~tensordict.NonTensorStack`:

    >>> td = TensorDict(
    ...     a=NonTensorStack("a string!", "another string!", "a third string!"),
    ...     b=torch.zeros(3),
    ...     batch_size=[3]
    ... )
    >>> print(td)
    TensorDict(
        fields={
            a: NonTensorStack(
                ['a string!', 'another string!', 'a third string!'...,
                batch_size=torch.Size([3]),
                device=None),
            b: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False)},
        batch_size=torch.Size([3]),
        device=None,
        is_shared=False)

Index the tensordict to get one element. The first element has the first
string:

    >>> print(td[0])
    TensorDict(
        fields={
            a: NonTensorData(data=a string!, batch_size=torch.Size([]), device=None),
            b: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False)},
        batch_size=torch.Size([]),
        device=None,
        is_shared=False)

For a stack, ``td["a"]`` returns a ``tensordict.utils.LinkedList``. It is a
subclass of :class:`list`, and an assignment to one of its items writes into
the tensordict. :meth:`~tensordict.NonTensorStack.tolist` returns a plain
:class:`list`:

    >>> values = td["a"]
    >>> isinstance(values, list)
    True
    >>> values[1] = "a new string!"
    >>> td[1]["a"]
    'a new string!'
    >>> td.get("a").tolist()
    ['a string!', 'a new string!', 'a third string!']

In contrast, :class:`~tensordict.NonTensorData` with a list does not give
the same result. A non-tensor value can be any Python object, so the list is
one value for the whole batch:

    >>> td = TensorDict(
    ...     a=NonTensorData(["a string!", "another string!", "a third string!"]),
    ...     b=torch.zeros(3),
    ...     batch_size=[3]
    ... )
    >>> print(td[0])
    TensorDict(
        fields={
            a: NonTensorData(data=['a string!', 'another string!', 'a third string!'], batch_size=torch.Size([]), device=None),
            b: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False)},
        batch_size=torch.Size([]),
        device=None,
        is_shared=False)

Stacking tensordicts with non-tensor data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:func:`~torch.stack` makes a :class:`~tensordict.NonTensorStack` from
:class:`~tensordict.NonTensorData` entries, also if their values are the
same:

    >>> td = TensorDict(
    ...     a=NonTensorData("a string!"),
    ...     b=torch.zeros(()),
    ... )
    >>> print(torch.stack([td, td]))
    TensorDict(
        fields={
            a: NonTensorStack(
                ['a string!', 'a string!'],
                batch_size=torch.Size([2]),
                device=None),
            b: Tensor(shape=torch.Size([2]), device=cpu, dtype=torch.float32, is_shared=False)},
        batch_size=torch.Size([2]),
        device=None,
        is_shared=False)

For :class:`~tensordict.MetaData` entries, :func:`~torch.stack` returns one
:class:`~tensordict.MetaData` if all the stacked values are the same Python
object. It compares them with ``is``, not with ``==``. Otherwise, it returns
a :class:`~tensordict.NonTensorStack`, as for
:class:`~tensordict.NonTensorData`:

    >>> td = TensorDict(
    ...     a=MetaData("a string!"),
    ...     b=torch.zeros(()),
    ... )
    >>> print(torch.stack([td, td]))
    TensorDict(
        fields={
            a: MetaData(data=a string!, batch_size=torch.Size([2]), device=None),
            b: Tensor(shape=torch.Size([2]), device=cpu, dtype=torch.float32, is_shared=False)},
        batch_size=torch.Size([2]),
        device=None,
        is_shared=False)
    >>> equal_lists = [TensorDict(a=MetaData([1, 2])), TensorDict(a=MetaData([1, 2]))]
    >>> type(torch.stack(equal_lists).get("a")).__name__
    'NonTensorStack'

:func:`~torch.unsqueeze` also keeps one :class:`~tensordict.MetaData`, and
makes a :class:`~tensordict.NonTensorStack` from a
:class:`~tensordict.NonTensorData`. Use :class:`~tensordict.MetaData` for a
value that is the same for every element, such as a label or a configuration
dict, and that must not become a stack.

Non-tensor data in other operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Pointwise operations, :meth:`~tensordict.TensorDict.apply` and
  :meth:`~tensordict.TensorDict.to` do not change non-tensor entries. The
  result has them as they are (see :ref:`operations`).
- ``keys(leaves_only=True)``, ``values(leaves_only=True)`` and
  ``items(leaves_only=True)`` skip the non-tensor entries. To include them,
  pass ``is_leaf=tensordict.is_leaf_nontensor``:

    >>> from tensordict import is_leaf_nontensor
    >>> td = TensorDict(text="x", t=torch.zeros(()))
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
  drops them (see :ref:`printing`).

.. _unbatched-tensor:

UnbatchedTensor
---------------

Some tensors belong to the whole batch, not to each element. Examples are a
configuration tensor, a mask, or parameters that all the elements share. The
shape of such a tensor does not start with the batch size. To store it in a
tensordict, wrap it in an :class:`~tensordict.UnbatchedTensor`.

Contract
~~~~~~~~

:class:`~tensordict.UnbatchedTensor` follows these rules:

- It is a tensor. The wrapped data is a real ``torch.Tensor`` and can be used
  as one.
- Operations that change only the batch shape do not change it.
  ``reshape``, ``view``, ``unbind``, ``split``, ``squeeze``, ``unsqueeze``,
  ``permute``, ``transpose``, ``flatten``, ``unflatten``, ``chunk``,
  ``expand``, ``cat``, ``stack`` and indexing return the same
  :class:`~tensordict.UnbatchedTensor`, with the same storage.
- Other operations work as for a tensor: device and dtype casts (``to``),
  pointwise arithmetic (``+``, ``-``, ``*``, ``/``), ``clone``, gradient
  computation and access by key.
- Python conversions follow PyTorch. ``str()``, ``bool()``, ``int()``,
  ``float()``, ``complex()`` and conversions to an integer index go to the
  wrapped tensor.

Usage
~~~~~

    >>> from tensordict import TensorDict, UnbatchedTensor
    >>> import torch
    >>> td = TensorDict(
    ...     a=torch.randn(2, 3),
    ...     config=UnbatchedTensor(torch.tensor([1.0, 2.0, 3.0])),
    ...     batch_size=[2, 3],
    ... )

Shape operations and indexing on the tensordict keep the storage of the
:class:`~tensordict.UnbatchedTensor`:

    >>> reshaped = td.reshape(6)
    >>> reshaped["config"].data_ptr() == td["config"].data_ptr()
    True
    >>> parts = td.unbind(0)
    >>> parts[0]["config"].data_ptr() == parts[1]["config"].data_ptr()
    True
    >>> td[0]["config"].shape
    torch.Size([3])

Pointwise arithmetic applies to the wrapped data:

    >>> td2 = td * 2
    >>> torch.equal(td2["config"], torch.tensor([2.0, 4.0, 6.0]))
    True

An :class:`~tensordict.UnbatchedTensor` with one element converts like a
regular PyTorch tensor:

    >>> int(UnbatchedTensor(torch.tensor([3])))
    3
    >>> f"{UnbatchedTensor(torch.tensor(1.25)):.1f}"
    '1.2'

Accessing values
~~~~~~~~~~~~~~~~

For a plain tensordict, both :meth:`~tensordict.TensorDict.get` and
``td[key]`` return the :class:`~tensordict.UnbatchedTensor` wrapper:

    >>> type(td.get("config"))
    <class 'tensordict._unbatched.UnbatchedTensor'>
    >>> type(td["config"])
    <class 'tensordict._unbatched.UnbatchedTensor'>

For a tensorclass, both attribute access and ``get()`` also return the
:class:`~tensordict.UnbatchedTensor` wrapper (see :ref:`tensorclass-guide`):

    >>> from tensordict import TensorClass
    >>> class MyClass(TensorClass):
    ...     config: torch.Tensor
    ...     value: torch.Tensor
    >>> tc = MyClass(
    ...     config=UnbatchedTensor(torch.tensor([1.0, 2.0])),
    ...     value=torch.randn(3),
    ...     batch_size=[3],
    ... )
    >>> type(tc.config)
    <class 'tensordict._unbatched.UnbatchedTensor'>
    >>> type(tc.get("config"))
    <class 'tensordict._unbatched.UnbatchedTensor'>

Limitations
~~~~~~~~~~~

- Memory-mapped serialization and consolidation do not support
  :class:`~tensordict.UnbatchedTensor` yet.
  :meth:`~tensordict.TensorDict.memmap` and
  :meth:`~tensordict.TensorDict.consolidate` raise ``NotImplementedError``
  when the tensordict contains one. :func:`torch.save` keeps them.
- If you stack tensordicts whose :class:`~tensordict.UnbatchedTensor`
  entries have different data, the result keeps only the data of the first
  one, and it warns.

See also
--------

- :ref:`api-tensorclass`: the reference entries of
  :class:`~tensordict.NonTensorData`, :class:`~tensordict.NonTensorStack`,
  :class:`~tensordict.MetaData` and :class:`~tensordict.UnbatchedTensor`.
- :ref:`stacking`: stacks of tensordicts, and variable-size data.
- :ref:`tensorclass-guide`: tensorclass fields that hold non-tensor values.
- :ref:`saving`: how non-tensor entries are saved.
