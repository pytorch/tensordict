.. _keys:

Keys, nested keys, and locking
==============================

A tensordict groups related tensors in a tree. One key reads any of them: a
string for the top level, a tuple of strings for a nested entry. A lock fixes
the set of keys, so that no code adds, replaces, or removes an entry by
mistake.

Nested tensordicts and nested keys
----------------------------------

A key is a string. The value of an entry can be another TensorDict: the
constructor converts each nested dictionary into one, with the batch size of
the parent. This training batch has two levels:

  >>> import torch
  >>> from tensordict import TensorDict
  >>> data = TensorDict(
  ...     {
  ...         "inputs": {
  ...             "image": torch.rand(100, 28, 28),
  ...             "mask": torch.randint(2, (100, 28, 28), dtype=torch.uint8),
  ...         },
  ...         "outputs": {"logits": torch.randn(100, 10)},
  ...     },
  ...     batch_size=[100],
  ... )
  >>> type(data["inputs"]).__name__, data["inputs"].batch_size
  ('TensorDict', torch.Size([100]))

A nested key is a tuple of strings, and each string selects one level of the
tree. ``data["a", "b"]`` is the same as ``data[("a", "b")]`` and as
``data["a"]["b"]``. A write with a nested key creates the nested tensordicts
that do not exist yet:

  >>> image = data["inputs", "image"]
  >>> logits = data.get(("outputs", "logits"))
  >>> data["outputs", "probabilities"] = torch.sigmoid(logits)
  >>> data["extra", "count"] = torch.zeros(100)
  >>> data["extra"].batch_size
  torch.Size([100])

A tuple key in the constructor also creates nested entries:
``TensorDict({("x", "y"): torch.zeros(3)}, batch_size=[3])`` has a nested
tensordict ``"x"`` with the key ``"y"``.

Keyword arguments are a shorthand for top-level keys:
``TensorDict(obs=torch.zeros(3), batch_size=[3])`` is the same as
``TensorDict({"obs": torch.zeros(3)}, batch_size=[3])``. In the shorthand, the
names of the constructor arguments (``source``, ``batch_size``, ``device``,
``names``, ``non_blocking``, and ``lock``) are not keys, and a key must be a
Python identifier. A key such as ``"obs.next"`` or a tuple key needs the dict.
These docs build every TensorDict from a dict.

If you know the keys when you write the code, a
:class:`~tensordict.TensorClass` gives you typed attributes in place of string
keys, and your editor completes them (see :ref:`tensorclass-guide`).

Valid keys
----------

The type name :class:`~tensordict.NestedKey` covers the two forms of a key: a
string, or a tuple of strings. A tuple inside a tuple is flattened, and a
tuple of one string is the same as the string:

  >>> from tensordict import unravel_key
  >>> unravel_key(("a", ("b", "c")))
  ('a', 'b', 'c')
  >>> unravel_key(("a",))
  'a'

Any other key raises a ``KeyError``. A string with a dot, such as ``"a.b"``,
is one plain key, not a nested key.

  >>> TensorDict().get(("a", 1))
  Traceback (most recent call last):
  ...
  KeyError: "Only NestedKeys are supported: a key must be a string or a non-empty, possibly nested tuple of strings. Got key ('a', 1)."

List the keys
-------------

:meth:`~tensordict.TensorDict.keys` lists the keys of the top level. Two
arguments change this:

- ``include_nested=True`` also lists the keys inside the nested tensordicts,
  as tuples.
- ``leaves_only=True`` lists only the leaves. A leaf is an entry that is not a
  tensordict, such as a tensor or a :class:`~tensordict.NonTensorData`.

The keys come in insertion order. With ``include_nested=True``, the keys inside
a nested tensordict come before the key of that tensordict. This robot state
has a timestamp, an arm, and a gripper on the arm:

  >>> td = TensorDict(
  ...     {
  ...         "time": torch.zeros(3),
  ...         "arm": {"pos": torch.zeros(3), "gripper": {"force": torch.zeros(3)}},
  ...     },
  ...     batch_size=[3],
  ... )
  >>> list(td.keys())
  ['time', 'arm']
  >>> list(td.keys(include_nested=True))
  ['time', ('arm', 'pos'), ('arm', 'gripper', 'force'), ('arm', 'gripper'), 'arm']
  >>> list(td.keys(include_nested=True, leaves_only=True))
  ['time', ('arm', 'pos'), ('arm', 'gripper', 'force')]
  >>> list(td.keys(leaves_only=True))
  ['time']

:meth:`~tensordict.TensorDict.values` and :meth:`~tensordict.TensorDict.items`
take the same arguments.

To test a key, use ``key in td``. It accepts nested keys. ``key in td.keys()``
finds a nested key only with ``include_nested=True``:

  >>> ("arm", "pos") in td
  True
  >>> ("arm", "pos") in td.keys(include_nested=True)
  True
  >>> ("arm", "pos") in td.keys()
  False

On a :class:`~tensordict.LazyStackedTensorDict`, a
:class:`~tensordict.store.TensorDictStore` and a
:class:`~tensordict.store.LazyStackedTensorDictStore`,
``("arm", "pos") in td.keys()`` is ``True`` and emits a ``FutureWarning``. This
result is deprecated and will be removed in TensorDict 0.17: from then on, the
check returns ``False``.

On a :class:`~tensordict.TensorDict`, and on most other types,
``key in td.keys(include_nested=True, leaves_only=True)`` checks that the key
exists, not that it is a leaf. A sub-tensordict is in the view, although
iterating the view does not list it (a
:class:`~tensordict.LazyStackedTensorDict` and a
:class:`~tensordict.PersistentTensorDict` check that the key is a leaf):

  >>> "arm" in td.keys(include_nested=True, leaves_only=True)
  True
  >>> "arm" in list(td.keys(include_nested=True, leaves_only=True))
  False

To test that a key is a leaf, pass the class of its entry to the ``is_leaf``
function of the view, by default :func:`~tensordict.default_is_leaf`:

  >>> from tensordict import default_is_leaf
  >>> default_is_leaf(td.entry_class("arm"))
  False
  >>> default_is_leaf(td.entry_class(("arm", "pos")))
  True

Read an entry
-------------

``td[key]`` and ``td.get(key)`` act differently on a missing key:

- ``td[key]`` raises a ``KeyError``.
- ``td.get(key)`` returns ``None``, as :meth:`dict.get` does.
- ``td.get(key, default)`` returns ``default``.

This is also true for a nested key, at any depth:

  >>> td.get("missing") is None
  True
  >>> td.get(("arm", "missing"), torch.ones(3))
  tensor([1., 1., 1.])
  >>> td["missing"]
  Traceback (most recent call last):
  ...
  KeyError: 'key "missing" not found in TensorDict with keys [\'arm\', \'time\']'

``get`` takes more arguments for lazy stacks whose entries have different
shapes. See :ref:`stacking-variable-size`.

.. _keys-write:

Write an entry
--------------

There are three ways to write a value under a key. They differ in what happens
to the tensor that is already there:

- ``td[key] = value`` and ``td.set(key, value)`` put ``value`` in the
  tensordict. If the key exists, the old tensor is replaced, not changed. If
  the key does not exist, it is added.
- ``td.set(key, value, inplace=True)`` copies ``value`` into the existing
  tensor if the key exists. If the key does not exist, it is added.
- ``td.set_(key, value)`` copies ``value`` into the existing tensor. If the key
  does not exist, it raises a ``KeyError``. For a nested key such as
  ``("a", "b")`` where ``"a"`` does not exist, a tensordict, a lazy stack and a
  tensorclass create ``"a"`` and emit a ``FutureWarning``: starting with
  TensorDict 0.17, the default will change to raising the ``KeyError``.

For example:

  >>> td = TensorDict({"pos": torch.zeros(3)}, batch_size=[3])
  >>> old = td["pos"]
  >>> _ = td.set("pos", torch.ones(3))
  >>> old  # replaced: the old tensor did not change
  tensor([0., 0., 0.])
  >>> old = td["pos"]
  >>> _ = td.set_("pos", torch.full((3,), 2.0))
  >>> old  # copied into the same tensor
  tensor([2., 2., 2.])
  >>> td.set_("vel", torch.zeros(3))
  Traceback (most recent call last):
  ...
  KeyError: 'key "vel" not found in TensorDict with keys [\'pos\']'

:ref:`inplace-conventions` gives the rules for the other in-place operations.

Write several entries
---------------------

:meth:`~tensordict.TensorDict.update` calls ``set`` for each entry of a
dictionary or a TensorDict. A nested dictionary updates the nested tensordict:
the keys that it does not give stay as they are. With ``inplace=True``, it
copies into the existing tensors, as ``set(..., inplace=True)`` does.
:meth:`~tensordict.TensorDict.update_` copies into existing tensors only, and
raises a ``KeyError`` for a missing key.

  >>> td = TensorDict({"time": torch.zeros(3), "arm": {"pos": torch.zeros(3)}}, batch_size=[3])
  >>> _ = td.update({"time": torch.ones(3), "arm": {"vel": torch.ones(3)}})
  >>> list(td.keys(include_nested=True, leaves_only=True))
  ['time', ('arm', 'pos'), ('arm', 'vel')]

``update`` writes the entries one after the other. If one entry raises, for
example on a batch size mismatch, the entries before it are already written.

Select, exclude, and remove entries
-----------------------------------

:meth:`~tensordict.TensorDict.select` returns a new tensordict with the keys
that you name. :meth:`~tensordict.TensorDict.exclude` returns a new tensordict
without them. Both accept nested keys. They do not copy the tensors: the new
tensordict holds the same tensors as the original.

  >>> td = TensorDict(
  ...     {
  ...         "time": torch.zeros(3),
  ...         "arm": {"pos": torch.zeros(3), "gripper": {"force": torch.zeros(3)}},
  ...     },
  ...     batch_size=[3],
  ... )
  >>> list(td.select("time", ("arm", "pos")).keys(include_nested=True))
  ['time', ('arm', 'pos'), 'arm']
  >>> list(td.exclude("time", ("arm", "gripper")).keys(include_nested=True))
  [('arm', 'pos'), 'arm']
  >>> td.select("time")["time"] is td["time"]
  True

``select`` raises a ``KeyError`` for a missing key, unless you pass
``strict=False``. ``exclude`` ignores missing keys. With ``inplace=True``,
both change the tensordict and return it.

To remove one entry, use ``del td[key]`` or :meth:`~tensordict.TensorDict.pop`.
``pop(key)`` raises a ``KeyError`` for a missing key, and ``pop(key, default)``
returns ``default``.

Rename a key
------------

:meth:`~tensordict.TensorDict.rename_key_` renames a key in place and returns
the tensordict. The old and the new keys can be nested, and at different
levels:

  >>> td = TensorDict({"time": torch.zeros(3), "arm": {"pos": torch.zeros(3)}}, batch_size=[3])
  >>> list(td.rename_key_(("arm", "pos"), "pos").keys(include_nested=True))
  ['time', 'arm', 'pos']

If the new key exists, ``rename_key_`` replaces its value. With
``safe=True``, it raises a ``KeyError`` instead. A new key inside the old one,
such as ``("arm", "arm")`` for ``"arm"``, moves the entry there, as
``td.set(("arm", "arm"), td.pop("arm"))`` does. Nothing is replaced, so
``safe=True`` does not raise for it.

Flatten and unflatten the keys
------------------------------

:meth:`~tensordict.TensorDict.flatten_keys` returns a tensordict with one
level. It joins each nested key into one string, with ``"."`` as the default
separator. :meth:`~tensordict.TensorDict.unflatten_keys` does the reverse.
The two methods do not copy the tensors. Use them for code or file formats
that accept only flat string keys.

  >>> td = TensorDict({"time": torch.zeros(3), "arm": {"pos": torch.zeros(3)}}, batch_size=[3])
  >>> flat = td.flatten_keys()
  >>> list(flat.keys())
  ['time', 'arm.pos']
  >>> list(flat.unflatten_keys().keys(include_nested=True))
  ['time', ('arm', 'pos'), 'arm']
  >>> list(td.flatten_keys("/").keys())
  ['time', 'arm/pos']

.. _keys-locking:

Locking and temporary changes
-----------------------------

A locked tensordict does not accept a change to its keys. Code can still
change the values of the existing tensors.

What a lock blocks
~~~~~~~~~~~~~~~~~~

:meth:`~tensordict.TensorDict.lock_` locks a tensordict and all its nested
tensordicts. A locked tensordict raises a ``RuntimeError`` for each of these
operations:

- Add a key, at any level.
- Replace an entry: ``td[key] = value``, ``set``, and ``update`` without
  ``inplace=True``.
- Remove or rename an entry: ``del``, ``pop``, ``rename_key_``, and
  ``select`` or ``exclude`` with ``inplace=True``.

For example, a new key in a nested tensordict raises:

  >>> td = TensorDict({"time": torch.zeros(3), "arm": {"pos": torch.zeros(3)}}, batch_size=[3])
  >>> _ = td.lock_()
  >>> td["arm", "vel"] = torch.zeros(3)
  Traceback (most recent call last):
  ...
  RuntimeError: Cannot modify locked TensorDict. For in-place modification, consider using the `set_()` method and make sure the key is present.

These operations write into the existing entries, so they work on a locked
tensordict, for tensor and non-tensor entries alike:

- ``set_`` and ``update_``.
- ``set(..., inplace=True)`` and ``update(..., inplace=True)``, if the key
  exists.
- An assignment to a batch index, such as ``td[0] = other``.
- An in-place method of the tensordict, such as ``td.add_(1)`` or
  ``td.zero_()``, or of one tensor, such as ``td["time"].add_(1)``.

For example:

  >>> _ = td.set_("time", torch.ones(3))
  >>> td["time"].add_(1)
  tensor([2., 2., 2.])

``td["time"] += 1`` raises on a locked tensordict, but only after it changed
the tensor. Python runs it as ``td["time"] = td["time"].__iadd__(1)``: the
in-place add works, and the assignment then raises. Use
``td["time"].add_(1)`` instead.

To read the lock state, use :attr:`~tensordict.TensorDict.is_locked`. To
remove it, call :meth:`~tensordict.TensorDict.unlock_` on the root
tensordict. If you unlock a nested tensordict of a locked tensordict alone, it
raises a ``RuntimeError``.

Who locks a tensordict
~~~~~~~~~~~~~~~~~~~~~~

You lock a tensordict with ``lock_()`` or with ``TensorDict(..., lock=True)``.
These methods also lock it:

- :meth:`~tensordict.TensorDict.memmap_` locks the tensordict, and
  :meth:`~tensordict.TensorDict.memmap` returns a locked tensordict.
  :meth:`~tensordict.TensorDict.load_memmap` returns an unlocked tensordict.
- :meth:`~tensordict.TensorDict.share_memory_` locks the tensordict (see
  :ref:`devices`).
- :meth:`~tensordict.TensorDict.consolidate` returns a locked tensordict.
- :func:`tensordict.from_module` returns a locked tensordict.
  :meth:`TensorDict.from_module <tensordict.TensorDict.from_module>` returns an
  unlocked one.

A new tensordict that a method returns is not always locked like the original.
On a locked tensordict, ``clone()``, ``select()``, ``apply()``, indexing, and
:func:`torch.stack` return an unlocked tensordict. ``view()``, ``reshape()``,
``unsqueeze()``, and ``to()`` return a locked one. Read ``is_locked`` on the
result if your code depends on it.

Temporary changes with ``with``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Some methods of a tensordict also work as context managers. At the end of the
``with`` block, they undo their change.

``with td.unlock_():`` unlocks the tensordict for the block, and locks it again
at the end. ``with td.lock_():`` does the reverse.

  >>> td = TensorDict({"pos": torch.zeros(3)}, batch_size=[3], lock=True)
  >>> with td.unlock_():
  ...     td["vel"] = torch.ones(3)
  >>> td.is_locked, sorted(td.keys())
  (True, ['pos', 'vel'])

``with params.to_module(module):`` puts the parameters and buffers of the
tensordict ``params`` into ``module`` for the block. At the end, the module
gets its original parameters back:

  >>> from torch import nn
  >>> module = nn.Linear(3, 2)
  >>> params = TensorDict.from_module(module).clone()
  >>> _ = params.zero_()
  >>> with params.to_module(module):
  ...     y = module(torch.ones(1, 3))
  >>> y.detach()
  tensor([[0., 0.]])
  >>> bool((module.weight == 0).all())
  False

``unlock_``, ``lock_``, and ``to_module`` also undo their change when the
``with`` block raises an exception.

These shape operations also work as context managers:
:meth:`~tensordict.TensorDict.view`, :meth:`~tensordict.TensorDict.permute`,
:meth:`~tensordict.TensorDict.transpose`,
:meth:`~tensordict.TensorDict.unsqueeze`,
:meth:`~tensordict.TensorDict.flatten`, and
:meth:`~tensordict.TensorDict.unflatten`. At the end of the block, the entries
that you wrote into the result go back into the original, with the reverse
operation. :meth:`~tensordict.TensorDict.reshape` does not write back.
:meth:`~tensordict.TensorDict.to` also writes back, and casts the existing
entries back to their original device and dtype (see
:ref:`devices-temporary-moves`). If the block raises an exception, these
operations write nothing back.

  >>> td = TensorDict({"x": torch.zeros(3, 4)}, batch_size=[3, 4])
  >>> with td.unsqueeze(-1) as tdu:
  ...     tdu["y"] = torch.ones(3, 4, 1)
  >>> td["y"].shape
  torch.Size([3, 4])

See also
--------

- :class:`~tensordict.NestedKey`, :meth:`~tensordict.TensorDict.keys`,
  :meth:`~tensordict.TensorDict.get`, :meth:`~tensordict.TensorDict.set`,
  :meth:`~tensordict.TensorDict.set_`, and :meth:`~tensordict.TensorDict.lock_`
  in the :ref:`api-reference`.
- :ref:`api-methods-keys` and :ref:`api-methods-locking`: the methods for keys
  and locking.
- The tutorial :doc:`/tutorials/tensordict_keys`.
