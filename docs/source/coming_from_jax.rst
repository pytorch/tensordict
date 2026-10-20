.. currentmodule:: tensordict

.. _coming-from-jax:

Coming from JAX pytrees
=======================

In JAX, you often put the arrays of a model or a dataset in a dataclass
registered as a pytree, and you transform them with the functions of
``jax.tree_util``. In PyTorch, a :class:`~tensordict.TensorClass` is that
dataclass, and a :class:`~tensordict.TensorDict` is its dict form, for keys
that you choose at run time. Both are registered with
``torch.utils._pytree``, so :func:`torch.func.grad`, :func:`torch.vmap`,
:func:`torch.compile`, and the pytree functions of PyTorch accept them. Both
also have a *batch size*: leading dimensions that all the leaves share, which
JAX pytrees do not have (see `What tensordict adds`_).

This page maps the pytree idioms of JAX libraries to tensordict, with a short
example for each one.

The mapping
-----------

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - In JAX
     - In tensordict
   * - A dataclass registered as a pytree
     - A subclass of :class:`~tensordict.TensorClass`, or a
       :class:`~tensordict.TensorDict` for keys chosen at run time
   * - ``jax.tree_util.tree_map(f, tree)``, ``tree_map(f, a, b)``
     - :meth:`a.apply(f) <tensordict.TensorDict.apply>`, ``a.apply(f, b)``
   * - ``jax.tree_util.tree_leaves(tree)``
     - ``tree.values(include_nested=True, leaves_only=True)``
   * - Partition a tree with a predicate, and combine the parts again
     - :meth:`~tensordict.TensorDict.split_keys` with a callable,
       :meth:`~tensordict.TensorDict.select`,
       :meth:`~tensordict.TensorDict.exclude`, then
       :meth:`~tensordict.TensorDict.update`
   * - An out-of-place update of some leaves (``dataclasses.replace``, or a
       function that replaces the leaf at a path)
     - :meth:`~tensordict.TensorDict.replace`, also with nested keys
   * - A static field (a field that is not a leaf)
     - A non-tensor field. It is part of the pytree spec.
   * - Load saved leaves into a skeleton of the same structure
     - :func:`~tensordict.load_memmap`, which needs no skeleton;
       ``load_memmap(path, out=skeleton)``,
       :meth:`~tensordict.TensorDict.load_memmap_`, and
       :meth:`~tensordict.TensorDict.from_schema` to build a skeleton
   * - ``jax.vmap`` over stacked parameters
     - :meth:`~tensordict.TensorDict.from_modules`,
       :meth:`~tensordict.TensorDict.to_module`, and :func:`torch.vmap`
   * - Key paths: ``jax.tree_util.tree_flatten_with_path(tree)``
     - ``torch.utils._pytree.tree_flatten_with_path(tree)``, and
       ``tree.keys(include_nested=True, leaves_only=True)``
   * - A frozen dataclass
     - ``TensorClass["frozen"]``. The tensors stay writable.
   * - ``tree_map(lambda *xs: jnp.stack(xs), *trees)``
     - ``torch.stack(trees)``
   * - ``jax.device_put(tree, device)``
     - ``tree.to(device)``

A dataclass that is a pytree
----------------------------

Declare the fields with type annotations. A tensor field is a leaf of the
pytree, and a ``str`` field is not:

  >>> import torch
  >>> from torch.utils import _pytree as pytree
  >>> from tensordict import TensorClass, TensorDict
  >>> class Linear(TensorClass):
  ...     weight: torch.Tensor
  ...     bias: torch.Tensor
  ...     activation: str
  ...
  ...     def __call__(self, x):
  ...         y = x @ self.weight.T + self.bias
  ...         return y.relu() if self.activation == "relu" else y.tanh()
  >>> _ = torch.manual_seed(0)
  >>> params = Linear(weight=torch.randn(2, 3), bias=torch.zeros(2), activation="relu")
  >>> leaves, spec = pytree.tree_flatten(params)
  >>> [leaf.shape for leaf in leaves]
  [torch.Size([2, 3]), torch.Size([2])]
  >>> type(pytree.tree_unflatten(leaves, spec)).__name__
  'Linear'

:func:`torch.func.grad` returns the gradient as a ``Linear``, as
``jax.grad`` does for a pytree:

  >>> x = torch.randn(5, 3)
  >>> def loss(params, x):
  ...     return params(x).pow(2).sum()
  >>> grads = torch.func.grad(loss)(params, x)
  >>> type(grads).__name__, grads.weight.shape, grads.activation
  ('Linear', torch.Size([2, 3]), 'relu')

Map a function over the leaves
------------------------------

:meth:`~tensordict.TensorDict.apply` is ``tree_map``. It calls the function
on each tensor leaf, nested ones too, and returns a tree of the same class.
Pass other trees of the same structure after the function, as for
``tree_map(f, a, b)``. This is one step of gradient descent:

  >>> new_params = params.apply(lambda p, g: p - 0.1 * g, grads)
  >>> type(new_params).__name__, new_params.activation
  ('Linear', 'relu')
  >>> bool(loss(new_params, x) < loss(params, x))
  True

``torch.utils._pytree.tree_map`` gives the same result. ``apply`` has more
options, such as ``filter_empty`` and ``call_on_nested``. To read the leaves
as a list, as with ``tree_leaves``, use ``values``:

  >>> state = TensorDict(
  ...     layer={"weight": torch.ones(2, 2), "bias": torch.zeros(2)},
  ...     step=torch.tensor(3),
  ... )
  >>> [v.shape for v in state.values(include_nested=True, leaves_only=True)]
  [torch.Size([2, 2]), torch.Size([2]), torch.Size([])]

Partition and combine
---------------------

To split a tree into the leaves that a predicate selects and the rest, pass
the predicate to :meth:`~tensordict.TensorDict.split_keys`. It receives the
nested key and the value of each leaf, and the last tree that
``split_keys`` returns holds the leaves that it did not select. The values
are not copied. :meth:`~tensordict.TensorDict.update` combines the parts
again:

  >>> floats, rest = state.split_keys(lambda key, value: value.is_floating_point())
  >>> sorted(floats.keys(include_nested=True, leaves_only=True))
  [('layer', 'bias'), ('layer', 'weight')]
  >>> list(rest.keys())
  ['step']
  >>> combined = floats.apply(lambda t: t - 1).update(rest)
  >>> combined["layer", "weight"][0], combined["step"]
  (tensor([0., 0.]), tensor(3))

To split by key instead, :meth:`~tensordict.TensorDict.select` keeps the
keys that you name, and :meth:`~tensordict.TensorDict.exclude` drops them:

  >>> list(state.select(("layer", "weight")).keys(include_nested=True, leaves_only=True))
  [('layer', 'weight')]
  >>> list(state.exclude("layer").keys())
  ['step']

Out-of-place updates
--------------------

:meth:`~tensordict.TensorDict.replace` returns a shallow copy with some
entries replaced, as ``dataclasses.replace`` does. The original does not
change, and the leaves that you do not replace are shared:

  >>> tanh_params = params.replace(activation="tanh", bias=torch.ones(2))
  >>> tanh_params.activation, params.activation
  ('tanh', 'relu')
  >>> tanh_params.weight is params.weight
  True

On a :class:`~tensordict.TensorDict`, pass a dict to replace a nested leaf
by its key, as a tuple of strings:

  >>> new_state = state.replace({("layer", "bias"): torch.ones(2)})
  >>> new_state["layer", "bias"], state["layer", "bias"]
  (tensor([1., 1.]), tensor([0., 0.]))

Static fields
-------------

A static field of a JAX pytree is part of the tree structure rather than a
leaf. A non-tensor field of a tensorclass, such as ``activation`` above, is
the same: it is stored in the pytree spec, so two trees that differ only in
that field have different specs, and functions such as ``tree_map`` and
:func:`torch.func.grad` pass it through unchanged:

  >>> spec_relu = pytree.tree_structure(params)
  >>> spec_relu == pytree.tree_structure(params.replace(weight=torch.zeros(2, 3)))
  True
  >>> spec_relu == pytree.tree_structure(params.replace(activation="tanh"))
  False

:func:`torch.compile` specializes a function on the non-tensor values that it
reads or returns, as ``jax.jit`` does on static fields: a call with a new
value compiles the function again. :ref:`compile` lists this with the other
causes of recompiles.

:func:`torch.stack` on tensorclasses with a ``str`` field makes a list of the
values, one for each element (see :ref:`non-tensor-data`). To keep one value
for the whole stack, as for a static field, wrap it in
:class:`~tensordict.MetaData`:

  >>> from tensordict import MetaData
  >>> layer = Linear(weight=torch.randn(2, 3), bias=torch.zeros(2), activation=MetaData("relu"))
  >>> torch.stack([layer, layer]).activation
  'relu'

Load into a skeleton
--------------------

To load saved arrays, JAX libraries often need a skeleton: a tree of the same
structure that receives the leaves. tensordict saves the structure and the
class with the data, so :func:`~tensordict.load_memmap` needs no skeleton:

  >>> import tempfile
  >>> from tensordict import load_memmap
  >>> tmpdir = tempfile.TemporaryDirectory()
  >>> _ = params.memmap(tmpdir.name)
  >>> loaded = load_memmap(tmpdir.name)
  >>> type(loaded).__name__, loaded.activation
  ('Linear', 'relu')

To load into a tree that you already have, pass it as ``out``, or call
:meth:`~tensordict.TensorDict.load_memmap_` on it. Both return that tree. Its
leaves are replaced by the loaded tensors:

  >>> skeleton = params.apply(torch.zeros_like)
  >>> load_memmap(tmpdir.name, out=skeleton) is skeleton
  True
  >>> torch.equal(skeleton.weight, params.weight)
  True
  >>> other = params.apply(torch.zeros_like)
  >>> other.load_memmap_(tmpdir.name) is other
  True
  >>> tmpdir.cleanup()

To build a skeleton from shapes and dtypes, without data, use
:meth:`~tensordict.TensorDict.from_schema`. It fills the leaves with zeros,
in memory or on disk (see :ref:`storage`):

  >>> buffer = TensorDict.from_schema(
  ...     {"obs": ([84, 84, 3], torch.uint8), "reward": ([], torch.float32)},
  ...     batch_size=[1000],
  ... )
  >>> buffer["obs"].shape, buffer["reward"].shape
  (torch.Size([1000, 84, 84, 3]), torch.Size([1000]))

:ref:`saving` covers the formats and the options of each load.

vmap over stacked modules
-------------------------

In JAX, you make an ensemble by stacking the parameters of several models
along a new leading axis, and you call the model with ``jax.vmap`` over that
axis. :meth:`~tensordict.TensorDict.from_modules` stacks the parameters of
several ``torch.nn`` modules of the same structure.
:meth:`~tensordict.TensorDict.to_module` puts a set of parameters into a
module for the duration of a ``with`` block, so :func:`torch.vmap` calls the
module once for each set:

  >>> from torch import nn
  >>> nets = [nn.Linear(3, 1) for _ in range(4)]
  >>> stacked = TensorDict.from_modules(*nets)
  >>> stacked.batch_size
  torch.Size([4])
  >>> base = nn.Linear(3, 1)
  >>> def call(params, x):
  ...     with params.to_module(base):
  ...         return base(x)
  >>> y = torch.vmap(call, (0, None))(stacked, x)
  >>> y.shape
  torch.Size([4, 5, 1])
  >>> torch.testing.assert_close(y[2], nets[2](x))

:ref:`modules-parameters` explains ``from_module`` and ``to_module``, and
:ref:`compile` shows the same pattern under :func:`torch.compile`.

Key paths
---------

``torch.utils._pytree.tree_flatten_with_path`` returns the path of each leaf,
as ``jax.tree_util.tree_flatten_with_path`` does. A key of a tensordict is a
``MappingKey`` in the path:

  >>> for path, leaf in pytree.tree_flatten_with_path(state)[0]:
  ...     print(pytree.keystr(path), tuple(leaf.shape))
  ['layer']['weight'] (2, 2)
  ['layer']['bias'] (2,)
  ['step'] ()

Without the pytree functions, ``keys(include_nested=True)`` lists the nested
keys as tuples, and a tuple reads its entry (see :ref:`keys`):

  >>> list(state.keys(include_nested=True, leaves_only=True))
  [('layer', 'weight'), ('layer', 'bias'), 'step']
  >>> state["layer", "weight"].shape
  torch.Size([2, 2])

Immutability
------------

JAX arrays cannot change, so a frozen dataclass of arrays cannot change
either. In PyTorch, tensors can change. With the ``frozen`` option, assigning
a field raises an error, as with ``@dataclass(frozen=True)``, but the tensors
of the instance stay writable (see :ref:`tensorclass-frozen`):

  >>> class State(TensorClass["frozen"]):
  ...     pos: torch.Tensor
  ...     vel: torch.Tensor
  >>> s = State(pos=torch.zeros(4, 3), vel=torch.ones(4, 3), batch_size=[4])
  >>> s.pos = s.pos + s.vel
  Traceback (most recent call last):
  ...
  dataclasses.FrozenInstanceError: cannot assign to field 'pos'
  >>> s = s.replace(pos=s.pos + s.vel)  # out of place
  >>> _ = s.pos.add_(1)  # in place: the tensor is writable
  >>> s.pos[0]
  tensor([2., 2., 2.])

What tensordict adds
--------------------

A tensordict has a batch size, the leading dimensions that all its leaves
share (see :ref:`batch-size`). Indexing, reshaping, and stacking apply to
these dimensions of every leaf, without ``tree_map`` or ``vmap``:

  >>> s.batch_size
  torch.Size([4])
  >>> s[1:3].pos.shape, s.reshape(2, 2).batch_size
  (torch.Size([2, 3]), torch.Size([2, 2]))
  >>> torch.stack([s, s]).batch_size
  torch.Size([2, 4])

A method written for one element then runs on a batch of any shape (see
:ref:`tensorclass-guide`). The rest of tensordict builds on the batch size:

- Lazy stacks hold elements of different shapes, such as sequences of
  different lengths, and pad them on request (see
  :ref:`stacking-variable-size`):

    >>> from tensordict import lazy_stack
    >>> ragged = lazy_stack([TensorDict(x=torch.zeros(n, 2)) for n in (3, 5)])
    >>> ragged[1]["x"].shape, ragged.get("x", as_padded_tensor=True).shape
    (torch.Size([5, 2]), torch.Size([2, 5, 2]))

- :meth:`~tensordict.TensorDict.memmap` saves a tree as memory-mapped
  files, which you load lazily and index without reading the whole dataset
  (see :ref:`saving`).
- ``to(device)`` moves all the leaves to a device, with one transfer when
  the tree is consolidated (see :ref:`devices`).

See also
--------

- :ref:`tensorclass-guide`: the dataclass form in full.
- :ref:`choosing-a-container`: TensorDict, TensorClass, and the other
  containers.
- :ref:`operations`: ``apply`` and the other operations on all the leaves.
- :ref:`compile`: :func:`torch.compile`, :mod:`torch.func`, and
  :func:`torch.vmap` with tensordicts.
