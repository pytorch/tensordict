.. currentmodule:: tensordict

TypedTensorDict
===============

:class:`~tensordict.TypedTensorDict` is a :class:`~tensordict.TensorDictBase` subclass
with typed field declarations and backend composition.  It brings ``TypedDict``-style
class definitions to ``TensorDict``: you declare fields as class annotations and get
typed construction, typed attribute access, inheritance, ``NotRequired`` fields,
``**state`` spreading, and the ability to wrap any ``TensorDictBase`` backend
(H5, Redis, lazy stacks, etc.) via ``from_tensordict``.

.. code-block:: python

  >>> import torch
  >>> from tensordict import TypedTensorDict
  >>> from torch import Tensor
  >>>
  >>> class PredictorState(TypedTensorDict):
  ...     eta: Tensor
  ...     X: Tensor
  ...     beta: Tensor
  >>>
  >>> state = PredictorState(
  ...     eta=torch.randn(5, 3),
  ...     X=torch.randn(5, 4),
  ...     beta=torch.randn(5, 1),
  ...     batch_size=[5],
  ... )
  >>> state.eta.shape
  torch.Size([5, 3])
  >>> state["X"].shape
  torch.Size([5, 4])

Why TypedTensorDict?
--------------------

Typed pipelines often build up state one step at a time:

.. code-block:: python

  class PredictorState(TypedTensorDict):
      eta: Tensor
      X: Tensor
      beta: Tensor

  class ObservedState(PredictorState):
      y: Tensor
      mu: Tensor

  def gaussian(state: PredictorState, std: float) -> ObservedState:
      eta = state.eta
      y = eta + torch.randn_like(eta) * std
      return ObservedState(**state, y=y, mu=eta, batch_size=state.batch_size)

Each stage inherits the previous one's fields and adds new ones. The
``**state`` spreading pattern lets transition functions stay short regardless
of how many fields the state has. And because ``TypedTensorDict`` inherits
from ``TensorDictBase``, every operation -- ``.to(device)``, ``.clone()``,
slicing, ``torch.stack``, ``memmap`` -- works at every stage.

TypedTensorDict vs TensorClass
------------------------------

Both ``TypedTensorDict`` and ``TensorClass`` provide typed tensor containers.
They share the same class-option syntax (``["shadow"]``, ``["frozen"]``, etc.)
and both use ``@dataclass_transform()`` for IDE support. The key difference is
in the underlying model:

.. list-table::
   :header-rows: 1
   :widths: 35 30 30

   * - Feature
     - ``TypedTensorDict``
     - ``TensorClass``
   * - Inherits from
     - ``TensorDictBase`` (delegates to ``_source``)
     - ``TensorCollection`` (wraps a ``TensorDict`` internally)
   * - Can wrap any backend
     - Yes (``from_tensordict``)
     - Yes (``from_tensordict``)
   * - Inheritance
     - Standard Python (``class Child(Parent): ...``)
     - Supported via metaclass
   * - ``**state`` spreading
     - Works natively (``MutableMapping``)
     - Requires manual field-by-field repacking
   * - ``state["key"]``
     - Works natively (``TensorDictBase.__getitem__``)
     - Raises ``ValueError`` -- use ``state.key`` or ``state.get("key")``
   * - ``NotRequired`` fields
     - Supported
     - Not supported
   * - Non-tensor fields
     - Not supported (tensor-only)
     - Supported (strings, ints, arbitrary objects)
   * - Custom methods
     - Supported (regular class methods)
     - Supported (regular class methods)
   * - ``@tensorclass`` decorator
     - Not needed (uses metaclass via inheritance)
     - Required (or ``class Foo(TensorClass): ...``)

**When to use which:**

- Use ``TypedTensorDict`` when you have a typed pipeline with progressive state
  accumulation, need ``**state`` spreading, want standard Python inheritance
  for schema composition, or need to wrap persistent backends while keeping
  full ``TensorDictBase`` API compatibility.

- Use ``TensorClass`` when you need non-tensor fields (strings, metadata),
  custom ``__init__`` logic, or your codebase already uses ``@tensorclass``
  extensively.

Inheritance and field accumulation
----------------------------------

Fields accumulate through the MRO. Each subclass adds its own fields while
inheriting all parent fields:

.. code-block:: python

  >>> from typing import NotRequired
  >>>
  >>> class PredictorState(TypedTensorDict):
  ...     eta: Tensor
  ...     X: Tensor
  ...     beta: Tensor
  >>>
  >>> class ObservedState(PredictorState):
  ...     y: Tensor
  ...     mu: Tensor
  ...     noise: NotRequired[Tensor]
  >>>
  >>> class SurvivalState(ObservedState):
  ...     event_time: Tensor
  ...     indicator: Tensor
  ...     observed_time: Tensor

  >>> ObservedState.__required_keys__
  frozenset({'eta', 'X', 'beta', 'y', 'mu'})
  >>> ObservedState.__optional_keys__
  frozenset({'noise'})

Inheritance works as standard Python: ``isinstance(obs, PredictorState)``
returns ``True`` for an ``ObservedState`` instance, and a function typed as
``f(state: PredictorState)`` accepts any subclass.

NotRequired fields
------------------

Mark fields as optional with :data:`~typing.NotRequired`:

.. code-block:: python

  >>> from typing import NotRequired
  >>>
  >>> class ObservedState(PredictorState):
  ...     y: Tensor
  ...     mu: Tensor
  ...     noise: NotRequired[Tensor]

  >>> obs = ObservedState(
  ...     eta=torch.randn(5, 3), X=torch.randn(5, 4), beta=torch.randn(5, 1),
  ...     y=torch.randn(5, 3), mu=torch.randn(5, 3),
  ...     batch_size=[5],
  ... )
  >>> "noise" in obs
  False

If a ``NotRequired`` field is not provided, it is simply absent from the
underlying ``TensorDict``. Accessing it via attribute raises
``AttributeError``.

Spreading (``**state``)
-----------------------

Because ``TypedTensorDict`` is a ``MutableMapping``, the ``**`` operator
unpacks it into keyword arguments. This makes state transitions concise:

.. code-block:: python

  >>> state = PredictorState(
  ...     eta=torch.randn(5, 3), X=torch.randn(5, 4), beta=torch.randn(5, 1),
  ...     batch_size=[5],
  ... )
  >>> obs = ObservedState(
  ...     **state,
  ...     y=torch.randn(5, 3),
  ...     mu=torch.randn(5, 3),
  ...     batch_size=state.batch_size,
  ... )
  >>> set(obs.keys()) == {"eta", "X", "beta", "y", "mu"}
  True

Adding a new field to a pipeline stage is one line in the class definition --
no transition function needs updating.

Class options
-------------

``TypedTensorDict`` supports the same bracket-syntax options as ``TensorClass``:

.. code-block:: python

  class MyModel(TypedTensorDict["shadow"]):
      data: Tensor      # "data" shadows TensorDict.data -- allowed

  class Immutable(TypedTensorDict["frozen"]):
      x: Tensor         # locked after construction

  class Combined(TypedTensorDict["shadow", "frozen"]):
      data: Tensor

- ``"shadow"`` -- Allow field names that clash with ``TensorDictBase`` attributes.
  Without this, conflicting names raise ``AttributeError`` at class definition
  time.
- ``"frozen"`` -- Lock the ``TensorDict`` after construction (read-only).
- ``"autocast"`` -- Automatically cast assigned values.
- ``"nocast"`` -- Disable type casting on assignment.
- ``"tensor_only"`` -- Restrict fields to tensor types only.

Options propagate through inheritance: a subclass of a ``"frozen"`` class is
also frozen.

Backend composition (``from_tensordict``)
-----------------------------------------

``TypedTensorDict`` can wrap any ``TensorDictBase`` backend via
``from_tensordict(td)``.  The backend is stored live (zero-copy); mutations
through the typed wrapper go directly to the underlying storage:

.. code-block:: python

  >>> from tensordict import TensorDict
  >>>
  >>> td = TensorDict(
  ...     eta=torch.randn(5, 3), X=torch.randn(5, 4), beta=torch.randn(5, 1),
  ...     batch_size=[5],
  ... )
  >>> state = PredictorState.from_tensordict(td)
  >>> state.eta.shape
  torch.Size([5, 3])
  >>> state.eta = torch.ones(5, 3)  # writes to td
  >>> (td["eta"] == 1).all()
  True

This works with any backend: ``PersistentTensorDict`` (H5),
``TensorDictStore`` (Redis), ``LazyStackedTensorDict``, memory-mapped
``TensorDict``, etc.  See the :doc:`../compatibility` page for full
details and examples.

TensorDict operations
---------------------

Every ``TensorDictBase`` operation works on ``TypedTensorDict`` instances:

.. code-block:: python

  >>> state = PredictorState(
  ...     eta=torch.randn(5, 3), X=torch.randn(5, 4), beta=torch.randn(5, 1),
  ...     batch_size=[5],
  ... )
  >>> state.to("cpu").device
  device(type='cpu')
  >>> state.clone()["eta"].shape
  torch.Size([5, 3])
  >>> state[0:3].batch_size
  torch.Size([3])
  >>> torch.stack([state, state], dim=0).batch_size
  torch.Size([2, 5])

This includes ``.memmap()``, ``.apply()``, ``torch.cat``, ``torch.stack``,
``.unbind()``, ``.select()``, ``.exclude()``, ``.update()``, and all other
``TensorDictBase`` methods.

Type checking
-------------

``TypedTensorDict`` uses ``@dataclass_transform()`` (PEP 681) on its metaclass.
This means type checkers (pyright, mypy) understand:

- **Constructor signatures** -- missing or extra fields are flagged.
- **Attribute access** -- ``state.eta`` is typed as ``Tensor``, and typos like
  ``state.etta`` produce errors.
- **Inheritance** -- subclass fields include parent fields.

String-key access (``state["eta"]``) works at runtime but does not get type
narrowing without a dedicated type checker plugin. For typed access, prefer
dot notation (``state.eta``).

.. autosummary::
    :toctree: generated/
    :template: td_template.rst

    TypedTensorDict
