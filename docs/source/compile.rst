.. _compile:

torch.compile, torch.func, torch.export and CUDA graphs
=======================================================

This page lists what works when tensordicts and tensorclasses go through
:func:`torch.compile`, :mod:`torch.func`, :func:`torch.export.export` and CUDA
graphs, what makes a function recompile or fail, and how to work around each
limit.

How tensordicts compile
-----------------------

:class:`~tensordict.TensorDict`, :class:`~tensordict.LazyStackedTensorDict`
and tensorclasses are registered as pytree nodes, so :mod:`torch.func` and
:func:`torch.export.export` see their tensors as leaves. Inside
:func:`torch.compile`, Dynamo traces the Python code of tensordict like user
code: indexing, ``apply``, arithmetic, reductions, shape operations, stacking
and construction all end up in the graph. Most operations compile with
``fullgraph=True``. With that flag, a graph break raises an error instead of
splitting the function into several graphs.

  >>> import torch
  >>> from tensordict import TensorDict, tensorclass
  >>> @tensorclass
  ... class Transition:
  ...     obs: torch.Tensor
  ...     reward: torch.Tensor
  >>> @torch.compile(fullgraph=True)
  ... def scale(td):
  ...     return td.apply(lambda x: x / x.abs().max()) * 2
  >>> @torch.compile(fullgraph=True)
  ... def step(data):
  ...     return Transition(data.obs + 1, data.reward * 0.5, batch_size=data.batch_size)
  >>> td = TensorDict(
  ...     obs=torch.randn(8, 3),
  ...     next=TensorDict(obs=torch.randn(8, 3), batch_size=[8]),
  ...     batch_size=[8],
  ... )
  >>> scale(td).batch_size
  torch.Size([8])
  >>> step(Transition(torch.zeros(8, 3), torch.ones(8), batch_size=[8])).reward[0]
  tensor(0.5000)

To find out why a function recompiles or breaks, run it with
``TORCH_LOGS="recompiles"`` or ``TORCH_LOGS="graph_breaks"`` in the
environment, or call ``torch._logging.set_logs(recompiles=True)`` first.

Recompiles
----------

Dynamo specializes a compiled function on the structure of its inputs and
installs guards that check it on every call. A call that fails a guard
compiles the function again. By default, a function compiles at most 8 times
(``torch._dynamo.config.recompile_limit``). After that, it runs in eager mode,
or raises ``FailOnRecompileLimitHit`` with ``fullgraph=True``. With
tensordicts, a function recompiles for the following reasons.

Another key set
  An operation that goes over all the entries (arithmetic, ``apply``,
  ``clone``, reductions) is compiled for the keys it saw. An input with another
  key set compiles a new graph. Reading one entry (``td["obs"]``) does not
  depend on the other keys.

Writing into an input tensordict
  ``td["y"] = ...``, ``td.set``, ``td.update`` and a
  :class:`~tensordict.nn.TensorDictModule` (which writes its ``out_keys`` into
  its input) recompile whenever the key set of the input changes, even when
  the key written already exists. Dynamo guards on all the keys of a dict that
  the function writes into
  (`#1606 <https://github.com/pytorch/tensordict/issues/1606>`_,
  `pytorch/pytorch#175858 <https://github.com/pytorch/pytorch/issues/175858>`_).
  A write into a nested tensordict only depends on the keys of that nested
  tensordict. If the key sets vary, build a new tensordict in the compiled
  function and write it into the input outside:

    >>> @torch.compile(fullgraph=True)
    ... def policy(td):
    ...     return TensorDict(action=td["obs"].tanh(), batch_size=td.batch_size)
    >>> td = TensorDict(obs=torch.randn(4, 3), batch_size=[4])
    >>> td["action"] = policy(td)["action"]

Another batch size
  The first call with a new batch size recompiles once with dynamic shapes
  (automatic dynamic shapes), and later sizes reuse that graph.
  ``torch.compile(fn, dynamic=True)`` avoids the second compilation.
  ``torch._dynamo.mark_dynamic`` on an entry does not: the batch size holds
  Python ints, and a function that builds a tensordict with
  ``batch_size=td.batch_size`` then raises ``ConstraintViolationError``
  (pytorch/tensordict#2077).
  ``split``, ``chunk``, ``unbind`` and ``tolist`` recompile for each batch
  size, also after that.

Lazy stacks of another length
  A :class:`~tensordict.LazyStackedTensorDict` is specialized on the number of
  tensordicts it stacks, also with ``dynamic=True``. A new stack of the same
  length and structure reuses the graph.

Strings in the output
  Each distinct ``str`` value of a tensorclass field or of a
  :class:`~tensordict.NonTensorData` entry recompiles the function when the
  function returns it, for instance through ``apply``, ``clone`` or the
  constructor of a tensorclass. Dynamo specializes the graph on the value.
  Reading only the tensors does not recompile. Return tensors from the
  compiled function and rebuild the tensorclass outside, or keep the strings
  out of the compiled function.

New tensorclasses and pytree flattening
  A compiled function that flattens a tensordict as a pytree (a compiled
  ``torch.vmap``, ``torch.utils._pytree.tree_map``) recompiles after a new
  tensorclass or :class:`~tensordict.TypedTensorDict` subclass is defined,
  because PyTorch guards on the size of its pytree registry. Define the
  classes before the first call.

With ``TORCH_LOGS="recompiles"``, the guard that failed names the cause. A
guard on ``len(td._tensordict)`` means that the key set of a tensordict that
the function writes into changed. A guard on ``td._batch_size[0]`` means a new
batch size.

Known limits
------------

The following patterns break the graph (an error with ``fullgraph=True``).
The error messages in the table are those of PyTorch 2.13, 2.14 and the 2.16
nightly. Other versions can word them differently.

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Pattern
     - What happens
     - Workaround
   * - ``td * t``, ``t * td``, ``td + t``, ``td -= t`` where ``t`` is a
       :class:`~torch.Tensor` (also with tensorclasses)
     - Dynamo turns the operator into a ``Tensor`` method and does not call
       ``TensorDict.__mul__``
       (`pytorch/pytorch#200455 <https://github.com/pytorch/pytorch/issues/200455>`_):
       ``Unsupported: All __torch_function__ overrides returned
       NotImplemented``. Operators with a Python number or another tensordict
       work.
     - Call the method: ``td.mul(t)``, ``td.add(t)``, ``td.sub_(t)``.
   * - ``td.all()``, ``td.any()`` without ``dim``, and
       ``if (td == other).all():``
     - These return a Python ``bool``, so the graph depends on tensor values:
       ``Unsupported: Data-dependent branching``.
     - Reduce in the graph:
       ``torch.stack([v.all() for v in td.values(include_nested=True,
       leaves_only=True)]).all()``,
       and select with :func:`torch.where` instead of ``if``. ``td.all(dim=0)``
       returns a tensordict and compiles.
   * - Boolean-mask indexing: ``td[mask]``, ``td[mask] = other[mask]``
     - The result has a data-dependent shape, also with
       ``torch._dynamo.config.capture_dynamic_output_shape_ops = True``:
       ``Could not guard on data-dependent expression``.
     - Keep the shapes fixed: ``torch.where(mask, td, other)``,
       ``td.where(mask, other)`` or ``td.masked_fill(~mask, 0.0)``, or index
       outside the compiled function.
   * - ``memmap``, ``memmap_like``, ``share_memory_``, ``to_namedtuple``,
       ``data_ptr``
     - Eager-only operations (files, shared memory, Python classes built at
       run time).
     - Call them outside the compiled function.
   * - ``loss.backward()`` or ``torch.autograd.grad`` inside the compiled
       function
     - Dynamo does not trace the backward pass:
       ``Unsupported Tensor.backward() call``.
     - Return the loss and call ``loss.backward()`` outside.
   * - Building a :class:`~tensordict.nn.TensorDictSequential` inside the
       compiled function (``seq[:2](td)``, ``seq.select_subsequence(...)``)
     - Dynamo cannot build the new module inside the graph:
       ``Unexpected type in sourceless builder``.
     - Build the subsequence once, outside, and call it inside.

torch.func
----------

:func:`torch.vmap` maps over the batch dimensions of tensordicts and
tensorclasses: ``in_dims`` and ``out_dims`` refer to batch dimensions, and
tensordict methods, arithmetic included, work on the per-sample tensordict.
``torch.compile(torch.vmap(fn))`` compiles the same code.

  >>> td = TensorDict(a=torch.randn(5, 3), b=torch.randn(5, 3), batch_size=[5])
  >>> torch.vmap(lambda t: (t * 2 + 1).exp())(td).batch_size
  torch.Size([5])
  >>> torch.vmap(lambda t, x: t["a"] @ x, (0, None))(td, torch.randn(3)).shape
  torch.Size([5])

Under :mod:`torch.func` transforms, pointwise tensordict methods run one
operation per entry, so they are slower than in eager mode
(`pytorch/pytorch#200456 <https://github.com/pytorch/pytorch/issues/200456>`_).

:func:`torch.func.grad`, :func:`~torch.func.grad_and_value`,
:func:`~torch.func.vjp` and :func:`~torch.func.jacrev` accept tensordict
inputs. The gradient is a tensordict with the structure of the input:

  >>> grads = torch.func.grad(lambda t: (t["a"] ** 2).sum())(td)
  >>> torch.testing.assert_close(grads["a"], 2 * td["a"])

To call a module with other parameters, extract them with
:meth:`~tensordict.TensorDict.from_module` and swap them in with
:meth:`~tensordict.TensorDict.to_module`. This works in eager mode, compiled
and under :func:`torch.vmap`, which gives model ensembles and per-sample
gradients:

  >>> from torch import nn
  >>> module = nn.Linear(3, 1)
  >>> def call(params, x):
  ...     with params.to_module(module):
  ...         return module(x)
  >>> x = torch.randn(10, 3)
  >>> ensemble = TensorDict.from_modules(*[nn.Linear(3, 1) for _ in range(4)])
  >>> torch.vmap(call, (0, None))(ensemble, x).shape
  torch.Size([4, 10, 1])
  >>> def loss(params, x):
  ...     return call(params, x).pow(2).sum()
  >>> params = TensorDict.from_module(module).detach()
  >>> per_sample = torch.vmap(torch.func.grad(loss), (None, 0))(params, x)
  >>> per_sample["weight"].shape
  torch.Size([10, 1, 3])

:func:`torch.func.functional_call` only accepts a ``dict``: pass
``params.flatten_keys(".").to_dict()`` instead of the tensordict. See the
:doc:`tutorials/functional` tutorial for more. If the process also creates a
:class:`~tensordict.nn.CudaGraphModule`, see the warning in
:ref:`compile-cudagraphs`.

torch.export
------------

:func:`torch.export.export` accepts tensordict and tensorclass inputs and
outputs, in strict and non-strict mode (non-strict is the default). To keep
the batch dimension dynamic, give ``dynamic_shapes`` one entry per entry of
the tensordict, in the order of ``td.keys()`` (insertion order), with a nested
list for a nested tensordict:

  >>> class Model(nn.Module):
  ...     def forward(self, td):
  ...         return TensorDict(c=td["a"] * 2 + td["n", "b"], batch_size=td.batch_size)
  >>> def make(batch):
  ...     return TensorDict(
  ...         a=torch.randn(batch, 3),
  ...         n=TensorDict(b=torch.randn(batch, 3), batch_size=[batch]),
  ...         batch_size=[batch],
  ...     )
  >>> batch = torch.export.Dim("batch")
  >>> program = torch.export.export(
  ...     Model(), (make(4),), dynamic_shapes=([{0: batch}, [{0: batch}]],)
  ... )
  >>> program.module()(make(6)).batch_size
  torch.Size([6])

Without ``dynamic_shapes``, a call with another batch size fails a guard of
the exported program (``AssertionError: Guard failed``). Non-strict export
does not support tensordicts with dimension names or non-tensor entries
(pytorch/tensordict#2073, pytorch/tensordict#2074): use ``strict=True`` for
them, and keep the non-tensor entries out of the output.
To export a :class:`~tensordict.nn.TensorDictModule` so that it takes and
returns plain tensors, see the :doc:`tutorials/export` tutorial. If the process
also creates a :class:`~tensordict.nn.CudaGraphModule`, see the warning in
:ref:`compile-cudagraphs`.

.. _compile-cudagraphs:

CUDA graphs
-----------

Two tools record CUDA graphs. :class:`~tensordict.nn.CudaGraphModule` works
with or without :func:`torch.compile`, records one graph, and checks the
inputs of each call against it. ``torch.compile(..., mode="reduce-overhead")``
needs compilation and manages its own graphs: it records a new graph for each
new batch size.

CudaGraphModule
~~~~~~~~~~~~~~~

:class:`~tensordict.nn.CudaGraphModule` captures a function, or a
:class:`~tensordict.nn.TensorDictModule`, in a
:class:`torch.cuda.CUDAGraph` and replays it. The function can be compiled.
Give it enough ``warmup`` calls to finish compiling, for instance
``CudaGraphModule(torch.compile(module), warmup=3)``. The first
``warmup - 1`` calls (``warmup=2`` by default) run the function on a side
stream. The next call runs it once more and captures the graph, and later
calls copy their inputs into the graph's buffers and replay it. See
:class:`~tensordict.nn.CudaGraphModule` for the other requirements: no
data-dependent control flow, and inputs that do not require gradients.

  >>> from tensordict.nn import CudaGraphModule, TensorDictModule
  >>> module = TensorDictModule(lambda x: x + 1, in_keys=["x"], out_keys=["y"])
  >>> graph_module = CudaGraphModule(module)
  >>> results = [
  ...     graph_module(TensorDict(x=torch.full((3,), float(i), device="cuda"), batch_size=[3]))
  ...     for i in range(4)
  ... ]
  >>> [result["y"][0].item() for result in results]
  [1.0, 2.0, 3.0, 4.0]

The replay follows these rules:

* The inputs must keep the shapes and batch size of the capture. A
  tensordict input without one of the ``in_keys`` raises a ``KeyError``, and
  a tensordict input with another batch size or entry shape raises a
  ``ValueError``, as a tensor input of another shape does. Extra keys are
  ignored. After the capture, a CPU input is copied to the device.
* The outputs are new tensors: the values written into the input
  tensordict, and the outputs of a function that returns a new tensordict or
  tensors, are clones of the graph's buffers. A result kept from an earlier
  call keeps its values.
* In-place writes to the inputs inside the function (``x.add_(1)``) do not
  reach the caller's tensors after the capture: the graph writes its own
  copies.
* The inputs are copied, and the graph replayed, on the current stream. If
  the inputs are written on another stream, order that work before the call,
  for instance with ``torch.cuda.current_stream().wait_stream(stream)``.
  Issue consecutive calls from one stream, or order them in the same way.
* Arguments that are not tensors (a ``float``, a ``str``) must not change
  after the capture. If the function takes a tensordict, the replay uses the
  captured values and ignores the new ones without an error
  (pytorch/tensordict#2075), and every tensor must be an entry of the
  tensordict, not a separate argument. If the function takes tensors, a
  changed value raises a ``ValueError``.

.. warning::

  Creating a :class:`~tensordict.nn.CudaGraphModule` removes
  :class:`~tensordict.TensorDict`, :class:`~tensordict.LazyStackedTensorDict`
  and :class:`~tensordict.PersistentTensorDict` from PyTorch's pytree registry
  for the rest of the process, and warns about it (set
  ``EXCLUDE_TD_FROM_PYTREE=1`` to silence the warning). After that,
  :func:`torch.export.export` and :func:`torch.func.grad` no longer accept
  these tensordicts. :func:`torch.vmap` still does, and tensorclasses stay
  registered (pytorch/tensordict#2076).

``mode="reduce-overhead"``
~~~~~~~~~~~~~~~~~~~~~~~~~~

``torch.compile(module, mode="reduce-overhead")`` records CUDA graphs with
the cudagraph trees of Inductor. It works with tensordict inputs and outputs,
and the PyTorch rules apply:

* The output tensors of a call are overwritten by the next call. Reading them
  afterwards raises ``RuntimeError: Error: accessing tensor output of
  CUDAGraphs that has been overwritten by a subsequent run``. Clone the
  outputs that you keep (``out = compiled(td).clone()``).
* A function that writes in place into an input tensor (``td["x"].add_(1)``)
  runs without CUDA graphs (``skipping cudagraphs due to mutated inputs`` with
  ``TORCH_LOGS="cudagraphs"``).
* For inference, call the function under :class:`torch.no_grad`. With
  gradients enabled and no backward pass, the calls run without recording a
  CUDA graph (``Running eager function`` with ``TORCH_LOGS="cudagraphs"``).

For example:

  >>> policy = torch.compile(
  ...     TensorDictModule(nn.Linear(3, 4).cuda(), in_keys=["obs"], out_keys=["action"]),
  ...     mode="reduce-overhead",
  ... )
  >>> with torch.no_grad():
  ...     outputs = [
  ...         policy(TensorDict(obs=torch.randn(8, 3, device="cuda"), batch_size=[8])).clone()
  ...         for _ in range(4)
  ...     ]
  >>> outputs[0]["action"].shape
  torch.Size([8, 4])
