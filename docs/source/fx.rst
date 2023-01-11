Tracing TensorDictModule
========================

We support tracing execution of :obj:`TensorDictModule` to create FX graphs. Simply import :obj:`symbolic_trace` from ``tensordict.prototype.fx`` instead of ``torch.fx``.

.. note:: Support for ``torch.fx`` is highly experimental and subject to change. Use with caution, and raise an issue if you try it out and encounter problems.

Tracing a :obj:`TensorDictModule`
---------------------------------

We'll illustrate with an example from the overview. We create a :obj:`TensorDictModule`, trace it, and inspect the graph and generated code.

.. code-block::
   :caption: Tracing a TensorDictModule

   >>> import torch
   >>> import torch.nn as nn
   >>> from tensordict import TensorDict
   >>> from tensordict.nn import TensorDictModule
   >>> from tensordict.prototype.fx import symbolic_trace

   >>> class Net(nn.Module):
   ...     def __init__(self):
   ...         super().__init__()
   ...         self.linear = nn.LazyLinear(1)
   ...
   ...     def forward(self, x):
   ...         logits = self.linear(x)
   ...         return logits, torch.sigmoid(logits)
   >>> module = TensorDictModule(
   ...     Net(),
   ...     in_keys=["input"],
   ...     out_keys=[("outputs", "logits"), ("outputs", "probabilities")],
   ... )
   >>> graph_module = symbolic_trace(module)
   >>> print(graph_module.graph)
   graph():
       %tensordict : [#users=1] = placeholder[target=tensordict]
       %getitem : [#users=1] = call_function[target=operator.getitem](args = (%tensordict, input), kwargs = {})
       %linear : [#users=2] = call_module[target=linear](args = (%getitem,), kwargs = {})
       %sigmoid : [#users=1] = call_function[target=torch.sigmoid](args = (%linear,), kwargs = {})
       return (linear, sigmoid)
   >>> print(graph_module.code)

   def forward(self, tensordict):
       getitem = tensordict['input'];  tensordict = None
       linear = self.linear(getitem);  getitem = None
       sigmoid = torch.sigmoid(linear)
       return (linear, sigmoid)

We can check that a forward pass with each module results in the same outputs.

   >>> tensordict = TensorDict({"input": torch.randn(32, 100)}, [32])
   >>> module_out = module(tensordict, tensordict_out=TensorDict({}, []))
   >>> graph_module_out = graph_module(tensordict, tensordict_out=TensorDict({}, []))
   >>> assert (
   ...     module_out["outputs", "logits"] == graph_module_out["outputs", "logits"]
   ... ).all()
   >>> assert (
   ...     module_out["outputs", "probabilities"]
   ...     == graph_module_out["outputs", "probabilities"]
   ... ).all()

Tracing a :obj:`TensorDictSequential`
-------------------------------------

We can also trace :obj:`TensorDictSequential`. In this case the entire execution of the module is traced into a single graph, eliminating intermediate reads and writes on the input :obj:`TensorDict`.

We demonstrate by tracing the sequential example from the overview.

.. code-block::
   :caption: Tracing TensorDictSequential

   >>> import torch
   >>> import torch.nn as nn
   >>> from tensordict import TensorDict
   >>> from tensordict.nn import TensorDictModule, TensorDictSequential
   >>> from tensordict.prototype.fx import symbolic_trace

   >>> class Net(nn.Module):
   ...     def __init__(self, input_size=100, hidden_size=50, output_size=10):
   ...         super().__init__()
   ...         self.fc1 = nn.Linear(input_size, hidden_size)
   ...         self.fc2 = nn.Linear(hidden_size, output_size)
   ...
   ...     def forward(self, x):
   ...         x = torch.relu(self.fc1(x))
   ...         return self.fc2(x)
   ...
   ... class Masker(nn.Module):
   ...     def forward(self, x, mask):
   ...         return torch.softmax(x * mask, dim=1)
   >>> net = TensorDictModule(
   ...     Net(), in_keys=[("input", "x")], out_keys=[("intermediate", "x")]
   ... )
   >>> masker = TensorDictModule(
   ...     Masker(),
   ...     in_keys=[("intermediate", "x"), ("input", "mask")],
   ...     out_keys=[("output", "probabilities")],
   ... )
   >>> module = TensorDictSequential(net, masker)
   >>> graph_module = symbolic_trace(module)
   >>> print(graph_module.code)

   def forward(self, tensordict):
       getitem = tensordict[('input', 'x')]
       _0_fc1 = getattr(self, "0").module.fc1(getitem);  getitem = None
       relu = torch.relu(_0_fc1);  _0_fc1 = None
       _0_fc2 = getattr(self, "0").module.fc2(relu);  relu = None
       getitem_1 = tensordict[('input', 'mask')];  tensordict = None
       mul = _0_fc2 * getitem_1;  getitem_1 = None
       softmax = torch.softmax(mul, dim = 1);  mul = None
       return (_0_fc2, softmax)

In this case the generated graph and code is a bit more complicated. We can visualize it as follows (requires ``pydot``)

.. code-block::
   :caption: Visualising the graph

   >>> from torch.fx.passes.graph_drawer import FxGraphDrawer
   >>> g = FxGraphDrawer(graph_module, "sequential")
   >>> with open("graph.svg", "wb") as f:
   ...     f.write(g.get_dot_graph().create_svg())

Which results in the following visualisation

.. image:: _static/img/graph.svg
   :alt: Visualization of the traced graph.
