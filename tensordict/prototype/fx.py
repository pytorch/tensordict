# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import operator
from itertools import filterfalse, tee
from typing import Any, Callable, Iterable

from tensordict._td import _unravel_key_to_tuple
from tensordict.nn import TensorDictModule, TensorDictModuleBase, TensorDictSequential
from tensordict.tensordict import TensorDictBase
from tensordict.utils import _zip_strict, NestedKey
from torch import fx, nn

__all__ = ["TDGraphModule", "symbolic_trace"]


class TDGraphModule(nn.Module):
    """The module that :func:`symbolic_trace` returns: an FX graph that reads and writes a tensordict.

    A call reads the ``in_keys`` of the traced module from the input
    tensordict, runs the graph, and writes the outputs at ``out_keys``, into
    ``tensordict_out`` if given, else into the input. The ``graph`` and
    ``code`` attributes are those of the underlying
    :class:`torch.fx.GraphModule`.

    This class is a prototype: its API can change without a deprecation
    period. See :func:`symbolic_trace` for an example.

    Args:
        graph_module (torch.fx.GraphModule): the traced graph.
        out_keys (list of NestedKey): the keys where the outputs are written.
    """

    def __init__(
        self,
        graph_module: fx.GraphModule,
        out_keys: list[NestedKey],
    ) -> None:
        super().__init__()
        self.out_keys = [_unravel_key_to_tuple(ok) for ok in out_keys]
        self._gm = graph_module

    def forward(
        self,
        tensordict: TensorDictBase,
        tensordict_out: TensorDictBase | None = None,
        **kwargs,
    ) -> TensorDictBase:
        """Runs the graph on ``tensordict`` and writes the outputs at ``out_keys``.

        Args:
            tensordict (TensorDictBase): the input tensordict.
            tensordict_out (TensorDictBase, optional): the tensordict to write
                the outputs into. Defaults to ``None``, which writes them into
                ``tensordict``.
            **kwargs: passed to the graph.

        Returns:
            The tensordict that holds the outputs.
        """
        outputs = self._gm(tensordict, **kwargs)
        # One output is the tensor itself. A tuple or a list is several outputs.
        # Iterating a tensor walks dim 0, so zip would not pair out_keys with
        # outputs.
        if not isinstance(outputs, (tuple, list)):
            outputs = (outputs,)

        if tensordict_out is None:
            tensordict_out = tensordict

        for out_key, output in _zip_strict(self.out_keys, outputs):
            # Stored keys are unraveled, so the skip sentinel is ("_",).
            # ("_", "y") is a real nested key and must be written.
            if out_key != ("_",):
                tensordict_out._set_tuple(
                    out_key, output, inplace=False, validated=True, non_blocking=False
                )

        return tensordict_out

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._gm, name)


def symbolic_trace(td_module: TensorDictModuleBase) -> TDGraphModule:
    """Traces a TensorDictModule or a TensorDictSequential into an FX graph.

    This function is a prototype: its API can change without a deprecation
    period. See :ref:`fx` for a guide.

    It supports a :class:`~tensordict.nn.TensorDictModule` that wraps an
    :class:`~torch.nn.Module` that :func:`torch.fx.symbolic_trace` can trace,
    and a :class:`~tensordict.nn.TensorDictSequential` of such modules. A
    module that wraps a plain function or a lambda raises a ``RuntimeError``
    ("Unsupported type ... passed for root!").

    Args:
        td_module (TensorDictModuleBase): the module to trace.

    Returns:
        A :class:`TDGraphModule` with the same inputs and outputs as
        ``td_module``.

    Raises:
        TypeError: if ``td_module`` is not a
            :class:`~tensordict.nn.TensorDictModule` or a
            :class:`~tensordict.nn.TensorDictSequential`.

    Examples:
        >>> import torch
        >>> from torch import nn
        >>> from tensordict import TensorDict
        >>> from tensordict.nn import TensorDictModule, TensorDictSequential
        >>> from tensordict.prototype.fx import symbolic_trace
        >>> seq = TensorDictSequential(
        ...     TensorDictModule(nn.Linear(3, 4), in_keys=["x"], out_keys=["hidden"]),
        ...     TensorDictModule(nn.ReLU(), in_keys=["hidden"], out_keys=["y"]),
        ... )
        >>> graph_module = symbolic_trace(seq)
        >>> td = graph_module(TensorDict(x=torch.ones(2, 3), batch_size=[2]))
        >>> sorted(td.keys())
        ['hidden', 'x', 'y']
        >>> expected = seq(TensorDict(x=torch.ones(2, 3), batch_size=[2]))["y"]
        >>> torch.equal(td["y"], expected)
        True
    """
    if isinstance(td_module, TensorDictSequential):
        return _trace_tensordictsequential(td_module)
    elif isinstance(td_module, TensorDictModule):
        return _trace_tensordictmodule(td_module)
    raise TypeError(f"Unsupported type {type(td_module)}")


# cf. https://docs.python.org/3/library/itertools.html#itertools-recipes
def _partition(
    pred: Callable[..., bool], iterable: Iterable[Any]
) -> tuple[Iterable[Any], Iterable[Any]]:
    """Use a predicate to partition entries into false entries and true entries."""
    # partition(is_odd, range(10)) --> 0 2 4 6 8   and  1 3 5 7 9
    t1, t2 = tee(iterable)
    return filterfalse(pred, t1), filter(pred, t2)


def _parse_input_nodes(
    in_keys: list[NestedKey], nodes, td: TensorDictBase, inputs: tuple[Any, ...], env
):
    for in_key, node in zip(in_keys, nodes):
        if in_key in inputs:
            new_node = inputs[in_key]
        else:
            output_proxy = operator.getitem(td, in_key)
            new_node = output_proxy.node
            inputs[in_key] = new_node
        env[node.name] = new_node


def _trace_tensordictmodule(td_module: TensorDictModuleBase) -> TDGraphModule:
    # this graph manipulation is based heavily on example in the PyTorch docs
    # https://pytorch.org/docs/stable/fx.html#proxy-retracing

    # trace the graph of the underlying module
    graph = fx.Tracer().trace(td_module.module)

    # create a new graph which we will populate from the old one
    new_graph = fx.Graph()
    env = {}

    # create a new placeholder for the input tensordict
    td = fx.Proxy(new_graph.placeholder("tensordict"))

    node_iter = iter(graph.nodes)

    # the first nodes, in order, are placeholders for the in_keys. We consume them and
    # convert them to "call_function" nodes with target=operator.getitem.
    _parse_input_nodes(td_module.in_keys, node_iter, td, {}, env)

    # the remaining nodes we simply clone, pulling any arguments from the env
    for node in node_iter:
        new_node = new_graph.node_copy(node, lambda x: env[x.name])
        env[node.name] = new_node

    return TDGraphModule(
        fx.GraphModule(td_module.module, new_graph), td_module.out_keys
    )


def _trace_tensordictsequential(td_sequential: TensorDictSequential) -> TDGraphModule:
    # we track values previously read from / written to the tensordict by storing the
    # nodes / proxy values in the inputs / outputs dictionaries
    inputs = {}
    outputs = {}
    # env is a lookup for nodes in the new graph using names from the old graph
    env = {}

    new_graph = fx.Graph()
    td = fx.Proxy(new_graph.placeholder("tensordict"))

    for i, td_module in enumerate(td_sequential.module):
        # trace the submodule
        if isinstance(td_module, TensorDictSequential):
            graph = _trace_tensordictsequential(td_module).graph
            node_iter = iter(graph.nodes)
            _td = next(node_iter)  # tensordict placeholder from submodule graph

            # in the graph of TensorDictSequential, the getitem calls to the tensordict
            # need not come first, so we partition nodes into getitem calls on the
            # placeholder tensordict (input_nodes) and the remaining nodes
            node_iter, input_nodes = _partition(
                lambda node, _td=_td: (
                    node.op == "call_function"
                    and node.target == operator.getitem
                    and node.args[0] == _td
                ),
                node_iter,
            )
            _parse_input_nodes(td_module.in_keys, input_nodes, td, inputs, env)

        else:
            graph = fx.Tracer().trace(td_module.module)
            # in the trace of a regular nn.Module the placeholder nodes all come first,
            # so we just consume them in order
            node_iter = iter(graph.nodes)
            _parse_input_nodes(td_module.in_keys, node_iter, td, inputs, env)

        # clone the remaining nodes
        for node in node_iter:
            if node.op == "output":
                # capture the outputs but don't clone the output node (this would
                # result in prematurely returning intermediate values)

                # need to unpack the args in the case that the submodule is itself a
                # TensorDictSequential that returns a tuple of arguments
                args = (
                    node.args[0]
                    if isinstance(td_module, TensorDictSequential)
                    else node.args
                )

                # if the submodule has multiple outputs, args has structure
                # ((out1, out2,),), so we need to do some extra unpacking
                args = args[0] if isinstance(args[0], tuple) else args

                for out_key, arg in _zip_strict(td_module.out_keys, args):
                    # any outputs of submodules will need to be returned at the end
                    outputs[out_key] = env[arg.name]
                    # we also need to make outputs of submodules available as inputs
                    # to subsequent submodules
                    inputs[out_key] = env[arg.name]
            else:
                new_node = new_graph.node_copy(node, lambda x: env[x.name])
                if new_node.op in ("call_module", "get_attr"):
                    # since we traced the submodule in isolation, we need to patch the
                    # targets of any calls to methods on the module or attribute access
                    new_node.target = f"{i}.module.{new_node.target}"
                    new_node.name = f"_{i}_{new_node.name}"
                env[node.name] = new_node

    # finally we add a new output node that collects all of the output values from
    # submodules in the graph and returns them together
    new_graph.output(tuple(outputs.values()))

    return TDGraphModule(
        fx.GraphModule(td_sequential.module, new_graph), tuple(outputs.keys())
    )
