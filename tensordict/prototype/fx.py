import operator
from itertools import filterfalse, tee

import torch.fx as fx
import torch.nn as nn

from tensordict.nn import TensorDictModule, TensorDictSequential


__all__ = ["symbolic_trace"]


class TDGraphModule(nn.Module):
    def __init__(self, graph_module, out_keys):
        super().__init__()
        self.out_keys = out_keys
        self._gm = graph_module

    def forward(self, tensordict, tensordict_out=None, **kwargs):
        outputs = self._gm(tensordict, **kwargs)

        if tensordict_out is None:
            tensordict_out = tensordict

        for out_key, output in zip(self.out_keys, outputs):
            if out_key != "_":
                tensordict_out.set(out_key, output, _run_checks=False)

        return tensordict_out

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._gm, name)


def symbolic_trace(td_module):
    if isinstance(td_module, TensorDictSequential):
        return _trace_tensordictsequential(td_module)
    elif isinstance(td_module, TensorDictModule):
        return _trace_tensordictmodule(td_module)
    raise TypeError(f"Unsupported type {type(td_module)}")


# cf. https://docs.python.org/3/library/itertools.html#itertools-recipes
def _partition(pred, iterable):
    "Use a predicate to partition entries into false entries and true entries"
    # partition(is_odd, range(10)) --> 0 2 4 6 8   and  1 3 5 7 9
    t1, t2 = tee(iterable)
    return filterfalse(pred, t1), filter(pred, t2)


def _parse_input_nodes(in_keys, nodes, td, inputs, env):
    for in_key, node in zip(in_keys, nodes):
        if in_key in inputs:
            new_node = inputs[in_key]
        else:
            output_proxy = operator.getitem(td, in_key)
            new_node = output_proxy.node
            inputs[in_key] = new_node
        env[node.name] = new_node


def _trace_tensordictmodule(td_module):
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


def _trace_tensordictsequential(td_sequential):
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

                for out_key, arg in zip(td_module.out_keys, args):
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
