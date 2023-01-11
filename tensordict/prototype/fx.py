import operator

import torch.fx as fx

# from tensordict import TensorDict
from tensordict.nn import TensorDictModule, TensorDictSequential


__all__ = ["symbolic_trace"]


def symbolic_trace(td_module):
    if isinstance(td_module, TensorDictSequential):
        return _trace_tensordictsequential(td_module)
    elif isinstance(td_module, TensorDictModule):
        return _trace_tensordictmodule(td_module)
    raise TypeError(f"Unsupported type {type(td_module)}")


def _trace_tensordictmodule(td_module):
    # this graph manipulation is based heavily on example in the PyTorch docs
    # https://pytorch.org/docs/stable/fx.html#proxy-retracing

    # trace the graph of the underlying module
    graph = fx.Tracer().trace(td_module.module)

    # create a new graph which we will populate from the old one
    new_graph = fx.Graph()
    env = {}
    # tracer = fx.proxy.GraphAppendingTracer(new_graph)

    # create a new placeholder for the input tensordict
    td = fx.Proxy(new_graph.placeholder("tensordict"))

    node_iter = iter(graph.nodes)

    # the first nodes, in order, are placeholders for the in_keys. We consume them and
    # convert them to "call_function" nodes with target=operator.getitem.
    for in_key, node in zip(td_module.in_keys, node_iter):
        output_proxy = operator.getitem(td, in_key)
        new_node = output_proxy.node
        env[node.name] = new_node

    # the remaining nodes we simply clone, pulling any arguments from the env
    for node in node_iter:
        # TODO: we would like to intercept the output node, populate the tensordict
        # with the outputs, and then return the tensordict as the output, however the
        # numerous runtime checks that are performed when setting a key are not easily
        # traceable...
        # if node.op == "output":
        #     # output node, intercept args and repopulate tensordict
        #     for out_key, arg in zip(td_module.out_keys, node.args):
        #         TensorDict.set(
        #             td,
        #             out_key,
        #             fx.Proxy(env[arg.name], tracer)
        #             if isinstance(arg, fx.Node)
        #             else arg,
        #         )
        #     new_graph.output(td, TensorDict)
        # else:
        #     new_node = new_graph.node_copy(node, lambda x: env[x.name])
        #     env[node.name] = new_node
        new_node = new_graph.node_copy(node, lambda x: env[x.name])
        env[node.name] = new_node

    return fx.GraphModule(td_module.module, new_graph)


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
            next(node_iter)  # discard the tensordict placeholder from submodule graph
        else:
            graph = fx.Tracer().trace(td_module.module)
            node_iter = iter(graph.nodes)

        for in_key, node in zip(td_module.in_keys, node_iter):
            # the first nodes, in order, are placeholders for the in_keys. We instead
            # check if they have been read before, if so we load from inputs, otherwise
            # we read from the tensordict proxy using operator.getitem
            if in_key in inputs:
                new_node = inputs[in_key]
            else:
                output_proxy = operator.getitem(td, in_key)
                new_node = output_proxy.node
                inputs[in_key] = new_node
            env[node.name] = new_node

        # clone the remaining nodes
        for node in node_iter:
            if node.op == "output":
                # capture the outputs but don't clone the output node (this would
                # result in prematurely returning intermediate values)
                # need to unpack the args in the case that the submodule is itself a
                # TensorDictSequential that returns a tuple of arguments
                for out_key, arg in zip(
                    td_module.out_keys,
                    node.args[0]
                    if isinstance(td_module, TensorDictSequential)
                    else node.args,
                ):
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

    return fx.GraphModule(td_sequential.module, new_graph)
