import operator

import torch.fx as fx

# from tensordict import TensorDict


def trace_tdmodule(td_module):
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
