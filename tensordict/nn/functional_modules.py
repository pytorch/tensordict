# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import inspect
from functools import wraps

import torch
from torch import nn

from tensordict import TensorDict
from tensordict.tensordict import TensorDictBase

_RESET_OLD_TENSORDICT = True
try:
    import functorch._src.vmap

    _has_functorch = True
except ImportError:
    _has_functorch = False

# Monky-patch functorch, mainly for cases where a "isinstance(obj, Tensor) is invoked
if _has_functorch:
    from functorch._src.vmap import (
        _get_name,
        tree_flatten,
        _broadcast_to_and_flatten,
        Tensor,
        _validate_and_get_batch_size,
        _add_batch_dim,
        tree_unflatten,
        _remove_batch_dim,
    )

    # Monkey-patches

    def _process_batched_inputs(in_dims, args, func):
        if not isinstance(in_dims, int) and not isinstance(in_dims, tuple):
            raise ValueError(
                f"""vmap({_get_name(func)}, in_dims={in_dims}, ...)(<inputs>):
expected `in_dims` to be int or a (potentially nested) tuple
matching the structure of inputs, got: {type(in_dims)}."""
            )
        if len(args) == 0:
            raise ValueError(
                f"""vmap({_get_name(func)})(<inputs>): got no inputs. Maybe you forgot to add
inputs, or you are trying to vmap over a function with no inputs.
The latter is unsupported."""
            )

        flat_args, args_spec = tree_flatten(args)
        flat_in_dims = _broadcast_to_and_flatten(in_dims, args_spec)
        if flat_in_dims is None:
            raise ValueError(
                f"""vmap({_get_name(func)}, in_dims={in_dims}, ...)(<inputs>):
in_dims is not compatible with the structure of `inputs`.
in_dims has structure {tree_flatten(in_dims)[1]} but inputs
has structure {args_spec}."""
            )

        for i, (arg, in_dim) in enumerate(zip(flat_args, flat_in_dims)):
            if not isinstance(in_dim, int) and in_dim is not None:
                raise ValueError(
                    f"""vmap({_get_name(func)}, in_dims={in_dims}, ...)(<inputs>):
Got in_dim={in_dim} for an input but in_dim must be either
an integer dimension or None."""
                )
            if isinstance(in_dim, int) and not isinstance(
                arg, (Tensor, TensorDictBase)
            ):
                raise ValueError(
                    f"""vmap({_get_name(func)}, in_dims={in_dims}, ...)(<inputs>):
Got in_dim={in_dim} for an input but the input is of type
{type(arg)}. We cannot vmap over non-Tensor arguments,
please use None as the respective in_dim"""
                )
            if in_dim is not None and (in_dim < -arg.dim() or in_dim >= arg.dim()):
                raise ValueError(
                    f"""vmap({_get_name(func)}, in_dims={in_dims}, ...)(<inputs>):
Got in_dim={in_dim} for some input, but that input is a Tensor
of dimensionality {arg.dim()} so expected in_dim to satisfy
-{arg.dim()} <= in_dim < {arg.dim()}."""
                )
            if in_dim is not None and in_dim < 0:
                flat_in_dims[i] = in_dim % arg.dim()

        return (
            _validate_and_get_batch_size(flat_in_dims, flat_args),
            flat_in_dims,
            flat_args,
            args_spec,
        )

    functorch._src.vmap._process_batched_inputs = _process_batched_inputs

    def _create_batched_inputs(flat_in_dims, flat_args, vmap_level: int, args_spec):
        # See NOTE [Ignored _remove_batch_dim, _add_batch_dim]
        # If tensordict, we remove the dim at batch_size[in_dim] such that the TensorDict can accept
        # the batched tensors. This will be added in _unwrap_batched
        batched_inputs = [
            arg
            if in_dim is None
            else arg.apply(
                lambda _arg: _add_batch_dim(_arg, in_dim, vmap_level),
                batch_size=[b for i, b in enumerate(arg.batch_size) if i != in_dim],
            )
            if isinstance(arg, TensorDictBase)
            else _add_batch_dim(arg, in_dim, vmap_level)
            for in_dim, arg in zip(flat_in_dims, flat_args)
        ]
        return tree_unflatten(batched_inputs, args_spec)

    functorch._src.vmap._create_batched_inputs = _create_batched_inputs

    def _unwrap_batched(
        batched_outputs, out_dims, vmap_level: int, batch_size: int, func
    ):
        flat_batched_outputs, output_spec = tree_flatten(batched_outputs)

        for out in flat_batched_outputs:
            # Change here:
            if isinstance(out, (TensorDictBase, torch.Tensor)):
                continue
            raise ValueError(
                f"vmap({_get_name(func)}, ...): `{_get_name(func)}` must only return "
                f"Tensors, got type {type(out)} as a return."
            )

        def incompatible_error():
            raise ValueError(
                f"vmap({_get_name(func)}, ..., out_dims={out_dims})(<inputs>): "
                f"out_dims is not compatible with the structure of `outputs`. "
                f"out_dims has structure {tree_flatten(out_dims)[1]} but outputs "
                f"has structure {output_spec}."
            )

        # Here:
        if isinstance(batched_outputs, (TensorDictBase, torch.Tensor)):
            # Some weird edge case requires us to spell out the following
            # see test_out_dims_edge_case
            if isinstance(out_dims, int):
                flat_out_dims = [out_dims]
            elif isinstance(out_dims, tuple) and len(out_dims) == 1:
                flat_out_dims = out_dims
                out_dims = out_dims[0]
            else:
                incompatible_error()
        else:
            flat_out_dims = _broadcast_to_and_flatten(out_dims, output_spec)
            if flat_out_dims is None:
                incompatible_error()

        flat_outputs = []
        for batched_output, out_dim in zip(flat_batched_outputs, flat_out_dims):
            if not isinstance(batched_output, TensorDictBase):
                out = _remove_batch_dim(batched_output, vmap_level, batch_size, out_dim)
            else:
                out = batched_output.apply(
                    lambda x: _remove_batch_dim(x, vmap_level, batch_size, out_dim),
                    batch_size=[batch_size, *batched_output.batch_size],
                )
            flat_outputs.append(out)
        return tree_unflatten(flat_outputs, output_spec)

    functorch._src.vmap._unwrap_batched = _unwrap_batched

# Tensordict-compatible Functional modules


def extract_weights_and_buffers(model: nn.Module):
    """Extracts the weights and buffers of a model in a tensordict, and adapts the modules to read those inputs."""
    tensordict = TensorDict({}, [])
    for name, param in list(model.named_parameters(recurse=False)):
        setattr(model, name, None)
        tensordict[name] = param

    for name, param in list(model.named_buffers(recurse=False)):
        setattr(model, name, None)
        tensordict[name] = param

    for name, module in model.named_children():
        module_tensordict = extract_weights_and_buffers(module)
        if module_tensordict is not None:
            tensordict[name] = module_tensordict
    model.forward = _forward_decorator(model)

    if len(tensordict.keys()):
        return tensordict
    else:
        return None


def _swap_state(model, tensordict, return_old_tensordict=False, old_tensordict=None):

    if return_old_tensordict and old_tensordict is None:
        old_tensordict = TensorDict(
            {}, torch.Size([]), device=tensordict.device, _run_checks=False
        )

    for key, value in list(tensordict.items()):
        if isinstance(value, TensorDictBase):
            if return_old_tensordict:
                _old_value = old_tensordict.get(key, None)
            _old_value = _swap_state(
                getattr(model, key),
                value,
                return_old_tensordict=return_old_tensordict,
                old_tensordict=_old_value,
            )
            old_tensordict._tensordict[key] = _old_value
        else:
            if return_old_tensordict:
                old_attr = getattr(model, key)
                if old_attr is None:
                    old_attr = torch.zeros(*value.shape, 0)
                old_tensordict._tensordict[key] = old_attr
            delattr(model, key)
            setattr(model, key, value)
    if return_old_tensordict:
        return old_tensordict


def make_functional(module):
    return extract_weights_and_buffers(module)


def _forward_decorator(module):
    forward = module.forward

    # we need to update the signature so that params can be the last positional arg
    oldsig = inspect.signature(forward)
    # search if a VAR_POSITIONAL or VAR_KEYWORD is present
    # if yes insert step parameter before it, else insert it in last position
    params = list(oldsig.parameters.values())
    for i, param in enumerate(params):
        if param.kind == inspect.Parameter.VAR_POSITIONAL:
            break
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            break
        if (
            param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
            and param.default is not inspect._empty
        ):
            break
    else:
        i = len(params)
    # new parameter name is params or params_[_...] if params if already present
    name = "params"
    while name in oldsig.parameters:
        name += "_"
    newparam = inspect.Parameter(
        name, inspect.Parameter.POSITIONAL_OR_KEYWORD, default=None
    )
    params.insert(i, newparam)
    # we can now build the signature for the wrapper function
    sig = oldsig.replace(parameters=params)

    @wraps(forward)
    def new_forward(*args, **kwargs):
        # 3 use cases: (1) params is the last arg, (2) params is in kwargs, (3) no params
        if len(args) == i + 1:
            params = args[-1]
            args = args[:-1]
        else:
            params = kwargs.pop("params", None)
        old_params = _assign_params(module, params)
        out = forward(*args, **kwargs)
        _assign_params(module, old_params)
        return out

    new_forward.__signature__ = sig
    return new_forward


def _assign_params(module, params):
    if params is not None:
        return _swap_state(module, params, True)
