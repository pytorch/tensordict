# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import inspect
import types
from copy import deepcopy
from functools import wraps
from typing import Any, Callable, Iterable

import torch
from tensordict import TensorDict
from tensordict.tensordict import TensorDictBase
from torch import nn

_RESET_OLD_TENSORDICT = True
try:
    import torch._functorch.vmap as vmap_src
    from torch._functorch.vmap import (
        _add_batch_dim,
        _broadcast_to_and_flatten,
        _get_name,
        _remove_batch_dim,
        _validate_and_get_batch_size,
        Tensor,
        tree_flatten,
        tree_unflatten,
    )

    _has_functorch = True
except ImportError:
    try:
        from functorch._src.vmap import (
            _add_batch_dim,
            _broadcast_to_and_flatten,
            _get_name,
            _remove_batch_dim,
            _validate_and_get_batch_size,
            Tensor,
            tree_flatten,
            tree_unflatten,
        )

        _has_functorch = True
        import functorch._src.vmap as vmap_src
    except ImportError:
        _has_functorch = False

# Monkey-patch functorch, mainly for cases where a "isinstance(obj, Tensor) is invoked
if _has_functorch:
    # Monkey-patches

    def _process_batched_inputs(
        in_dims: int | tuple[int, ...], args: Any, func: Callable
    ) -> tuple[Any, ...]:
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

    vmap_src._process_batched_inputs = _process_batched_inputs

    def _create_batched_inputs(
        flat_in_dims: list[int], flat_args: list[Any], vmap_level: int, args_spec
    ) -> Any:
        # See NOTE [Ignored _remove_batch_dim, _add_batch_dim]
        # If tensordict, we remove the dim at batch_size[in_dim] such that the TensorDict can accept
        # the batched tensors. This will be added in _unwrap_batched
        batched_inputs = [
            arg
            if in_dim is None
            else arg.apply(
                lambda _arg, in_dim=in_dim: _add_batch_dim(_arg, in_dim, vmap_level),
                batch_size=[b for i, b in enumerate(arg.batch_size) if i != in_dim],
            )
            if isinstance(arg, TensorDictBase)
            else _add_batch_dim(arg, in_dim, vmap_level)
            for in_dim, arg in zip(flat_in_dims, flat_args)
        ]
        return tree_unflatten(batched_inputs, args_spec)

    vmap_src._create_batched_inputs = _create_batched_inputs

    def _unwrap_batched(
        batched_outputs: Any,
        out_dims: int | tuple[int, ...],
        vmap_level: int,
        batch_size: int,
        func: Callable,
    ) -> Any:
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
                new_batch_size = list(batched_output.batch_size)
                new_batch_size.insert(out_dim, batch_size)
                out = batched_output.apply(
                    lambda x, out_dim=out_dim: _remove_batch_dim(
                        x, vmap_level, batch_size, out_dim
                    ),
                    batch_size=new_batch_size,
                )
            flat_outputs.append(out)
        return tree_unflatten(flat_outputs, output_spec)

    vmap_src._unwrap_batched = _unwrap_batched


# Tensordict-compatible Functional modules


def extract_weights_and_buffers(
    model: nn.Module,
    funs_to_decorate: Iterable[str] | None = None,
    recurse: bool = True,
) -> TensorDict:
    """Extracts the weights and buffers of a model in a tensordict, and adapts the modules to read those inputs."""
    tensordict = {}
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

    if funs_to_decorate is None:
        funs_to_decorate = ["forward"]

    if not model.__dict__.get("_functionalized", False):
        for fun_to_decorate in funs_to_decorate:
            try:
                setattr(
                    model,
                    fun_to_decorate,
                    types.MethodType(_make_decorator(model, fun_to_decorate), model),
                )
            except AttributeError:
                continue
    model.__dict__["_functionalized"] = True
    model.__dict__["_is_stateless"] = True
    return TensorDict(tensordict, [], _run_checks=False)


def _swap_state(
    model: nn.Module,
    tensordict: TensorDict,
    is_stateless: bool,
    return_old_tensordict: bool = False,
    old_tensordict: dict[str, torch.Tensor] | TensorDict | None = None,
) -> dict[str, torch.Tensor] | TensorDict | None:
    model.__dict__["_is_stateless"] = is_stateless
    if return_old_tensordict and old_tensordict is None:
        old_tensordict = {}
    keys = set(tensordict.keys())
    children = []
    for key, child in model.named_children():
        try:
            keys.remove(key)
        except KeyError:
            # if params are built externally, this could lead to a KeyError as some
            # modules do not have params
            pass
        children.append(key)
        value = tensordict.get(key, None)
        if value is None:
            # faster than get(key, Tensordict(...))
            value = {}

        _old_value = old_tensordict.get(key, None) if return_old_tensordict else None
        _old_value = _swap_state(
            child,
            value,
            return_old_tensordict=return_old_tensordict,
            old_tensordict=_old_value,
            is_stateless=is_stateless,
        )
        if old_tensordict is not None:
            old_tensordict[key] = _old_value
    for key in keys:
        value = tensordict.get(key)
        is_param = key in model.__dict__.get("_parameters")
        if return_old_tensordict:
            old_attr = getattr(model, key)
            if old_attr is None:
                old_attr = torch.zeros(*value.shape, 0)
            old_tensordict[key] = old_attr
        if is_param:
            delattr(model, key)
        setattr(model, key, value)
    if return_old_tensordict:
        return old_tensordict
    return None


def make_functional(
    module: nn.Module,
    funs_to_decorate: Iterable[str] | None = None,
) -> TensorDict:
    params = extract_weights_and_buffers(module, funs_to_decorate=funs_to_decorate)
    return params


def get_functional(
    module: nn.Module,
    funs_to_decorate: Iterable[str] | None = None,
) -> nn.Module:
    params = make_functional(module, funs_to_decorate=funs_to_decorate)
    out = deepcopy(module)
    repopulate_module(module, params)
    return out


def _make_decorator(module: nn.Module, fun_name: str) -> Callable:
    fun = getattr(module, fun_name)
    # we need to update the signature so that params can be the last positional arg
    oldsig = inspect.signature(fun)
    if "_forward_unimplemented" in fun.__name__:
        raise AttributeError("_forward_unimplemented not supported")
    # search if a VAR_POSITIONAL or VAR_KEYWORD is present
    # if yes insert step parameter before it, else insert it in last position
    params = list(oldsig.parameters.values())
    for i, param in enumerate(params):
        if param.kind == inspect.Parameter.KEYWORD_ONLY:
            out_type = inspect.Parameter.POSITIONAL_OR_KEYWORD
            break
        if param.kind == inspect.Parameter.VAR_POSITIONAL:
            out_type = inspect.Parameter.KEYWORD_ONLY
            i = i + 1
            break
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            out_type = inspect.Parameter.POSITIONAL_OR_KEYWORD
            break
        if (
            param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
            and param.default is not inspect._empty
        ):
            out_type = inspect.Parameter.POSITIONAL_OR_KEYWORD
            break
    else:
        out_type = inspect.Parameter.POSITIONAL_OR_KEYWORD
        i = len(params)
    # new parameter name is params or params_[_...] if params if already present
    name = "params"
    while name in oldsig.parameters:
        name += "_"
    newparam = inspect.Parameter(name, out_type, default=None)
    params.insert(i, newparam)
    # we can now build the signature for the wrapper function
    sig = oldsig.replace(parameters=params)

    @wraps(fun)
    def new_fun(self, *args, **kwargs):
        # 3 use cases: (1) params is the last arg, (2) params is in kwargs, (3) no params
        if self.__dict__.get("_is_stateless", False):
            params = kwargs.pop("params", None)
            if params is None:
                params = args[-1]
                args = args[:-1]
                # get the previous params, and tell the submodules not to look for params anymore
            old_params = _assign_params(
                self, params, make_stateless=False, return_old_tensordict=True
            )
            try:
                out = getattr(type(self), fun_name)(self, *args, **kwargs)
            finally:
                # reset the previous params, and tell the submodules to look for params
                _assign_params(
                    self, old_params, make_stateless=True, return_old_tensordict=True
                )
            return out
        else:
            return getattr(type(self), fun_name)(self, *args, **kwargs)

    new_fun.__signature__ = sig
    return new_fun


def _assign_params(
    module: nn.Module,
    params: TensorDict,
    make_stateless: bool,
    return_old_tensordict: bool,
) -> TensorDict | None:
    if params is not None:
        out = _swap_state(module, params, make_stateless, return_old_tensordict)
        if out is not None:
            return TensorDict(out, [], _run_checks=False)
        return None
    return None


def repopulate_module(model: nn.Module, tensordict: TensorDict) -> nn.Module:
    _swap_state(model, tensordict, is_stateless=False)
    return model
