# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import os
from inspect import signature
from typing import Any, Callable, Iterable

import torch
import torch.utils._pytree
from tensordict._pytree import PYTREE_REGISTERED_LAZY_TDS, PYTREE_REGISTERED_TDS

from tensordict._td import TensorDict
from tensordict.base import is_tensor_collection

from tensordict.utils import implement_for, strtobool
from torch import nn
from torch.utils._pytree import SUPPORTED_NODES

try:
    from torch.nn.modules.module import _global_parameter_registration_hooks
except ImportError:
    # old torch version, passing
    pass

__base__setattr__ = nn.Module.__setattr__

PYTREE_HAS_ISLEAF = "is_leaf" in signature(torch.utils._pytree.tree_map).parameters


@implement_for("torch", "2.0", None)
def _register_params(self, name, param):
    """A simplified version of register_param where checks are skipped."""
    for hook in _global_parameter_registration_hooks.values():
        output = hook(self, name, param)
        if output is not None:
            param = output
    self._parameters[name] = param


@implement_for("torch", None, "2.0")
def _register_params(self, name, param):  # noqa: F811
    self.register_parameter(name, param)


def set_tensor(module: "torch.nn.Module", name: str, tensor: torch.Tensor) -> None:
    """Simplified version of torch.nn.utils._named_member_accessor."""
    if name in module._parameters:
        del module._parameters[name]  # type: ignore[assignment]
    was_buffer = name in module._buffers
    if was_buffer:
        del module._buffers[name]
    if isinstance(tensor, nn.Parameter):
        module.__dict__.pop(name, None)
        # module.register_parameter(name, tensor)
        _register_params(module, name, tensor)
    elif was_buffer and isinstance(tensor, Tensor):
        module._buffers[name] = tensor
    else:
        module.__dict__[name] = tensor


@implement_for("torch", "2.0", None)
def set_tensor_dict(  # noqa: F811
    module_dict, module, name: str, tensor: torch.Tensor
) -> None:
    """Simplified version of torch.nn.utils._named_member_accessor."""
    if name in module_dict["_parameters"]:
        del module_dict["_parameters"][name]  # type: ignore[assignment]
    was_buffer = name in module_dict["_buffers"]
    if was_buffer:
        del module_dict["_buffers"][name]
    if isinstance(tensor, nn.Parameter):
        module_dict.pop(name, None)
        # module.register_parameter(name, tensor)
        for hook in _global_parameter_registration_hooks.values():
            output = hook(module, name, tensor)
            if output is not None:
                tensor = output
        module_dict["_parameters"][name] = tensor
    elif was_buffer and isinstance(tensor, Tensor):
        module_dict["_buffers"][name] = tensor
    else:
        module_dict[name] = tensor


@implement_for("torch", None, "2.0")
def set_tensor_dict(  # noqa: F811
    module_dict, module, name: str, tensor: torch.Tensor
) -> None:
    """Simplified version of torch.nn.utils._named_member_accessor."""
    if name in module_dict["_parameters"]:
        del module_dict["_parameters"][name]  # type: ignore[assignment]
    was_buffer = name in module_dict["_buffers"]
    if was_buffer:
        del module_dict["_buffers"][name]
    if isinstance(tensor, nn.Parameter):
        module_dict.pop(name, None)
        module.register_parameter(name, tensor)
    elif was_buffer and isinstance(tensor, Tensor):
        module_dict["_buffers"][name] = tensor
    else:
        module_dict[name] = tensor


_RESET_OLD_TENSORDICT = True
import torch._functorch.vmap as vmap_src  # @manual=fbcode//caffe2:torch
from torch._functorch.vmap import (  # @manual=fbcode//caffe2:torch
    _add_batch_dim,
    _broadcast_to_and_flatten,
    _get_name,
    _maybe_remove_batch_dim,
    _validate_and_get_batch_size,
    Tensor,
    tree_flatten,
    tree_unflatten,
)


class _exclude_td_from_pytree:
    def __init__(self):
        self.tdnodes = {}

    def __enter__(self):
        for tdtype in PYTREE_REGISTERED_TDS + PYTREE_REGISTERED_LAZY_TDS:
            node = SUPPORTED_NODES.pop(tdtype, None)
            if node is None:
                continue
            self.tdnodes[tdtype] = node

    def __exit__(self, exc_type, exc_val, exc_tb):
        for tdtype, node in self.tdnodes.items():
            SUPPORTED_NODES[tdtype] = node

    def set(self):
        self.__enter__()

    def unset(self):
        self.__exit__(None, None, None)


if not strtobool(os.getenv("PYTORCH_TENSORDICT_IMPORT_VMAP", "False")):
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

        # we want to escape TensorDicts as they take care of adding the batch dimension
        if PYTREE_HAS_ISLEAF:
            flat_args, args_spec = tree_flatten(args, is_leaf=is_tensor_collection)
            flat_in_dims = _broadcast_to_and_flatten(in_dims, args_spec)
            if flat_in_dims is None:
                raise ValueError(
                    f"""vmap({_get_name(func)}, in_dims={in_dims}, ...)(<inputs>):
    in_dims is not compatible with the structure of `inputs`.
    in_dims has structure {tree_flatten(in_dims)[1]} but inputs
    has structure {args_spec}."""
                )
        else:
            with _exclude_td_from_pytree():
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
            if (
                isinstance(in_dim, int)
                and not isinstance(arg, Tensor)
                and not is_tensor_collection(arg)
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

        batched_inputs = []
        for in_dim, arg in zip(flat_in_dims, flat_args):
            if in_dim is None:
                if is_tensor_collection(arg):
                    # this may be a perf bottleneck and could benefit from caching
                    # arg = cache(arg.clone)(False)
                    arg = arg.clone(False)

                batched_input = arg
            else:
                if is_tensor_collection(arg):
                    batched_input = arg._add_batch_dim(
                        in_dim=in_dim, vmap_level=vmap_level
                    )
                else:
                    batched_input = _add_batch_dim(arg, in_dim, vmap_level)
            batched_inputs.append(batched_input)
        return tree_unflatten(batched_inputs, args_spec)

    vmap_src._create_batched_inputs = _create_batched_inputs

    def _unwrap_batched(
        batched_outputs: Any,
        out_dims: int | tuple[int, ...],
        vmap_level: int,
        batch_size: int,
        func: Callable,
    ) -> Any:
        if PYTREE_HAS_ISLEAF:
            flat_batched_outputs, output_spec = tree_flatten(
                batched_outputs, is_leaf=is_tensor_collection
            )
        else:
            with _exclude_td_from_pytree():
                flat_batched_outputs, output_spec = tree_flatten(batched_outputs)

        def incompatible_error():
            raise ValueError(
                f"vmap({_get_name(func)}, ..., out_dims={out_dims})(<inputs>): "
                f"out_dims is not compatible with the structure of `outputs`. "
                f"out_dims has structure {tree_flatten(out_dims)[1]} but outputs "
                f"has structure {output_spec}."
            )

        if isinstance(batched_outputs, torch.Tensor) or is_tensor_collection(
            batched_outputs
        ):
            # Some weird edge case requires us to spell out the following
            # see test_out_dims_edge_case
            if isinstance(out_dims, int):
                flat_out_dims = [out_dims]
            elif isinstance(out_dims, tuple) and len(out_dims) == 1:
                flat_out_dims = out_dims
            elif out_dims is None:
                flat_out_dims = [out_dims]
            else:
                incompatible_error()
        else:
            flat_out_dims = _broadcast_to_and_flatten(out_dims, output_spec)
            if flat_out_dims is None:
                incompatible_error()
        flat_outputs = []
        for batched_output, out_dim in zip(flat_batched_outputs, flat_out_dims):
            if not is_tensor_collection(batched_output):
                out = _maybe_remove_batch_dim(
                    _get_name(func), batched_output, vmap_level, batch_size, out_dim
                )
            else:
                out = batched_output._maybe_remove_batch_dim(
                    _get_name(func),
                    vmap_level=vmap_level,
                    batch_size=batch_size,
                    out_dim=out_dim,
                )
            flat_outputs.append(out)
        return tree_unflatten(flat_outputs, output_spec)

    vmap_src._unwrap_batched = _unwrap_batched


def extract_weights_and_buffers(
    model: nn.Module,
) -> TensorDict:  # noqa
    raise RuntimeError("extract_weights_and_buffers has been removed from tensordict.")


def is_functional(module: nn.Module):  # noqa
    raise RuntimeError("is_functional has been removed from tensordict.")


def make_functional(
    module: nn.Module,
    funs_to_decorate: Iterable[str] | None = None,
    keep_params: bool = False,
    return_params: bool = True,
) -> TensorDict:  # noqa
    raise RuntimeError("make_functional has been removed from tensordict.")


def get_functional(
    module: nn.Module,
    funs_to_decorate: Iterable[str] | None = None,
) -> nn.Module:  # noqa
    raise RuntimeError("get_functional has been removed from tensordict.")


def repopulate_module(model: nn.Module, tensordict: TensorDict) -> nn.Module:  # noqa
    raise RuntimeError("repopulate_module has been removed from tensordict.")


if strtobool(os.environ.get("EXCLUDE_TD_FROM_PYTREE", "0")):
    _exclude_td_from_pytree().set()
