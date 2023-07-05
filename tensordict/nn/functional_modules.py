# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import inspect
import re
import types
import warnings
from copy import deepcopy
from functools import wraps
from typing import Any, Callable, Iterable

import torch
from tensordict import TensorDict
from tensordict.tensordict import _is_tensor_collection, TensorDictBase

from tensordict.utils import implement_for
from torch import nn

try:
    from torch.nn.modules.module import _global_parameter_registration_hooks
except ImportError:
    # old torch version, passing
    pass

__base__setattr__ = nn.Module.__setattr__


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

        batched_inputs = []
        for in_dim, arg in zip(flat_in_dims, flat_args):
            if in_dim is None:
                if isinstance(arg, TensorDictBase):
                    # this may be a perf bottleneck and could benefit from caching
                    # arg = cache(arg.clone)(False)
                    arg = arg.clone(False)

                batched_input = arg
            else:
                if isinstance(arg, TensorDictBase):
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
                out = batched_output._remove_batch_dim(
                    vmap_level=vmap_level, batch_size=batch_size, out_dim=out_dim
                )
            flat_outputs.append(out)
        return tree_unflatten(flat_outputs, output_spec)

    vmap_src._unwrap_batched = _unwrap_batched


# Tensordict-compatible Functional modules


def _decorate_funs(
    model: nn.Module,
    make_stateless: bool,
    funs_to_decorate: Iterable[str] | None = None,
) -> None:
    if funs_to_decorate is None:
        funs_to_decorate = ["forward"]
    _is_functional = model.__dict__.get("_functionalized", False)
    if not _is_functional:
        model.__dict__["_functionalized"] = True
        model.__dict__["_decorated_funs"] = set()

    for fun_to_decorate in funs_to_decorate:
        if fun_to_decorate in model.__dict__["_decorated_funs"]:
            continue
        try:
            setattr(
                model,
                fun_to_decorate,
                types.MethodType(_make_decorator(model, fun_to_decorate), model),
            )
            model.__dict__["_decorated_funs"].add(fun_to_decorate)
        except AttributeError:
            continue
    if not model.__dict__.get("_is_stateless", False):
        model.__dict__["_is_stateless"] = make_stateless

    for module in model.children():
        # we decorate forward for the sub-modules
        _decorate_funs(module, make_stateless=make_stateless)


def extract_weights_and_buffers(
    model: nn.Module,
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
    model.__dict__["_is_stateless"] = True
    return TensorDict(tensordict, batch_size=torch.Size([]), _run_checks=False)


# For bookkeeping: this function seems to have the same runtime but will not access
# modules that don't have parameters if they're not registered as empty tensordicts
# in the input. Hence they won't be turned as stateful, which could cause some bugs.
def _swap_state(
    model: nn.Module,
    tensordict: TensorDict,
    is_stateless: bool,
    return_old_tensordict: bool = False,
    old_tensordict: dict[str, torch.Tensor] | TensorDict | None = None,
) -> dict[str, torch.Tensor] | TensorDict | None:
    __dict__ = model.__dict__
    was_stateless = __dict__.get("_is_stateless", None)
    if was_stateless is None:
        raise Exception(f"{model}\nhas no stateless attribute.")
    __dict__["_is_stateless"] = is_stateless
    # return_old_tensordict = return_old_tensordict and not was_stateless
    if old_tensordict is None:
        old_tensordict_dict = old_tensordict = {}
    else:
        old_tensordict_dict = {}
    for key, value in tensordict.items():
        cls = value.__class__
        if _is_tensor_collection(cls) or issubclass(cls, dict):
            _old_value = old_tensordict.get(key, None)
            _old_value = _swap_state(
                __dict__["_modules"][key],
                value,
                is_stateless=is_stateless,
                old_tensordict=_old_value,
                return_old_tensordict=return_old_tensordict,
            )
            old_tensordict_dict[key] = _old_value
        else:
            _old_value = None
            if return_old_tensordict:
                _old_value = __dict__["_parameters"].get(key, None)
                if _old_value is None:
                    _old_value = __dict__["_buffers"].get(key, None)
                if _old_value is None:
                    _old_value = __dict__.get(key, None)
                if _old_value is None:
                    pass
                    # _old_value = torch.zeros(*value.shape, 0)
                old_tensordict_dict[key] = _old_value
                # old_tensordict_dict[key] = _old_value
            if model.__class__.__setattr__ is __base__setattr__:
                set_tensor_dict(__dict__, model, key, value)
            else:
                setattr(model, key, value)
    old_tensordict.update(old_tensordict_dict)
    if was_stateless or not return_old_tensordict:
        return old_tensordict
    else:
        return TensorDict(old_tensordict, [], _run_checks=False)


# def _swap_state(
#     model: nn.Module,
#     tensordict: TensorDict,
#     is_stateless: bool,
#     return_old_tensordict: bool = False,
#     old_tensordict: dict[str, torch.Tensor] | TensorDict | None = None,
# ) -> dict[str, torch.Tensor] | TensorDict | None:
#     __dict__ = model.__dict__
#     was_stateless = __dict__.get("_is_stateless", None)
#     if was_stateless is None:
#         raise Exception(f"{model}\nhas no stateless attribute.")
#     __dict__["_is_stateless"] = is_stateless
#     # return_old_tensordict = return_old_tensordict and not was_stateless
#     if old_tensordict is None:
#         old_tensordict_dict = old_tensordict = {}
#     else:
#         old_tensordict_dict = {}
#     # keys = set(tensordict.keys())
#     children = set()
#     # this loop ignores the memo from named children
#     for key, child in __dict__["_modules"].items():  # model.named_children():
#         children.add(key)
#         value = tensordict.get(key, None)
#         if value is None:
#             # faster than get(key, Tensordict(...))
#             value = {}
#
#         _old_value = old_tensordict.get(key, None)
#         _old_value = _swap_state(
#             child,
#             value,
#             return_old_tensordict=return_old_tensordict,
#             old_tensordict=_old_value,
#             is_stateless=is_stateless,
#         )
#         old_tensordict_dict[key] = _old_value
#     for key in tensordict.keys():
#         if key in children:
#             continue
#         value = tensordict.get(key)
#         if return_old_tensordict:
#             old_attr = __dict__["_parameters"].get(key, None)
#             if old_attr is None:
#                 old_attr = __dict__["_buffers"].get(key, None)
#             if old_attr is None:
#                 old_attr = __dict__.get(key, None)
#             if old_attr is None:
#                 old_attr = torch.zeros(*value.shape, 0)
#             old_tensordict_dict[key] = old_attr
#         # is_param = key in model.__dict__.get("_parameters")
#         # if is_param:
#         #     delattr(model, key)
#         #     print(value)
#         set_tensor_dict(__dict__, model, key, value)
#     old_tensordict.update(old_tensordict_dict)
#     if was_stateless or not return_old_tensordict:
#         return old_tensordict
#     else:
#         return TensorDict(old_tensordict, [])


def is_functional(module: nn.Module):
    """Checks if :func:`make_functional` has been called on the module."""
    return "_functionalized" in module.__dict__


def make_functional(
    module: nn.Module,
    funs_to_decorate: Iterable[str] | None = None,
    keep_params: bool = False,
    return_params: bool = True,
) -> TensorDict:
    """Converts a nn.Module to a functional module in-place, and returns its params.

    Args:
        module (torch.nn.Module): module that is to be made functional.
        funs_to_decorate (iterable of str, optional): each string must correspond
            to a function belonging to module. For nested modules, the
            :meth:`torch.nn.Module.forward` method will be decorated.
            Defaults to ``"forward"``.
        keep_params (bool, optional): if ``True``, the module will keep its
            parameters. Defaults to ``False``.
        return_params (bool, optional): if ``True``, the parameters will
            be collected in a nested tensordict and returned. If ``False``,
            the module will be made functional but still be stateful.

    """
    _is_stateless = module.__dict__.get("_is_stateless", False)
    _decorate_funs(
        module,
        funs_to_decorate=funs_to_decorate,
        make_stateless=not keep_params,
    )
    if return_params and not _is_stateless:
        params = extract_weights_and_buffers(
            module,
        )
        if keep_params:
            repopulate_module(module, params)
        return params.lock_()
    elif return_params and _is_stateless:
        raise RuntimeError(
            "Calling make_functional with return_params=True on a functional, stateless module. "
        )
    elif not keep_params:
        extract_weights_and_buffers(module)


def get_functional(
    module: nn.Module,
    funs_to_decorate: Iterable[str] | None = None,
) -> nn.Module:
    """Converts a nn.Module to a functional module in-place, and returns a stateful version of this module that can be used in functional settings."""
    params = make_functional(module, funs_to_decorate=funs_to_decorate)
    out = deepcopy(module)
    repopulate_module(module, params)
    return out


def _make_decorator(module: nn.Module, fun_name: str) -> Callable:
    fun = getattr(module, fun_name)

    from tensordict.nn.common import TensorDictModuleBase

    @wraps(fun)
    def new_fun(self, *args, **kwargs):
        # 3 use cases: (1) params is the last arg, (2) params is in kwargs, (3) no params
        _is_stateless = self.__dict__.get("_is_stateless", False)
        params = kwargs.pop("params", None)

        if isinstance(self, TensorDictModuleBase):
            if (
                params is None
                and len(args) == 2
                and all(_is_tensor_collection(item.__class__) for item in args)
            ):
                params = args[1]
                args = args[:1]
        elif (
            len(args) and _is_tensor_collection(args[0].__class__)
        ) or "tensordict" in kwargs:
            warnings.warn(
                "You are passing a tensordict/tensorclass instance to a module that "
                "does not inherit from TensorDictModuleBase. This may lead to unexpected "
                "behaviours with functional calls."
            )
        if _is_stateless or params is not None:
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
                    self,
                    old_params,
                    make_stateless=_is_stateless,
                    return_old_tensordict=False,
                )
            return out
        else:
            try:
                return getattr(type(self), fun_name)(self, *args, **kwargs)
            except TypeError as err:
                pattern = r".*takes \d+ positional arguments but \d+ were given|got multiple values for argument"
                pattern = re.compile(pattern)
                if pattern.search(str(err)) and isinstance(args[-1], TensorDictBase):
                    # this is raised whenever the module is an nn.Module (not a TensorDictModuleBase)
                    raise TypeError(
                        "It seems you tried to provide the parameters as an argument to the module when the module was not stateless. "
                        "If this is the case, this error should vanish by providing the parameters using the ``module(..., params=params)`` "
                        "syntax."
                    ) from err
                else:
                    raise err

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

    new_fun.__signature__ = sig
    return new_fun


def _assign_params(
    module: nn.Module,
    params: TensorDict,
    make_stateless: bool,
    return_old_tensordict: bool,
) -> TensorDict | None:
    if params is not None:
        return _swap_state(module, params, make_stateless, return_old_tensordict)

    return None


def repopulate_module(model: nn.Module, tensordict: TensorDict) -> nn.Module:
    """Repopulates a module with its parameters, presented as a nested TensorDict."""
    _swap_state(model, tensordict, is_stateless=False)
    return model
