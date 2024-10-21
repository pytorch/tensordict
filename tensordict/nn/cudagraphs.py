# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import contextlib
import functools
import os
import warnings

from textwrap import indent
from typing import Any, Callable, List

import torch

from tensordict._nestedkey import NestedKey
from tensordict.base import is_tensor_collection, TensorDictBase
from tensordict.nn.common import dispatch
from tensordict.nn.functional_modules import (
    _exclude_td_from_pytree,
    PYTREE_REGISTERED_LAZY_TDS,
    PYTREE_REGISTERED_TDS,
)
from tensordict.utils import strtobool
from torch import Tensor

from torch.utils._pytree import SUPPORTED_NODES, tree_map

try:
    from torch.utils._pytree import tree_flatten, tree_leaves, tree_unflatten
except ImportError:
    from torch.utils._pytree import tree_flatten, tree_unflatten

    def tree_leaves(pytree):
        """Torch 2.0 compatible version of tree_leaves."""
        return tree_flatten(pytree)[0]


class CudaGraphModule:
    """A cudagraph wrapper for PyTorch callables.

    ``CudaGraphModule`` is a wrapper class that provides a user-friendly interface to CUDA graphs for PyTorch callables.

    .. warning::
        ``CudaGraphModule`` is a prototype feature and its API restrictions are likely to change in the future.

    This class provides a user-friendly interface to cudagraphs, allowing for a fast, CPU-overhead free execution of
    operations on GPU.
    It runs essential checks for the inputs to the function, and gives an nn.Module-like API to run

    .. warning::
      This module requires the wrapped function to meet a few requirements. It is the user responsibility
      to make sure that all of these are fullfilled.

        - The function cannot have dynamic control flow. For instance, the following code snippet will fail to be
          wrapped in `CudaGraphModule`:

            >>> def func(x):
            ...     if x.norm() > 1:
            ...         return x + 1
            ...     else:
            ...         return x - 1

          Fortunately, PyTorch offers solutions in most cases:

            >>> def func(x):
            ...     return torch.where(x.norm() > 1, x + 1, x - 1)

        - The function must execute a code that can be exactly re-run using the same buffers. This means that
          dynamic shapes (changing shape in the input or during the code execution) is not supported. In other words,
          the input must have a constant shape.
        - The output of the function must be detached. If a call to the optimizers is required, put it in the input
          function. For instance, the following function is a valid operator:

            >>> def func(x, y):
            ...     optim.zero_grad()
            ...     loss_val = loss_fn(x, y)
            ...     loss_val.backward()
            ...     optim.step()
            ...     return loss_val.detach()

        - The input should not be differntiable. If you need to use `nn.Parameters` (or differentiable tensors in general),
          just write a function that uses them as global values rather than passing them as input:

            >>> x = nn.Parameter(torch.randn(()))
            >>> optim = Adam([x], lr=1)
            >>> def func(): # right
            ...     optim.zero_grad()
            ...     (x+1).backward()
            ...     optim.step()
            >>> def func(x): # wrong
            ...     optim.zero_grad()
            ...     (x+1).backward()
            ...     optim.step()

        - Args and kwargs that are tensors or tensordict may change (provided that device and shape match), but non-tensor
          args and kwargs should not change. For instance, if the function receives a string input and the input is changed
          at any point, the module will silently execute the code with the string used during the capture of the cudagraph.
          The only supported keyword argument is `tensordict_out` in case the input is a tensordict.

        - If the module is a :class:`~tensordict.nn.TensorDictModuleBase` instance and the output id matches the input
          id, then this identity will be preserved during a call to ``CudaGraphModule``. In all other cases, the output
          will be cloned, irrespective of whether its elements match or do not match one of the inputs.

    .. warning::
        ``CudaGraphModule`` is not an :class:`~torch.nn.Module` by design, to discourage gathering parameters
        of the input module and passing them to an optimizer.

    Args:
        module (Callable): a function that receives tensors (or tensordict) as input and outputs a
            PyTreeable collection of tensors. If a tensordict is provided, the module can be run with keyword arguments
            too (see example below).
        warmup (int, optional): the number of warmup steps in case the module is compiled (compiled modules should be
            run a couple of times before being captured by cudagraphs). Defaults to ``2`` for all modules.
        in_keys (list of NestedKeys): the input keys, if the module takes a TensorDict as input.
            Defaults to ``module.in_keys`` if this value exists, otherwise ``None``.

            .. note::
                If ``in_keys`` is provided but empty, the module is assumed to receive a tensordict as input.
                This is sufficient to make ``CudaGraphModule`` aware that the function should be treated as a
                `TensorDictModule`, but keyword arguments will not be dispatched. See below for some examples.

        out_keys (list of NestedKeys): the output keys, if the module takes and outputs TensorDict as output.
            Defaults to ``module.out_keys`` if this value exists, otherwise ``None``.

    Examples:
        >>> # Wrap a simple function
        >>> def func(x):
        ...     return x + 1
        >>> func = CudaGraphModule(func)
        >>> x = torch.rand((), device='cuda')
        >>> out = func(x)
        >>> assert isinstance(out, torch.Tensor)
        >>> assert out == x+1
        >>> # Wrap a tensordict module
        >>> func = TensorDictModule(lambda x: x+1, in_keys=["x"], out_keys=["y"])
        >>> func = CudaGraphModule(func)
        >>> # This can be called either with a TensorDict or regular keyword arguments alike
        >>> y = func(x=x)
        >>> td = TensorDict(x=x)
        >>> td = func(td)


    """

    _REQUIRES_GRAD_ERROR = (
        "CudaGraphModule cannot be part of a graph (or leaves variables). If you need tensors to "
        "require gradients, please pass them as constant inputs to the module instead (e.g. "
        "`def func(a, b, c=c): ...` where c is the tensor requiring gradients)."
    )

    def __init__(
        self,
        module: Callable[[List[Tensor] | TensorDictBase], None],
        warmup: int = 2,
        in_keys: List[NestedKey] = None,
        out_keys: List[NestedKey] = None,
    ):
        self._has_cuda = torch.cuda.is_available()
        if not self._has_cuda:
            warnings.warn(
                "This module is instantiated on a machine without CUDA device available. "
                "We permit this usage, but calls to this instance will just pass through it and execute "
                "the provided function without additional performance gain."
            )
        self.module = module
        self.counter = 0
        if not isinstance(warmup, int) or warmup < 1:
            raise ValueError("warmup must be an integer greater than 0.")
        self._warmup = warmup
        if torch.cuda.is_available():
            self._warmup_stream = torch.cuda.Stream()
            self._warmup_stream_cm = torch.cuda.stream(self._warmup_stream)
        else:
            self._warmup_stream = None
            self._warmup_stream_cm = contextlib.nullcontext()

        if hasattr(module, "in_keys"):
            self.in_keys = module.in_keys
        else:
            self.in_keys = in_keys
        if hasattr(module, "out_keys"):
            self.out_keys = module.out_keys
        else:
            if out_keys is None and self.in_keys is not None:
                out_keys = []
            self.out_keys = out_keys
        self._is_tensordict_module = self.in_keys is not None
        self._out_matches_in = None
        for tdtype in PYTREE_REGISTERED_TDS + PYTREE_REGISTERED_LAZY_TDS:
            if tdtype in SUPPORTED_NODES:
                if not strtobool(os.environ.get("EXCLUDE_TD_FROM_PYTREE", "0")):
                    warnings.warn(
                        f"Tensordict is registered in PyTree. This is incompatible with {self.__class__.__name__}. "
                        f"Removing TDs from PyTree. To silence this warning, call tensordict.nn.functional_module._exclude_td_from_pytree().set() "
                        f"or set the environment variable `EXCLUDE_TD_FROM_PYTREE=1`. "
                        f"This operation is irreversible."
                    )
                _exclude_td_from_pytree().set()

        if self._is_tensordict_module:

            @dispatch(source=self.in_keys, dest=self.out_keys, auto_batch_size=False)
            def _call(
                tensordict: TensorDictBase,
                *args,
                tensordict_out: TensorDictBase | None = None,
                **kwargs: Any,
            ) -> Any:
                if self.counter >= self._warmup:
                    self._tensordict.update_(tensordict, non_blocking=True)
                    torch.cuda.synchronize()
                    self.graph.replay()
                    if self._out_matches_in:
                        result = tensordict.update(
                            self._out, keys_to_update=self._selected_keys
                        )
                    elif tensordict_out is not None:
                        result = tensordict_out.update(self._out, clone=True)
                    else:
                        result = self._out.clone() if self._out is not None else None
                    return result

                if not self._has_cuda or self.counter < self._warmup - 1:
                    # We must clone the data because providing non-contiguous data will fail later when we clone
                    tensordict.apply(self._clone, out=tensordict)
                    if self._has_cuda:
                        torch.cuda.synchronize()
                    with self._warmup_stream_cm:
                        if tensordict_out is not None:
                            kwargs["tensordict_out"] = tensordict_out
                        out = self.module(tensordict, *args, **kwargs)
                        if self._out_matches_in is None:
                            self._out_matches_in = out is tensordict
                    self.counter += self._has_cuda
                    if self._has_cuda:
                        torch.cuda.synchronize()
                    return out
                else:
                    if tensordict.device is None:
                        tensordict.apply(self._check_device_and_grad, filter_empty=True)
                    elif tensordict.device.type != "cuda":
                        raise ValueError(
                            "The input tensordict device must be of the 'cuda' type."
                        )

                    tree_map(self._check_non_tensor, (args, kwargs))
                    tensordict.apply(self._clone, out=tensordict)
                    self._tensordict = tensordict.copy()
                    if tensordict_out is not None:
                        td_out_save = tensordict_out.copy()
                        kwargs["tensordict_out"] = tensordict_out

                    torch.cuda.synchronize()
                    this_out = self.module(tensordict, *args, **kwargs)
                    torch.cuda.synchronize()

                    self.graph = torch.cuda.CUDAGraph()
                    if tensordict_out is not None:
                        kwargs["tensordict_out"] = td_out_save
                    with torch.cuda.graph(self.graph):
                        out = self.module(self._tensordict, *args, **kwargs)

                    if not is_tensor_collection(out) and out is not None:
                        raise RuntimeError(
                            "The output of the function must be a tensordict, a tensorclass or None. Got "
                            f"type(out)={type(out)}."
                        )
                    if is_tensor_collection(out):
                        out.lock_()
                    self._out = out
                    self.counter += 1
                    if self._out_matches_in:
                        # We need to know what keys to update in out
                        if self.out_keys:
                            self._selected_keys = self.out_keys
                        else:
                            # Gather keys that have changed during execution
                            self._selected_keys = []

                            def check_tensor_id(name, t0, t1):
                                if t0 is not t1:
                                    self._selected_keys.append(name)

                            self._out.named_apply(
                                check_tensor_id,
                                tensordict,
                                default=None,
                                filter_empty=True,
                            )
                    return this_out

        else:

            def _call(*args: torch.Tensor, **kwargs: torch.Tensor):
                if self.counter >= self._warmup:
                    srcs, dests = [], []
                    for arg_src, arg_dest in zip(
                        tree_leaves((args, kwargs)), self._flat_tree
                    ):
                        self._maybe_copy_onto_(arg_src, arg_dest, srcs, dests)
                    if dests:
                        torch._foreach_copy_(dests, srcs)
                    torch.cuda.synchronize()
                    self.graph.replay()
                    if self._return_unchanged:
                        result = self._out
                    else:
                        result = tree_unflatten(
                            [
                                out.clone() if hasattr(out, "clone") else out
                                for out in self._out
                            ],
                            self._out_struct,
                        )
                    return result

                if not self._has_cuda or self.counter < self._warmup - 1:
                    args, kwargs = tree_map(self._clone, (args, kwargs))
                    if self._has_cuda:
                        torch.cuda.synchronize()
                    with self._warmup_stream_cm:
                        out = self.module(*args, **kwargs)
                    if self._has_cuda:
                        torch.cuda.synchronize()
                    self.counter += self._has_cuda
                    return out
                else:
                    self._flat_tree, self._tree_spec = tree_flatten((args, kwargs))

                    self._flat_tree = tuple(
                        self._check_device_and_clone(arg) for arg in self._flat_tree
                    )
                    args, kwargs = self._args, self._kwargs = tree_unflatten(
                        self._flat_tree, self._tree_spec
                    )

                    torch.cuda.synchronize()
                    this_out = self.module(*args, **kwargs)
                    torch.cuda.synchronize()

                    self.graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(self.graph):
                        out = self.module(*self._args, **self._kwargs)
                    self._out, self._out_struct = tree_flatten(out)
                    self.counter += 1
                    # Check that there is not intersection between the indentity of inputs and outputs, otherwise warn
                    # user.
                    tree_leaves_input = tree_leaves((args, kwargs))
                    tree_leaves_output = tree_leaves(out)
                    if not isinstance(tree_leaves_output, tuple):
                        tree_leaves_output = (tree_leaves_output,)
                    for inp in tree_leaves_input:
                        if isinstance(inp, torch.Tensor) and inp in tree_leaves_output:
                            warnings.warn(
                                "An input tensor is also present in the output of the wrapped function. "
                                f"{type(self).__name__} will not copy tensors in place: the output will be cloned "
                                f"and the identity between input and output will not match anymore. "
                                f"Make sure you don't rely on input-output identity further in the code."
                            )
                    if not self._out:
                        self._return_unchanged = True
                    else:
                        self._out = [
                            out.lock_() if is_tensor_collection(out) else out
                            for out in self._out
                        ]
                        self._return_unchanged = False
                    return this_out

        _call_func = functools.wraps(self.module)(_call)
        self._call_func = _call_func

    @staticmethod
    def _maybe_copy_onto_(src, dest, srcs, dests):
        if isinstance(src, torch.Tensor):
            srcs.append(src)
            dests.append(dest)
            return
        if is_tensor_collection(src):
            dest.copy_(src)
            return
        isdiff = False
        try:
            isdiff = src != dest
        except Exception as err:
            raise RuntimeError(
                "Couldn't assess input value. Make sure your function only takes tensor inputs or that "
                "the input value can be easily checked and is constant. For a better efficiency, avoid "
                "passing non-tensor inputs to your function."
            ) from err
        if isdiff:
            raise ValueError("Varying inputs must be torch.Tensor subclasses.")

    @classmethod
    def _check_device_and_clone(cls, x):
        if isinstance(x, torch.Tensor) or is_tensor_collection(x):
            if x.requires_grad:
                raise RuntimeError(cls._REQUIRES_GRAD_ERROR)
            if x.device is None:
                # Check device of leaves of tensordict
                x.apply(cls._check_device_and_grad, filter_empty=True)

            elif x.device.type != "cuda":
                raise ValueError(
                    f"All tensors must be stored on CUDA. Got {x.device.type}."
                )

            return x.clone()
        return x

    @classmethod
    def _clone(cls, x):
        if isinstance(x, torch.Tensor) or is_tensor_collection(x):
            if x.requires_grad:
                raise RuntimeError(cls._REQUIRES_GRAD_ERROR)
            return x.clone()
        return x

    @classmethod
    def _check_device_and_grad(cls, x):
        if isinstance(x, torch.Tensor):
            if x.device.type != "cuda":
                raise ValueError(
                    f"All tensors must be stored on CUDA. Got {x.device.type}."
                )
            if x.requires_grad:
                raise RuntimeError(cls._REQUIRES_GRAD_ERROR)

    @staticmethod
    def _check_non_tensor(arg):
        if isinstance(arg, torch.Tensor):
            raise ValueError(
                "All tensors must be passed in the tensordict, not as arg or kwarg."
            )

    def __call__(self, *args, **kwargs):
        return self._call_func(*args, **kwargs)

    def __repr__(self):
        module = indent(f"module={self.module}", 4 * " ")
        warmup = warmup = {self._warmup}
        in_keys = indent(f"in_keys={self.in_keys}", 4 * " ")
        out_keys = indent(f"out_keys={self.out_keys}", 4 * " ")
        return f"{self.__class__.__name__}(\n{module}, \n{warmup}, \n{in_keys}, \n{out_keys}\n)"
