# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import functools
import warnings

from textwrap import indent
from typing import Any, Callable, List

import torch

from tensordict._nestedkey import NestedKey
from tensordict.base import is_tensor_collection, TensorDictBase
from tensordict.nn.common import dispatch
from torch import Tensor
from torch.utils._pytree import tree_map


class CudaGraphModule:
    """A cudagraph wrapper for PyTorch callables.

    ``CudaGraphModule`` is a wrapper class that provides a user-friendly interface to CUDA graphs for PyTorch callables.

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

        - Args and kwargs that are tensors or tensordict may change (provided that device and shape match), but non-tensor
          args and kwargs should not change. For instance, if the function receives a string input and the input is changed
          at any point, the module will silently execute the code with the string used during the capture of the cudagraph.
          The only supported keyword argument is `tensordict_out` in case the input is a tensordict.

    .. warning:: ``CudaGraphModule`` is not an :class:`~torch.nn.Module` by design, to discourage gathering parameters
        of the input module and passing them to an optimizer.

    Args:
        module (Callable): a function that receives tensors (or tensordict) as input and outputs a
            PyTreeable collection of tensors. If a tensordict is provided, the module can be run with keyword arguments
            too (see example below).
        warmup (int, optional): the number of warmup steps in case the module is compiled (compiled modules should be
            run a couple of times before being captured by cudagraphs). Defaults to ``2`` for all modules.
        in_keys (list of NestedKeys): the input keys, if the module takes a TensorDict as input.
            Defaults to ``module.in_keys`` if this value exists, otherwise ``None``.

            .. note:: If ``in_keys`` is provided but empty, the module is assumed to receive a tensordict as input.
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
        self._is_tensordict_module = (
            self.in_keys is not None
        )
        self._out_matches_in = None

        if self._is_tensordict_module:

            @dispatch(source=self.in_keys, dest=self.out_keys, auto_batch_size=False)
            def _call(tensordict: TensorDictBase, *args, tensordict_out: TensorDictBase | None=None, **kwargs: Any) -> Any:
                if self.counter < self._warmup:
                    if tensordict_out is not None:
                        kwargs["tensordict_out"] = tensordict_out
                    out = self.module(tensordict, *args, **kwargs)
                    if self._out_matches_in is None:
                        self._out_matches_in = out is tensordict
                    self.counter += self._has_cuda
                    return out
                elif self.counter == self._warmup:
                    if tensordict.device is None:
                        def check_device(x):
                            if isinstance(x, torch.Tensor):
                                if x.device.type != "cuda":
                                    raise ValueError(
                                        f"All tensors must be stored on CUDA. Got {x.device.type}."
                                    )
                        tensordict.apply(check_device, filter_empty=True)
                    elif tensordict.device.type != "cuda":
                        raise ValueError(
                            "The input tensordict device must be of the 'cuda' type."
                        )

                    def check_non_tensor(arg):
                        if isinstance(arg, torch.Tensor):
                            raise ValueError(
                                "All tensors must be passed in the tensordict, not as arg or kwarg."
                            )

                    tree_map(check_non_tensor, (args, kwargs))

                    self.graph = torch.cuda.CUDAGraph()
                    self._tensordict = tensordict.copy()
                    with torch.cuda.graph(self.graph):
                        if tensordict_out is not None:
                            kwargs["tensordict_out"] = tensordict_out
                        out = self.module(self._tensordict, *args, **kwargs)
                    self.graph.replay()

                    if not is_tensor_collection(out) and out is not None:
                        raise RuntimeError(
                            "The output of the function must be a tensordict, a tensorclass or None. Got "
                            f"type(out)={type(out)}."
                        )
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
                                print('checking', name, t0, t1)
                                if t0 is not t1:
                                    print('adding')
                                    self._selected_keys.append(name)
                            self._out.named_apply(check_tensor_id, tensordict, default=None, filter_empty=True)
                        return tensordict.update(
                            self._out, keys_to_update=self._selected_keys
                        )
                    if tensordict_out is not None:
                        return tensordict_out.update(out, clone=True)
                    print('else')
                    return out.clone() if self._out is not None else None
                else:
                    self._tensordict.update_(tensordict)
                    self.graph.replay()
                    if self._out_matches_in:
                        return tensordict.update(
                            self._out, keys_to_update=self._selected_keys
                        )
                    if tensordict_out is not None:
                        return tensordict_out.update(self._out, clone=True)
                    return self._out.clone() if self._out is not None else None

        else:

            def _call(*args: torch.Tensor, **kwargs: torch.Tensor):
                if self.counter < self._warmup:
                    out = self.module(*args, **kwargs)
                    self.counter += self._has_cuda
                    return out
                elif self.counter == self._warmup:
                    self.graph = torch.cuda.CUDAGraph()

                    def check_device_and_clone(x):
                        if isinstance(x, torch.Tensor):
                            if x.device.type != "cuda":
                                raise ValueError(
                                    f"All tensors must be stored on CUDA. Got {x.device.type}."
                                )
                            return x.clone()
                        return x

                    self._args, self._kwargs = tree_map(
                        check_device_and_clone, (args, kwargs)
                    )
                    print('record')
                    with torch.cuda.graph(self.graph):
                        out = self.module(*self._args, **self._kwargs)
                    self.graph.replay()
                    self._out = out
                    self.counter += 1
                    return tree_map(lambda x: x.clone(), out) if out is not None else out
                else:
                    tree_map(
                        lambda x, y: x.copy_(y),
                        (self._args, self._kwargs),
                        (args, kwargs),
                    )
                    print('replay')
                    self.graph.replay()
                    return tree_map(
                        lambda x: x.clone() if x is not None else x, self._out
                    )

        _call_func = functools.wraps(self.module)(_call)
        self._call_func = _call_func

    def __call__(self, *args, **kwargs):
        return self._call_func(*args, **kwargs)

    def __repr__(self):
        module = indent(f"module={self.module}", 4 * " ")
        warmup = warmup = {self._warmup}
        in_keys = indent(f"in_keys={self.in_keys}", 4 * " ")
        out_keys = indent(f"out_keys={self.out_keys}", 4 * " ")
        return f"{self.__class__.__name__}(\n{module}, \n{warmup}, \n{in_keys}, \n{out_keys}\n)"
