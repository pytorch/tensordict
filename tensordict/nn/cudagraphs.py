# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import warnings
from typing import Callable, List

import torch

from tensordict._nestedkey import NestedKey
from tensordict.base import is_tensor_collection, TensorDictBase
from tensordict.nn.common import dispatch
from torch import Tensor
from torch.utils._pytree import tree_map


class CudaGraphModule:
    """A cudagraph wrapper for PyTorch callables.

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
            self.in_keys is not None and self.out_keys is not None
        )

    @property
    def __call__(self):
        if self._is_tensordict_module:
            return self._call_tdmodule
        else:
            return self._call_regular

    @dispatch(auto_batch_size=False)
    def _call_tdmodule(self, tensordict: TensorDictBase, *args, **kwargs):
        if self.counter < self._warmup:
            out = self.module(tensordict, *args, **kwargs)
            self.counter += self._has_cuda
            return out
        elif self.counter == self._warmup:
            if tensordict.device.type != "cuda":
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
                out = self.module(self._tensordict, *args, **kwargs)
            if not is_tensor_collection(out) and out is not None:
                raise RuntimeError(
                    "The output of the function must be a tensordict, a tensorclass or None. Got "
                    f"type(out)={type(out)}."
                )
            self._out_matches_in = out is tensordict
            if self._out_matches_in:
                if self.out_keys:
                    self._selected_keys = self.out_keys
                else:
                    # Gather keys that have changed during execution
                    self._selected_keys = []

                    def check_tensor_id(name, t0, t1):
                        if t0 is not t1:
                            self._selected_keys.append(name)

                    out.apply(check_tensor_id, self._tensordict, default=None)
            self._out = out
            self.counter += 1
            return out.clone()
        else:
            self._tensordict.update_(tensordict)
            self.graph.replay()
            if self._out_matches_in:
                return tensordict.update(self._out, keys_to_updte=self._selected_keys)
            return self._out.clone() if self._out is not None else None

    def _call_regular(self, *args, **kwargs):
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

            self._args, self._kwargs = tree_map(check_device_and_clone, (args, kwargs))
            with torch.cuda.graph(self.graph):
                out = self.module(*self._args, **self._kwargs)
            self._out = out
            self.counter += 1
            return out.clone()
        else:
            tree_map(
                lambda x, y: x.copy_(y), (self._args, self._kwargs), (args, kwargs)
            )
            self.graph.replay()
            return tree_map(lambda x: x.clone() if x is not None else x, self._out)
