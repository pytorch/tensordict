# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import List

import torch
from torch.utils._pytree import tree_map

from tensordict._nestedkey import NestedKey
from tensordict.nn.common import dispatch


class CudaGraphCompiledModule:
    def __init__(self, compiled_module, warmup: int=2, in_keys: List[NestedKey]=None, out_keys: List[NestedKey]=None):
        self.compiled_module = compiled_module
        self.counter = 0
        self.warmup = warmup

        if hasattr(compiled_module, "in_keys"):
            self.in_keys = compiled_module.in_keys
        else:
            self.in_keys = in_keys
        if hasattr(compiled_module, "out_keys"):
            self.out_keys = compiled_module.out_keys
        else:
            self.out_keys = out_keys
        self._is_tensordict_module = self.in_keys is not None and self.out_keys is not None

    @property
    def __call__(self):
        if self._is_tensordict_module:
            return self._call_tdmodule
        else:
            return self._call_regular

    @dispatch(auto_batch_size=False)
    def _call_tdmodule(self, tensordict, *args, **kwargs):
        if self.counter < self.warmup:
            out = self.compiled_module(tensordict, *args, **kwargs)
            self.counter += 1
            return out
        elif self.counter == self.warmup:
            self.graph = torch.cuda.CUDAGraph()
            self._tensordict = tensordict
            with torch.cuda.graph(self.graph):
                out = self.compiled_module(tensordict, *args, **kwargs)
            self._out = out
            self.counter += 1
            return out
        else:
            self._tensordict.update_(tensordict)
            self.graph.replay()
            return self._out.clone() if self._out is not None else None

    def _call_regular(self, *args, **kwargs):
        if self.counter < self.warmup:
            out = self.compiled_module(*args, **kwargs)
            self.counter += 1
            return out
        elif self.counter == self.warmup:
            self.graph = torch.cuda.CUDAGraph()
            self._args, self._kwargs = tree_map(lambda x: x.clone(), (args, kwargs))
            with torch.cuda.graph(self.graph):
                out = self.compiled_module(*self._args, **self._kwargs)
            self._out = out
            self.counter += 1
            return out
        else:
            tree_map(lambda x, y: x.copy_(y), (self._args, self._kwargs), (args, kwargs))
            self.graph.replay()
            return tree_map(lambda x: x.clone() if x is not None else x, self._out)
