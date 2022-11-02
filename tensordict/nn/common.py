# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import warnings
from textwrap import indent
from typing import (
    Any,
    Iterable,
    List,
    Optional,
    Sequence,
    Union,
)

try:
    import functorch

    _has_functorch = True
except ImportError:
    _has_functorch = False

import torch
from functorch import FunctionalModule, FunctionalModuleWithBuffers
from torch import nn, Tensor

from tensordict.tensordict import TensorDictBase

__all__ = [
    "TensorDictModule",
    "TensorDictModuleWrapper",
]


def _check_all_str(list_of_str):
    if isinstance(list_of_str, str):
        raise RuntimeError(
            f"Expected a list of strings but got a string: {list_of_str}"
        )
    if any(not isinstance(key, str) for key in list_of_str):
        raise TypeError(f"Expected a list of strings but got: {list_of_str}")


class TensorDictModule(nn.Module):
    """A TensorDictModule, is a python wrapper around a :obj:`nn.Module` that reads and writes to a TensorDict.

    Args:
        module (nn.Module): a nn.Module used to map the input to the output parameter space. Can be a functional
            module (FunctionalModule or FunctionalModuleWithBuffers), in which case the :obj:`forward` method will expect
            the params (and possibly) buffers keyword arguments.
        in_keys (iterable of str): keys to be read from input tensordict and passed to the module. If it
            contains more than one element, the values will be passed in the order given by the in_keys iterable.
        out_keys (iterable of str): keys to be written to the input tensordict. The length of out_keys must match the
            number of tensors returned by the embedded module. Using "_" as a key avoid writing tensor to output.

    Embedding a neural network in a TensorDictModule only requires to specify the input
    and output keys. TensorDictModule support functional and regular :obj:`nn.Module`
    objects. In the functional case, the 'params' (and 'buffers') keyword argument must
    be specified:

    Examples:
        >>> import torch, functorch
        >>> from tensordict import TensorDict
        >>> from tensordict.nn import TensorDictModule
        >>> td = TensorDict({"input": torch.randn(3, 4), "hidden": torch.randn(3, 8)}, [3,])
        >>> module = torch.nn.GRUCell(4, 8)
        >>> fmodule, params, buffers = functorch.make_functional_with_buffers(module)
        >>> td_fmodule = TensorDictModule(
        ...    module=fmodule, in_keys=["input", "hidden"], out_keys=["output"]
        ... )
        >>> td_functional = td_fmodule(td.clone(), params=params, buffers=buffers)
        >>> print(td_functional)
        TensorDict(
            fields={input: Tensor(torch.Size([3, 4]), dtype=torch.float32),
                hidden: Tensor(torch.Size([3, 8]), dtype=torch.float32),
                output: Tensor(torch.Size([3, 8]), dtype=torch.float32)},
            shared=False,
            batch_size=torch.Size([3]),
            device=cpu)

    In the stateful case:
        >>> td_module = TensorDictModule(
        ...    module=module, in_keys=["input", "hidden"], out_keys=["output"]
        ... )
        >>> td_stateful = td_module(td.clone())
        >>> print(td_stateful)
        TensorDict(
            fields={input: Tensor(torch.Size([3, 4]), dtype=torch.float32),
                hidden: Tensor(torch.Size([3, 8]), dtype=torch.float32),
                output: Tensor(torch.Size([3, 8]), dtype=torch.float32)},
            shared=False,
            batch_size=torch.Size([3]),
            device=cpu)

    One can use a vmap operator to call the functional module. In this case the tensordict is expanded to match the
    batch size (i.e. the tensordict isn't modified in-place anymore):
        >>> # Model ensemble using vmap
        >>> params_repeat = tuple(param.expand(4, *param.shape).contiguous().normal_() for param in params)
        >>> buffers_repeat = tuple(param.expand(4, *param.shape).contiguous().normal_() for param in buffers)
        >>> td_vmap = td_fmodule(td.clone(), params=params_repeat, buffers=buffers_repeat, vmap=True)
        >>> print(td_vmap)
        TensorDict(
            fields={input: Tensor(torch.Size([4, 3, 4]), dtype=torch.float32),
                hidden: Tensor(torch.Size([4, 3, 8]), dtype=torch.float32),
                output: Tensor(torch.Size([4, 3, 8]), dtype=torch.float32)},
            shared=False,
            batch_size=torch.Size([4, 3]),
            device=cpu)

    """

    def __init__(
        self,
        module: Union[
            FunctionalModule, FunctionalModuleWithBuffers, TensorDictModule, nn.Module
        ],
        in_keys: Iterable[str],
        out_keys: Iterable[str],
    ):

        super().__init__()

        if not out_keys:
            raise RuntimeError(f"out_keys were not passed to {self.__class__.__name__}")
        if not in_keys:
            raise RuntimeError(f"in_keys were not passed to {self.__class__.__name__}")
        self.out_keys = out_keys
        _check_all_str(self.out_keys)
        self.in_keys = in_keys
        _check_all_str(self.in_keys)

        if "_" in in_keys:
            warnings.warn(
                'key "_" is for ignoring output, it should not be used in input keys'
            )

        self.module = module

    @property
    def is_functional(self):
        return isinstance(
            self.module,
            (functorch.FunctionalModule, functorch.FunctionalModuleWithBuffers),
        )

    def _write_to_tensordict(
        self,
        tensordict: TensorDictBase,
        tensors: List,
        tensordict_out: Optional[TensorDictBase] = None,
        out_keys: Optional[Iterable[str]] = None,
    ) -> TensorDictBase:

        if out_keys is None:
            out_keys = self.out_keys
        if tensordict_out is None:
            tensordict_out = tensordict
        for _out_key, _tensor in zip(out_keys, tensors):
            if _out_key != "_":
                tensordict_out.set(_out_key, _tensor)
        return tensordict_out

    def _call_module(
        self,
        tensors: Sequence[Tensor],
        **kwargs,
    ) -> Union[Tensor, Sequence[Tensor]]:
        out = self.module(*tensors, **kwargs)
        return out

    def forward(
        self,
        tensordict: TensorDictBase,
        tensordict_out: Optional[TensorDictBase] = None,
        **kwargs,
    ) -> TensorDictBase:
        tensors = tuple(tensordict.get(in_key, None) for in_key in self.in_keys)
        tensors = self._call_module(tensors, **kwargs)
        if not isinstance(tensors, tuple):
            tensors = (tensors,)
        tensordict_out = self._write_to_tensordict(
            tensordict,
            tensors,
            tensordict_out,
        )
        return tensordict_out

    @property
    def device(self):
        for p in self.parameters():
            return p.device
        return torch.device("cpu")

    def __repr__(self) -> str:
        fields = indent(
            f"module={self.module}, \n"
            f"device={self.device}, \n"
            f"in_keys={self.in_keys}, \n"
            f"out_keys={self.out_keys}",
            4 * " ",
        )

        return f"{self.__class__.__name__}(\n{fields})"

    @property
    def num_params(self):
        if isinstance(
            self.module,
            (functorch.FunctionalModule, functorch.FunctionalModuleWithBuffers),
        ):
            return len(self.module.param_names)
        else:
            return 0

    @property
    def num_buffers(self):
        if isinstance(self.module, (functorch.FunctionalModuleWithBuffers,)):
            return len(self.module.buffer_names)
        else:
            return 0


class TensorDictModuleWrapper(nn.Module):
    """Wrapper calss for TensorDictModule objects.

    Once created, a TensorDictModuleWrapper will behave exactly as the TensorDictModule it contains except for the methods that are
    overwritten.

    Args:
        td_module (TensorDictModule): operator to be wrapped.

    """

    def __init__(self, td_module: TensorDictModule):
        super().__init__()
        self.td_module = td_module
        if len(self.td_module._forward_hooks):
            for pre_hook in self.td_module._forward_hooks:
                self.register_forward_hook(self.td_module._forward_hooks[pre_hook])

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            if name not in self.__dict__ and not name.startswith("__"):
                return getattr(self._modules["td_module"], name)
            else:
                raise AttributeError(
                    f"attribute {name} not recognised in {type(self).__name__}"
                )

    def forward(self, *args, **kwargs):
        return self.td_module.forward(*args, **kwargs)
