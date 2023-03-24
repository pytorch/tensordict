# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterable

_has_functorch = False
try:
    import functorch

    _has_functorch = True
except ImportError:
    print(
        "failed to import functorch. TensorDict's features that do not require "
        "functional programming should work, but functionality and performance "
        "may be affected. Consider installing functorch and/or upgrating pytorch."
    )
    FUNCTORCH_ERROR = "functorch not installed. Consider installing functorch to use this functionality."


from tensordict.nn.common import dispatch, TensorDictModule
from tensordict.tensordict import LazyStackedTensorDict, TensorDictBase
from tensordict.utils import _normalize_key, NestedKey
from torch import nn

__all__ = ["TensorDictSequential"]


class TensorDictSequential(TensorDictModule):
    """A sequence of TensorDictModules.

    By default, :class:`TensorDictSequential` subclasses are always functional,
    meaning that they support the ``td_module(input, params=params)`` function
    call signature.

    Similarly to :obj:`nn.Sequence` which passes a tensor through a chain of mappings that read and write a single tensor
    each, this module will read and write over a tensordict by querying each of the input modules.
    When calling a :obj:`TensorDictSequencial` instance with a functional module, it is expected that the parameter lists (and
    buffers) will be concatenated in a single list.

    Args:
         modules (iterable of TensorDictModules): ordered sequence of TensorDictModule instances to be run sequentially.
         partial_tolerant (bool, optional): if True, the input tensordict can miss some of the input keys.
            If so, the only module that will be executed are those who can be executed given the keys that
            are present.
            Also, if the input tensordict is a lazy stack of tensordicts AND if partial_tolerant is :obj:`True` AND if the
            stack does not have the required keys, then TensorDictSequential will scan through the sub-tensordicts
            looking for those that have the required keys, if any.

    TensorDictSequence supports functional, modular and vmap coding:
    Examples:
        >>> import torch
        >>> from tensordict import TensorDict
        >>> from tensordict.nn import (
        ...     ProbabilisticTensorDictModule,
        ...     ProbabilisticTensorDictSequential,
        ...     TensorDictModule,
        ...     TensorDictSequential,
        ... )
        >>> from tensordict.nn.distributions import NormalParamExtractor
        >>> from tensordict.nn.functional_modules import make_functional
        >>> from torch.distributions import Normal
        >>> td = TensorDict({"input": torch.randn(3, 4)}, [3,])
        >>> net1 = torch.nn.Linear(4, 8)
        >>> module1 = TensorDictModule(net1, in_keys=["input"], out_keys=["params"])
        >>> normal_params = TensorDictModule(
        ...      NormalParamExtractor(), in_keys=["params"], out_keys=["loc", "scale"]
        ...  )
        >>> td_module1 = ProbabilisticTensorDictSequential(
        ...     module1,
        ...     normal_params,
        ...     ProbabilisticTensorDictModule(
        ...         in_keys=["loc", "scale"],
        ...         out_keys=["hidden"],
        ...         distribution_class=Normal,
        ...         return_log_prob=True,
        ...     )
        ... )
        >>> module2 = torch.nn.Linear(4, 8)
        >>> td_module2 = TensorDictModule(
        ...    module=module2, in_keys=["hidden"], out_keys=["output"]
        ... )
        >>> td_module = TensorDictSequential(td_module1, td_module2)
        >>> params = make_functional(td_module)
        >>> _ = td_module(td, params=params)
        >>> print(td)
        TensorDict(
            fields={
                hidden: Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.float32, is_shared=False),
                input: Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.float32, is_shared=False),
                loc: Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.float32, is_shared=False),
                output: Tensor(shape=torch.Size([3, 8]), device=cpu, dtype=torch.float32, is_shared=False),
                params: Tensor(shape=torch.Size([3, 8]), device=cpu, dtype=torch.float32, is_shared=False),
                sample_log_prob: Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.float32, is_shared=False),
                scale: Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.float32, is_shared=False)},
            batch_size=torch.Size([3]),
            device=None,
            is_shared=False)

    In the vmap case:
        >>> from torch import vmap
        >>> params = params.expand(4)
        >>> td_vmap = vmap(td_module, (None, 0))(td, params)
        >>> print(td_vmap)
        TensorDict(
            fields={
                hidden: Tensor(shape=torch.Size([4, 3, 4]), device=cpu, dtype=torch.float32, is_shared=False),
                input: Tensor(shape=torch.Size([4, 3, 4]), device=cpu, dtype=torch.float32, is_shared=False),
                loc: Tensor(shape=torch.Size([4, 3, 4]), device=cpu, dtype=torch.float32, is_shared=False),
                output: Tensor(shape=torch.Size([4, 3, 8]), device=cpu, dtype=torch.float32, is_shared=False),
                params: Tensor(shape=torch.Size([4, 3, 8]), device=cpu, dtype=torch.float32, is_shared=False),
                sample_log_prob: Tensor(shape=torch.Size([4, 3, 4]), device=cpu, dtype=torch.float32, is_shared=False),
                scale: Tensor(shape=torch.Size([4, 3, 4]), device=cpu, dtype=torch.float32, is_shared=False)},
            batch_size=torch.Size([4, 3]),
            device=None,
            is_shared=False)

    """

    module: nn.ModuleList

    def __init__(
        self,
        *modules: TensorDictModule,
        partial_tolerant: bool = False,
    ) -> None:
        in_keys, out_keys = self._compute_in_and_out_keys(modules)

        super().__init__(
            module=nn.ModuleList(list(modules)), in_keys=in_keys, out_keys=out_keys
        )

        self.partial_tolerant = partial_tolerant

    def _compute_in_and_out_keys(
        self, modules: list[TensorDictModule]
    ) -> tuple[list[NestedKey], list[NestedKey]]:
        in_keys = []
        out_keys = []
        for module in modules:
            # we sometimes use in_keys to select keys of a tensordict that are
            # necessary to run a TensorDictModule. If a key is an intermediary in
            # the chain, there is no reason why it should belong to the input
            # TensorDict.
            for in_key in module.in_keys:
                if in_key not in (out_keys + in_keys):
                    in_keys.append(in_key)
            out_keys += module.out_keys

        out_keys = [
            out_key
            for i, out_key in enumerate(out_keys)
            if out_key not in out_keys[i + 1 :]
        ]
        return in_keys, out_keys

    @staticmethod
    def _find_functional_module(module: TensorDictModule) -> nn.Module:
        if not _has_functorch:
            raise ImportError(FUNCTORCH_ERROR)
        fmodule = module
        while not isinstance(
            fmodule, (functorch.FunctionalModule, functorch.FunctionalModuleWithBuffers)
        ):
            try:
                fmodule = fmodule.module
            except AttributeError:
                raise AttributeError(
                    f"couldn't find a functional module in module of type {type(module)}"
                )
        return fmodule

    def select_subsequence(
        self,
        in_keys: Iterable[NestedKey] | None = None,
        out_keys: Iterable[NestedKey] | None = None,
    ) -> TensorDictSequential:
        """Returns a new TensorDictSequential with only the modules that are necessary to compute the given output keys with the given input keys.

        Args:
            in_keys: input keys of the subsequence we want to select
            out_keys: output keys of the subsequence we want to select

        Returns:
            A new TensorDictSequential with only the modules that are necessary acording to the given input and output keys.
        """
        if in_keys is None:
            in_keys = deepcopy(self.in_keys)
        else:
            in_keys = [_normalize_key(key) for key in in_keys]
        if out_keys is None:
            out_keys = deepcopy(self.out_keys)
        else:
            out_keys = [_normalize_key(key) for key in out_keys]
        id_to_keep = set(range(len(self.module)))
        for i, module in enumerate(self.module):
            if all(key in in_keys for key in module.in_keys):
                in_keys.extend(module.out_keys)
            else:
                id_to_keep.remove(i)
        for i, module in reversed(list(enumerate(self.module))):
            if i in id_to_keep:
                if any(key in out_keys for key in module.out_keys):
                    out_keys.extend(module.in_keys)
                else:
                    id_to_keep.remove(i)
        id_to_keep = sorted(id_to_keep)

        modules = [self.module[i] for i in id_to_keep]

        if modules == []:
            raise ValueError(
                "No modules left after selection. Make sure that in_keys and out_keys are coherent."
            )

        return self.__class__(*modules)

    def _run_module(
        self,
        module: TensorDictModule,
        tensordict: TensorDictBase,
        **kwargs: Any,
    ) -> Any:
        tensordict_keys = set(tensordict.keys(include_nested=True))
        if not self.partial_tolerant or all(
            key in tensordict_keys for key in module.in_keys
        ):
            tensordict = module(tensordict, **kwargs)
        elif self.partial_tolerant and isinstance(tensordict, LazyStackedTensorDict):
            for sub_td in tensordict.tensordicts:
                tensordict_keys = set(sub_td.keys(include_nested=True))
                if all(key in tensordict_keys for key in module.in_keys):
                    module(sub_td, **kwargs)
            tensordict._update_valid_keys()
        return tensordict

    @dispatch
    def forward(
        self,
        tensordict: TensorDictBase,
        tensordict_out: TensorDictBase | None = None,
        **kwargs: Any,
    ) -> TensorDictBase:
        if not len(kwargs):
            for module in self.module:
                tensordict = self._run_module(module, tensordict, **kwargs)
        else:
            raise RuntimeError(
                "TensorDictSequential does not support keyword arguments other than 'tensordict_out', 'in_keys' and 'out_keys'"
            )
        if tensordict_out is not None:
            tensordict_out.update(tensordict, inplace=True)
            return tensordict_out
        return tensordict

    def __len__(self) -> int:
        return len(self.module)

    def __getitem__(self, index: int | slice) -> TensorDictModule:
        if isinstance(index, int):
            return self.module.__getitem__(index)
        else:
            return self.__class__(*self.module.__getitem__(index))

    def __setitem__(self, index: int, tensordict_module: TensorDictModule) -> None:
        return self.module.__setitem__(idx=index, module=tensordict_module)

    def __delitem__(self, index: int | slice) -> None:
        self.module.__delitem__(idx=index)
