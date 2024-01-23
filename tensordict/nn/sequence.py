# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import logging

from copy import deepcopy
from typing import Any, Iterable

from tensordict.nn.utils import set_skip_existing

_has_functorch = False
try:
    import functorch

    _has_functorch = True
except ImportError:
    logging.info(
        "failed to import functorch. TensorDict's features that do not require "
        "functional programming should work, but functionality and performance "
        "may be affected. Consider installing functorch and/or upgrating pytorch."
    )
    FUNCTORCH_ERROR = "functorch not installed. Consider installing functorch to use this functionality."


from tensordict._tensordict import unravel_key_list
from tensordict.nn.common import dispatch, TensorDictModule
from tensordict.tensordict import LazyStackedTensorDict, TensorDictBase
from tensordict.utils import NestedKey
from torch import nn

__all__ = ["TensorDictSequential"]


class TensorDictSequential(TensorDictModule):
    """A sequence of TensorDictModules.

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

    Examples:
        >>> import torch
        >>> from tensordict import TensorDict
        >>> from tensordict.nn import TensorDictModule, TensorDictSequential
        >>> torch.manual_seed(0)
        >>> module = TensorDictSequential(
        ...     TensorDictModule(lambda x: x+1, in_keys=["x"], out_keys=["x+1"]),
        ...     TensorDictModule(nn.Linear(3, 4), in_keys=["x+1"], out_keys=["w*(x+1)+b"]),
        ... )
        >>> # with tensordict input
        >>> print(module(TensorDict({"x": torch.zeros(3)}, [])))
        TensorDict(
            fields={
                w*(x+1)+b: Tensor(shape=torch.Size([4]), device=cpu, dtype=torch.float32, is_shared=False),
                x+1: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False),
                x: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False)},
            batch_size=torch.Size([]),
            device=None,
            is_shared=False)
        >>> # with tensor input: returns all the output keys in the order of the modules, ie "x+1" and "w*(x+1)+b"
        >>> module(x=torch.zeros(3))
        (tensor([1., 1., 1.]), tensor([-0.7214, -0.8748,  0.1571, -0.1138], grad_fn=<AddBackward0>))
        >>> module(torch.zeros(3))
        (tensor([1., 1., 1.]), tensor([-0.7214, -0.8748,  0.1571, -0.1138], grad_fn=<AddBackward0>))

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
        >>> params = TensorDict.from_module(td_module)
        >>> with params.to_module(td_module):
        ...     _ = td_module(td)
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
        >>> def func(td, params):
        ...     with params.to_module(td_module):
        ...         return td_module(td)
        >>> td_vmap = vmap(func, (None, 0))(td, params)
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
            in_keys: input keys of the subsequence we want to select.
                All the keys absent from ``in_keys`` will be considered as
                non-relevant, and modules that *just* take these keys as inputs
                will be discarded.
                The resulting sequential module will follow the pattern "all
                the modules which output will be affected by a different value
                for any in <in_keys>".
                If none is provided, the module's ``in_keys`` are assumed.
            out_keys: output keys of the subsequence we want to select.
                Only the modules that are necessary to get the ``out_keys``
                will be found in the resulting sequence.
                The resulting sequential module will follow the pattern "all
                the modules that condition the value or <out_keys> entries."
                If none is provided, the module's ``out_keys`` are assumed.

        Returns:
            A new TensorDictSequential with only the modules that are necessary acording to the given input and output keys.

        Examples:
            >>> from tensordict.nn import TensorDictSequential as Seq, TensorDictModule as Mod
            >>> idn = lambda x: x
            >>> module = Seq(
            ...     Mod(idn, in_keys=["a"], out_keys=["b"]),
            ...     Mod(idn, in_keys=["b"], out_keys=["c"]),
            ...     Mod(idn, in_keys=["c"], out_keys=["d"]),
            ...     Mod(idn, in_keys=["a"], out_keys=["e"]),
            ... )
            >>> # select all modules whose output depend on "a"
            >>> module.select_subsequence(in_keys=["a"])
            TensorDictSequential(
                module=ModuleList(
                  (0): TensorDictModule(
                      module=<function <lambda> at 0x126ed1ca0>,
                      device=cpu,
                      in_keys=['a'],
                      out_keys=['b'])
                  (1): TensorDictModule(
                      module=<function <lambda> at 0x126ed1ca0>,
                      device=cpu,
                      in_keys=['b'],
                      out_keys=['c'])
                  (2): TensorDictModule(
                      module=<function <lambda> at 0x126ed1ca0>,
                      device=cpu,
                      in_keys=['c'],
                      out_keys=['d'])
                  (3): TensorDictModule(
                      module=<function <lambda> at 0x126ed1ca0>,
                      device=cpu,
                      in_keys=['a'],
                      out_keys=['e'])
                ),
                device=cpu,
                in_keys=['a'],
                out_keys=['b', 'c', 'd', 'e'])
            >>> # select all modules whose output depend on "c"
            >>> module.select_subsequence(in_keys=["c"])
            TensorDictSequential(
                module=ModuleList(
                  (0): TensorDictModule(
                      module=<function <lambda> at 0x126ed1ca0>,
                      device=cpu,
                      in_keys=['c'],
                      out_keys=['d'])
                ),
                device=cpu,
                in_keys=['c'],
                out_keys=['d'])
            >>> # select all modules that affect the value of "c"
            >>> module.select_subsequence(out_keys=["c"])
            TensorDictSequential(
                module=ModuleList(
                  (0): TensorDictModule(
                      module=<function <lambda> at 0x126ed1ca0>,
                      device=cpu,
                      in_keys=['a'],
                      out_keys=['b'])
                  (1): TensorDictModule(
                      module=<function <lambda> at 0x126ed1ca0>,
                      device=cpu,
                      in_keys=['b'],
                      out_keys=['c'])
                ),
                device=cpu,
                in_keys=['a'],
                out_keys=['b', 'c'])
            >>> # select all modules that affect the value of "e"
            >>> module.select_subsequence(out_keys=["e"])
            TensorDictSequential(
                module=ModuleList(
                  (0): TensorDictModule(
                      module=<function <lambda> at 0x126ed1ca0>,
                      device=cpu,
                      in_keys=['a'],
                      out_keys=['e'])
                ),
                device=cpu,
                in_keys=['a'],
                out_keys=['e'])

        This method propagates to nested sequential:

            >>> module = Seq(
            ...     Seq(
            ...         Mod(idn, in_keys=["a"], out_keys=["b"]),
            ...         Mod(idn, in_keys=["b"], out_keys=["c"]),
            ...     ),
            ...     Seq(
            ...         Mod(idn, in_keys=["b"], out_keys=["d"]),
            ...         Mod(idn, in_keys=["d"], out_keys=["e"]),
            ...     ),
            ... )
            >>> # select submodules whose output will be affected by a change in "b" or "d" AND which output is "e"
            >>> module.select_subsequence(in_keys=["b", "d"], out_keys=["e"])
            TensorDictSequential(
                module=ModuleList(
                  (0): TensorDictSequential(
                      module=ModuleList(
                        (0): TensorDictModule(
                            module=<function <lambda> at 0x129efae50>,
                            device=cpu,
                            in_keys=['b'],
                            out_keys=['d'])
                        (1): TensorDictModule(
                            module=<function <lambda> at 0x129efae50>,
                            device=cpu,
                            in_keys=['d'],
                            out_keys=['e'])
                      ),
                      device=cpu,
                      in_keys=['b'],
                      out_keys=['d', 'e'])
                ),
                device=cpu,
                in_keys=['b'],
                out_keys=['d', 'e'])

        """
        if in_keys is None:
            in_keys = deepcopy(self.in_keys)
        in_keys = unravel_key_list(in_keys)
        if out_keys is None:
            out_keys = deepcopy(self.out_keys)
        out_keys = unravel_key_list(out_keys)

        module_list = list(self.module)
        id_to_keep = set(range(len(module_list)))
        for i, module in enumerate(module_list):
            if (
                type(module) is TensorDictSequential
            ):  # no isinstance because we don't want to mess up subclasses
                try:
                    module = module_list[i] = module.select_subsequence(in_keys=in_keys)
                except ValueError:
                    # then the module can be removed
                    id_to_keep.remove(i)
                    continue

            if all(key in in_keys for key in module.in_keys):
                in_keys.extend(module.out_keys)
            else:
                id_to_keep.remove(i)
        for i, module in reversed(list(enumerate(module_list))):
            if i in id_to_keep:
                if any(key in out_keys for key in module.out_keys):
                    if (
                        type(module) is TensorDictSequential
                    ):  # no isinstance because we don't want to mess up subclasses
                        module = module_list[i] = module.select_subsequence(
                            out_keys=out_keys
                        )
                    out_keys.extend(module.in_keys)
                else:
                    id_to_keep.remove(i)
        id_to_keep = sorted(id_to_keep)

        modules = [module_list[i] for i in id_to_keep]

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
        if not self.partial_tolerant or all(
            key in tensordict.keys(include_nested=True) for key in module.in_keys
        ):
            tensordict = module(tensordict, **kwargs)
        elif self.partial_tolerant and isinstance(tensordict, LazyStackedTensorDict):
            for sub_td in tensordict.tensordicts:
                if all(
                    key in sub_td.keys(include_nested=True) for key in module.in_keys
                ):
                    module(sub_td, **kwargs)
        return tensordict

    @dispatch(auto_batch_size=False)
    @set_skip_existing(None)
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
                f"TensorDictSequential does not support keyword arguments other than 'tensordict_out' or in_keys: {self.in_keys}. Got {kwargs.keys()} instead."
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
