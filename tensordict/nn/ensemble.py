# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import warnings

import torch
from tensordict._lazy import LazyStackedTensorDict
from tensordict._td import TensorDict
from tensordict.base import TensorDictBase
from tensordict.nn.common import TensorDictModuleBase
from tensordict.nn.params import TensorDictParams

__all__ = ["EnsembleModule"]


class EnsembleModule(TensorDictModuleBase):
    """Runs ``num_copies`` copies of a module, each with its own parameters, on the same input.

    The constructor makes ``num_copies`` sets of parameters, each one reset
    with the ``reset_parameters`` method of the submodules. It stacks them in
    ``ensemble.params_td``, a :class:`~tensordict.nn.TensorDictParams` with
    batch size ``[num_copies]``. A call runs the module once per copy with
    :func:`torch.vmap`, so every output gets a leading dimension of size
    ``num_copies``.

    ``ensemble.parameters()`` returns the stacked copies, which an optimizer
    can train. It also returns the parameters of the original ``module``,
    which the call does not use.

    Args:
        module (TensorDictModuleBase): the module to copy. Its ``in_keys`` and
            ``out_keys`` become those of the ensemble.
        num_copies (int): the number of copies.
        expand_input (bool, optional): if ``True``, every copy receives the
            same input. If ``False``, the input must already have a leading
            dimension of size ``num_copies``, and copy ``i`` receives entry
            ``i``. Use ``False`` for an EnsembleModule placed after another
            one, for example ``EnsembleModule(cnn)`` followed by
            ``EnsembleModule(mlp)``. Defaults to ``True``.

    Examples:
        >>> import torch
        >>> from torch import nn
        >>> from tensordict.nn import TensorDictModule, EnsembleModule
        >>> from tensordict import TensorDict
        >>> net = nn.Sequential(nn.Linear(4, 32), nn.ReLU(), nn.Linear(32, 2))
        >>> mod = TensorDictModule(net, in_keys=['a'], out_keys=['b'])
        >>> ensemble = EnsembleModule(mod, num_copies=3)
        >>> data = TensorDict({'a': torch.randn(10, 4)}, batch_size=[10])
        >>> ensemble(data)
        TensorDict(batch_size=[3,10], fields={a: f32[4], b: f32[2]})

    To chain EnsembleModules, set ``expand_input=False`` on the second one and on those after it.

        >>> import torch
        >>> from tensordict.nn import TensorDictModule, TensorDictSequential, EnsembleModule
        >>> from tensordict import TensorDict
        >>> module = TensorDictModule(torch.nn.Linear(2,3), in_keys=['x'], out_keys=['hidden'])
        >>> next_module = TensorDictModule(torch.nn.Linear(3,1), in_keys=['hidden'], out_keys=['y'])
        >>> e0 = EnsembleModule(module, num_copies=4, expand_input=True)
        >>> e1 = EnsembleModule(next_module, num_copies=4, expand_input=False)
        >>> seq = TensorDictSequential(e0, e1)
        >>> data = TensorDict({'x': torch.randn(5,2)}, batch_size=[5])
        >>> seq(data)
        TensorDict(batch_size=[4,5], fields={hidden: f32[3], x: f32[2], y: f32[1]})
    """

    def __init__(
        self,
        module: TensorDictModuleBase,
        num_copies: int,
        expand_input: bool = True,
    ):
        super().__init__()
        self.in_keys = module.in_keys
        self.out_keys = module.out_keys
        params_td = TensorDict.from_module(module).expand(num_copies).to_tensordict()

        self.module = module
        if expand_input:
            self.vmapped_forward = torch.vmap(self._func_module_call, (None, 0))
        else:
            self.vmapped_forward = torch.vmap(self._func_module_call, 0)

        self.reset_parameters_recursive(params_td)
        self.params_td = TensorDictParams(params_td)

    def _func_module_call(self, input, params):
        with params.to_module(self.module, preserve_module_state=False):
            return self.module(input)

    def forward(self, tensordict: TensorDict) -> TensorDict:
        """Calls every copy on ``tensordict`` and returns the outputs, stacked along a new first dimension."""
        return self.vmapped_forward(tensordict, self.params_td)

    def reset_parameters_recursive(
        self, parameters: TensorDictBase = None
    ) -> TensorDictBase:
        """Resets the values of ``parameters`` in place, with one set of values per copy of the module.

        The constructor calls it to give each copy its own initial values. To
        reset the parameters of the ensemble, pass ``ensemble.params_td``.

        Args:
            parameters (TensorDictBase): the parameters of ``module``, with
                the structure of :meth:`~tensordict.TensorDict.from_module`.
                Each index of the batch dimensions is one copy.

        Returns:
            A lazy stack of the reset parameters, with the batch size of
            ``parameters``, or for an empty batch size what
            :meth:`~tensordict.nn.TensorDictModuleBase.reset_parameters_recursive`
            of ``module`` returns.

        Raises:
            ValueError: if ``parameters`` is ``None``.
        """
        if parameters is None:
            raise ValueError(
                "Ensembles are functional and require passing a TensorDict of parameters to reset_parameters_recursive"
            )
        if parameters.ndim:
            if parameters.ndim > 1:
                # The outputs of unbind are views of one multi-output op:
                # once one is reset in place, the others cannot be unbound
                # again. Indexing gives single-output views.
                params_copies = [parameters[i] for i in range(parameters.shape[0])]
            else:
                params_copies = parameters.unbind(0)
            params_pointers = []
            for params_copy in params_copies:
                self.reset_parameters_recursive(params_copy)
                params_pointers.append(params_copy)
            return LazyStackedTensorDict.lazy_stack(params_pointers, 0)
        else:
            # In case the user has added other neural networks to the EnsembleModule
            # besides those in self.module
            child_mods = [
                mod
                for name, mod in self.named_children()
                if name not in ("module", "ensemble_parameters", "params_td")
            ]
            if child_mods:
                warnings.warn(
                    "EnsembleModule.reset_parameters_recursive() only resets parameters of self.module, but other parameters were detected. These parameters will not be reset."
                )
            # Reset all self.module descendant parameters
            return self.module.reset_parameters_recursive(parameters)
