# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import warnings
from typing import Any, Literal

import torch
from tensordict._lazy import LazyStackedTensorDict
from tensordict._td import TensorDict
from tensordict.base import is_tensor_collection, TensorDictBase
from tensordict.nn.common import TensorDictModuleBase
from tensordict.nn.params import TensorDictParams
from torch import nn
from torch.utils._pytree import tree_map

__all__ = ["EnsembleModule", "functional_vmap"]


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


def functional_vmap(
    module: nn.Module,
    *,
    params: TensorDictBase,
    args: tuple = (),
    kwargs: dict[str, Any] | None = None,
    in_dims: dict[str, int] | None = None,
    randomness: Literal["error", "different", "same"] = "error",
    chunk_size: int | None = None,
    method: str | None = None,
    loop: bool = False,
) -> Any:
    """Calls ``module`` once per set of parameters in ``params`` and stacks the outputs.

    The batch dimensions of ``params`` index the sets, for example the
    parameters of several modules stacked with
    :meth:`~tensordict.TensorDict.from_modules`. They are flattened into one
    dimension and :func:`torch.vmap` runs over it once, so every output gets
    ``params.batch_size`` as its leading dimensions. With an empty batch
    size, ``module`` is called once with ``params``.

    Args:
        module (nn.Module): the module to call. ``params`` must have the
            structure of :meth:`~tensordict.TensorDict.from_module` of it.

    Keyword Args:
        params (TensorDictBase): the stacked parameters and buffers.
        args (tuple, optional): the positional inputs of ``module``. Every set
            receives the same ones. Defaults to ``()``.
        kwargs (dict, optional): the keyword inputs of ``module``.
        in_dims (dict, optional): maps keys of ``kwargs`` to the dimension that
            indexes the sets. That input needs ``params.batch_size`` at this
            dimension, and set ``i`` receives its slice ``i``. The other
            inputs are shared.
        randomness (str, optional): Defaults to ``"error"``. Specifies whether the
            randomness in this vmap should be the same or different across batches.
            Look at torch.vmap docs for more info.
        chunk_size (int, optional): the number of sets computed at once.
        method (str, optional): the name of the method of ``module`` to call
            instead of ``forward``.
        loop (bool, optional): if ``True``, calls the sets one at a time and
            stacks the results, for modules that :func:`torch.vmap` cannot
            handle. Defaults to ``False``.

    Returns:
        The outputs of ``module``, with ``params.batch_size`` as leading
        dimensions.

    Raises:
        KeyError: if a key of ``in_dims`` is not in ``kwargs``.
        ValueError: if ``in_dims`` is not empty and ``params`` has no batch
            dimension, if an input named in ``in_dims`` does not have
            ``params.batch_size`` at its dimension, or if ``loop=True`` with
            ``randomness="same"``.

    Examples:
        >>> import torch
        >>> from torch import nn
        >>> from tensordict import TensorDict
        >>> from tensordict.nn import functional_vmap
        >>> nets = [nn.Linear(3, 1) for _ in range(4)]
        >>> params = TensorDict.from_modules(*nets)
        >>> x = torch.randn(10, 3)
        >>> functional_vmap(nets[0], params=params, args=(x,)).shape
        torch.Size([4, 10, 1])
        >>> x_per_set = torch.randn(4, 10, 3)
        >>> functional_vmap(
        ...     nets[0],
        ...     params=params,
        ...     kwargs={"input": x_per_set},
        ...     in_dims={"input": 0},
        ... ).shape
        torch.Size([4, 10, 1])
    """
    kwargs = {} if kwargs is None else kwargs
    in_dims = {} if in_dims is None else in_dims
    if loop and randomness == "same":
        raise ValueError("loop=True cannot share random values across sets.")
    missing = in_dims.keys() - kwargs.keys()
    if missing:
        raise KeyError(
            f"in_dims names inputs that are not in kwargs: {sorted(missing)}."
        )
    batch_size = params.batch_size
    if in_dims and not batch_size:
        raise ValueError("in_dims needs params with at least one batch dimension.")
    ndim = len(batch_size)
    dims = {}
    for key, dim in in_dims.items():
        shape = kwargs[key].shape
        dim = dim % len(shape) if shape else dim
        if shape[dim : dim + ndim] != batch_size:
            raise ValueError(
                f"kwargs[{key!r}] has shape {tuple(shape)}, but "
                f"in_dims[{key!r}]={in_dims[key]} needs "
                f"params.batch_size={tuple(batch_size)} at that dimension."
            )
        dims[key] = dim
    func = module if method is None else getattr(module, method)

    def call(params, kwargs):
        with params.to_module(module, preserve_module_state=False):
            return func(*args, **kwargs)

    if not batch_size:
        return call(params, kwargs)

    params = params.reshape(-1)
    kwargs = {
        key: value.flatten(dims[key], dims[key] + ndim - 1) if key in dims else value
        for key, value in kwargs.items()
    }
    if loop:
        outs = [
            call(
                params[i],
                {
                    key: (
                        value[(slice(None),) * dims[key] + (i,)]
                        if key in dims
                        else value
                    )
                    for key, value in kwargs.items()
                },
            )
            for i in range(params.shape[0])
        ]
        out = tree_map(lambda *xs: torch.stack(xs), *outs, is_leaf=is_tensor_collection)
    else:
        vmapped = torch.vmap(
            call,
            (0, {key: dims.get(key) for key in kwargs}),
            randomness=randomness,
            chunk_size=chunk_size,
        )
        try:
            out = vmapped(params, kwargs)
        except RuntimeError as err:
            if "Batching rule not implemented" in str(err):
                raise RuntimeError(
                    f"{err}\nfunctional_vmap(..., loop=True) calls the sets one "
                    "at a time instead."
                ) from err
            raise
    return tree_map(
        lambda x: x.unflatten(0, batch_size), out, is_leaf=is_tensor_collection
    )
