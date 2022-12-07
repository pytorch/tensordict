# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import re
from textwrap import indent
from typing import Optional, Sequence, Tuple, Type, Union

import torch.nn as nn

from tensordict.nn.common import _check_all_str, TensorDictModule
from tensordict.nn.distributions import Delta, distributions_maps
from tensordict.nn.functional_modules import repopulate_module
from tensordict.nn.probabilistic import interaction_mode
from tensordict.nn.sequence import TensorDictSequential
from tensordict.tensordict import TensorDictBase
from torch import distributions as d, Tensor

__all__ = ["ProbabilisticTensorDictModule"]


class ProbabilisticTensorDictModule(nn.Module):
    """A probabilistic TD Module.

    `ProbabilisticTensorDictModule` is a non-parametric module representing a
    probability distribution. It reads the distribution parameters from an input
    TensorDict using the specified `in_keys`. The output is sampled given some rule,
    specified by the input :obj:`default_interaction_mode` argument and the
    :obj:`interaction_mode()` global function.

    :obj:`ProbabilisticTensorDictModule` can be used to construct the distribution
    (through the :obj:`get_dist()` method) and/or sampling from this distribution
    (through a regular :obj:`__call__()` to the module).

    A :obj:`ProbabilisticTensorDictModule` instance has two main features:
    - It reads and writes TensorDict objects
    - It uses a real mapping R^n -> R^m to create a distribution in R^d from
    which values can be sampled or computed.

    When the :obj:`__call__` / :obj:`forward` method is called, a distribution is
    created, and a value computed (using the 'mean', 'mode', 'median' attribute or
    the 'rsample', 'sample' method). The sampling step is skipped if the supplied
    TensorDict has all of the desired key-value pairs already.

    By default, ProbabilisticTensorDictModule distribution class is a Delta
    distribution, making ProbabilisticTensorDictModule a simple wrapper around
    a deterministic mapping function.

    Args:
        in_keys (str or iterable of str or dict): key(s) that will be read from the
            input TensorDict and used to build the distribution. Importantly, if it's an
            iterable of string or a string, those keys must match the keywords used by
            the distribution class of interest, e.g. :obj:`"loc"` and :obj:`"scale"` for
            the Normal distribution and similar. If in_keys is a dictionary,, the keys
            are the keys of the distribution and the values are the keys in the
            tensordict that will get match to the corresponding distribution keys.
        out_keys (str or iterable of str): keys where the sampled values will be
            written. Importantly, if these keys are found in the input TensorDict, the
            sampling step will be skipped.
        default_interaction_mode (str, optional): default method to be used to retrieve
            the output value. Should be one of: 'mode', 'median', 'mean' or 'random'
            (in which case the value is sampled randomly from the distribution). Default
            is 'mode'.
            Note: When a sample is drawn, the :obj:`ProbabilisticTDModule` instance will
            fist look for the interaction mode dictated by the `interaction_mode()`
            global function. If this returns `None` (its default value), then the
            `default_interaction_mode` of the `ProbabilisticTDModule` instance will be
            used. Note that DataCollector instances will use `set_interaction_mode` to
            `"random"` by default.
        distribution_class (Type, optional): a torch.distributions.Distribution class to
            be used for sampling. Default is Delta.
        distribution_kwargs (dict, optional): kwargs to be passed to the distribution.
        return_log_prob (bool, optional): if True, the log-probability of the
            distribution sample will be written in the
            tensordict with the key `'sample_log_prob'`. Default is `False`.
        cache_dist (bool, optional): EXPERIMENTAL: if True, the parameters of the distribution (i.e. the output of the module)
            will be written to the tensordict along with the sample. Those parameters can be used to
            re-compute the original distribution later on (e.g. to compute the divergence between the distribution
            used to sample the action and the updated distribution in PPO).
            Default is `False`.
        n_empirical_estimate (int, optional): number of samples to compute the empirical mean when it is not available.
            Default is 1000

    Examples:
        >>> import torch
        >>> from tensordict import TensorDict
        >>> from tensordict.nn import ProbabilisticTensorDictModule, TensorDictModule
        >>> from tensordict.nn.distributions import NormalParamWrapper
        >>> from tensordict.nn.functional_modules import make_functional
        >>> from torch.distributions import Normal
        >>> td = TensorDict({"input": torch.randn(3, 4), "hidden": torch.randn(3, 8)}, [3,])
        >>> net = NormalParamWrapper(torch.nn.GRUCell(4, 8))
        >>> module = TensorDictModule(net, in_keys=["input", "hidden"], out_keys=["loc", "scale"])
        >>> td_module = ProbabilisticTensorDictModule(
        ...     module=module,
        ...     in_keys=["loc", "scale"],
        ...     out_keys=["action"],
        ...     distribution_class=Normal,
        ...     return_log_prob=True,
        ... )
        >>> params = make_functional(td_module, funs_to_decorate=["forward", "get_dist"])
        >>> _ = td_module(td, params=params)
        >>> print(td)
        TensorDict(
            fields={
                action: Tensor(torch.Size([3, 4]), dtype=torch.float32),
                hidden: Tensor(torch.Size([3, 8]), dtype=torch.float32),
                input: Tensor(torch.Size([3, 4]), dtype=torch.float32),
                loc: Tensor(torch.Size([3, 4]), dtype=torch.float32),
                sample_log_prob: Tensor(torch.Size([3, 4]), dtype=torch.float32),
                scale: Tensor(torch.Size([3, 4]), dtype=torch.float32)},
            batch_size=torch.Size([3]),
            device=None,
            is_shared=False)
        >>> dist, *_ = td_module.get_dist(td, params=params)
        >>> print(dist)
        Normal(loc: torch.Size([3, 4]), scale: torch.Size([3, 4]))

        >>> # we can also apply the module to the TensorDict with vmap
        >>> from functorch import vmap
        >>> params = params.expand(4)
        >>> td_vmap = vmap(td_module, (None, 0))(td, params)
        >>> print(td_vmap)
        TensorDict(
            fields={
                action: Tensor(torch.Size([4, 3, 4]), dtype=torch.float32),
                hidden: Tensor(torch.Size([4, 3, 8]), dtype=torch.float32),
                input: Tensor(torch.Size([4, 3, 4]), dtype=torch.float32),
                loc: Tensor(torch.Size([4, 3, 4]), dtype=torch.float32),
                sample_log_prob: Tensor(torch.Size([4, 3, 4]), dtype=torch.float32),
                scale: Tensor(torch.Size([4, 3, 4]), dtype=torch.float32)},
            batch_size=torch.Size([4, 3]),
            device=None,
            is_shared=False)

    """

    def __init__(
        self,
        in_keys: Union[str, Sequence[str], dict],
        out_keys: Optional[Union[str, Sequence[str]]] = None,
        default_interaction_mode: str = "mode",
        distribution_class: Type = Delta,
        distribution_kwargs: Optional[dict] = None,
        return_log_prob: bool = False,
        cache_dist: bool = False,
        n_empirical_estimate: int = 1000,
    ):
        super().__init__()
        if isinstance(in_keys, str):
            in_keys = [in_keys]
        if isinstance(out_keys, str):
            out_keys = [out_keys]
        elif out_keys is None:
            out_keys = []
        if not isinstance(in_keys, dict):
            in_keys = {param_key: param_key for param_key in in_keys}

        self.out_keys = out_keys
        _check_all_str(self.out_keys)
        self.in_keys = in_keys
        _check_all_str(self.in_keys.keys())
        _check_all_str(self.in_keys.values())

        self.default_interaction_mode = default_interaction_mode
        if isinstance(distribution_class, str):
            distribution_class = distributions_maps.get(distribution_class.lower())
        self.distribution_class = distribution_class
        self.distribution_kwargs = (
            distribution_kwargs if distribution_kwargs is not None else {}
        )
        self.n_empirical_estimate = n_empirical_estimate
        self._dist = None
        self.cache_dist = cache_dist if hasattr(distribution_class, "update") else False
        self.return_log_prob = return_log_prob

    def get_dist(self, tensordict: TensorDictBase) -> d.Distribution:
        try:
            dist_kwargs = {
                dist_key: tensordict[td_key]
                for dist_key, td_key in self.in_keys.items()
            }
            dist = self.distribution_class(**dist_kwargs)
        except TypeError as err:
            if "an unexpected keyword argument" in str(err):
                raise TypeError(
                    "distribution keywords and tensordict keys indicated by ProbabilisticTensorDictModule.in_keys must match."
                    f"Got this error message: \n{indent(str(err), 4 * ' ')}\nwith in_keys={self.in_keys}"
                )
            elif re.search(r"missing.*required positional arguments", str(err)):
                raise TypeError(
                    f"TensorDict with keys {tensordict.keys()} does not match the distribution {self.distribution_class} keywords."
                )
            else:
                raise err
        return dist

    def forward(
        self,
        tensordict: TensorDictBase,
        tensordict_out: Optional[TensorDictBase] = None,
    ) -> TensorDictBase:
        if tensordict_out is None:
            tensordict_out = tensordict

        dist = self.get_dist(tensordict)
        td_keys = set(tensordict.keys())
        # if the tensordict contains the sampled keys we wont be sampling them again
        # in that case ProbabilisticTensorDictModule is presumably used to return the
        # distribution using `get_dist` or to sample log_probabilities
        if not all(key in td_keys for key in self.out_keys):
            out_tensors = self._dist_sample(dist, interaction_mode=interaction_mode())
            if isinstance(out_tensors, Tensor):
                out_tensors = (out_tensors,)
            tensordict_out.update(
                {key: value for key, value in zip(self.out_keys, out_tensors)}
            )
            if self.return_log_prob:
                log_prob = dist.log_prob(*out_tensors)
                tensordict_out.set("sample_log_prob", log_prob)
        elif self.return_log_prob:
            out_tensors = [tensordict.get(key) for key in self.out_keys]
            log_prob = dist.log_prob(*out_tensors)
            tensordict_out.set("sample_log_prob", log_prob)
            # raise RuntimeError(
            #     "ProbabilisticTensorDictModule.return_log_prob = True is incompatible with settings in which "
            #     "the submodule is responsible for sampling. To manually gather the log-probability, call first "
            #     "\n>>> dist, tensordict = tensordict_module.get_dist(tensordict)"
            #     "\n>>> tensordict.set('sample_log_prob', dist.log_prob(tensordict.get(sample_key))"
            # )
        return tensordict_out

    def _dist_sample(
        self, dist: d.Distribution, interaction_mode: bool = None
    ) -> Union[Tuple[Tensor], Tensor]:
        if interaction_mode is None or interaction_mode == "":
            interaction_mode = self.default_interaction_mode
        if not isinstance(dist, d.Distribution):
            raise TypeError(f"type {type(dist)} not recognised by _dist_sample")

        if interaction_mode == "mode":
            if hasattr(dist, "mode"):
                return dist.mode
            else:
                raise NotImplementedError(
                    f"method {type(dist)}.mode is not implemented"
                )

        elif interaction_mode == "median":
            if hasattr(dist, "median"):
                return dist.median
            else:
                raise NotImplementedError(
                    f"method {type(dist)}.median is not implemented"
                )

        elif interaction_mode == "mean":
            try:
                return dist.mean
            except (AttributeError, NotImplementedError):
                if dist.has_rsample:
                    return dist.rsample((self.n_empirical_estimate,)).mean(0)
                else:
                    return dist.sample((self.n_empirical_estimate,)).mean(0)

        elif interaction_mode == "random":
            if dist.has_rsample:
                return dist.rsample()
            else:
                return dist.sample()
        else:
            raise NotImplementedError(f"unknown interaction_mode {interaction_mode}")


class ProbabilisticTensorDictSequential(TensorDictSequential):
    def __init__(
        self,
        *modules: Union[TensorDictModule, ProbabilisticTensorDictModule],
        partial_tolerant: bool = False,
    ) -> None:
        if len(modules) == 0:
            raise ValueError(
                "ProbabilisticTensorDictSequential must consist of zero or more "
                "TensorDictModules followed by a ProbabilisticTensorDictModule"
            )
        if not isinstance(modules[-1], ProbabilisticTensorDictModule):
            raise TypeError(
                "The final module passed to ProbabilisticTensorDictSequential must be "
                "an instance of ProbabilisticTensorDictModule"
            )
        super().__init__(*modules, partial_tolerant=partial_tolerant)

    def get_dist(
        self,
        tensordict: TensorDictBase,
        tensordict_out: Optional[TensorDictBase] = None,
        **kwargs,
    ) -> d.Distribution:
        tds = TensorDictSequential(*self.module[:-1])
        if self.__dict__.get("_is_stateless", False):
            tds = repopulate_module(tds, kwargs.pop("params"))
        tensordict_out = tds(tensordict, tensordict_out, **kwargs)
        return self.module[-1].get_dist(tensordict_out)
