# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import re
import warnings
from enum import auto, Enum
from textwrap import indent
from typing import Any, Callable, Dict, List, Optional
from warnings import warn

from tensordict._contextlib import _DecoratorContextManager
from tensordict.nn import CompositeDistribution

from tensordict.nn.common import dispatch, TensorDictModule, TensorDictModuleBase
from tensordict.nn.distributions import Delta, distributions_maps
from tensordict.nn.sequence import TensorDictSequential

from tensordict.nn.utils import set_skip_existing
from tensordict.tensordict import TensorDictBase
from tensordict.utils import _zip_strict, NestedKey
from torch import distributions as D, Tensor

__all__ = ["ProbabilisticTensorDictModule", "ProbabilisticTensorDictSequential"]


class InteractionType(Enum):
    """A list of possible interaction types with a distribution.

    MODE, MEDIAN and MEAN point to the property / attribute with the same name.
    RANDOM points to ``rsample()`` if that method exists or ``sample()`` if not.

    DETERMINISTIC can be used as a generic fallback if ``MEAN`` or ``MODE`` are not guaranteed to
    be analytically tractable. In such cases, a rude deterministic estimate can be used
    in some cases even if it lacks a true algebraic meaning.
    This value will trigger a query to the ``deterministic_sample`` attribute in the distribution
    and if it does not exist, the ``mean`` will be used.

    """

    MODE = auto()
    MEDIAN = auto()
    MEAN = auto()
    RANDOM = auto()
    DETERMINISTIC = auto()

    @classmethod
    def from_str(cls, type_str: str) -> InteractionType:
        """Return the interaction_type with name matched to the provided string (case insensitive)."""
        for member_type in cls:
            if member_type.name == type_str.upper():
                return member_type
        raise ValueError(f"The provided interaction type {type_str} is unsupported!")


_INTERACTION_TYPE: InteractionType | None = None


def _insert_interaction_mode_deprecation_warning(
    prefix: str = "",
) -> Callable[[str, Warning, int], None]:
    return warn(
        (
            f"{prefix}interaction_mode is deprecated for naming clarity and will be removed in v0.6. "
            f"Please use {prefix}interaction_type with InteractionType enum instead."
        ),
        DeprecationWarning,
        stacklevel=2,
    )


def interaction_type() -> InteractionType | None:
    """Returns the current sampling type."""
    return _INTERACTION_TYPE


def interaction_mode() -> str | None:
    """*Deprecated* Returns the current sampling mode."""
    _insert_interaction_mode_deprecation_warning()
    type = interaction_type()
    return type.name.lower() if type else None


class set_interaction_mode(_DecoratorContextManager):
    """*Deprecated* Sets the sampling mode of all ProbabilisticTDModules to the desired mode.

    Args:
        mode (str): mode to use when the policy is being called.
    """

    def __init__(self, mode: str | None = "mode") -> None:
        _insert_interaction_mode_deprecation_warning("set_")
        super().__init__()
        self.mode = InteractionType.from_str(mode) if mode else None

    def clone(self) -> set_interaction_mode:
        # override this method if your children class takes __init__ parameters
        return type(self)(self.mode)

    def __enter__(self) -> None:
        global _INTERACTION_TYPE
        self.prev = _INTERACTION_TYPE
        _INTERACTION_TYPE = self.mode

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        global _INTERACTION_TYPE
        _INTERACTION_TYPE = self.prev


class set_interaction_type(_DecoratorContextManager):
    """Sets all ProbabilisticTDModules sampling to the desired type.

    Args:
        type (InteractionType): sampling type to use when the policy is being called.
    """

    def __init__(
        self, type: InteractionType | None = InteractionType.DETERMINISTIC
    ) -> None:
        super().__init__()
        self.type = type

    def clone(self) -> set_interaction_type:
        # override this method if your children class takes __init__ parameters
        return type(self)(self.type)

    def __enter__(self) -> None:
        global _INTERACTION_TYPE
        self.prev = _INTERACTION_TYPE
        _INTERACTION_TYPE = self.type

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        global _INTERACTION_TYPE
        _INTERACTION_TYPE = self.prev


class ProbabilisticTensorDictModule(TensorDictModuleBase):
    """A probabilistic TD Module.

    `ProbabilisticTensorDictModule` is a non-parametric module representing a
    probability distribution. It reads the distribution parameters from an input
    TensorDict using the specified `in_keys`. The output is sampled given some rule,
    specified by the input :obj:`default_interaction_type` argument and the
    :obj:`interaction_type()` global function.

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
        in_keys (NestedKey or list of NestedKey or dict): key(s) that will be read from the
            input TensorDict and used to build the distribution. Importantly, if it's an
            list of NestedKey or a NestedKey, the leaf (last element) of those keys must match the keywords used by
            the distribution class of interest, e.g. :obj:`"loc"` and :obj:`"scale"` for
            the Normal distribution and similar. If in_keys is a dictionary, the keys
            are the keys of the distribution and the values are the keys in the
            tensordict that will get match to the corresponding distribution keys.
        out_keys (NestedKey or list of NestedKey): keys where the sampled values will be
            written. Importantly, if these keys are found in the input TensorDict, the
            sampling step will be skipped.
        default_interaction_mode (str, optional): *Deprecated* keyword-only argument.
            Please use default_interaction_type instead.
        default_interaction_type (InteractionType, optional): keyword-only argument.
            Default method to be used to retrieve
            the output value. Should be one of InteractionType: MODE, MEDIAN, MEAN or RANDOM
            (in which case the value is sampled randomly from the distribution). Default
            is MODE.

            .. note:: When a sample is drawn, the
              :class:`ProbabilisticTensorDictModule` instance will
              first look for the interaction mode dictated by the
              :func:`~tensordict.nn.probabilistic.interaction_type`
              global function. If this returns `None` (its default value), then the
              `default_interaction_type` of the `ProbabilisticTDModule`
              instance will be used. Note that
              :class:`~torchrl.collectors.collectors.DataCollectorBase`
              instances will use `set_interaction_type` to
              :class:`tensordict.nn.InteractionType.RANDOM` by default.

            .. note:: In some cases, the mode, median or mean value may not be
              readily available through the corresponding attribute.
              To paliate this, :class:`~ProbabilisticTensorDictModule` will first attempt
              to get the value through a call to ``get_mode()``, ``get_median()`` or ``get_mean()``
              if the method exists.

        distribution_class (Type, optional): keyword-only argument.
            A :class:`torch.distributions.Distribution` class to
            be used for sampling.
            Default is :class:`~tensordict.nn.distributions.Delta`.

            .. note:: If the distribution class is of type
                :class:`~tensordict.nn.distributions.CompositeDistribution`, the ``out_keys``
                can be inferred directly form the ``"distribution_map"`` or ``"name_map"``
                keywork arguments provided through this class' ``distribution_kwargs``
                keyword argument, making the ``out_keys`` optional in such cases.

        distribution_kwargs (dict, optional): keyword-only argument.
            Keyword-argument pairs to be passed to the distribution.
        return_log_prob (bool, optional): keyword-only argument.
            If ``True``, the log-probability of the
            distribution sample will be written in the tensordict with the key
            `log_prob_key`. Default is ``False``.
        log_prob_key (NestedKey, optional): key where to write the log_prob if return_log_prob = True.
            Defaults to `'sample_log_prob'`.
        cache_dist (bool, optional): keyword-only argument.
            EXPERIMENTAL: if ``True``, the parameters of the
            distribution (i.e. the output of the module) will be written to the
            tensordict along with the sample. Those parameters can be used to re-compute
            the original distribution later on (e.g. to compute the divergence between
            the distribution used to sample the action and the updated distribution in
            PPO). Default is ``False``.
        n_empirical_estimate (int, optional): keyword-only argument.
            Number of samples to compute the empirical
            mean when it is not available. Defaults to 1000.

    Examples:
        >>> import torch
        >>> from tensordict import TensorDict
        >>> from tensordict.nn import (
        ...     ProbabilisticTensorDictModule,
        ...     ProbabilisticTensorDictSequential,
        ...     TensorDictModule,
        ... )
        >>> from tensordict.nn.distributions import NormalParamExtractor
        >>> from tensordict.nn.functional_modules import make_functional
        >>> from torch.distributions import Normal, Independent
        >>> td = TensorDict(
        ...     {"input": torch.randn(3, 4), "hidden": torch.randn(3, 8)}, [3]
        ... )
        >>> net = torch.nn.GRUCell(4, 8)
        >>> module = TensorDictModule(
        ...     net, in_keys=["input", "hidden"], out_keys=["params"]
        ... )
        >>> normal_params = TensorDictModule(
        ...     NormalParamExtractor(), in_keys=["params"], out_keys=["loc", "scale"]
        ... )
        >>> def IndepNormal(**kwargs):
        ...     return Independent(Normal(**kwargs), 1)
        >>> prob_module = ProbabilisticTensorDictModule(
        ...     in_keys=["loc", "scale"],
        ...     out_keys=["action"],
        ...     distribution_class=IndepNormal,
        ...     return_log_prob=True,
        ... )
        >>> td_module = ProbabilisticTensorDictSequential(
        ...     module, normal_params, prob_module
        ... )
        >>> params = TensorDict.from_module(td_module)
        >>> with params.to_module(td_module):
        ...     _ = td_module(td)
        >>> print(td)
        TensorDict(
            fields={
                action: Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.float32, is_shared=False),
                hidden: Tensor(shape=torch.Size([3, 8]), device=cpu, dtype=torch.float32, is_shared=False),
                input: Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.float32, is_shared=False),
                loc: Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.float32, is_shared=False),
                params: Tensor(shape=torch.Size([3, 8]), device=cpu, dtype=torch.float32, is_shared=False),
                sample_log_prob: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False),
                scale: Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.float32, is_shared=False)},
            batch_size=torch.Size([3]),
            device=None,
            is_shared=False)
        >>> with params.to_module(td_module):
        ...     dist = td_module.get_dist(td)
        >>> print(dist)
        Independent(Normal(loc: torch.Size([3, 4]), scale: torch.Size([3, 4])), 1)
        >>> # we can also apply the module to the TensorDict with vmap
        >>> from torch import vmap
        >>> params = params.expand(4)
        >>> def func(td, params):
        ...     with params.to_module(td_module):
        ...         return td_module(td)
        >>> td_vmap = vmap(func, (None, 0))(td, params)
        >>> print(td_vmap)
        TensorDict(
            fields={
                action: Tensor(shape=torch.Size([4, 3, 4]), device=cpu, dtype=torch.float32, is_shared=False),
                hidden: Tensor(shape=torch.Size([4, 3, 8]), device=cpu, dtype=torch.float32, is_shared=False),
                input: Tensor(shape=torch.Size([4, 3, 4]), device=cpu, dtype=torch.float32, is_shared=False),
                loc: Tensor(shape=torch.Size([4, 3, 4]), device=cpu, dtype=torch.float32, is_shared=False),
                params: Tensor(shape=torch.Size([4, 3, 8]), device=cpu, dtype=torch.float32, is_shared=False),
                sample_log_prob: Tensor(shape=torch.Size([4, 3]), device=cpu, dtype=torch.float32, is_shared=False),
                scale: Tensor(shape=torch.Size([4, 3, 4]), device=cpu, dtype=torch.float32, is_shared=False)},
            batch_size=torch.Size([4, 3]),
            device=None,
            is_shared=False)

    """

    def __init__(
        self,
        in_keys: NestedKey | List[NestedKey] | Dict[str, NestedKey],
        out_keys: NestedKey | List[NestedKey] | None = None,
        *,
        default_interaction_mode: str | None = None,
        default_interaction_type: InteractionType = InteractionType.DETERMINISTIC,
        distribution_class: type = Delta,
        distribution_kwargs: dict | None = None,
        return_log_prob: bool = False,
        log_prob_key: Optional[NestedKey] = "sample_log_prob",
        cache_dist: bool = False,
        n_empirical_estimate: int = 1000,
    ) -> None:
        super().__init__()
        distribution_kwargs = (
            distribution_kwargs if distribution_kwargs is not None else {}
        )
        if isinstance(in_keys, (str, tuple)):
            in_keys = [in_keys]
        if isinstance(out_keys, (str, tuple)):
            out_keys = [out_keys]
        elif out_keys is None:
            if distribution_class is CompositeDistribution:
                distribution_map = distribution_kwargs.get("distribution_map")
                if distribution_map is None:
                    raise KeyError(
                        "'distribution_map' must be provided within "
                        "distribution_kwargs whenever the distribution is of type CompositeDistribution."
                    )
                name_map = distribution_kwargs.get("name_map", None)
                if name_map is not None:
                    out_keys = list(name_map.values())
                else:
                    out_keys = list(distribution_map.keys())
            else:
                out_keys = ["_"]
        if isinstance(in_keys, dict):
            dist_keys, in_keys = zip(*in_keys.items())
            if set(map(type, dist_keys)) != {str}:
                raise ValueError(
                    f"If in_keys is dict, its keys must be strings matching to the distribution kwargs."
                    f"{type(self).__name__} got {dist_keys}"
                )
        else:
            dist_keys = in_keys

        self.out_keys = out_keys
        self.in_keys = in_keys
        self.dist_keys = dist_keys
        if log_prob_key is None:
            log_prob_key = "sample_log_prob"
        self.log_prob_key = log_prob_key

        if default_interaction_mode is not None:
            _insert_interaction_mode_deprecation_warning("default_")
            self.default_interaction_type = InteractionType.from_str(
                default_interaction_mode
            )
        else:
            self.default_interaction_type = default_interaction_type

        if isinstance(distribution_class, str):
            distribution_class = distributions_maps.get(distribution_class.lower())
        self.distribution_class = distribution_class
        self.distribution_kwargs = distribution_kwargs
        self.n_empirical_estimate = n_empirical_estimate
        self._dist = None
        self.cache_dist = cache_dist if hasattr(distribution_class, "update") else False
        self.return_log_prob = return_log_prob
        if self.return_log_prob and self.log_prob_key not in self.out_keys:
            self.out_keys.append(self.log_prob_key)

    def get_dist(self, tensordict: TensorDictBase) -> D.Distribution:
        """Creates a :class:`torch.distribution.Distribution` instance with the parameters provided in the input tensordict."""
        try:
            dist_kwargs = {}
            for dist_key, td_key in _zip_strict(self.dist_keys, self.in_keys):
                if isinstance(dist_key, tuple):
                    dist_key = dist_key[-1]
                dist_kwargs[dist_key] = tensordict.get(td_key)
            dist = self.distribution_class(**dist_kwargs, **self.distribution_kwargs)
        except TypeError as err:
            if "an unexpected keyword argument" in str(err):
                raise TypeError(
                    "distribution keywords and tensordict keys indicated by ProbabilisticTensorDictModule.dist_keys must match."
                    f"Got this error message: \n{indent(str(err), 4 * ' ')}\nwith dist_keys={self.dist_keys}"
                )
            elif re.search(r"missing.*required positional arguments", str(err)):
                raise TypeError(
                    f"TensorDict with keys {tensordict.keys()} does not match the distribution {self.distribution_class} keywords."
                )
            else:
                raise err
        return dist

    def log_prob(self, tensordict):
        """Writes the log-probability of the distribution sample."""
        dist = self.get_dist(tensordict)
        if isinstance(dist, CompositeDistribution):
            tensordict = dist.log_prob(tensordict)
            return tensordict.get("sample_log_prob")
        else:
            return dist.log_prob(tensordict.get(self.out_keys[0]))

    @property
    def SAMPLE_LOG_PROB_KEY(self):
        warnings.warn(
            "SAMPLE_LOG_PROB_KEY will be deprecated in v0.6."
            "Use 'obj.log_prob_key' instead",
            category=DeprecationWarning,
        )
        return self.log_prob_key

    @dispatch(auto_batch_size=False)
    @set_skip_existing(None)
    def forward(
        self,
        tensordict: TensorDictBase,
        tensordict_out: TensorDictBase | None = None,
        _requires_sample: bool = True,
    ) -> TensorDictBase:
        if tensordict_out is None:
            tensordict_out = tensordict

        dist = self.get_dist(tensordict)
        if _requires_sample:
            out_tensors = self._dist_sample(dist, interaction_type=interaction_type())
            if isinstance(out_tensors, TensorDictBase):
                tensordict_out.update(out_tensors)
                if self.return_log_prob:
                    tensordict_out = dist.log_prob(tensordict_out)
            else:
                if isinstance(out_tensors, Tensor):
                    out_tensors = (out_tensors,)
                tensordict_out.update(
                    {key: value for key, value in zip(self.out_keys, out_tensors)}
                )
                if self.return_log_prob:
                    log_prob = dist.log_prob(*out_tensors)
                    tensordict_out.set(self.log_prob_key, log_prob)
        elif self.return_log_prob:
            out_tensors = [
                tensordict.get(key) for key in self.out_keys if key != self.log_prob_key
            ]
            log_prob = dist.log_prob(*out_tensors)
            tensordict_out.set(self.log_prob_key, log_prob)
            # raise RuntimeError(
            #     "ProbabilisticTensorDictModule.return_log_prob = True is incompatible with settings in which "
            #     "the submodule is responsible for sampling. To manually gather the log-probability, call first "
            #     "\n>>> dist, tensordict = tensordict_module.get_dist(tensordict)"
            #     "\n>>> tensordict.set('sample_log_prob', dist.log_prob(tensordict.get(sample_key))"
            # )
        return tensordict_out

    def _dist_sample(
        self,
        dist: D.Distribution,
        interaction_type: InteractionType | None = None,
    ) -> tuple[Tensor, ...] | Tensor:
        if interaction_type is None:
            interaction_type = self.default_interaction_type

        if interaction_type is InteractionType.DETERMINISTIC:
            try:
                return dist.deterministic_sample
            except AttributeError:
                try:
                    support = dist.support
                    fallback = (
                        "mean" if isinstance(support, D.constraints._Real) else "mode"
                    )
                except NotImplementedError:
                    # Some custom dists don't have a support
                    # We arbitrarily fall onto 'mean' in these cases
                    fallback = "mean"
                try:
                    if fallback == "mean":
                        return dist.mean
                    elif fallback == "mode":
                        # Categorical dists don't have an average
                        return dist.mode
                    else:
                        raise AttributeError
                except AttributeError:
                    raise NotImplementedError(
                        f"method {type(dist)}.deterministic_sample is not implemented, no replacement found."
                    )
                finally:
                    warnings.warn(
                        f"deterministic_sample wasn't found when queried on {type(dist)}. "
                        f"{type(self).__name__} is falling back on {fallback} instead. "
                        f"For better code quality and efficiency, make sure to either "
                        f"provide a distribution with a deterministic_sample attribute or "
                        f"to change the InteractionMode to the desired value.",
                        category=UserWarning,
                    )

        if interaction_type is InteractionType.MODE:
            try:
                return dist.mode
            except AttributeError:
                raise NotImplementedError(
                    f"method {type(dist)}.mode is not implemented"
                )

        elif interaction_type is InteractionType.MEDIAN:
            try:
                return dist.median
            except AttributeError:
                raise NotImplementedError(
                    f"method {type(dist)}.median is not implemented"
                )

        elif interaction_type is InteractionType.MEAN:
            try:
                return dist.mean
            except (AttributeError, NotImplementedError):
                if dist.has_rsample:
                    return dist.rsample((self.n_empirical_estimate,)).mean(0)
                else:
                    return dist.sample((self.n_empirical_estimate,)).mean(0)

        elif interaction_type is InteractionType.RANDOM:
            if dist.has_rsample:
                return dist.rsample()
            else:
                return dist.sample()
        else:
            raise NotImplementedError(f"unknown interaction_type {interaction_type}")


class ProbabilisticTensorDictSequential(TensorDictSequential):
    """A sequence of TensorDictModules ending in a ProbabilistictTensorDictModule.

    Similarly to :obj:`TensorDictSequential`, but enforces that the final module in the
    sequence is an :obj:`ProbabilisticTensorDictModule` and also exposes ``get_dist``
    method to recover the distribution object from the ``ProbabilisticTensorDictModule``

    Args:
         modules (sequence of TensorDictModules): ordered sequence of TensorDictModule
            instances, terminating in ProbabilisticTensorDictModule, to be run
            sequentially.
         partial_tolerant (bool, optional): if True, the input tensordict can miss some
            of the input keys. If so, the only module that will be executed are those
            who can be executed given the keys that are present. Also, if the input
            tensordict is a lazy stack of tensordicts AND if partial_tolerant is
            :obj:`True` AND if the stack does not have the required keys, then
            TensorDictSequential will scan through the sub-tensordicts looking for those
            that have the required keys, if any.

    """

    def __init__(
        self,
        *modules: TensorDictModule | ProbabilisticTensorDictModule,
        partial_tolerant: bool = False,
    ) -> None:
        if len(modules) == 0:
            raise ValueError(
                "ProbabilisticTensorDictSequential must consist of zero or more "
                "TensorDictModules followed by a ProbabilisticTensorDictModule"
            )
        if not isinstance(
            modules[-1],
            (ProbabilisticTensorDictModule, ProbabilisticTensorDictSequential),
        ):
            raise TypeError(
                "The final module passed to ProbabilisticTensorDictSequential must be "
                "an instance of ProbabilisticTensorDictModule or another "
                "ProbabilisticTensorDictSequential"
            )
        # if the modules not including the final probabilistic module return the sampled
        # key we wont be sampling it again, in that case
        # ProbabilisticTensorDictSequential is presumably used to return the
        # distribution using `get_dist` or to sample log_probabilities
        _, out_keys = self._compute_in_and_out_keys(modules[:-1])
        self._requires_sample = modules[-1].out_keys[0] not in set(out_keys)
        super().__init__(*modules, partial_tolerant=partial_tolerant)

    @property
    def det_part(self):
        if not hasattr(self, "_det_part"):
            # we use a list to avoid having the submodules listed in module.modules()
            self._det_part = [TensorDictSequential(*self.module[:-1])]
        return self._det_part[0]

    def get_dist_params(
        self,
        tensordict: TensorDictBase,
        tensordict_out: TensorDictBase | None = None,
        **kwargs,
    ) -> tuple[D.Distribution, TensorDictBase]:
        tds = self.det_part
        type = interaction_type()
        if type is None:
            type = self.module[-1].default_interaction_type
        with set_interaction_type(type):
            return tds(tensordict, tensordict_out, **kwargs)

    def get_dist(
        self,
        tensordict: TensorDictBase,
        tensordict_out: TensorDictBase | None = None,
        **kwargs,
    ) -> D.Distribution:
        """Get the distribution that results from passing the input tensordict through the sequence, and then using the resulting parameters."""
        tensordict_out = self.get_dist_params(tensordict, tensordict_out, **kwargs)
        return self.build_dist_from_params(tensordict_out)

    def log_prob(
        self, tensordict, tensordict_out: TensorDictBase | None = None, **kwargs
    ):
        tensordict_out = self.get_dist_params(
            tensordict,
            tensordict_out,
            **kwargs,
        )
        return self.module[-1].log_prob(tensordict_out)

    def build_dist_from_params(self, tensordict: TensorDictBase) -> D.Distribution:
        """Construct a distribution from the input parameters. Other modules in the sequence are not evaluated.

        This method will look for the last ProbabilisticTensorDictModule contained in the
        sequence and use it to build the distribution.

        """
        dest_module = None
        for module in reversed(list(self.modules())):
            if isinstance(module, ProbabilisticTensorDictModule):
                dest_module = module
                break
        if dest_module is None:
            raise RuntimeError(
                "Could not find any ProbabilisticTensorDictModule in the sequence."
            )
        return dest_module.get_dist(tensordict)

    @dispatch(auto_batch_size=False)
    @set_skip_existing(None)
    def forward(
        self,
        tensordict: TensorDictBase,
        tensordict_out: TensorDictBase | None = None,
        **kwargs,
    ) -> TensorDictBase:
        tensordict_out = self.get_dist_params(tensordict, tensordict_out, **kwargs)
        return self.module[-1](tensordict_out, _requires_sample=self._requires_sample)
