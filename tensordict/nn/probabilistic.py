# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import collections
import re
import warnings
from collections.abc import MutableSequence
from textwrap import indent
from typing import Any, Dict, List, OrderedDict, overload

import torch

from tensordict._nestedkey import NestedKey
from tensordict._td import TensorDict
from tensordict.base import is_tensor_collection
from tensordict.nn.common import dispatch, TensorDictModuleBase
from tensordict.nn.distributions import distributions_maps
from tensordict.nn.distributions.composite import _add_suffix, CompositeDistribution
from tensordict.nn.distributions.continuous import Delta
from tensordict.nn.distributions.discrete import OneHotCategorical
from tensordict.nn.distributions.truncated_normal import (
    TruncatedNormal,
    TruncatedStandardNormal,
)
from tensordict.nn.sequence import TensorDictSequential
from tensordict.nn.utils import (
    _composite_lp_aggregate,
    _set_skip_existing_None,
    composite_lp_aggregate,
    set_composite_lp_aggregate,
)
from tensordict.tensorclass import is_non_tensor
from tensordict.tensordict import TensorDictBase
from tensordict.utils import _ContextManager, _zip_strict, unravel_key
from torch import distributions as D, Tensor
from torch.utils._contextlib import _DecoratorContextManager

try:
    from torch.compiler import is_compiling
except ImportError:
    from torch._dynamo import is_compiling

try:
    from enum import StrEnum
except ImportError:
    from .utils import StrEnum

__all__ = ["ProbabilisticTensorDictModule", "ProbabilisticTensorDictSequential"]


class InteractionType(StrEnum):
    """A list of possible interaction types with a distribution.

    MODE, MEDIAN and MEAN point to the property / attribute with the same name.
    RANDOM points to ``rsample()`` if that method exists or ``sample()`` if not.

    DETERMINISTIC can be used as a generic fallback if ``MEAN`` or ``MODE`` are not guaranteed to
    be analytically tractable. In such cases, a rude deterministic estimate can be used
    in some cases even if it lacks a true algebraic meaning.
    This value will trigger a query to the ``deterministic_sample`` attribute in the distribution
    and if it does not exist, the ``mean`` will be used.

    """

    MODE = "mode"
    MEDIAN = "median"
    MEAN = "mean"
    RANDOM = "random"
    DETERMINISTIC = "deterministic"

    @classmethod
    def from_str(cls, type_str: str) -> InteractionType:
        """Return the interaction_type with name matched to the provided string (case insensitive)."""
        return cls(type_str.lower())


_interaction_type = _ContextManager()


DETERMINISTIC_REGISTER = {}

dist_has_enum_support = {}
# Iterate over all distribution classes in torch.distributions
for dist_name in dir(D):
    dist_cls = getattr(D, dist_name)

    # Check if it's a class (not a function or variable) and is a subclass of Distribution
    if isinstance(dist_cls, type) and issubclass(dist_cls, D.Distribution):
        if dist_cls is D.LogisticNormal:
            DETERMINISTIC_REGISTER[dist_cls] = InteractionType.DETERMINISTIC
        elif dist_cls.has_enumerate_support:
            DETERMINISTIC_REGISTER[dist_cls] = InteractionType.MODE
        else:
            DETERMINISTIC_REGISTER[dist_cls] = InteractionType.MEAN


DETERMINISTIC_REGISTER[Delta] = InteractionType.DETERMINISTIC
DETERMINISTIC_REGISTER[OneHotCategorical] = InteractionType.MODE

DETERMINISTIC_REGISTER[TruncatedNormal] = InteractionType.MEAN
DETERMINISTIC_REGISTER[TruncatedStandardNormal] = InteractionType.MEAN


def interaction_type() -> InteractionType | None:
    """Returns the current sampling type."""
    return _interaction_type.get_mode()


class set_interaction_type(_DecoratorContextManager):
    """Sets all ProbabilisticTDModules sampling to the desired type.

    Args:
        type (InteractionType or str): sampling type to use when the policy is being called.

    """

    def __init__(
        self, type: InteractionType | str | None = InteractionType.DETERMINISTIC
    ) -> None:
        super().__init__()
        if not isinstance(type, InteractionType) and type is not None:
            if isinstance(type, str):
                type = InteractionType(type.lower())
            else:
                raise ValueError(f"{type} is not a valid InteractionType")
        self.type = type

    def clone(self) -> set_interaction_type:
        # override this method if your children class takes __init__ parameters
        return type(self)(self.type)

    def __enter__(self) -> None:
        self.prev = _interaction_type.get_mode()
        _interaction_type.set_mode(self.type)

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        _interaction_type.set_mode(self.prev)


class ProbabilisticTensorDictModule(TensorDictModuleBase):
    """A probabilistic TD Module.

    `ProbabilisticTensorDictModule` is a non-parametric module embedding a
    probability distribution constructor. It reads the distribution parameters from an input
    TensorDict using the specified `in_keys` and outputs a sample (loosely speaking) of the
    distribution.

    The output "sample" is produced given some rule, specified by the input ``default_interaction_type``
    argument and the ``interaction_type()`` global function.

    `ProbabilisticTensorDictModule` can be used to construct the distribution
    (through the :meth:`~.get_dist` method) and/or sampling from this distribution
    (through a regular :meth:`~.__call__` to the module).

    A `ProbabilisticTensorDictModule` instance has two main features:

    - It reads and writes from and to TensorDict objects;
    - It uses a real mapping R^n -> R^m to create a distribution in R^d from
      which values can be sampled or computed.

    When the :meth:`~.__call__` and :meth:`~.forward` method are called, a distribution is
    created, and a value computed (depending on the ``interaction_type`` value, 'dist.mean',
    'dist.mode', 'dist.median' attributes could be used, as well as
    the 'dist.rsample', 'dist.sample' method). The sampling step is skipped if the supplied
    TensorDict has all the desired key-value pairs already.

    By default, `ProbabilisticTensorDictModule` distribution class is a :class:`~torchrl.modules.distributions.Delta`
    distribution, making `ProbabilisticTensorDictModule` a simple wrapper around
    a deterministic mapping function.


    Args:
        in_keys (NestedKey | List[NestedKey] | Dict[str, NestedKey]): key(s) that will be read from the input TensorDict
            and used to build the distribution.
            Importantly, if it's a list of NestedKey or a NestedKey, the leaf (last element) of those keys must match the keywords used by
            the distribution class of interest, e.g. ``"loc"`` and ``"scale"`` for
            the :class:`~torch.distributions.Normal` distribution and similar.
            If in_keys is a dictionary, the keys are the keys of the distribution and the values are the keys in the
            tensordict that will get match to the corresponding distribution keys.
        out_keys (NestedKey | List[NestedKey] | None): key(s) where the sampled values will be written.
            Importantly, if these keys are found in the input TensorDict, the sampling step will be skipped.

    Keyword Args:
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

            .. note::
                In some cases, the mode, median or mean value may not be
                readily available through the corresponding attribute.
                To paliate this, :class:`~ProbabilisticTensorDictModule` will first attempt
                to get the value through a call to ``get_mode()``, ``get_median()`` or ``get_mean()``
                if the method exists.

        distribution_class (Type or Callable[[Any], Distribution], optional): keyword-only argument.
            A :class:`torch.distributions.Distribution` class to
            be used for sampling.
            Default is :class:`~tensordict.nn.distributions.Delta`.

            .. note::
                If the distribution class is of type
                :class:`~tensordict.nn.distributions.CompositeDistribution`, the ``out_keys``
                can be inferred directly form the ``"distribution_map"`` or ``"name_map"``
                keywork arguments provided through this class' ``distribution_kwargs``
                keyword argument, making the ``out_keys`` optional in such cases.

        distribution_kwargs (dict, optional): keyword-only argument.
            Keyword-argument pairs to be passed to the distribution.

            .. note:: if your kwargs contain tensors that you would like to transfer to device with the module, or
                tensors that should see their dtype modified when calling `module.to(dtype)`, you can wrap the kwargs
                in a :class:`~tensordict.nn.TensorDictParams` to do this automatically.

        return_log_prob (bool, optional): keyword-only argument.
            If ``True``, the log-probability of the
            distribution sample will be written in the tensordict with the key
            `log_prob_key`. Default is ``False``.
        log_prob_keys (List[NestedKey], optional): keys where to write the log_prob if ``return_log_prob=True``.
            Defaults to `'<sample_key_name>_log_prob'`, where `<sample_key_name>` is each of the :attr:`out_keys`.

            .. note:: This is only available when :func:`~tensordict.nn.probabilistic.composite_lp_aggregate` is set to ``False``.

        log_prob_key (NestedKey, optional): key where to write the log_prob if ``return_log_prob=True``.
            Defaults to `'sample_log_prob'` when :func:`~tensordict.nn.probabilistic.composite_lp_aggregate` is set to `True`
            or `'<sample_key_name>_log_prob'` otherwise.

            .. note:: When there is more than one sample, this is only available when :func:`~tensordict.nn.probabilistic.composite_lp_aggregate` is set to ``True``.

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

    # To be removed in v0.9
    _trigger_warning_lpk: bool = False

    def __init__(
        self,
        in_keys: NestedKey | List[NestedKey] | Dict[str, NestedKey],
        out_keys: NestedKey | List[NestedKey] | None = None,
        *,
        default_interaction_type: InteractionType = InteractionType.DETERMINISTIC,
        distribution_class: type = Delta,
        distribution_kwargs: dict | None = None,
        return_log_prob: bool = False,
        log_prob_keys: List[NestedKey] | None = None,
        log_prob_key: NestedKey | None = None,
        cache_dist: bool = False,
        n_empirical_estimate: int = 1000,
        num_samples: int | torch.Size | None = None,
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
                name_map = distribution_kwargs.get("name_map")
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

        self._out_keys = [unravel_key(k) for k in out_keys]
        self.in_keys = in_keys
        self.dist_keys = dist_keys
        if log_prob_keys is None:
            if len(out_keys) == 1 and log_prob_key is not None:
                log_prob_keys = [log_prob_key]
            elif composite_lp_aggregate(nowarn=True):
                if len(out_keys) == 1:
                    log_prob_keys = ["sample_log_prob"]
                else:
                    log_prob_keys = None
            else:
                log_prob_keys = [
                    _add_suffix(key, "_log_prob") for key in self._out_keys
                ]
        elif composite_lp_aggregate(nowarn=True):
            raise RuntimeError(
                "composite_lp_aggregate is set to True but log_prob_keys were passed. "
                "When composite_lp_aggregate() returns ``True``, log_prob_key must be used instead."
            )
        self._trigger_warning_lpk = len(self._out_keys) > 1
        if log_prob_key is None:
            if composite_lp_aggregate(nowarn=True):
                log_prob_key = "sample_log_prob"
                self._trigger_warning_lpk = True
            elif len(out_keys) == 1:
                log_prob_key = _add_suffix(out_keys[0], "_log_prob")
        elif len(out_keys) > 1 and not composite_lp_aggregate(nowarn=True):
            raise RuntimeError(
                "composite_lp_aggregate is set to `False` but a `log_prob_key` was passed. "
                "When composite_lp_aggregate() returns ``False``, log_prob_keys must be used instead."
            )
        self._log_prob_key = log_prob_key
        self._log_prob_keys = log_prob_keys

        self.default_interaction_type = InteractionType(default_interaction_type)

        if isinstance(distribution_class, str):
            distribution_class = distributions_maps.get(distribution_class.lower())
        self.distribution_class = distribution_class
        self.distribution_kwargs = distribution_kwargs
        self.n_empirical_estimate = n_empirical_estimate
        self._dist = None
        self.cache_dist = cache_dist if hasattr(distribution_class, "update") else False
        self.return_log_prob = return_log_prob
        if isinstance(num_samples, (int, torch.SymInt)):
            num_samples = torch.Size((num_samples,))
        self.num_samples = num_samples
        self._composite_lp_aggreate_at_init = composite_lp_aggregate(nowarn=True)

    @property
    def out_keys(self) -> List[NestedKey]:
        out_keys = list(self._out_keys)
        if self.return_log_prob:
            if not composite_lp_aggregate(nowarn=True):
                if out_keys[-len(self.log_prob_keys) :] != self.log_prob_keys:
                    out_keys.extend(self.log_prob_keys)
            elif self.log_prob_key not in out_keys:
                out_keys.append(self.log_prob_key)
        return out_keys

    @property
    def log_prob_key(self):
        clpa = composite_lp_aggregate(nowarn=True)
        if clpa != self._composite_lp_aggreate_at_init:
            raise RuntimeError(
                f"composite_lp_aggregate is set to `{clpa}`, but the class was instantiated with `{self._composite_lp_aggreate_at_init}` "
                f"which may affect the log_prob_key property."
            )
        if not clpa:
            if len(self.log_prob_keys) == 1:
                return self.log_prob_keys[0]
            raise RuntimeError(
                f"composite_lp_aggregate is set to ``False``, hence {type(self).__name__}.log_prob_key cannot be accessed "
                f"unless there is one and only one element in log_prob_keys (got log_prob_keys={self.log_prob_keys}). "
                f"When composite_lp_aggregate() returns ``False``, try to use {type(self).__name__}.log_prob_keys instead."
            )
        if _composite_lp_aggregate.get_mode() is None and self._trigger_warning_lpk:
            warnings.warn(
                f"You are querying the log-probability key of a {type(self).__name__} where the "
                f"composite_lp_aggregate has not been set and the log-prob key has not been chosen. "
                f"Currently, it is assumed that composite_lp_aggregate() will return True: the log-probs will be aggregated "
                f"in a {self._log_prob_key} entry. "
                f"From v0.9, this behaviour will be changed and individual log-probs will "
                f"be written in `('path', 'to', 'leaf', '<sample_name>_log_prob')`. "
                f"To prepare for this change, "
                f"call `set_composite_lp_aggregate(mode: bool).set()` at the beginning of your script (or set the "
                f"COMPOSITE_LP_AGGREGATE env variable). Use mode=True "
                f"to keep the current behaviour, and mode=False to use per-leaf log-probs.",
                category=DeprecationWarning,
            )

        return self._log_prob_key

    @property
    def log_prob_keys(self):
        clpa = composite_lp_aggregate(nowarn=True)
        if clpa != self._composite_lp_aggreate_at_init:
            raise RuntimeError(
                f"composite_lp_aggregate is set to `{clpa}`, but the class was instantiated with `{self._composite_lp_aggreate_at_init}` "
                f"which may affect the log_prob_keys property."
            )
        if clpa:
            return [self.log_prob_key]
        return self._log_prob_keys

    @property
    def dist_params_keys(self) -> List[NestedKey]:
        """Returns all the keys pointing at the distribution params."""
        return list(self.in_keys)

    @property
    def dist_sample_keys(self) -> List[NestedKey]:
        """Returns all the keys pointing at the distribution samples."""
        return list(self._out_keys)

    def get_dist(self, tensordict: TensorDictBase) -> D.Distribution:
        """Creates a :class:`torch.distribution.Distribution` instance with the parameters provided in the input tensordict.

        Args:
            tensordict (TensorDictBase): The input tensordict containing the distribution parameters.

        Returns:
            A :class:`torch.distribution.Distribution` instance created from the input tensordict.

        Raises:
            TypeError: If the input tensordict does not match the distribution keywords.
        """
        try:
            dist_kwargs = {}
            for dist_key, td_key in _zip_strict(self.dist_keys, self.in_keys):
                if isinstance(dist_key, tuple):
                    dist_key = dist_key[-1]
                dist_kwargs[dist_key] = tensordict.get(td_key, None)
            dist = self.distribution_class(
                **dist_kwargs, **_dynamo_friendly_to_dict(self.distribution_kwargs)
            )
        except TypeError as err:
            if "an unexpected keyword argument" in str(err):
                raise TypeError(
                    "distribution keywords and tensordict keys indicated by ProbabilisticTensorDictModule.dist_keys must match. "
                    f"Got this error message: \n{indent(str(err), 4 * ' ')}\nwith dist_keys={self.dist_keys}"
                )
            elif re.search(r"missing.*required positional arguments", str(err)):
                raise TypeError(
                    f"TensorDict with keys {tensordict.keys()} does not match the distribution {self.distribution_class} keywords."
                )
            else:
                raise err
        return dist

    build_dist_from_params = get_dist

    _CHANGE_IN_C_LP_A = "The value returned by composite_lp_aggregate changed between init of {} and its execution ({} -> {}). Make sure the mode matches."

    def log_prob(
        self,
        tensordict,
        *,
        dist: torch.distributions.Distribution | None = None,
    ):
        """Computes the log-probability of the distribution sample.

        Args:
            tensordict (TensorDictBase): The input tensordict containing the distribution parameters.
            dist (torch.distributions.Distribution, optional): The distribution instance. Defaults to ``None``.
                If ``None``, the distribution will be computed using the `get_dist` method.

        Returns:
            A tensor representing the log-probability of the distribution sample.
        """
        if dist is None:
            dist = self.get_dist(tensordict)
        if isinstance(dist, CompositeDistribution):
            clpa = composite_lp_aggregate()
            if clpa != self._composite_lp_aggreate_at_init:
                raise RuntimeError(
                    self._CHANGE_IN_C_LP_A.format(
                        type(self).__name__, self._composite_lp_aggreate_at_init, clpa
                    )
                )
            if clpa:
                # Old behaviour - discouraged
                with set_composite_lp_aggregate(False):
                    td = dist.log_prob(tensordict)
                    tensordict.update(td)
                lp = sum(td.sum(dim="feature").values(True, True))
                return lp
            else:
                # Check the values within the dist - if not set, choose defaults
                lp = dist.log_prob(tensordict)
                self._update_td_lp(lp)
                return lp
        else:
            return dist.log_prob(tensordict.get(self.out_keys[0]))

    def _update_td_lp(self, lp):
        for out_key, lp_key in _zip_strict(self.dist_sample_keys, self.log_prob_keys):
            lp_key_expected = _add_suffix(out_key, "_log_prob")
            if lp_key != lp_key_expected:
                lp.rename_key_(lp_key_expected, lp_key)

    @property
    def SAMPLE_LOG_PROB_KEY(self):
        raise RuntimeError(
            "SAMPLE_LOG_PROB_KEY is fully deprecated. Use `obj.log_prob_key` instead."
        )

    @dispatch(auto_batch_size=False)
    @_set_skip_existing_None()
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
            if self.num_samples is not None:
                # TODO: capture contiguous error here
                tensordict_out = tensordict_out.expand(
                    self.num_samples + tensordict_out.shape
                )
            if isinstance(out_tensors, TensorDictBase):
                if self.return_log_prob:
                    clpa = composite_lp_aggregate()
                    if clpa != self._composite_lp_aggreate_at_init:
                        raise RuntimeError(
                            self._CHANGE_IN_C_LP_A.format(
                                type(self).__name__,
                                self._composite_lp_aggreate_at_init,
                                clpa,
                            )
                        )
                    if clpa:
                        with set_composite_lp_aggregate(False):
                            # We want the tensordict to do the sum and such
                            log_prob = dist.log_prob(out_tensors)
                        if log_prob is not out_tensors:
                            # Composite dists return the tensordict_out directly when aggrgate_prob is False
                            out_tensors.update(log_prob)
                        out_tensors.set(
                            self.log_prob_key,
                            sum(log_prob.sum(dim="feature").values(True, True)),
                        )
                    else:
                        log_prob = dist.log_prob(out_tensors)
                        out_tensors.update(log_prob)
                        self._update_td_lp(log_prob)
                tensordict_out.update(out_tensors)
            else:
                if isinstance(out_tensors, Tensor):
                    out_tensors = (out_tensors,)
                tensordict_out.update(
                    dict(_zip_strict(self.dist_sample_keys, out_tensors))
                )
                if self.return_log_prob:
                    log_prob = dist.log_prob(*out_tensors)
                    tensordict_out.set(self.log_prob_key, log_prob)
        elif self.return_log_prob:
            out_tensors = [
                tensordict.get(key)
                for key in self.out_keys
                if key not in self.log_prob_keys
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
        if not isinstance(dist, D.Distribution):
            raise TypeError("Expected Distribution, but got {}".format(type(dist)))
        if interaction_type is None:
            interaction_type = self.default_interaction_type
        if isinstance(dist, D.LKJCholesky) and interaction_type in (
            InteractionType.DETERMINISTIC,
            InteractionType.MEAN,
            InteractionType.MODE,
        ):
            raise RuntimeError(
                f"DETERMINISTIC, MEAN and MODE are not implemented for {type(dist).__name__}."
            )
        if interaction_type is InteractionType.DETERMINISTIC:
            if hasattr(dist, "deterministic_sample"):
                return dist.deterministic_sample
            else:
                # Fallbacks
                tdist = type(dist)
                if issubclass(tdist, D.Independent):
                    tdist = type(dist.base_dist)
                interaction_type = DETERMINISTIC_REGISTER.get(tdist)
                if interaction_type is None:
                    try:
                        support = dist.support
                        fallback = (
                            "mean"
                            if isinstance(support, D.constraints._Real)
                            else "mode"
                        )
                    except NotImplementedError:
                        # Some custom dists don't have a support
                        # We arbitrarily fall onto 'mean' in these cases
                        fallback = "mean"
                    try:
                        if fallback == "mean":
                            interaction_type = InteractionType.MEAN
                        elif fallback == "mode":
                            # Categorical dists don't have an average
                            interaction_type = InteractionType.MODE
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
            if hasattr(dist, "mean"):
                try:
                    return dist.mean
                except NotImplementedError:
                    pass
            if dist.has_rsample:
                return dist.rsample((self.n_empirical_estimate,)).mean(0)
            else:
                return dist.sample((self.n_empirical_estimate,)).mean(0)

        elif interaction_type is InteractionType.RANDOM:
            num_samples = self.num_samples
            if num_samples is None:
                num_samples = torch.Size(())
            if dist.has_rsample:
                return dist.rsample(num_samples)
            else:
                return dist.sample(num_samples)
        else:
            raise NotImplementedError(f"unknown interaction_type {interaction_type}")


class ProbabilisticTensorDictSequential(TensorDictSequential):
    """A sequence of :class:`~tensordict.nn.TensorDictModules` containing at least one :class:`~tensordict.nn.ProbabilisticTensorDictModule`.

    This class extends :class:`~tensordict.nn.TensorDictSequential` and is typically configured with a sequence of
    modules where the final module is an instance of :class:`~tensordict.nn.ProbabilisticTensorDictModule`.
    However, it also supports configurations where one or more intermediate modules are instances of
    :class:`~tensordict.nn.ProbabilisticTensorDictModule`, while the last module may or may not be probabilistic.
    In all cases, it exposes the :meth:`~.get_dist` method to recover the distribution object from the
    :class:`~tensordict.nn.ProbabilisticTensorDictModule` instances in the sequence.

    Multiple probabilistic modules can co-exist in a single ``ProbabilisticTensorDictSequential``.
    If `return_composite` is ``False`` (default), only the last one will produce a distribution and the others
    will be executed as regular :class:`~tensordict.nn.TensorDictModule` instances.
    However, if a `ProbabilisticTensorDictModule` is not the last module in the sequence and `return_composite=False`,
    a `ValueError` will be raised when trying to query the module. If `return_composite=True`,
    all intermediate `ProbabilisticTensorDictModule` instances will contribute to a single
    :class:`~tensordict.nn.CompositeDistribution` instance.

    Resulting log-probabilities will be conditional probabilities if samples are interdependent:
    whenever

        .. math::
            Z = F(X, Y)

    then the log-probability of Z will be

        .. math::
            log(p(z | x, y))

    Args:
        *modules (sequence or OrderedDict of TensorDictModuleBase or ProbabilisticTensorDictModule): An ordered sequence of
            :class:`~tensordict.nn.TensorDictModule` instances, usually terminating in a :class:`~tensordict.nn.ProbabilisticTensorDictModule`,
            to be run sequentially.
            The modules can be instances of TensorDictModuleBase or any other function that matches this signature.
            Note that if a non-TensorDictModuleBase callable is used, its input and output keys will not be tracked,
            and thus will not affect the `in_keys` and `out_keys` attributes of the TensorDictSequential.

    Keyword Args:
        partial_tolerant (bool, optional): If ``True``, the input tensordict can miss some
            of the input keys. If so, only the modules that can be executed given the
            keys that are present will be executed. Also, if the input tensordict is a
            lazy stack of tensordicts AND if partial_tolerant is ``True`` AND if the stack
            does not have the required keys, then TensorDictSequential will scan through
            the sub-tensordicts looking for those that have the required keys, if any.
            Defaults to ``False``.
        return_composite (bool, optional): If True and multiple
            :class:`~tensordict.nn.ProbabilisticTensorDictModule` or
            :class:`~tensordict.nn.ProbabilisticTensorDictSequential` instances are found,
            a :class:`~tensordict.nn.CompositeDistribution` instance will be used.
            Otherwise, only the last module will be used to build the distribution.
            Defaults to ``False``.

            .. warning:: The behaviour of :attr:`return_composite` will change in v0.9
                and default to True from there on.
        inplace (bool, optional): if `True`, the input tensordict is modified in-place. If `False`, a new empty
            :class:`~tensordict.TensorDict` instance is created. If `"empty"`, `input.empty()` is used instead (ie, the
            output preserves type, device and batch-size). Defaults to `None` (relies on sub-modules).

    Raises:
        ValueError: If the input sequence of modules is empty.
        TypeError: If the final module is not an instance of
            :obj:`ProbabilisticTensorDictModule` or
            :obj:`ProbabilisticTensorDictSequential`.

    Examples:
        >>> from tensordict.nn import ProbabilisticTensorDictModule as Prob, ProbabilisticTensorDictSequential as Seq
        >>> import torch
        >>> # Typical usage: a single distribution is computed last in the sequence
        >>> import torch
        >>> from tensordict import TensorDict
        >>> from tensordict.nn import ProbabilisticTensorDictModule as Prob, ProbabilisticTensorDictSequential as Seq, \
        ...     TensorDictModule as Mod
        >>> torch.manual_seed(0)
        >>>
        >>> module = Seq(
        ...     Mod(lambda x: x + 1, in_keys=["x"], out_keys=["loc"]),
        ...     Prob(in_keys=["loc"], out_keys=["sample"], distribution_class=torch.distributions.Normal,
        ...          distribution_kwargs={"scale": 1}),
        ... )
        >>> input = TensorDict(x=torch.ones(3))
        >>> td = module(input.copy())
        >>> print(td)
        TensorDict(
            fields={
                loc: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False),
                sample: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False),
                x: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False)},
            batch_size=torch.Size([]),
            device=None,
            is_shared=False)
        >>> print(module.get_dist(input))
        Normal(loc: torch.Size([3]), scale: torch.Size([3]))
        >>> print(module.log_prob(td))
        tensor([-0.9189, -0.9189, -0.9189])
        >>> # Intermediate distributions are ignored when return_composite=False
        >>> module = Seq(
        ...     Mod(lambda x: x + 1, in_keys=["x"], out_keys=["loc"]),
        ...     Prob(in_keys=["loc"], out_keys=["sample0"], distribution_class=torch.distributions.Normal,
        ...          distribution_kwargs={"scale": 1}),
        ...     Mod(lambda x: x + 1, in_keys=["sample0"], out_keys=["loc2"]),
        ...     Prob(in_keys={"loc": "loc2"}, out_keys=["sample1"], distribution_class=torch.distributions.Normal,
        ...          distribution_kwargs={"scale": 1}),
        ...     return_composite=False,
        ... )
        >>> td = module(TensorDict(x=torch.ones(3)))
        >>> print(td)
        TensorDict(
            fields={
                loc2: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False),
                loc: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False),
                sample0: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False),
                sample1: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False),
                x: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False)},
            batch_size=torch.Size([]),
            device=None,
            is_shared=False)
        >>> print(module.get_dist(input))
        Normal(loc: torch.Size([3]), scale: torch.Size([3]))
        >>> print(module.log_prob(td))
        tensor([-0.9189, -0.9189, -0.9189])
        >>> # Intermediate distributions produce a CompositeDistribution when return_composite=True
        >>> module = Seq(
        ...     Mod(lambda x: x + 1, in_keys=["x"], out_keys=["loc"]),
        ...     Prob(in_keys=["loc"], out_keys=["sample0"], distribution_class=torch.distributions.Normal,
        ...          distribution_kwargs={"scale": 1}),
        ...     Mod(lambda x: x + 1, in_keys=["sample0"], out_keys=["loc2"]),
        ...     Prob(in_keys={"loc": "loc2"}, out_keys=["sample1"], distribution_class=torch.distributions.Normal,
        ...          distribution_kwargs={"scale": 1}),
        ...     return_composite=True,
        ... )
        >>> input = TensorDict(x=torch.ones(3))
        >>> td = module(input.copy())
        >>> print(td)
        TensorDict(
            fields={
                loc2: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False),
                loc: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False),
                sample0: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False),
                sample1: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False),
                x: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False)},
            batch_size=torch.Size([]),
            device=None,
            is_shared=False)
        >>> print(module.get_dist(input))
        CompositeDistribution({'sample0': Normal(loc: torch.Size([3]), scale: torch.Size([3])), 'sample1': Normal(loc: torch.Size([3]), scale: torch.Size([3]))})
        >>> print(module.log_prob(td))
        TensorDict(
            fields={
                sample0_log_prob: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False),
                sample1_log_prob: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False)},
            batch_size=torch.Size([]),
            device=None,
            is_shared=False)
        >>> # Even a single intermediate distribution is wrapped in a CompositeDistribution when
        >>> # return_composite=True
        >>> module = Seq(
        ...     Mod(lambda x: x + 1, in_keys=["x"], out_keys=["loc"]),
        ...     Prob(in_keys=["loc"], out_keys=["sample0"], distribution_class=torch.distributions.Normal,
        ...          distribution_kwargs={"scale": 1}),
        ...     Mod(lambda x: x + 1, in_keys=["sample0"], out_keys=["y"]),
        ...     return_composite=True,
        ... )
        >>> td = module(TensorDict(x=torch.ones(3)))
        >>> print(td)
        TensorDict(
            fields={
                loc: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False),
                sample0: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False),
                x: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False),
                y: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False)},
            batch_size=torch.Size([]),
            device=None,
            is_shared=False)
        >>> print(module.get_dist(input))
        CompositeDistribution({'sample0': Normal(loc: torch.Size([3]), scale: torch.Size([3]))})
        >>> print(module.log_prob(td))
        TensorDict(
            fields={
                sample0_log_prob: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False)},
            batch_size=torch.Size([]),
            device=None,
            is_shared=False)

    """

    @overload
    def __init__(
        self,
        modules: OrderedDict[str, TensorDictModuleBase | ProbabilisticTensorDictModule],
        partial_tolerant: bool = False,
        return_composite: bool | None = None,
        *,
        inplace: bool | None = None,
    ) -> None: ...

    @overload
    def __init__(
        self,
        modules: List[TensorDictModuleBase | ProbabilisticTensorDictModule],
        partial_tolerant: bool = False,
        return_composite: bool | None = None,
        *,
        inplace: bool | None = None,
    ) -> None: ...

    def __init__(
        self,
        *modules: TensorDictModuleBase | ProbabilisticTensorDictModule,
        partial_tolerant: bool = False,
        return_composite: bool | None = None,
        inplace: bool | None = None,
    ) -> None:
        if len(modules) == 0:
            raise ValueError(
                "ProbabilisticTensorDictSequential must consist of zero or more "
                "TensorDictModules followed by a ProbabilisticTensorDictModule"
            )
        self._ordered_dict = False
        if len(modules) == 1 and isinstance(
            modules[0], (collections.OrderedDict, MutableSequence)
        ):
            if isinstance(modules[0], collections.OrderedDict):
                modules_list = list(modules[0].values())
                self._ordered_dict = True
            else:
                modules = modules_list = list(modules[0])
        elif len(modules) == 1 and isinstance(modules[0], dict):
            modules = [collections.OrderedDict(modules[0])]
            return self.__init__(
                *modules,
                partial_tolerant=partial_tolerant,
                return_composite=return_composite,
                inplace=inplace,
            )
        elif not return_composite and not isinstance(
            modules[-1],
            (ProbabilisticTensorDictModule, ProbabilisticTensorDictSequential),
        ):
            raise TypeError(
                "The final module passed to ProbabilisticTensorDictSequential must be "
                "an instance of ProbabilisticTensorDictModule or another "
                "ProbabilisticTensorDictSequential (unless return_composite is set to ``True``)."
            )
        else:
            modules_list = list(modules)
        modules_list = self._convert_modules(modules_list)

        # if the modules not including the final probabilistic module return the sampled
        # key we won't be sampling it again, in that case
        # ProbabilisticTensorDictSequential is presumably used to return the
        # distribution using `get_dist` or to sample log_probabilities
        _, out_keys = self._compute_in_and_out_keys(modules_list[:-1])
        self._requires_sample = any(
            key not in set(out_keys)
            for key in getattr(modules_list[-1], "dist_sample_keys", [None])
        )
        if self._ordered_dict:
            self.__dict__["_det_part"] = TensorDictSequential(
                collections.OrderedDict(list(modules[0].items())[:-1])
            )
        else:
            self.__dict__["_det_part"] = TensorDictSequential(*modules[:-1])

        super().__init__(*modules, partial_tolerant=partial_tolerant, inplace=inplace)
        self.return_composite = return_composite

    def __getitem__(self, index: int | slice | str) -> TensorDictModuleBase:
        if isinstance(index, (int, str)):
            return self.module.__getitem__(index)
        else:
            mods = self.module.__getitem__(index)
            if self.return_composite and any(
                isinstance(
                    item,
                    (ProbabilisticTensorDictModule, ProbabilisticTensorDictSequential),
                )
                for item in mods
            ):
                return type(self)(*mods, return_composite=self.return_composite)
            elif isinstance(
                mods[-1],
                (ProbabilisticTensorDictModule, ProbabilisticTensorDictSequential),
            ):
                return type(self)(*mods)
            else:
                return TensorDictSequential(*mods)

    _dist_sample = ProbabilisticTensorDictModule._dist_sample

    @property
    def det_part(self):
        return self._det_part

    def get_dist_params(
        self,
        tensordict: TensorDictBase,
        tensordict_out: TensorDictBase | None = None,
        **kwargs,
    ) -> tuple[D.Distribution, TensorDictBase]:
        """Returns the distribution parameters and output tensordict.

        This method runs the deterministic part of the :class:`~tensordict.nn.ProbabilisticTensorDictSequential`
        module to obtain the distribution parameters. The interaction type is set to the current global
        interaction type if available, otherwise it defaults to the interaction type of the last module.

        Args:
            tensordict (TensorDictBase): The input tensordict.
            tensordict_out (TensorDictBase, optional): The output tensordict. If ``None``, a new tensordict will be created.
                Defaults to ``None``.

        Keyword Args:
            **kwargs: Additional keyword arguments passed to the deterministic part of the module.

        Returns:
            tuple[D.Distribution, TensorDictBase]: A tuple containing the distribution object and the output tensordict.

        .. note:: The interaction type is temporarily set to the specified value during the execution of this method.
        """
        tds = self.det_part
        type = interaction_type()
        if type is None:
            for m in reversed(list(self._module_iter())):
                if hasattr(m, "default_interaction_type"):
                    type = m.default_interaction_type
                    break
            else:
                raise ValueError("Could not find a default interaction in the modules.")
        with set_interaction_type(type):
            return tds(tensordict, tensordict_out, **kwargs)

    @property
    def log_prob_keys(self):
        lpks = []
        for m in reversed(list(self._module_iter())):
            lpks.extend(getattr(m, "log_prob_keys", []))
        return lpks

    log_prob_key = ProbabilisticTensorDictModule.log_prob_key

    @property
    def dist_params_keys(self) -> List[NestedKey]:
        """Returns all the keys pointing at the distribution params."""
        result = []
        for m in reversed(list(self._module_iter())):
            result.extend(getattr(m, "dist_params_keys", []))
        return result

    @property
    def dist_sample_keys(self) -> List[NestedKey]:
        """Returns all the keys pointing at the distribution samples."""
        result = []
        for m in reversed(list(self._module_iter())):
            result.extend(getattr(m, "dist_sample_keys", []))
        return result

    @property
    def num_samples(self):
        num_samples = ()
        for tdm in self._module_iter():
            if isinstance(
                tdm, (ProbabilisticTensorDictModule, ProbabilisticTensorDictSequential)
            ):
                num_samples = tdm.num_samples + num_samples
        return num_samples

    def get_dist(
        self,
        tensordict: TensorDictBase,
        tensordict_out: TensorDictBase | None = None,
        **kwargs,
    ) -> D.Distribution:
        """Returns the distribution resulting from passing the input tensordict through the sequence.

        If `return_composite` is ``False`` (default), this method will only consider the last probabilistic module in the sequence.

        Otherwise, it will return a :class:`~tensordict.nn.CompositeDistribution` instance containing the distributions of all probabilistic modules.

        Args:
            tensordict (TensorDictBase): The input tensordict.
            tensordict_out (TensorDictBase, optional): The output tensordict. If ``None``, a new tensordict will be created.
                Defaults to ``None``.

        Keyword Args:
            **kwargs: Additional keyword arguments passed to the underlying modules.

        Returns:
            D.Distribution: The resulting distribution object.

        Raises:
            RuntimeError: If no probabilistic module is found in the sequence.

        .. note::
            When `return_composite` is ``True``, the distributions are conditioned on the previous samples in the sequence.
            This means that if a module depends on the output of a previous probabilistic module, its distribution will be conditional.

        """
        if not self.return_composite:
            tensordict_out = self.get_dist_params(tensordict, tensordict_out, **kwargs)
            return self.build_dist_from_params(tensordict_out)

        td_copy = tensordict.copy()
        dists = {}
        for i, tdm in enumerate(self._module_iter()):
            if isinstance(
                tdm, (ProbabilisticTensorDictModule, ProbabilisticTensorDictSequential)
            ):
                dist = tdm.get_dist(td_copy)
                if i < len(self.module) - 1:
                    sample = tdm._dist_sample(dist, interaction_type=interaction_type())
                    if tdm.num_samples not in ((), None):
                        td_copy = td_copy.expand(tdm.num_samples + td_copy.shape)
                    if isinstance(tdm, ProbabilisticTensorDictModule):
                        if isinstance(sample, torch.Tensor):
                            sample = [sample]
                        td_copy.update(dict(_zip_strict(tdm.dist_sample_keys, sample)))
                    else:
                        td_copy.update(sample)
                dists[tdm.out_keys[0]] = dist
            else:
                td_copy = tdm(td_copy)
        if len(dists) == 0:
            raise RuntimeError(f"No distribution module found in {self}.")
        # elif len(dists) == 1:
        #     return dist
        return CompositeDistribution.from_distributions(
            td_copy,
            dists,
        )

    @property
    def default_interaction_type(self):
        """Returns the `default_interaction_type` of the module using an iterative heuristic.

        This property iterates over all modules in reverse order, attempting to retrieve the
        `default_interaction_type` attribute from any child module. The first non-None value
        encountered is returned. If no such value is found, a default `interaction_type()` is returned.

        """
        for m in reversed(list(self._module_iter())):
            interaction = getattr(m, "default_interaction_type", None)
            if interaction is not None:
                return interaction
        return interaction_type()

    @property
    def _last_module(self):
        if not self._ordered_dict:
            return self.module[-1]
        mod = None
        for mod in self._module_iter():  # noqa: B007
            continue
        return mod

    def log_prob(
        self,
        tensordict,
        tensordict_out: TensorDictBase | None = None,
        *,
        dist: torch.distributions.Distribution | None = None,
        **kwargs,
    ) -> TensorDictBase | torch.Tensor:
        """Returns the log-probability of the input tensordict.

        If `self.return_composite` is ``True`` and the distribution is a :class:`~tensordict.nn.CompositeDistribution`,
        this method will return the log-probability of the entire composite distribution.

        Otherwise, it will only consider the last probabilistic module in the sequence.

        Args:
            tensordict (TensorDictBase): The input tensordict.
            tensordict_out (TensorDictBase, optional): The output tensordict. If ``None``, a new tensordict will be created.
                Defaults to ``None``.

        Keyword Args:
            dist (torch.distributions.Distribution, optional): The distribution object. If ``None``, it will be computed using `get_dist`.
                Defaults to ``None``.

        Returns:
            TensorDictBase or torch.Tensor: The log-probability of the input tensordict.

        .. warning::
            In future releases (v0.9), the default values of `aggregate_probabilities`, `inplace`, and `include_sum` will change.
            To avoid warnings, it is recommended to explicitly pass these arguments to the `log_prob` method or set them in the constructor.

        """
        if tensordict_out is not None:
            tensordict_inp = tensordict.copy()
        else:
            tensordict_inp = tensordict
        if dist is None:
            dist = self.get_dist(tensordict_inp)
        return_composite = self.return_composite
        if return_composite and isinstance(dist, CompositeDistribution):
            # Check the values within the dist - if not set, choose defaults
            return dist.log_prob(
                tensordict,
                **kwargs,
            )
        last_module: ProbabilisticTensorDictModule = self._last_module
        out = last_module.log_prob(tensordict_inp, dist=dist, **kwargs)
        if is_tensor_collection(out):
            if tensordict_out is not None:
                if out is tensordict_inp:
                    tensordict_out.update(
                        tensordict_inp.apply(
                            lambda x, y: x if x is not y else None,
                            tensordict,
                            filter_empty=True,
                        )
                    )
                else:
                    tensordict_out.update(out)
            else:
                tensordict_out = out
            return tensordict_out
        return out

    def build_dist_from_params(self, tensordict: TensorDictBase) -> D.Distribution:
        """Constructs a distribution from the input parameters without evaluating other modules in the sequence.

        This method searches for the last :class:`~tensordict.nn.ProbabilisticTensorDictModule` in the sequence and uses it to build the distribution.

        Args:
            tensordict (TensorDictBase): The input tensordict containing the distribution parameters.

        Returns:
            D.Distribution: The constructed distribution object.

        Raises:
            RuntimeError: If no :class:`~tensordict.nn.ProbabilisticTensorDictModule` is found in the sequence.
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
    @_set_skip_existing_None()
    def forward(
        self,
        tensordict: TensorDictBase,
        tensordict_out: TensorDictBase | None = None,
        **kwargs,
    ) -> TensorDictBase:
        if (tensordict_out is None and self._select_before_return) or (
            tensordict_out is not None
        ):
            tensordict_exec = tensordict.copy()
        else:
            tensordict_exec = tensordict
        if self.return_composite:
            for m in self._module_iter():
                try:
                    if isinstance(
                        m,
                        (ProbabilisticTensorDictModule, ProbabilisticTensorDictModule),
                    ):
                        tensordict_exec = m(
                            tensordict_exec, _requires_sample=self._requires_sample
                        )
                    else:
                        tensordict_exec = m(tensordict_exec, **kwargs)
                except Exception as e:
                    module_num_or_key = self._get_module_num_or_key(m)
                    raise RuntimeError(
                        f"Failed while executing module '{module_num_or_key}'. Scroll up for more info."
                    ) from e
        else:
            tensordict_exec = self.get_dist_params(tensordict_exec, **kwargs)
            tensordict_exec = self._last_module(
                tensordict_exec, _requires_sample=self._requires_sample
            )

        if self.inplace is True:
            tensordict_out = tensordict
        elif self.inplace is False:
            tensordict_out = TensorDict()
        elif self.inplace == "empty":
            tensordict_out = tensordict.empty()

        if tensordict_out is not None:
            result = tensordict_out
            result.update(tensordict_exec, keys_to_update=self.out_keys)
        else:
            result = tensordict_exec
            if self._select_before_return:
                # We must also update any value that has been updated during the course of execution
                # from the input data.
                if is_compiling():
                    keys = [  # noqa: C416
                        k
                        for k in {k for k in self.out_keys}.union(  # noqa: C416
                            {k for k in tensordict.keys(True, True)}  # noqa: C416
                        )
                    ]
                else:
                    keys = list(set(self.out_keys + list(tensordict.keys(True, True))))
                return tensordict.update(result, keys_to_update=keys)
        return result


def _dynamo_friendly_to_dict(data):
    if not is_compiling():
        return data
    if isinstance(data, TensorDictBase):
        # to_dict is recursive and we don't want that
        items = dict(data.items())
        for k, v in items.items():
            if is_non_tensor(v):
                items[k] = v.data
        return items
    return data
