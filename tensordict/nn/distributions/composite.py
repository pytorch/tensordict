# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import warnings
from collections.abc import Mapping
from typing import Any, Dict, TYPE_CHECKING

import torch
from tensordict._td import TensorDict
from tensordict.base import TensorDictBase
from tensordict.nn.utils import composite_lp_aggregate, set_composite_lp_aggregate
from tensordict.utils import IndexType, NestedKey, unravel_key, unravel_keys
from torch import distributions as d

if TYPE_CHECKING:
    from typing import Self
else:
    Self = Any


class CompositeDistribution(d.Distribution, Mapping):
    """A composite distribution that groups multiple distributions together using the TensorDict interface.

    This class allows for operations such as `log_prob_composite`, `entropy_composite`, `cdf`, `icdf`, `rsample`, and `sample`
    to be performed on a collection of distributions, returning a TensorDict. The input TensorDict may be modified in-place.

    Args:
        params (TensorDictBase): A nested key-tensor map where the root entries correspond to sample names, and the leaves
            are the distribution parameters. Entry names must match those specified in `distribution_map`.
        distribution_map (Dict[NestedKey, Type[torch.distribution.Distribution]]): Specifies the distribution types to be used.
            The names of the distributions should match the sample names in the `TensorDict`.

    Keyword Arguments:
        name_map (Dict[NestedKey, NestedKey], optional): A mapping of where each sample should be written. If not provided,
            the key names from `distribution_map` will be used.
        extra_kwargs (Dict[NestedKey, Dict], optional): A dictionary of additional keyword arguments for constructing the distributions.
        log_prob_key (NestedKey, optional): The key where the aggregated log probability will be stored.
            Defaults to `'sample_log_prob'`.

            .. note:: if :func:`tensordict.nn.probabilistic.composite_lp_aggregate` returns ``False``, tbe log-probabilities will
                be written under `("path", "to", "leaf", "<sample_name>_log_prob")`
                where `("path", "to", "leaf", "<sample_name>")` is the :class:`~tensordict.NestedKey` corresponding to
                the leaf tensor being sampled. In that case, the ``log_prob_key`` argument will be ignored.

        entropy_key (NestedKey, optional): The key where the entropy will be stored. Defaults to `'entropy'`

            .. note:: if :func:`tensordict.nn.probabilistic.composite_lp_aggregate` returns ``False``, tbe entropies will
                be written under `("path", "to", "leaf", "<sample_name>_entropy")`
                where `("path", "to", "leaf", "<sample_name>")` is the :class:`~tensordict.NestedKey` corresponding to
                the leaf tensor being sampled. In that case, the ``entropy_key`` argument will be ignored.

    .. note:: The batch size of the input TensorDict containing the parameters (`params`) determines the batch shape of
        the distribution. For example, the `"sample_log_prob"` entry resulting from a call to `log_prob` will have the
        shape of the parameters plus any additional batch dimensions.

    .. seealso:: :class:`~tensordict.nn.ProbabilisticTensorDictModule` and :class:`~tensordict.nn.ProbabilisticTensorDictSequential`
        to learn how to use this class as part of a model.

    .. seealso:: :class:`~tensordict.nn.set_composite_lp_aggregate` to control the aggregation of the log-probabilities.

    Examples:
        >>> params = TensorDict({
        ...     "cont": {"loc": torch.randn(3, 4), "scale": torch.rand(3, 4)},
        ...     ("nested", "disc"): {"logits": torch.randn(3, 10)}
        ... }, [3])
        >>> dist = CompositeDistribution(params,
        ...     distribution_map={"cont": d.Normal, ("nested", "disc"): d.Categorical})
        >>> sample = dist.sample((4,))
        >>> with set_composite_lp_aggregate(False):
        ...     sample = dist.log_prob(sample)
        ...     print(sample)
        TensorDict(
            fields={
                cont: Tensor(shape=torch.Size([4, 3, 4]), device=cpu, dtype=torch.float32, is_shared=False),
                cont_log_prob: Tensor(shape=torch.Size([4, 3, 4]), device=cpu, dtype=torch.float32, is_shared=False),
                nested: TensorDict(
                    fields={
                        disc: Tensor(shape=torch.Size([4, 3]), device=cpu, dtype=torch.int64, is_shared=False),
                        disc_log_prob: Tensor(shape=torch.Size([4, 3]), device=cpu, dtype=torch.float32, is_shared=False)},
                    batch_size=torch.Size([4]),
                    device=None,
                    is_shared=False)},
            batch_size=torch.Size([4]),
            device=None,
            is_shared=False)

    """

    def __init__(
        self,
        params: TensorDictBase,
        distribution_map: dict,
        *,
        name_map: dict | None = None,
        extra_kwargs=None,
        log_prob_key: NestedKey | None = None,
        entropy_key: NestedKey | None = None,
    ):
        self._batch_shape = params.shape
        if extra_kwargs is None:
            extra_kwargs = {}
        dists = {}
        if name_map is not None:
            name_map = {
                unravel_key(key): unravel_key(other_key)
                for key, other_key in name_map.items()
            }
        for name, dist_class in distribution_map.items():
            name_unravel = unravel_key(name)
            if name_map:
                try:
                    write_name = unravel_key(name_map.get(name, name_unravel))
                except KeyError:
                    raise KeyError(
                        f"Failed to retrieve the key {name} from the name_map with keys {name_map.keys()}."
                    )
            else:
                write_name = name_unravel
            name = name_unravel
            dist_params = params.get(name)
            kwargs = extra_kwargs.get(name, {})
            if dist_params is None:
                raise KeyError(
                    f"no param {name} found in params with keys {params.keys(True, True)}"
                )
            dist = dist_class(**dist_params, **kwargs)
            dists[write_name] = dist
        self.dists = dists
        self.log_prob_key = log_prob_key
        self.entropy_key = entropy_key

    def __iter__(self):
        yield from self.dists

    def __getitem__(self, item: IndexType) -> Self:
        return self.dists[item]

    def __len__(self):
        return len(self.dists)

    @property
    def log_prob_key(self):
        log_prob_key = self._log_prob_key
        if log_prob_key is None:
            log_prob_key = "sample_log_prob"
        return log_prob_key

    @log_prob_key.setter
    def log_prob_key(self, value):
        self._log_prob_key = value

    @property
    def entropy_key(self):
        entropy_key = self._entropy_key
        if entropy_key is None:
            entropy_key = "entropy"
        return entropy_key

    @entropy_key.setter
    def entropy_key(self, value):
        self._entropy_key = value

    @classmethod
    def from_distributions(
        cls,
        params,
        distributions: Dict[NestedKey, d.Distribution],
        *,
        name_map: dict | None = None,
        log_prob_key: NestedKey | None = None,
        entropy_key: NestedKey | None = None,
    ) -> CompositeDistribution:
        """Create a `CompositeDistribution` instance from existing distribution objects.

        This class method allows for the creation of a `CompositeDistribution` by directly providing
        a dictionary of distribution instances, rather than specifying distribution types and parameters separately.

        Args:
            params (TensorDictBase): A TensorDict that defines the batch shape for the composite distribution.
                The params will not be used by this method, but the tensordict will be used to gather the key names of
                the distributions.
            distributions (Dict[NestedKey, d.Distribution]): A dictionary mapping nested keys to distribution instances.
                These distributions will be used directly in the composite distribution.

        Keyword Args:
            name_map (Dict[NestedKey, NestedKey], optional): A mapping of where each sample should be written. If not provided,
                the key names from `distribution_map` will be used.
            log_prob_key (NestedKey, optional): The key where the log probability will be stored.
                Defaults to `'sample_log_prob'`.
            entropy_key (NestedKey, optional): The key where the entropy will be stored. Defaults to `'entropy'`.

        Returns:
            CompositeDistribution: An instance of `CompositeDistribution` initialized with the provided distributions.

        Raises:
            KeyError: If a key in `name_map` cannot be found in the provided distributions.

        .. note:: The batch size of the `params` TensorDict determines the batch shape of the composite distribution.

        Example:
            >>> from tensordict.nn import CompositeDistribution, ProbabilisticTensorDictSequential, ProbabilisticTensorDictModule, TensorDictModule
            >>> import torch
            >>> from tensordict import TensorDict
            >>>
            >>> # Values are not used to build the dists
            >>> params = TensorDict({("0", "loc"): None, ("1", "loc"): None, ("0", "scale"): None, ("1", "scale"): None})
            >>> d0 = torch.distributions.Normal(0, 1)
            >>> d1 = torch.distributions.Normal(torch.zeros(1, 2), torch.ones(1, 2))
            >>>
            >>> d = CompositeDistribution.from_distributions(params, {"0": d0, "1": d1})
            >>> print(d.sample())
            TensorDict(
                fields={
                    0: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
                    1: Tensor(shape=torch.Size([1, 2]), device=cpu, dtype=torch.float32, is_shared=False)},
                batch_size=torch.Size([]),
                device=None,
                is_shared=False)
        """
        self = cls.__new__(cls)
        self._batch_shape = params.shape
        dists = {}
        if name_map is not None:
            name_map = {
                unravel_key(key): unravel_key(other_key)
                for key, other_key in name_map.items()
            }
        for name, dist in distributions.items():
            name_unravel = unravel_key(name)
            if name_map:
                try:
                    write_name = unravel_key(name_map.get(name, name_unravel))
                except KeyError:
                    raise KeyError(
                        f"Failed to retrieve the key {name} from the name_map with keys {name_map.keys()}."
                    )
            else:
                write_name = name_unravel
            dists[write_name] = dist
        self.dists = dists
        self.log_prob_key = log_prob_key
        self.entropy_key = entropy_key
        self.aggregate_probabilities = None

        return self

    def sample(self, shape=None) -> TensorDictBase:
        if shape is None:
            shape = torch.Size([])
        samples = {name: dist.sample(shape) for name, dist in self.dists.items()}
        return TensorDict(
            samples,
            shape + self.batch_shape,
        )

    @property
    def mode(self) -> TensorDictBase:
        samples = {name: dist.mode for name, dist in self.dists.items()}
        return TensorDict(
            samples,
            self.batch_shape,
        )

    @property
    def mean(self) -> TensorDictBase:
        samples = {name: dist.mean for name, dist in self.dists.items()}
        return TensorDict(
            samples,
            self.batch_shape,
        )

    @property
    def deterministic_sample(self) -> TensorDictBase:
        def maybe_deterministic_sample(dist):
            if hasattr(dist, "deterministic_sample"):
                return dist.deterministic_sample
            else:
                from tensordict.nn.probabilistic import DETERMINISTIC_REGISTER

                # Fallbacks
                tdist = type(dist)
                if issubclass(tdist, d.Independent):
                    tdist = type(dist.base_dist)
                interaction_type = DETERMINISTIC_REGISTER.get(tdist)
                if interaction_type == "mode":
                    return dist.mode
                if interaction_type == "mean":
                    return dist.mean
                if interaction_type == "random":
                    return dist.rsample() if dist.has_rsample else dist.sample()
                if interaction_type is None:
                    try:
                        support = dist.support
                        fallback = (
                            "mean"
                            if isinstance(support, d.constraints._Real)
                            else "mode"
                        )
                    except NotImplementedError:
                        # Some custom dists don't have a support
                        # We arbitrarily fall onto 'mean' in these cases
                        fallback = "mean"
                else:
                    raise RuntimeError(
                        f"InteractionType {interaction_type} is unaccounted for."
                    )
                try:
                    if fallback == "mean":
                        return dist.mean
                    elif fallback == "mode":
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

        samples = {
            name: maybe_deterministic_sample(dist) for name, dist in self.dists.items()
        }
        return TensorDict(
            samples,
            self.batch_shape,
        )

    def __repr__(self):
        return f"{type(self).__name__}({self.dists})"

    def rsample(self, shape=None) -> TensorDictBase:
        if shape is None:
            shape = torch.Size([])
        return TensorDict(
            {name: dist.rsample(shape) for name, dist in self.dists.items()},
            shape + self.batch_shape,
        )

    def log_prob(
        self, sample: TensorDictBase
    ) -> torch.Tensor | TensorDictBase:  # noqa: D417
        """Compute the summed log-probability of a given sample.

        Args:
            sample (TensorDictBase): The input sample to compute the log probability for.

        If ``self.aggregate_probabilities`` is ``True``, this method will return a single tensor with
        the summed log-probabilities. If ``self.aggregate_probabilities`` is ``False``, this method will
        call the `:meth:`~.log_prob_composite` method and return a tensordict with the log-probabilities
        of each sample in the input tensordict along with a ``sample_log_prob`` entry with the summed
        log-prob. In both cases, the output shape will be the shape of the input tensordict.
        """
        aggregate_probabilities = composite_lp_aggregate()
        if not aggregate_probabilities:
            with set_composite_lp_aggregate(False):
                return self.log_prob_composite(sample)
        slp = 0.0
        for name, dist in self.dists.items():
            local_sample = sample.get(name)
            lp = dist.log_prob(local_sample)
            if lp.ndim > sample.ndim:
                lp = lp.flatten(sample.ndim, -1).sum(-1)
            slp = slp + lp
        return slp

    def log_prob_composite(
        self,
        sample: TensorDictBase,
        *,
        include_sum: bool | None = None,
    ) -> TensorDictBase:
        """Computes the log-probability of each component in the input sample and return a TensorDict with individual log-probabilities.

        Args:
            sample (TensorDictBase): The input sample to compute the log probabilities for.

        Keyword Args:
            include_sum (bool, optional): Whether to include the summed log-probability in the output TensorDict.
                Defaults to ``composite_lp_aggregate()`` (``False`` by default).

        Returns:
            TensorDictBase: A TensorDict containing the individual log-probabilities for each component in the input sample,
                along with a "sample_log_prob" entry containing the summed log-probability if `include_sum` is True.
        """
        if include_sum is None:
            include_sum = composite_lp_aggregate()
        inplace = composite_lp_aggregate()

        if include_sum:
            slp = 0.0
        d = {}
        for name, dist in self.dists.items():
            try:
                d[_add_suffix(name, "_log_prob")] = lp = dist.log_prob(sample.get(name))
            except AttributeError:
                raise RuntimeError(
                    f"Expected a tensordict sample, but got a {type(sample).__name__} instead."
                )
            if include_sum:
                if lp.ndim > sample.ndim:
                    lp = lp.flatten(sample.ndim, -1).sum(-1)
                slp = slp + lp
        if include_sum:
            d[self.log_prob_key] = slp
        if inplace:
            return sample.update(d)
        else:
            return sample.empty(recurse=True).update(d).filter_empty_()

    def entropy(
        self,
        samples_mc: int = 1,
        *,
        aggregate_probabilities: bool | None = None,
        include_sum: bool | None = None,
    ) -> torch.Tensor | TensorDictBase:  # noqa: D417
        """Computes and returns the entropy of the composite distribution.

        This method calculates the entropy for each component distribution and optionally sums them.

        Args:
            samples_mc (int): The number of samples to draw if the entropy does not have a closed-form solution.
                Defaults to `1`.

        Keyword Args:
            include_sum (bool, optional): Whether to include the summed entropy in the output TensorDict.
                Defaults to `composite_lp_aggregate()`, which defaults to `False`.

        Returns:
            torch.Tensor or TensorDictBase: If `aggregate_probabilities` is `True`, returns a single tensor with
            the summed entropies. If `aggregate_probabilities` is `False`, returns a TensorDict with the entropies
            of each component distribution.

        .. note:: If a distribution does not implement a closed-form solution for entropy, Monte Carlo sampling is used
            to estimate it.
        """
        aggregate_probabilities = composite_lp_aggregate()
        if include_sum is None:
            include_sum = composite_lp_aggregate()

        if not aggregate_probabilities:
            return self.entropy_composite(samples_mc, include_sum=include_sum)
        se = 0.0
        for dist in self.dists.values():
            try:
                e = dist.entropy()
            except NotImplementedError:
                x = dist.rsample((samples_mc,))
                e = -dist.log_prob(x).mean(0)
            if e.ndim > len(self.batch_shape):
                e = e.flatten(len(self.batch_shape), -1).sum(-1)
            se = se + e
        return se

    def entropy_composite(
        self,
        samples_mc=1,
        *,
        include_sum: bool | None = None,
    ) -> TensorDictBase:
        """Computes the entropy for each component distribution and returns a TensorDict with individual entropies.

        This method is used by the `entropy` method when `self.aggregate_probabilities` is `False`.

        Args:
            samples_mc (int): The number of samples to draw if the entropy does not have a closed-form solution.
                Defaults to `1`.

        Keyword Args:
            include_sum (bool, optional): Whether to include the summed entropy in the output TensorDict.
                Defaults to `composite_lp_aggregate()`, which is defaults to `False`.

        Returns:
            TensorDictBase: A TensorDict containing the individual entropies for each component distribution,
            along with an "entropy" entry containing the summed entropies if `include_sum` is `True`.

        .. note:: If a distribution does not implement a closed-form solution for entropy, Monte Carlo sampling is used
            to estimate it.
        """
        if include_sum is None:
            include_sum = composite_lp_aggregate()

        se = 0.0
        d = {}
        for name, dist in self.dists.items():
            try:
                e = dist.entropy()
            except NotImplementedError:
                x = dist.rsample((samples_mc,))
                e = -dist.log_prob(x).mean(0)
            d[_add_suffix(name, "_entropy")] = e
            if include_sum:
                if e.ndim > len(self.batch_shape):
                    e = e.flatten(len(self.batch_shape), -1).sum(-1)
                se = se + e
        if include_sum:
            d[self.entropy_key] = se
        return TensorDict(
            d,
            self.batch_shape,
        )

    def cdf(self, sample: TensorDictBase) -> TensorDictBase:
        """Computes the cumulative distribution function (CDF) for each component distribution in the composite distribution.

        This method calculates the CDF for each component distribution and updates the input TensorDict with the results.

        Args:
            sample (TensorDictBase): A TensorDict containing samples for which to compute the CDF.

        Returns:
            TensorDictBase: The input TensorDict updated with `<sample_name>_cdf` entries for each component distribution.
        """
        cdfs = {
            _add_suffix(name, "_cdf"): dist.cdf(sample.get(name))
            for name, dist in self.dists.items()
        }
        sample.update(cdfs)
        return sample

    def icdf(self, sample: TensorDictBase) -> TensorDictBase:
        """Computes the inverse cumulative distribution function (inverse CDF) for each component distribution.

        This method requires the input TensorDict to have either a `<sample_name>_cdf` entry or a `<sample_name>` entry
        for each component distribution. It calculates the inverse CDF and updates the TensorDict with the results.

        Args:
            sample (TensorDictBase): A TensorDict containing either `<sample_name>_cdf` or `<sample_name>` entries
                for each component distribution.

        Returns:
            TensorDictBase: The input TensorDict updated with `<sample_name>_icdf` entries for each component distribution.

        Raises:
            KeyError: If neither `<sample_name>` nor `<sample_name>_cdf` can be found in the input TensorDict for a component distribution.
        """
        for name, dist in self.dists.items():
            prob = sample.get(_add_suffix(name, "_cdf"))
            if prob is None:
                try:
                    prob = self.cdf(sample.get(name))
                except KeyError:
                    raise KeyError(
                        f"Neither {name} nor {name + '_cdf'} could be found in the sampled tensordict. Make sure one of these is available to icdf."
                    )
            icdf = dist.icdf(prob)
            sample.set(_add_suffix(name, "_icdf"), icdf)
        return sample


def _add_suffix(key: NestedKey, suffix: str):
    key = unravel_keys(key)
    if isinstance(key, str):
        return key + suffix
    return key[:-1] + (key[-1] + suffix,)
