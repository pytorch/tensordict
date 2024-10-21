# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import warnings

import torch
from tensordict import TensorDict, TensorDictBase
from tensordict.utils import NestedKey, unravel_key, unravel_keys
from torch import distributions as d


class CompositeDistribution(d.Distribution):
    """A composition of distributions.

    Groups distributions together with the TensorDict interface. Methods
    (``log_prob_composite``, ``entropy_composite``, ``cdf``, ``icdf``, ``rsample``, ``sample`` etc.)
    will return a tensordict, possibly modified in-place if the input was a tensordict.

    Args:
        params (TensorDictBase): a nested key-tensor map where the root entries
            point to the sample names, and the leaves are the distribution parameters.
            Entry names must match those of ``distribution_map``.

        distribution_map (Dict[NestedKey, Type[torch.distribution.Distribution]]):
            indicated the distribution types to be used. The names of the distributions
            will match the names of the samples in the tensordict.

    Keyword Arguments:
        name_map (Dict[NestedKey, NestedKey]]): a dictionary representing where each
            sample should be written. If not provided, the key names from ``distribution_map``
            will be used.
        extra_kwargs (Dict[NestedKey, Dict]): a possibly incomplete dictionary of
            extra keyword arguments for the distributions to be built.
        aggregate_probabilities (bool): if ``True``, the :meth:`~.log_prob` and :meth:`~.entropy` methods will
            sum the probabilities and entropies of the individual distributions and return a single tensor.
            If ``False``, the single log-probabilities will be registered in the input tensordict (for :meth:`~.log_prob`)
            or retuned as leaves of the output tensordict (for :meth:`~.entropy`).
            This parameter can be overridden at runtime by passing the ``aggregate_probabilities`` argument to
            ``log_prob`` and ``entropy``.
            Defaults to ``False``.
        log_prob_key (NestedKey, optional): key where to write the log_prob.
            Defaults to `'sample_log_prob'`.
        entropy_key (NestedKey, optional): key where to write the entropy.
            Defaults to `'entropy'`.

    .. note::
        In this distribution class, the batch-size of the input tensordict containing the params
        (``params``) is indicative of the batch_shape of the distribution. For instance,
        the ``"sample_log_prob"`` entry resulting from a call to ``log_prob``
        will be of the shape of the params (+ any supplementary batch dimension).

    Examples:
        >>> params = TensorDict({
        ...     "cont": {"loc": torch.randn(3, 4), "scale": torch.rand(3, 4)},
        ...     ("nested", "disc"): {"logits": torch.randn(3, 10)}
        ... }, [3])
        >>> dist = CompositeDistribution(params,
        ...     distribution_map={"cont": d.Normal, ("nested", "disc"): d.Categorical})
        >>> sample = dist.sample((4,))
        >>> sample = dist.log_prob(sample)
        >>> print(sample)
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
        aggregate_probabilities: bool | None = None,
        log_prob_key: NestedKey = "sample_log_prob",
        entropy_key: NestedKey = "entropy",
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
            # TODO: v0.7: remove the None
            dist_params = params.get(name, None)
            kwargs = extra_kwargs.get(name, {})
            if dist_params is None:
                raise KeyError
            dist = dist_class(**dist_params, **kwargs)
            dists[write_name] = dist
        self.dists = dists
        self.log_prob_key = log_prob_key
        self.entropy_key = entropy_key

        self.aggregate_probabilities = aggregate_probabilities

    @property
    def aggregate_probabilities(self):
        aggregate_probabilities = self._aggregate_probabilities
        if aggregate_probabilities is None:
            warnings.warn(
                "The default value of `aggregate_probabilities` will change from `False` to `True` in v0.7. "
                "Please pass this value explicitly to avoid this warning.",
                FutureWarning,
            )
            aggregate_probabilities = self._aggregate_probabilities = False
        return aggregate_probabilities

    @aggregate_probabilities.setter
    def aggregate_probabilities(self, value):
        self._aggregate_probabilities = value

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
                try:
                    support = dist.support
                    fallback = (
                        "mean"
                        if isinstance(support, torch.distributions.constraints._Real)
                        else "mode"
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
                        f"deterministic_sample wasn't found when queried in {type(dist)}. "
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
        self, sample: TensorDictBase, *, aggregate_probabilities: bool | None = None
    ) -> torch.Tensor | TensorDictBase:  # noqa: D417
        """Computes and returns the summed log-prob.

        Args:
            sample (TensorDictBase): the sample to compute the log probability.

        Keyword Args:
            aggregate_probabilities (bool, optional): if provided, overrides the default ``aggregate_probabilities``
                from the class.

        If ``self.aggregate_probabilities`` is ``True``, this method will return a single tensor with
        the summed log-probabilities. If ``self.aggregate_probabilities`` is ``False``, this method will
        call the `:meth:`~.log_prob_composite` method and return a tensordict with the log-probabilities
        of each sample in the input tensordict along with a ``sample_log_prob`` entry with the summed
        log-prob. In both cases, the output shape will be the shape of the input tensordict.
        """
        if aggregate_probabilities is None:
            aggregate_probabilities = self.aggregate_probabilities
        if not aggregate_probabilities:
            return self.log_prob_composite(sample, include_sum=True)
        slp = 0.0
        for name, dist in self.dists.items():
            lp = dist.log_prob(sample.get(name))
            if lp.ndim > sample.ndim:
                lp = lp.flatten(sample.ndim, -1).sum(-1)
            slp = slp + lp
        return slp

    def log_prob_composite(
        self, sample: TensorDictBase, include_sum=True
    ) -> TensorDictBase:
        """Writes a ``<sample>_log_prob`` entry for each sample in the input tensordict, along with a ``"sample_log_prob"`` entry with the summed log-prob.

        This method is called by the :meth:`~.log_prob` method when ``self.aggregate_probabilities`` is ``False``.
        """
        slp = 0.0
        d = {}
        for name, dist in self.dists.items():
            d[_add_suffix(name, "_log_prob")] = lp = dist.log_prob(sample.get(name))
            if lp.ndim > sample.ndim:
                lp = lp.flatten(sample.ndim, -1).sum(-1)
            slp = slp + lp
        if include_sum:
            d[self.log_prob_key] = slp
        sample.update(d)
        return sample

    def entropy(
        self, samples_mc: int = 1, *, aggregate_probabilities: bool | None = None
    ) -> torch.Tensor | TensorDictBase:  # noqa: D417
        """Computes and returns the summed entropies.

        Args:
            samples_mc (int): the number samples to draw if the entropy does not have a closed form formula.
                Defaults to ``1``.

        Keyword Args:
            aggregate_probabilities (bool, optional): if provided, overrides the default ``aggregate_probabilities``
                from the class.

        If ``self.aggregate_probabilities`` is ``True``, this method will return a single tensor with
        the summed entropies. If ``self.aggregate_probabilities`` is ``False``, this method will call
        the `:meth:`~.entropy_composite` method and return a tensordict with the entropies of each sample
        in the input tensordict along with an ``entropy`` entry with the summed entropy. In both cases,
        the output shape will match the shape of the distribution ``batch_shape``.
        """
        if aggregate_probabilities is None:
            aggregate_probabilities = self.aggregate_probabilities
        if not aggregate_probabilities:
            return self.entropy_composite(samples_mc, include_sum=True)
        se = 0.0
        for _, dist in self.dists.items():
            try:
                e = dist.entropy()
            except NotImplementedError:
                x = dist.rsample((samples_mc,))
                e = -dist.log_prob(x).mean(0)
            if e.ndim > len(self.batch_shape):
                e = e.flatten(len(self.batch_shape), -1).sum(-1)
            se = se + e
        return se

    def entropy_composite(self, samples_mc=1, include_sum=True) -> TensorDictBase:
        """Writes a ``<sample>_entropy`` entry for each sample in the input tensordict, along with a ``"entropy"`` entry with the summed entropies.

        This method is called by the :meth:`~.entropy` method when ``self.aggregate_probabilities`` is ``False``.
        """
        se = 0.0
        d = {}
        for name, dist in self.dists.items():
            try:
                e = dist.entropy()
            except NotImplementedError:
                x = dist.rsample((samples_mc,))
                e = -dist.log_prob(x).mean(0)
            d[_add_suffix(name, "_entropy")] = e
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
        cdfs = {
            _add_suffix(name, "_cdf"): dist.cdf(sample.get(name))
            for name, dist in self.dists.items()
        }
        sample.update(cdfs)
        return sample

    def icdf(self, sample: TensorDictBase) -> TensorDictBase:
        """Computes the inverse CDF.

        Requires the input tensordict to have one of `<sample_name>+'_cdf'` entry
        or a `<sample_name>` entry.

        Args:
            sample (TensorDictBase): a tensordict containing `<sample>_log_prob` where
                `<sample>` is the name of the sample provided during construction.
        """
        for name, dist in self.dists.items():
            # TODO: v0.7: remove the None
            prob = sample.get(_add_suffix(name, "_cdf"), None)
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
