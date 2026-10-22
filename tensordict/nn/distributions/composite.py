# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import warnings
from collections.abc import Mapping
from typing import Any, Dict, TYPE_CHECKING

import torch
from tensordict._deprecation import warn_deprecated
from tensordict._td import TensorDict
from tensordict.base import TensorDictBase
from tensordict.nn.utils import composite_lp_aggregate, set_composite_lp_aggregate
from tensordict.utils import IndexType, NestedKey, unravel_key
from torch import distributions as d

if TYPE_CHECKING:
    from typing_extensions import Self
else:
    Self = Any

__all__ = ["CompositeDistribution"]


def _resolve_name_map(name_map: dict | None, names) -> dict[NestedKey, NestedKey]:
    """Map each distribution name to the key its sample is written under.

    Keys are matched after :func:`~tensordict.unravel_key`. Distributions
    omitted from ``name_map`` keep their own name.
    """
    unraveled_names = [unravel_key(name) for name in names]
    if not name_map:
        return {name: name for name in unraveled_names}
    name_map = {unravel_key(key): unravel_key(value) for key, value in name_map.items()}
    return {name: name_map.get(name, name) for name in unraveled_names}


class CompositeDistribution(d.Distribution, Mapping):
    """A distribution made of several named distributions, which samples and scores tensordicts.

    Each component is built from one entry of ``params``, as
    ``distribution_map[name](**params[name], **extra_kwargs[name])``.
    :meth:`sample`, :meth:`rsample`, :attr:`mode`, :attr:`mean` and
    :attr:`deterministic_sample` return a :class:`~tensordict.TensorDict` with
    one entry per component, at its name or at its ``name_map`` key.
    :meth:`log_prob`, :meth:`cdf` and :meth:`icdf` take such a tensordict.
    The batch size of ``params`` is the ``batch_shape`` of the distribution,
    and a sample of shape ``sample_shape`` has the batch size
    ``sample_shape + batch_shape``. The ``event_shape`` is ``torch.Size([])``:
    the event dimensions of each component stay in its entry of the sample.

    The distribution is also a read-only mapping from the sample keys to the
    components: ``dist["cont"]``, ``list(dist)``, ``len(dist)``.

    :meth:`log_prob` and :meth:`entropy` return a tensordict with one entry
    per component while :func:`~tensordict.nn.composite_lp_aggregate` is
    ``False`` (the default), and one summed tensor while it is ``True``. See
    the :ref:`log-probability key table <log-prob-keys>` of
    :class:`~tensordict.nn.ProbabilisticTensorDictModule` for the keys that
    probabilistic modules write.

    Args:
        params (TensorDictBase): the parameters of the components. The entry
            at each name of ``distribution_map`` is a tensordict of keyword
            arguments for that distribution, for example
            ``{"loc": ..., "scale": ...}`` for a :class:`~torch.distributions.Normal`.
        distribution_map (dict): maps each name (a ``NestedKey``) to a
            distribution class, or to any callable that takes the parameters
            as keyword arguments and returns a distribution.

    Keyword Args:
        name_map (dict, optional): maps a name of ``distribution_map`` to the
            key where its sample is written. Names that are not in it keep
            their own key. Keys are compared after
            :func:`~tensordict.unravel_key`. Defaults to ``None``.
        extra_kwargs (dict, optional): maps a name of ``distribution_map`` to a
            dict of extra keyword arguments for that distribution, for
            example constants that are not in ``params``. Defaults to ``None``.
        log_prob_key (NestedKey, optional): the key of the summed
            log-probability that :meth:`log_prob_composite` writes with
            ``include_sum=True``. The per-component entries are always at
            ``<sample key>_log_prob``. Defaults to ``"sample_log_prob"``.
        entropy_key (NestedKey, optional): the key of the summed entropy that
            :meth:`entropy_composite` writes with ``include_sum=True``. The
            per-component entries are always at ``<sample key>_entropy``.
            Defaults to ``"entropy"``.

    Raises:
        KeyError: if ``params`` has no entry for a name of ``distribution_map``.

    .. seealso:: :class:`~tensordict.nn.ProbabilisticTensorDictModule` and
        :class:`~tensordict.nn.ProbabilisticTensorDictSequential` to use this
        class in a model, and :class:`~tensordict.nn.set_composite_lp_aggregate`
        to control the aggregation of the log-probabilities.

    Examples:
        >>> import torch
        >>> from torch import distributions as d
        >>> from tensordict import TensorDict
        >>> from tensordict.nn import CompositeDistribution
        >>> params = TensorDict({
        ...     "cont": {"loc": torch.zeros(3, 4), "scale": torch.ones(3, 4)},
        ...     ("nested", "disc"): {"logits": torch.zeros(3, 10)},
        ... }, [3])
        >>> dist = CompositeDistribution(params,
        ...     distribution_map={"cont": d.Normal, ("nested", "disc"): d.Categorical})
        >>> dist["cont"]
        Normal(loc: torch.Size([3, 4]), scale: torch.Size([3, 4]))
        >>> sample = dist.sample((4,))
        >>> sample.batch_size, sample["cont"].shape, sample["nested", "disc"].shape
        (torch.Size([4, 3]), torch.Size([4, 3, 4]), torch.Size([4, 3]))
        >>> lp = dist.log_prob(sample)
        >>> lp["cont_log_prob"].shape, lp["nested", "disc_log_prob"].shape
        (torch.Size([4, 3, 4]), torch.Size([4, 3]))

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
        self._event_shape = torch.Size(())
        if extra_kwargs is None:
            extra_kwargs = {}
        dists = {}
        write_names = _resolve_name_map(name_map, distribution_map)
        for raw_name, dist_class in distribution_map.items():
            name = unravel_key(raw_name)
            write_name = write_names[name]
            dist_params = params.get(name)
            kwargs = extra_kwargs.get(name, extra_kwargs.get(raw_name, {}))
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
        """The key of the summed log-probability that :meth:`log_prob_composite` writes with ``include_sum=True``.

        Defaults to ``"sample_log_prob"``. Setting it to ``None`` restores the
        default.
        """
        log_prob_key = self._log_prob_key
        if log_prob_key is None:
            log_prob_key = "sample_log_prob"
        return log_prob_key

    @log_prob_key.setter
    def log_prob_key(self, value):
        self._log_prob_key = value

    @property
    def entropy_key(self):
        """The key of the summed entropy that :meth:`entropy_composite` writes with ``include_sum=True``.

        Defaults to ``"entropy"``. Setting it to ``None`` restores the default.
        """
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
        """Builds a composite distribution from distribution instances, instead of classes and parameters.

        The distributions are used as they are. :class:`~tensordict.nn.ProbabilisticTensorDictSequential`
        uses this method to combine the distributions of its modules.

        Args:
            params (TensorDictBase): a tensordict whose batch size is the
                ``batch_shape`` of the composite distribution. Its entries are
                not read.
            distributions (dict): maps each name (a ``NestedKey``) to a
                :class:`~torch.distributions.Distribution` instance. Each
                batch shape must start with the batch size of ``params``.

        Keyword Args:
            name_map (dict, optional): maps a name of ``distributions`` to the
                key where its sample is written. Names that are not in it keep
                their own key, and keys of ``name_map`` that are not in
                ``distributions`` are ignored. Keys are compared after
                :func:`~tensordict.unravel_key`. Defaults to ``None``.
            log_prob_key (NestedKey, optional): as in
                :class:`CompositeDistribution`. Defaults to ``"sample_log_prob"``.
            entropy_key (NestedKey, optional): as in
                :class:`CompositeDistribution`. Defaults to ``"entropy"``.

        Returns:
            A new :class:`CompositeDistribution`.

        Examples:
            >>> import torch
            >>> from torch import distributions as d
            >>> from tensordict import TensorDict
            >>> from tensordict.nn import CompositeDistribution
            >>> dist = CompositeDistribution.from_distributions(
            ...     TensorDict(batch_size=[3]),
            ...     {"a": d.Normal(torch.zeros(3), torch.ones(3)),
            ...      "b": d.Bernoulli(torch.full((3,), 0.5))},
            ... )
            >>> sample = dist.sample()
            >>> sample.batch_size, sorted(sample.keys())
            (torch.Size([3]), ['a', 'b'])
        """
        self = cls.__new__(cls)
        self._batch_shape = params.shape
        self._event_shape = torch.Size(())
        dists = {}
        write_names = _resolve_name_map(name_map, distributions)
        for name, dist in distributions.items():
            dists[write_names[unravel_key(name)]] = dist
        self.dists = dists
        self.log_prob_key = log_prob_key
        self.entropy_key = entropy_key
        self.aggregate_probabilities = None

        return self

    def sample(self, shape=None) -> TensorDictBase:
        """Draws a sample from each component and returns them in a tensordict.

        Args:
            shape (torch.Size, optional): the leading sample dimensions.
                Defaults to ``torch.Size([])``.

        Returns:
            A :class:`~tensordict.TensorDict` with one entry per component
            and the batch size ``shape + batch_shape``.
        """
        if shape is None:
            shape = torch.Size([])
        samples = {name: dist.sample(shape) for name, dist in self.dists.items()}
        return TensorDict(
            samples,
            torch.Size(shape) + self.batch_shape,
        )

    @property
    def mode(self) -> TensorDictBase:
        """A tensordict with the mode of each component, of batch size ``batch_shape``."""
        samples = {name: dist.mode for name, dist in self.dists.items()}
        return TensorDict(
            samples,
            self.batch_shape,
        )

    @property
    def mean(self) -> TensorDictBase:
        """A tensordict with the mean of each component, of batch size ``batch_shape``."""
        samples = {name: dist.mean for name, dist in self.dists.items()}
        return TensorDict(
            samples,
            self.batch_shape,
        )

    @property
    def deterministic_sample(self) -> TensorDictBase:
        """A tensordict with a deterministic value of each component, of batch size ``batch_shape``.

        This is the value that a probabilistic module writes with the
        ``DETERMINISTIC`` interaction type (see
        :class:`~tensordict.nn.InteractionType`). Each entry is the
        ``deterministic_sample`` attribute of the component. A component
        without one gives its mode if its support is enumerable
        (:class:`~torch.distributions.Categorical`,
        :class:`~torch.distributions.Bernoulli`, ...) and its mean otherwise.
        For a distribution class that tensordict does not know, the choice
        follows the support of the distribution, with a warning.

        Raises:
            NotImplementedError: if a component has no
                ``deterministic_sample`` and the mean or mode it falls back on
                is not implemented.
        """

        def maybe_deterministic_sample(dist):
            if hasattr(dist, "deterministic_sample"):
                return dist.deterministic_sample
            else:
                from tensordict.nn.probabilistic import _DETERMINISTIC_REGISTER

                # Fallbacks
                tdist = type(dist)
                if issubclass(tdist, d.Independent):
                    tdist = type(dist.base_dist)
                interaction_type = _DETERMINISTIC_REGISTER.get(tdist)
                if interaction_type == "random":
                    return dist.rsample() if dist.has_rsample else dist.sample()
                if (
                    interaction_type == "deterministic"
                    and isinstance(dist, d.Independent)
                    and hasattr(dist.base_dist, "deterministic_sample")
                ):
                    # Independent leaves the samples of its base unchanged
                    return dist.base_dist.deterministic_sample
                if interaction_type == "mode":
                    fallback = "mode"
                elif interaction_type == "mean":
                    fallback = "mean"
                elif interaction_type is None:
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
                # A registered dist's AttributeError propagates, as before
                errors = (
                    (AttributeError, NotImplementedError)
                    if interaction_type is None
                    else NotImplementedError
                )
                try:
                    result = dist.mean if fallback == "mean" else dist.mode
                except errors as err:
                    raise NotImplementedError(
                        f"{type(dist)} has no deterministic_sample, and its {fallback} "
                        f"is not implemented, so {type(self).__name__} cannot compute "
                        f"a deterministic sample for it. Use a distribution with a "
                        f"deterministic_sample attribute (for example TanhNormal), or "
                        f"another interaction type."
                    ) from err
                if interaction_type is None:
                    warnings.warn(
                        f"deterministic_sample wasn't found when queried on {type(dist)}. "
                        f"{type(self).__name__} is falling back on {fallback} instead. "
                        f"For better code quality and efficiency, make sure to either "
                        f"provide a distribution with a deterministic_sample attribute or "
                        f"to change the InteractionMode to the desired value.",
                        category=UserWarning,
                    )
                return result

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
        """Draws a reparameterized sample from each component and returns them in a tensordict.

        Every component must support ``rsample``.

        Args:
            shape (torch.Size, optional): the leading sample dimensions.
                Defaults to ``torch.Size([])``.

        Returns:
            A :class:`~tensordict.TensorDict` with one entry per component
            and the batch size ``shape + batch_shape``.
        """
        if shape is None:
            shape = torch.Size([])
        return TensorDict(
            {name: dist.rsample(shape) for name, dist in self.dists.items()},
            torch.Size(shape) + self.batch_shape,
        )

    def sample_and_log_prob(
        self, shape=None
    ) -> tuple[TensorDictBase, torch.Tensor | TensorDictBase]:
        """Draws a sample and returns it with the log-probability of the same draw.

        Each component is sampled and scored with its own ``sample_and_log_prob``
        method if it has one, and with ``sample`` then ``log_prob`` otherwise.
        :class:`~tensordict.nn.ProbabilisticTensorDictModule` calls this method
        when it samples with ``return_log_prob=True``.

        Args:
            shape (torch.Size, optional): the leading sample dimensions.
                Defaults to ``torch.Size([])``.

        Returns:
            A ``(sample, log_prob)`` tuple, with the sample as returned by
            :meth:`sample` and its log-probability in the layout of
            :meth:`log_prob`.

        Examples:
            >>> import torch
            >>> from torch import distributions as d
            >>> from tensordict import TensorDict
            >>> from tensordict.nn import CompositeDistribution
            >>> params = TensorDict({
            ...     "cont": {"loc": torch.zeros(3, 4), "scale": torch.ones(3, 4)},
            ...     ("nested", "disc"): {"logits": torch.zeros(3, 10)}
            ... }, [3])
            >>> dist = CompositeDistribution(params,
            ...     distribution_map={"cont": d.Normal, ("nested", "disc"): d.Categorical})
            >>> sample, lp = dist.sample_and_log_prob()
            >>> sorted(lp.keys(True, True), key=str)
            [('nested', 'disc_log_prob'), 'cont_log_prob']
        """
        return self._sample_and_log_prob(shape, reparameterize=False)

    def rsample_and_log_prob(
        self, shape=None
    ) -> tuple[TensorDictBase, torch.Tensor | TensorDictBase]:
        """Draws a reparameterized sample and returns it with the log-probability of the same draw.

        Each component is sampled and scored with its own
        ``rsample_and_log_prob`` method if it has one, and with ``rsample`` then
        ``log_prob`` otherwise. See :meth:`sample_and_log_prob`.

        Args:
            shape (torch.Size, optional): the leading sample dimensions.
                Defaults to ``torch.Size([])``.

        Returns:
            A ``(sample, log_prob)`` tuple, with the sample as returned by
            :meth:`rsample` and its log-probability in the layout of
            :meth:`log_prob`.
        """
        return self._sample_and_log_prob(shape, reparameterize=True)

    def _sample_and_log_prob(
        self, shape, *, reparameterize: bool
    ) -> tuple[TensorDictBase, torch.Tensor | TensorDictBase]:
        if shape is None:
            shape = torch.Size([])
        joint_name = "rsample_and_log_prob" if reparameterize else "sample_and_log_prob"
        samples = {}
        log_probs = {}
        for name, dist in self.dists.items():
            if callable(getattr(type(dist), joint_name, None)):
                samples[name], log_probs[name] = getattr(dist, joint_name)(shape)
            else:
                samples[name] = (
                    dist.rsample(shape) if reparameterize else dist.sample(shape)
                )
                log_probs[name] = dist.log_prob(samples[name])
        sample = TensorDict(samples, torch.Size(shape) + self.batch_shape)
        if not composite_lp_aggregate():
            log_prob = {
                _add_suffix(name, "_log_prob"): lp for name, lp in log_probs.items()
            }
            return sample, sample.empty(recurse=True).update(log_prob).filter_empty_()
        slp = 0.0
        for lp in log_probs.values():
            if lp.ndim > sample.ndim:
                lp = lp.flatten(sample.ndim, -1).sum(-1)
            slp = slp + lp
        return sample, slp

    def log_prob(self, sample: TensorDictBase) -> torch.Tensor | TensorDictBase:  # noqa: D417
        """Computes the log-probability of a sample.

        Args:
            sample (TensorDictBase): the sample, with the value of each
                component at its sample key, as returned by :meth:`sample`.

        Returns:
            The result depends on :func:`~tensordict.nn.composite_lp_aggregate`.

            - ``False`` (default): a new tensordict with one ``<name>_log_prob``
              entry per distribution, for example ``cont_log_prob`` and
              ``("nested", "disc_log_prob")`` (see :meth:`log_prob_composite`).
              ``sample`` is not modified.
            - ``True``: one tensor, the sum of the log-probabilities over all
              distributions and their event dimensions.

            In both cases the batch shape is the batch shape of ``sample``.

        Examples:
            >>> import torch
            >>> from torch import distributions as d
            >>> from tensordict import TensorDict
            >>> from tensordict.nn import CompositeDistribution, set_composite_lp_aggregate
            >>> params = TensorDict({
            ...     "cont": {"loc": torch.zeros(3, 4), "scale": torch.ones(3, 4)},
            ...     ("nested", "disc"): {"logits": torch.zeros(3, 10)}
            ... }, [3])
            >>> dist = CompositeDistribution(params,
            ...     distribution_map={"cont": d.Normal, ("nested", "disc"): d.Categorical})
            >>> sample = TensorDict({"cont": torch.zeros(3, 4), ("nested", "disc"): torch.zeros(3, dtype=torch.long)}, [3])
            >>> lp = dist.log_prob(sample)
            >>> sorted(lp.keys(True, True), key=str)
            [('nested', 'disc_log_prob'), 'cont_log_prob']
            >>> with set_composite_lp_aggregate(True):
            ...     dist.log_prob(sample)
            tensor([-5.9783, -5.9783, -5.9783])
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
        """Computes the log-probability of each component and returns them in a tensordict.

        The log-probability of the component at ``key`` is at
        ``<key>_log_prob``, with the suffix on the last element of a nested
        key: ``("nested", "disc_log_prob")``. :meth:`log_prob` calls this
        method while :func:`~tensordict.nn.composite_lp_aggregate` is
        ``False``.

        Args:
            sample (TensorDictBase): the sample, with the value of each
                component at its sample key.

        Keyword Args:
            include_sum (bool, optional): if ``True``, the result also holds
                the sum of the log-probabilities over the components and their
                event dimensions, at :attr:`log_prob_key`. Defaults to
                ``None``, which uses the value of
                :func:`~tensordict.nn.composite_lp_aggregate`.

        Returns:
            While :func:`~tensordict.nn.composite_lp_aggregate` is ``False``, a
            new tensordict with the log-probabilities. While it is ``True``,
            ``sample``, with the log-probabilities written into it.

        Examples:
            >>> import torch
            >>> from torch import distributions as d
            >>> from tensordict import TensorDict
            >>> from tensordict.nn import CompositeDistribution
            >>> params = TensorDict({
            ...     "a": {"loc": torch.zeros(2), "scale": torch.ones(2)},
            ...     "b": {"probs": torch.full((2,), 0.5)},
            ... }, [2])
            >>> dist = CompositeDistribution(params, distribution_map={"a": d.Normal, "b": d.Bernoulli})
            >>> sample = TensorDict(a=torch.zeros(2), b=torch.ones(2), batch_size=[2])
            >>> lp = dist.log_prob_composite(sample, include_sum=True)
            >>> sorted(lp.keys())
            ['a_log_prob', 'b_log_prob', 'sample_log_prob']
            >>> lp["sample_log_prob"]
            tensor([-1.6121, -1.6121])
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
        """Computes the entropy of each component, or their sum.

        A component whose ``entropy()`` raises ``NotImplementedError`` gets a
        Monte Carlo estimate from ``samples_mc`` reparameterized samples
        (``rsample``).

        Args:
            samples_mc (int, optional): the number of samples of a Monte Carlo
                estimate. Defaults to ``1``.

        Keyword Args:
            aggregate_probabilities (bool, optional): deprecated and ignored,
                will be removed in TensorDict 0.17. A value other than
                ``None`` emits a ``DeprecationWarning``. Use
                :func:`~tensordict.nn.set_composite_lp_aggregate` instead.
            include_sum (bool, optional): used while
                :func:`~tensordict.nn.composite_lp_aggregate` is ``False``: if
                ``True``, the tensordict also holds the summed entropy at
                :attr:`entropy_key`. Defaults to ``None``, which uses the value
                of :func:`~tensordict.nn.composite_lp_aggregate`.

        Returns:
            While :func:`~tensordict.nn.composite_lp_aggregate` is ``False``
            (the default), a tensordict with one ``<name>_entropy`` entry per
            component (see :meth:`entropy_composite`). While it is ``True``, one
            tensor, the sum of the entropies over the components and their
            event dimensions.

        Examples:
            >>> import torch
            >>> from torch import distributions as d
            >>> from tensordict import TensorDict
            >>> from tensordict.nn import CompositeDistribution, set_composite_lp_aggregate
            >>> params = TensorDict({
            ...     "cont": {"loc": torch.zeros(3, 4), "scale": torch.ones(3, 4)},
            ...     ("nested", "disc"): {"logits": torch.zeros(3, 10)}
            ... }, [3])
            >>> dist = CompositeDistribution(params,
            ...     distribution_map={"cont": d.Normal, ("nested", "disc"): d.Categorical})
            >>> sorted(dist.entropy().keys(True, True), key=str)
            [('nested', 'disc_entropy'), 'cont_entropy']
            >>> with set_composite_lp_aggregate(True):
            ...     dist.entropy()
            tensor([7.9783, 7.9783, 7.9783])
        """
        if aggregate_probabilities is not None:
            warn_deprecated(
                "CompositeDistribution.entropy(aggregate_probabilities=...), which has no effect,",
                removal="0.17",
                replacement="set_composite_lp_aggregate",
            )
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
        """Computes the entropy of each component and returns them in a new tensordict.

        The entropy of the component at ``key`` is at ``<key>_entropy``, with
        the suffix on the last element of a nested key. :meth:`entropy` calls
        this method while :func:`~tensordict.nn.composite_lp_aggregate` is
        ``False``. A component without a closed-form entropy gets a Monte Carlo
        estimate, as in :meth:`entropy`.

        Args:
            samples_mc (int, optional): the number of samples of a Monte Carlo
                estimate. Defaults to ``1``.

        Keyword Args:
            include_sum (bool, optional): if ``True``, the result also holds
                the sum of the entropies over the components and their event
                dimensions, at :attr:`entropy_key`. Defaults to ``None``, which
                uses the value of :func:`~tensordict.nn.composite_lp_aggregate`.

        Returns:
            A :class:`~tensordict.TensorDict` of batch size ``batch_shape``.

        Examples:
            >>> import torch
            >>> from torch import distributions as d
            >>> from tensordict import TensorDict
            >>> from tensordict.nn import CompositeDistribution
            >>> params = TensorDict({
            ...     "a": {"loc": torch.zeros(2), "scale": torch.ones(2)},
            ...     "b": {"probs": torch.full((2,), 0.5)},
            ... }, [2])
            >>> dist = CompositeDistribution(params, distribution_map={"a": d.Normal, "b": d.Bernoulli})
            >>> ent = dist.entropy_composite(include_sum=True)
            >>> sorted(ent.keys())
            ['a_entropy', 'b_entropy', 'entropy']
            >>> ent["entropy"]
            tensor([2.1121, 2.1121])
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
        """Computes the cumulative distribution function (CDF) of each component and writes it into ``sample``.

        The CDF of the component at ``key`` is written at ``<key>_cdf``, with
        the suffix on the last element of a nested key.

        Args:
            sample (TensorDictBase): the sample, with the value of each
                component at its sample key.

        Returns:
            ``sample``, with the ``<key>_cdf`` entries.

        Examples:
            >>> import torch
            >>> from torch import distributions as d
            >>> from tensordict import TensorDict
            >>> from tensordict.nn import CompositeDistribution
            >>> params = TensorDict({
            ...     "a": {"loc": torch.zeros(2), "scale": torch.ones(2)},
            ...     "b": {"loc": torch.ones(2), "scale": torch.ones(2)},
            ... }, [2])
            >>> dist = CompositeDistribution(params, distribution_map={"a": d.Normal, "b": d.Normal})
            >>> sample = TensorDict(a=torch.zeros(2), b=torch.zeros(2), batch_size=[2])
            >>> out = dist.cdf(sample)
            >>> out is sample
            True
            >>> sample["a_cdf"], sample["b_cdf"]
            (tensor([0.5000, 0.5000]), tensor([0.1587, 0.1587]))
            >>> _ = dist.icdf(sample)  # reads the "_cdf" entries
            >>> sample["a_icdf"]
            tensor([0., 0.])
        """
        cdfs = {
            _add_suffix(name, "_cdf"): dist.cdf(sample.get(name))
            for name, dist in self.dists.items()
        }
        sample.update(cdfs)
        return sample

    def icdf(self, sample: TensorDictBase) -> TensorDictBase:
        """Computes the inverse CDF of each component and writes it into ``sample``.

        For the component at ``key``, the method reads the probability at
        ``<key>_cdf`` (as written by :meth:`cdf`) and writes its inverse CDF at
        ``<key>_icdf``. If ``<key>_cdf`` is missing, it uses the CDF of the
        value at ``key``, so ``<key>_icdf`` is that value again, up to rounding.
        See the example of :meth:`cdf`.

        Args:
            sample (TensorDictBase): a tensordict with a ``<key>_cdf`` or a
                ``key`` entry for each component.

        Returns:
            ``sample``, with the ``<key>_icdf`` entries.

        Raises:
            KeyError: if neither ``key`` nor ``<key>_cdf`` is in ``sample``
                for a component.
        """
        for name, dist in self.dists.items():
            prob = sample.get(_add_suffix(name, "_cdf"))
            if prob is None:
                value = sample.get(name)
                if value is None:
                    raise KeyError(
                        f"Neither {name} nor {_add_suffix(name, '_cdf')} could be found in the sampled tensordict. Make sure one of these is available to icdf."
                    )
                prob = dist.cdf(value)
            icdf = dist.icdf(prob)
            sample.set(_add_suffix(name, "_icdf"), icdf)
        return sample


def _add_suffix(key: NestedKey, suffix: str):
    key = unravel_key(key)
    if isinstance(key, str):
        return key + suffix
    return key[:-1] + (key[-1] + suffix,)
