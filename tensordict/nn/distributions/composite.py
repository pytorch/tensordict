# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from tensordict import TensorDict, TensorDictBase
from tensordict._tensordict import unravel_keys
from tensordict.utils import NestedKey
from torch import distributions as d


class CompositeDistribution(d.Distribution):
    """A composition of distributions.

    Groups distributions together with the TensorDict interface. All methods
    (``log_prob``, ``cdf``, ``icdf``, ``rsample``, ``sample`` etc.) will return a
    tensordict, possibly modified in-place if the input was a tensordict.

    Args:
        params (TensorDictBase): a nested key-tensor map where the root entries
            point to the sample names, and the leaves are the distribution parameters.
            Entry names must match those of ``distribution_map``.

        distribution_map (Dict[NestedKey, Type[torch.distribution.Distribution]]):
            indicated the distribution types to be used. The names of the distributions
            will match the names of the samples in the tensordict.

    Keyword Arguments:
        extra_kwargs (Dict[NestedKey, Dict]): a possibly incomplete dictionary of
            extra keyword arguments for the distributions to be built.

    .. note:: In this distribution class, the batch-size of the input tensordict containing the params
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

    def __init__(self, params: TensorDictBase, distribution_map, *, extra_kwargs=None):
        self._batch_shape = params.shape
        if extra_kwargs is None:
            extra_kwargs = {}
        dists = {}
        for name, dist_class in distribution_map.items():
            dist_params = params.get(name, None)
            kwargs = extra_kwargs.get(name, {})
            if dist_params is None:
                raise KeyError
            dist = dist_class(**dist_params, **kwargs)
            dists[name] = dist
        self.dists = dists

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

    def rsample(self, shape=None) -> TensorDictBase:
        if shape is None:
            shape = torch.Size([])
        return TensorDict(
            {name: dist.rsample(shape) for name, dist in self.dists.items()},
            shape + self.batch_shape,
        )

    def log_prob(self, sample: TensorDictBase) -> TensorDictBase:
        """Writes a ``<sample>_log_prob entry`` for each sample in the input tensordit, along with a ``"sample_log_prob"`` entry with the summed log-prob."""
        slp = 0.0
        d = {}
        for name, dist in self.dists.items():
            d[_add_suffix(name, "_log_prob")] = lp = dist.log_prob(sample.get(name))
            while lp.ndim > sample.ndim:
                lp = lp.sum(-1)
            slp = slp + lp
        d["sample_log_prob"] = slp
        sample.update(d)
        return sample

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
