# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import collections
import contextlib
import re
import warnings
from collections.abc import MutableSequence
from textwrap import indent
from typing import Any, Callable, Dict, List, OrderedDict, overload, TYPE_CHECKING

import torch
from tensordict._deprecation import deprecated, deprecated_attributes, warn_deprecated
from tensordict._nestedkey import NestedKey
from tensordict._td import TensorDict
from tensordict.base import is_tensor_collection
from tensordict.nn.common import dispatch, TensorDictModuleBase
from tensordict.nn.distributions.composite import _add_suffix, CompositeDistribution
from tensordict.nn.distributions.continuous import Delta
from tensordict.nn.distributions.discrete import OneHotCategorical
from tensordict.nn.distributions.truncated_normal import (
    TruncatedNormal,
    TruncatedStandardNormal,
)
from tensordict.nn.sequence import TensorDictSequential
from tensordict.nn.utils import (
    _set_skip_existing_None,
    composite_lp_aggregate,
    set_composite_lp_aggregate,
)
from tensordict.tensorclass import is_non_tensor
from tensordict.tensordict import TensorDictBase
from tensordict.utils import _ContextManager, _zip_strict, unravel_key
from torch import distributions as D, Tensor
from torch.compiler import is_compiling
from torch.utils._contextlib import _DecoratorContextManager

try:
    from enum import StrEnum
except ImportError:
    from .utils import _StrEnum as StrEnum

if TYPE_CHECKING:
    from typing import Self
else:
    Self = Any

__all__ = [
    "InteractionType",
    "ProbabilisticTensorDictModule",
    "ProbabilisticTensorDictSequential",
    "interaction_type",
    "set_interaction_type",
]


class InteractionType(StrEnum):
    """The rule that a probabilistic module uses to get a value from a distribution.

    - ``RANDOM``: a sample, from ``rsample()`` if the distribution has it,
      else from ``sample()``.
    - ``MODE``, ``MEDIAN``, ``MEAN``: the attribute of the same name. If the
      distribution has no ``mean``, ``MEAN`` averages ``n_empirical_estimate``
      samples (see :class:`ProbabilisticTensorDictModule`).
    - ``DETERMINISTIC``: the ``deterministic_sample`` attribute. If the
      distribution does not have one, the mode for distributions with
      enumerable support (Categorical, Bernoulli,
      :class:`~tensordict.nn.distributions.OneHotCategorical`) and the mean
      for the others. For a distribution class that tensordict does not
      know, the module chooses the mean or the mode from the support of the
      distribution, and warns.

    Set the type for a block of code with :class:`set_interaction_type`, or
    for one module with its ``default_interaction_type`` argument. Strings
    are accepted, in any case: ``InteractionType("RANDOM")``.

    Examples:
        >>> from tensordict.nn import InteractionType
        >>> InteractionType("RANDOM")
        <InteractionType.RANDOM: 'random'>
        >>> InteractionType.from_str("Mode") is InteractionType.MODE
        True

    """

    MODE = "mode"
    MEDIAN = "median"
    MEAN = "mean"
    RANDOM = "random"
    DETERMINISTIC = "deterministic"

    @classmethod
    def from_str(cls, type_str: str) -> InteractionType:
        """Returns the interaction type whose name matches a string, in any case.

        Args:
            type_str (str): the name, for example ``"random"`` or ``"MODE"``.
        """
        return cls(type_str.lower())

    @classmethod
    def _missing_(cls, value):
        # Normalize input: if a string, lower it
        if isinstance(value, str):
            value = value.lower()
            # Try to match with the enum values
            for member in cls:
                if member.value == value:
                    return member
        return None  # Default behavior if not found


_interaction_type = _ContextManager()


def _torch_deterministic_register() -> dict[type, InteractionType]:
    register = {}
    # Iterate over all distribution classes in torch.distributions
    for dist_name in dir(D):
        dist_cls = getattr(D, dist_name)

        # Check if it's a class (not a function or variable) and is a subclass of Distribution
        if isinstance(dist_cls, type) and issubclass(dist_cls, D.Distribution):
            if dist_cls is D.LogisticNormal:
                register[dist_cls] = InteractionType.DETERMINISTIC
            elif getattr(dist_cls, "has_enumerate_support", False):
                register[dist_cls] = InteractionType.MODE
            else:
                register[dist_cls] = InteractionType.MEAN
    return register


_DETERMINISTIC_REGISTER = _torch_deterministic_register()

_DETERMINISTIC_REGISTER[Delta] = InteractionType.DETERMINISTIC
_DETERMINISTIC_REGISTER[OneHotCategorical] = InteractionType.MODE

_DETERMINISTIC_REGISTER[TruncatedNormal] = InteractionType.MEAN
_DETERMINISTIC_REGISTER[TruncatedStandardNormal] = InteractionType.MEAN


def interaction_type() -> InteractionType | None:
    """Returns the interaction type set by :class:`set_interaction_type`, or ``None`` outside of it.

    When it returns ``None``, each :class:`ProbabilisticTensorDictModule`
    uses its ``default_interaction_type``.

    Examples:
        >>> from tensordict.nn import interaction_type, set_interaction_type
        >>> print(interaction_type())
        None
        >>> with set_interaction_type("mode"):
        ...     print(interaction_type())
        mode
    """
    return _interaction_type.get_mode()


class set_interaction_type(_DecoratorContextManager):
    """Sets the interaction type of all probabilistic modules inside a block of code, or inside a decorated function.

    Inside the block, :func:`~tensordict.nn.interaction_type`
    returns ``type``, and every :class:`ProbabilisticTensorDictModule` uses it
    instead of its ``default_interaction_type``. When the block ends, the
    previous value comes back.

    Args:
        type (InteractionType or str, optional): the interaction type, for
            example ``InteractionType.RANDOM`` or ``"random"`` (in any case).
            ``None`` makes each module use its ``default_interaction_type``.
            Defaults to ``InteractionType.DETERMINISTIC``.

    Keyword Args:
        thread_local (bool, optional): if ``True``, the interaction type
            applies to the calling thread only, and blocks that other threads
            enter do not change it. Inside the block, blocks without this flag
            that the same thread enters stay in that thread too. Threads
            started inside the block see the process-wide interaction type.
            If ``False``, the interaction type applies to the whole process.
            Defaults to ``False``.

    .. note:: Threads. By default the interaction type is process-wide: a
        ``with`` block in one thread also changes what the modules running in
        other threads at that time use. Pass ``thread_local=True`` to keep a
        block to its own thread. A thread started inside such a block reads
        the process-wide value, not the block's.

    Examples:
        >>> import torch
        >>> from torch import distributions as d
        >>> from tensordict import TensorDict
        >>> from tensordict.nn import (
        ...     InteractionType,
        ...     ProbabilisticTensorDictModule,
        ...     set_interaction_type,
        ... )
        >>> module = ProbabilisticTensorDictModule(
        ...     in_keys=["loc", "scale"], out_keys=["action"], distribution_class=d.Normal
        ... )
        >>> td = TensorDict(loc=torch.zeros(3), scale=torch.ones(3))
        >>> module(td)["action"]  # default_interaction_type is DETERMINISTIC: the mean
        tensor([0., 0., 0.])
        >>> _ = torch.manual_seed(0)
        >>> with set_interaction_type("random"):
        ...     module(td)["action"]
        tensor([ 1.5410, -0.2934, -2.1788])

    It also works as a decorator:

        >>> @set_interaction_type(InteractionType.MODE)
        ... def act(td):
        ...     return module(td)
        >>> act(td)["action"]
        tensor([0., 0., 0.])

    """

    def __init__(
        self,
        type: InteractionType | str | None = InteractionType.DETERMINISTIC,
        *,
        thread_local: bool = False,
    ) -> None:
        super().__init__()
        if not isinstance(type, InteractionType) and type is not None:
            if isinstance(type, str):
                type = InteractionType(type.lower())
            else:
                raise ValueError(f"{type} is not a valid InteractionType")
        self.type = type
        self.thread_local = thread_local

    def clone(self) -> set_interaction_type:
        # override this method if your children class takes __init__ parameters
        return type(self)(self.type, thread_local=self.thread_local)

    def __enter__(self) -> None:
        if self.thread_local:
            self.prev = _interaction_type.get_thread_mode()
            _interaction_type.set_thread_mode(self.type)
        else:
            self.prev = _interaction_type.get_mode()
            _interaction_type.set_mode(self.type)

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        if self.thread_local:
            _interaction_type.set_thread_mode(self.prev)
        else:
            _interaction_type.set_mode(self.prev)


@contextlib.contextmanager
def _use_generator(generator: torch.Generator | None):
    """Temporarily route the global RNG through ``generator`` for the duration of the block.

    On entry the current global RNG state (CPU and, if applicable, the relevant CUDA device)
    is forked, the generator's state is loaded into the global RNG, and the block runs.
    On exit, the post-sample global state is written back to ``generator`` so its stream
    advances in place, then the original global state is restored.

    No-op when ``generator`` is ``None`` or when running under ``torch.compile``: in the
    compile case fork_rng is not traceable, so we leave the global RNG as-is.
    """
    if generator is None or is_compiling():
        yield
        return
    device = generator.device
    if device.type == "cuda":
        with torch.random.fork_rng(devices=[device.index]):
            torch.cuda.set_rng_state(generator.get_state(), device)
            try:
                yield
            finally:
                generator.set_state(torch.cuda.get_rng_state(device))
    else:
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(generator.get_state())
            try:
                yield
            finally:
                generator.set_state(torch.get_rng_state())


def _get_joint_sample_and_log_prob(
    dist: D.Distribution,
) -> Callable[[torch.Size], tuple[Any, Tensor | TensorDictBase]] | None:
    """Return the method of ``dist`` that draws a sample and scores the same draw.

    ``rsample_and_log_prob`` is used when ``dist.has_rsample``, and
    ``sample_and_log_prob`` otherwise. Returns ``None`` when the distribution
    does not define the method, as for the :mod:`torch.distributions` classes,
    and for an object that is not a distribution, which ``_dist_sample``
    rejects with a ``TypeError``.
    """
    if not isinstance(dist, D.Distribution):
        return None
    name = "rsample_and_log_prob" if dist.has_rsample else "sample_and_log_prob"
    if callable(getattr(type(dist), name, None)):
        return getattr(dist, name)
    return None


class ProbabilisticTensorDictModule(TensorDictModuleBase):
    """A module that builds a distribution from entries of a tensordict and writes a value from it.

    The module has no parameters. It reads the distribution parameters at
    ``in_keys``, builds ``distribution_class(**params, **distribution_kwargs)``,
    and writes one value at each of the ``out_keys``. To compute the parameters
    with a network, put a :class:`~tensordict.nn.TensorDictModule` before this
    module in a :class:`~tensordict.nn.ProbabilisticTensorDictSequential`. To
    get the distribution without a value, call :meth:`get_dist`. To get the
    log-probability of a value, call :meth:`log_prob`.

    The value depends on the interaction type. The module uses
    :func:`~tensordict.nn.interaction_type` if it is set (see
    :class:`~tensordict.nn.set_interaction_type`). Otherwise it uses
    ``default_interaction_type``. See :class:`~tensordict.nn.InteractionType`
    for what each type reads from the distribution. By default the
    interaction type is process-wide, so a :class:`~tensordict.nn.set_interaction_type`
    block in one thread also changes the value that modules running in other
    threads use. Pass ``thread_local=True`` to keep the block to its own
    thread.

    With ``return_log_prob=True`` and the ``RANDOM`` interaction type, the
    module draws and scores the value in one call if the distribution class
    defines the method: ``dist.rsample_and_log_prob(sample_shape)`` if
    ``dist.has_rsample``, ``dist.sample_and_log_prob(sample_shape)``
    otherwise. The method returns a ``(sample, log_prob)`` tuple, with
    ``log_prob`` in the layout of ``dist.log_prob``. Define it when
    ``dist.log_prob(sample)`` cannot recover the draw exactly, for example
    for a tanh-squashed normal that saturates. Otherwise the module calls
    ``dist.log_prob`` on the sample. The :mod:`torch.distributions` classes
    do not define these methods;
    :class:`~tensordict.nn.distributions.CompositeDistribution` defines both,
    and calls them on its components when they define them.

    The default ``distribution_class`` is
    :class:`~tensordict.nn.distributions.Delta`, which always returns its
    parameter. With it, the module copies its input to ``out_keys``.

    If every key in ``out_keys`` is already in the input and
    :func:`~tensordict.nn.skip_existing` is ``True`` (see
    :class:`~tensordict.nn.set_skip_existing`), the module returns the input
    unchanged. Otherwise it computes a new value and overwrites the entry.
    See :ref:`modules-guide` for a guide.

    .. _log-prob-keys:

    .. rubric:: Log-probability keys

    With ``return_log_prob=True``, the module also writes the log-probability
    of the value. The key depends on :func:`~tensordict.nn.composite_lp_aggregate`
    (``False`` by default, see :class:`~tensordict.nn.set_composite_lp_aggregate`)
    and on the number of sampled keys:

    .. list-table::
        :header-rows: 1

        * - Case
          - ``composite_lp_aggregate()`` is ``False`` (default)
          - ``composite_lp_aggregate()`` is ``True``
        * - One sampled key, ``"action"``
          - ``"action_log_prob"``
          - ``"sample_log_prob"``
        * - One nested key, ``("agent", "action")``
          - ``("agent", "action_log_prob")``
          - ``"sample_log_prob"``
        * - Several sampled keys from a
            :class:`~tensordict.nn.distributions.CompositeDistribution`,
            ``"cont"`` and ``("nested", "disc")``
          - ``"cont_log_prob"`` and ``("nested", "disc_log_prob")``
          - The same two entries, and ``"sample_log_prob"`` with their sum
        * - Several probabilistic modules in a
            :class:`~tensordict.nn.ProbabilisticTensorDictSequential`
          - One ``<key>_log_prob`` entry per sampled key
          - Each module writes ``"sample_log_prob"``, so the entry holds
            the log-probability of the last module only. Call
            :meth:`ProbabilisticTensorDictSequential.log_prob` to get the sum.

    The suffix ``_log_prob`` goes on the last element of a nested key. Pass
    ``log_prob_key`` (one sampled key) or ``log_prob_keys`` (several sampled
    keys) to choose other keys.

    :meth:`log_prob` returns a tensor, except for a
    :class:`~tensordict.nn.distributions.CompositeDistribution` while
    ``composite_lp_aggregate()`` is ``False``: then it returns a
    :class:`~tensordict.TensorDict` with the ``<key>_log_prob`` entries of the
    table. With ``True``, the tensor is the sum over the sampled keys and their
    event dimensions.

    Set :func:`~tensordict.nn.composite_lp_aggregate` before you build the
    module. If the value changed since then, :attr:`log_prob_key`,
    :attr:`log_prob_keys`, :meth:`log_prob` and a call with
    ``return_log_prob=True`` can raise a ``RuntimeError``.

    Args:
        in_keys (NestedKey | List[NestedKey] | Dict[str, NestedKey]): key(s) that will be read from the input TensorDict
            and used to build the distribution.
            If it is a list of NestedKey or a NestedKey, the leaf (last element) of those keys must match the keywords used by
            the distribution class of interest, e.g. ``"loc"`` and ``"scale"`` for
            the :class:`~torch.distributions.Normal` distribution and similar.
            If in_keys is a dictionary, the keys are the keys of the distribution and the values are the keys in the
            tensordict that will get match to the corresponding distribution keys.
        out_keys (NestedKey | List[NestedKey] | None): key(s) where the sampled values will be written.
            If the input already has these keys, the module overwrites them,
            unless :func:`~tensordict.nn.skip_existing` is ``True``.

    Keyword Args:
        default_interaction_type (InteractionType, optional): keyword-only argument.
            Default method to be used to retrieve
            the output value. Should be one of InteractionType: DETERMINISTIC, MODE, MEDIAN, MEAN or RANDOM
            (in which case the value is sampled randomly from the distribution). Default
            is DETERMINISTIC.

            .. note:: When a value is computed, the module first reads
                :func:`~tensordict.nn.interaction_type`. If it
                returns ``None`` (outside of :class:`set_interaction_type`),
                the module uses ``default_interaction_type``. TorchRL
                collectors sample with ``InteractionType.RANDOM`` by default
                (their ``exploration_type`` argument).

            .. note::
                ``MODE`` and ``MEDIAN`` read ``dist.mode`` and ``dist.median``,
                and raise ``NotImplementedError`` if the distribution does
                not have them. ``MEAN`` reads ``dist.mean``. If the
                distribution does not have a mean, the module averages
                ``n_empirical_estimate`` samples.

        distribution_class (Type or Callable[[Any], Distribution], optional): keyword-only argument.
            A :class:`torch.distributions.Distribution` class to
            be used for sampling.
            Default is :class:`~tensordict.nn.distributions.Delta`.

            .. note::
                If the distribution class is of type
                :class:`~tensordict.nn.distributions.CompositeDistribution`, the ``out_keys``
                can be inferred directly from the ``"distribution_map"`` or ``"name_map"``
                keyword arguments provided through this class' ``distribution_kwargs``
                keyword argument, making the ``out_keys`` optional in such cases.

        distribution_kwargs (dict, optional): keyword-only argument.
            Keyword-argument pairs to be passed to the distribution.

            .. note:: if your kwargs contain tensors that you would like to transfer to device with the module, or
                tensors that should see their dtype modified when calling `module.to(dtype)`, you can wrap the kwargs
                in a :class:`~tensordict.TensorDictParams` to do this automatically.

        return_log_prob (bool, optional): keyword-only argument.
            If ``True``, the log-probability of the
            distribution sample will be written in the tensordict, at the key
            given in the table above. Default is ``False``.
        log_prob_keys (List[NestedKey], optional): keys where to write the log_prob if ``return_log_prob=True``.
            Defaults to `'<sample_key_name>_log_prob'`, where `<sample_key_name>` is each of the :attr:`out_keys`.

            .. note:: This is only available when :func:`~tensordict.nn.composite_lp_aggregate` is set to ``False``.

        log_prob_key (NestedKey, optional): key where to write the log_prob if ``return_log_prob=True``.
            Defaults to `'sample_log_prob'` when :func:`~tensordict.nn.composite_lp_aggregate` is set to `True`
            or `'<sample_key_name>_log_prob'` otherwise.

            .. note:: When there is more than one sampled key, this is only available when :func:`~tensordict.nn.composite_lp_aggregate` is set to ``True``.

        cache_dist (bool, optional): keyword-only argument. Has no effect.
            Default is ``False``.

            .. deprecated:: 0.15
                Passing ``cache_dist=True`` warns, and the argument will be
                removed in TensorDict 0.17. Do not pass it.
        n_empirical_estimate (int, optional): keyword-only argument.
            Number of samples to compute the empirical
            mean when it is not available. Defaults to 1000.
        num_samples (int or torch.Size, optional): keyword-only argument.
            The number of values per call. The output tensordict is expanded
            to ``num_samples + batch_size``, and each sampled value gets these
            leading dimensions. With the ``RANDOM`` interaction type, each
            value is a new draw. With another interaction type, the module
            repeats its one deterministic value ``num_samples`` times.
            Defaults to ``None``: one value, no extra dimension.
        generator (torch.Generator, int, NestedKey, or None, optional): keyword-only argument.
            Controls the random number generator used when sampling from the distribution
            (relevant only for ``RANDOM`` and ``MEAN`` interaction types).

            - ``None`` (default): the global PyTorch RNG is used (current behaviour).
            - :class:`torch.Generator`: this generator is used for every sample. Its state
              advances in place across calls, so consecutive forward passes draw distinct
              samples. The generator must live on the same device as the distribution
              parameters.
            - :class:`int`: shorthand for ``torch.Generator().manual_seed(int)``; a CPU
              generator is created at construction time and used statefully thereafter.
            - ``NestedKey`` (``str`` or ``tuple``): the generator is fetched from the
              input tensordict at this key on every call. The value may be a
              :class:`torch.Generator` (used in place, state mutates) or an integer scalar
              (read as a stream-key, then a fresh ``next_seed`` is written back to the
              same key after sampling, JAX-PRNG-style).

            .. note:: PyTorch's :class:`~torch.distributions.Distribution` API does not
                accept a ``generator`` argument. Internally, sampling is wrapped in
                :func:`torch.random.fork_rng` and the global RNG state is swapped with the
                provided generator's state for the duration of the call. This makes the
                approach portable across all distributions (including reparameterized
                ``rsample`` and discrete ``sample``). Under ``torch.compile``,
                the argument is ignored, without a warning.

    Examples:
        >>> import torch
        >>> from tensordict import TensorDict
        >>> from tensordict.nn import (
        ...     ProbabilisticTensorDictModule,
        ...     ProbabilisticTensorDictSequential,
        ...     TensorDictModule,
        ... )
        >>> from tensordict.nn.distributions import NormalParamExtractor
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
                action_log_prob: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False),
                hidden: Tensor(shape=torch.Size([3, 8]), device=cpu, dtype=torch.float32, is_shared=False),
                input: Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.float32, is_shared=False),
                loc: Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.float32, is_shared=False),
                params: Tensor(shape=torch.Size([3, 8]), device=cpu, dtype=torch.float32, is_shared=False),
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
                action_log_prob: Tensor(shape=torch.Size([4, 3]), device=cpu, dtype=torch.float32, is_shared=False),
                hidden: Tensor(shape=torch.Size([4, 3, 8]), device=cpu, dtype=torch.float32, is_shared=False),
                input: Tensor(shape=torch.Size([4, 3, 4]), device=cpu, dtype=torch.float32, is_shared=False),
                loc: Tensor(shape=torch.Size([4, 3, 4]), device=cpu, dtype=torch.float32, is_shared=False),
                params: Tensor(shape=torch.Size([4, 3, 8]), device=cpu, dtype=torch.float32, is_shared=False),
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
        default_interaction_type: InteractionType = InteractionType.DETERMINISTIC,
        distribution_class: type = Delta,
        distribution_kwargs: dict | None = None,
        return_log_prob: bool = False,
        log_prob_keys: List[NestedKey] | None = None,
        log_prob_key: NestedKey | None = None,
        cache_dist: bool = False,
        n_empirical_estimate: int = 1000,
        num_samples: int | torch.Size | None = None,
        generator: torch.Generator | int | NestedKey | None = None,
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
            from tensordict.nn.distributions import _distributions_maps

            distribution_class = _distributions_maps.get(distribution_class.lower())
        self.distribution_class = distribution_class
        self.distribution_kwargs = distribution_kwargs
        self.n_empirical_estimate = n_empirical_estimate
        self._dist = None
        if cache_dist:
            warn_deprecated(
                "ProbabilisticTensorDictModule(cache_dist=True)", removal="0.17"
            )
        self.cache_dist = cache_dist if hasattr(distribution_class, "update") else False
        self.return_log_prob = return_log_prob
        if isinstance(num_samples, (int, torch.SymInt)):
            num_samples = torch.Size((num_samples,))
        self.num_samples = num_samples
        self._composite_lp_aggreate_at_init = composite_lp_aggregate()

        # Generator: either a stateful Generator owned by this module, or a NestedKey
        # to pull a Generator/seed from the input tensordict on every call.
        if generator is None:
            self._generator = None
            self._generator_key = None
        elif isinstance(generator, torch.Generator):
            self._generator = generator
            self._generator_key = None
        elif isinstance(generator, int) and not isinstance(generator, bool):
            owned = torch.Generator()
            owned.manual_seed(generator)
            self._generator = owned
            self._generator_key = None
        elif isinstance(generator, (str, tuple)):
            self._generator = None
            self._generator_key = unravel_key(generator)
        else:
            raise TypeError(
                f"generator must be a torch.Generator, int, NestedKey, or None; "
                f"got {type(generator).__name__}."
            )

    def __repr__(self):
        return (
            f"{type(self).__name__}("
            f"\n{indent(f'in_keys={self.in_keys}', 4 * ' ')},"
            f"\n{indent(f'out_keys={self.out_keys}', 4 * ' ')},"
            f"\n{indent(f'distribution_class={self.distribution_class}', 4 * ' ')}, "
            f"\n{indent(f'distribution_kwargs={self.distribution_kwargs})', 4 * ' ')},"
            f"\n{indent(f'default_interaction_type={self.default_interaction_type})', 4 * ' ')},"
            f"\n{indent(f'num_samples={self.num_samples})', 4 * ' ')})"
        )

    @property
    def out_keys(self) -> List[NestedKey]:
        out_keys = list(self._out_keys)
        if self.return_log_prob:
            if not composite_lp_aggregate():
                if out_keys[-len(self.log_prob_keys) :] != self.log_prob_keys:
                    out_keys.extend(self.log_prob_keys)
            elif self.log_prob_key not in out_keys:
                out_keys.append(self.log_prob_key)
        return out_keys

    @property
    def log_prob_key(self):
        """The key where the log-probability is written, when there is one such key.

        See the :ref:`log-probability key table <log-prob-keys>`.

        Raises:
            RuntimeError: if :func:`~tensordict.nn.composite_lp_aggregate`
                changed since the module was built, or if it is ``False`` and
                the module has several sampled keys (use :attr:`log_prob_keys`).
        """
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
        return self._log_prob_key

    @property
    def log_prob_keys(self):
        """The list of keys where the log-probabilities are written.

        One key per sampled key when :func:`~tensordict.nn.composite_lp_aggregate`
        is ``False``, and ``[log_prob_key]`` when it is ``True``. See the
        :ref:`log-probability key table <log-prob-keys>`.

        Raises:
            RuntimeError: if :func:`~tensordict.nn.composite_lp_aggregate`
                changed since the module was built.
        """
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
        """Creates a :class:`torch.distributions.Distribution` instance with the parameters provided in the input tensordict.

        Args:
            tensordict (TensorDictBase): The input tensordict containing the distribution parameters.

        Returns:
            A :class:`torch.distributions.Distribution` instance created from the input tensordict.

        Raises:
            TypeError: If the input tensordict does not match the distribution keywords.

        Examples:
            >>> import torch
            >>> from torch import distributions as d
            >>> from tensordict import TensorDict
            >>> from tensordict.nn import ProbabilisticTensorDictModule
            >>> module = ProbabilisticTensorDictModule(
            ...     in_keys=["loc", "scale"], out_keys=["action"], distribution_class=d.Normal
            ... )
            >>> td = TensorDict(loc=torch.zeros(3), scale=torch.ones(3))
            >>> dist = module.get_dist(td)
            >>> dist
            Normal(loc: torch.Size([3]), scale: torch.Size([3]))
            >>> dist.entropy()
            tensor([1.4189, 1.4189, 1.4189])
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
        """Computes the log-probability of the value stored in ``tensordict``.

        Args:
            tensordict (TensorDictBase): The input tensordict containing the
                distribution parameters and the value at ``out_keys``.

        Keyword Args:
            dist (torch.distributions.Distribution, optional): The distribution instance. Defaults to ``None``.
                If ``None``, the distribution will be computed using the `get_dist` method.

        Returns:
            A tensor. For a
            :class:`~tensordict.nn.distributions.CompositeDistribution` while
            :func:`~tensordict.nn.composite_lp_aggregate` is ``False``, a
            TensorDict with one ``<key>_log_prob`` entry per sampled key (see
            the :ref:`log-probability key table <log-prob-keys>`). For a
            composite distribution while it is ``True``, the sum of these
            entries, which are also written into ``tensordict``.

        Examples:
            >>> import torch
            >>> from torch import distributions as d
            >>> from tensordict import TensorDict
            >>> from tensordict.nn import ProbabilisticTensorDictModule
            >>> module = ProbabilisticTensorDictModule(
            ...     in_keys=["loc", "scale"], out_keys=["action"], distribution_class=d.Normal
            ... )
            >>> td = module(TensorDict(loc=torch.zeros(3), scale=torch.ones(3)))
            >>> td["action"]
            tensor([0., 0., 0.])
            >>> module.log_prob(td)
            tensor([-0.9189, -0.9189, -0.9189])
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
            generator, writeback = self._resolve_generator(tensordict)
            log_prob = None
            joint_sample_and_log_prob = None
            if (
                self.return_log_prob
                and (interaction_type() or self.default_interaction_type)
                is InteractionType.RANDOM
            ):
                joint_sample_and_log_prob = _get_joint_sample_and_log_prob(dist)
            if joint_sample_and_log_prob is not None:
                # Score the draw the distribution sampled, not a value recovered
                # from the sample (e.g. atanh of a saturated tanh)
                lp_context = (
                    set_composite_lp_aggregate(False)
                    if isinstance(dist, CompositeDistribution)
                    else contextlib.nullcontext()
                )
                with _use_generator(generator), lp_context:
                    out_tensors, log_prob = joint_sample_and_log_prob(
                        self.num_samples or torch.Size(())
                    )
            else:
                out_tensors = self._dist_sample(
                    dist, interaction_type=interaction_type(), generator=generator
                )
            if writeback is not None:
                writeback(generator)
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
                        if log_prob is None:
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
                        if log_prob is None:
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
                    if log_prob is None:
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

    def _resolve_generator(
        self, tensordict: TensorDictBase
    ) -> tuple[torch.Generator | None, Callable[[torch.Generator], None] | None]:
        """Resolve the generator to use for the current call.

        Returns a ``(generator, writeback)`` tuple. ``writeback`` is either ``None`` (no
        action needed after sampling) or a callable that, given the just-used generator,
        writes a fresh seed back to the input tensordict (stream-key form).
        """
        if self._generator is not None:
            return self._generator, None
        if self._generator_key is None:
            return None, None
        value = tensordict.get(self._generator_key, default=None)
        if value is None:
            return None, None
        unwrapped = value.data if is_non_tensor(value) else value
        if isinstance(unwrapped, torch.Generator):
            return unwrapped, None
        if isinstance(unwrapped, torch.Tensor):
            if unwrapped.numel() != 1:
                raise ValueError(
                    f"Generator key {self._generator_key!r} resolved to a tensor "
                    f"of {unwrapped.numel()} elements; expected a scalar seed."
                )
            seed = int(unwrapped.item())
            device = unwrapped.device
            seed_was_tensor = True
        elif isinstance(unwrapped, int) and not isinstance(unwrapped, bool):
            seed = unwrapped
            device = self._infer_param_device(tensordict)
            seed_was_tensor = False
        else:
            raise TypeError(
                f"Generator key {self._generator_key!r} resolved to a value of type "
                f"{type(unwrapped).__name__}; expected torch.Generator, int, or scalar Tensor."
            )
        local_gen = torch.Generator(device=device)
        local_gen.manual_seed(seed)
        key = self._generator_key

        def writeback(gen: torch.Generator) -> None:
            next_seed_t = torch.randint(
                0, 2**62, (), generator=gen, dtype=torch.int64, device=device
            )
            if seed_was_tensor:
                tensordict.set(key, next_seed_t)
            else:
                from tensordict.tensorclass import NonTensorData

                tensordict.set(key, NonTensorData(int(next_seed_t.item())))

        return local_gen, writeback

    def _infer_param_device(self, tensordict: TensorDictBase) -> torch.device:
        for k in self.in_keys:
            v = tensordict.get(k, default=None)
            if isinstance(v, torch.Tensor):
                return v.device
        return torch.device("cpu")

    def _dist_sample(
        self,
        dist: D.Distribution,
        interaction_type: InteractionType | None = None,
        *,
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, ...] | Tensor | TensorDictBase:
        if interaction_type is None:
            interaction_type = self.default_interaction_type
        sample = self._draw_dist_sample(dist, interaction_type, generator=generator)
        num_samples = self.num_samples
        if num_samples and interaction_type is not InteractionType.RANDOM:
            # Only RANDOM draws num_samples values: repeat the single
            # deterministic value so that it fits the expanded output.
            if isinstance(sample, tuple):
                return tuple(s.expand(num_samples + s.shape) for s in sample)
            return sample.expand(num_samples + sample.shape)
        return sample

    def _draw_dist_sample(
        self,
        dist: D.Distribution,
        interaction_type: InteractionType | None = None,
        *,
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, ...] | Tensor | TensorDictBase:
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
            # One read: ``deterministic_sample`` is often a computed property.
            deterministic_sample = getattr(dist, "deterministic_sample", None)
            if deterministic_sample is not None:
                return deterministic_sample
            else:
                # Fallbacks
                tdist = type(dist)
                if issubclass(tdist, D.Independent):
                    tdist = type(dist.base_dist)
                interaction_type = _DETERMINISTIC_REGISTER.get(tdist)
                if (
                    interaction_type is InteractionType.DETERMINISTIC
                    and isinstance(dist, D.Independent)
                    and hasattr(dist.base_dist, "deterministic_sample")
                ):
                    # Independent leaves the samples of its base unchanged
                    return dist.base_dist.deterministic_sample
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
            try:
                return dist.mean
            except (AttributeError, NotImplementedError):
                pass
            with _use_generator(generator):
                if dist.has_rsample:
                    return dist.rsample((self.n_empirical_estimate,)).mean(0)
                else:
                    return dist.sample((self.n_empirical_estimate,)).mean(0)

        elif interaction_type is InteractionType.RANDOM:
            num_samples = self.num_samples
            if num_samples is None:
                num_samples = torch.Size(())
            with _use_generator(generator):
                if dist.has_rsample:
                    return dist.rsample(num_samples)
                else:
                    return dist.sample(num_samples)
        else:
            raise NotImplementedError(f"unknown interaction_type {interaction_type}")


class ProbabilisticTensorDictSequential(TensorDictSequential):
    """A :class:`~tensordict.nn.TensorDictSequential` that contains one or more :class:`~tensordict.nn.ProbabilisticTensorDictModule` instances, and can return their distribution.

    Usually the last module is probabilistic and the modules before it
    compute its parameters. :meth:`get_dist` returns the distribution, and
    :meth:`log_prob` returns the log-probability of a sample. A call to the
    module runs every module and writes the sampled values (see
    :class:`~tensordict.nn.ProbabilisticTensorDictModule` for the keys of the
    log-probabilities).

    ``return_composite`` decides what :meth:`get_dist` returns when there is
    more than one probabilistic module:

    - ``False``: the distribution of the last module. The other probabilistic
      modules only write their values. The last module must be
      probabilistic, otherwise the constructor raises ``TypeError``.
    - ``True``: one :class:`~tensordict.nn.distributions.CompositeDistribution`
      that holds the distributions of all probabilistic modules.

    The default chooses ``True`` if there is more than one probabilistic
    module or if the last module is not probabilistic, and ``False``
    otherwise.

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
        partial_tolerant (bool, optional): stored as an attribute, but
            :meth:`forward` does not use it: every module runs on the whole
            input, as with ``partial_tolerant=False``. To skip the modules whose
            input keys are missing, or to run them on the members of a lazy
            stack that have the keys, put this module in a
            :class:`~tensordict.nn.TensorDictSequential` with
            ``partial_tolerant=True``. Defaults to ``False``.
        return_composite (bool, optional): If `True` and multiple
            :class:`~tensordict.nn.ProbabilisticTensorDictModule` or
            :class:`~tensordict.nn.ProbabilisticTensorDictSequential` instances are found,
            a :class:`~tensordict.nn.distributions.CompositeDistribution` instance will be used.
            Otherwise, only the last module will be used to build the distribution.
            Defaults to ``True`` whenever there are more than one probabilistic modules or the last module is not probabilistic.
            Errors if `return_composite` is `False` and the last module is not probabilistic.
        selected_out_keys (iterable of NestedKeys, optional): the list of out-keys to select. If not provided, all
            ``out_keys`` will be written.
        inplace (bool, optional): if `True`, the input tensordict is modified in-place. If `False`, a new empty
            :class:`~tensordict.TensorDict` instance is created. If `"empty"`, `input.empty()` is used instead (ie, the
            output preserves type, device and batch-size). Defaults to `None` (relies on sub-modules).

    Raises:
        ValueError: If the input sequence of modules is empty.
        TypeError: If ``return_composite=False`` and the final module is not an instance of
            :obj:`ProbabilisticTensorDictModule` or
            :obj:`ProbabilisticTensorDictSequential`.

    Examples:
        >>> # Typical usage: a single distribution is computed last in the sequence
        >>> import torch
        >>> from tensordict import TensorDict
        >>> from tensordict.nn import (
        ...     ProbabilisticTensorDictModule as Prob,
        ...     ProbabilisticTensorDictSequential as Seq,
        ...     TensorDictModule as Mod,
        ... )
        >>> _ = torch.manual_seed(0)
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
        selected_out_keys: List[NestedKey] | None = None,
        inplace: bool | None = None,
    ) -> None: ...

    @overload
    def __init__(
        self,
        modules: List[TensorDictModuleBase | ProbabilisticTensorDictModule],
        partial_tolerant: bool = False,
        return_composite: bool | None = None,
        *,
        selected_out_keys: List[NestedKey] | None = None,
        inplace: bool | None = None,
    ) -> None: ...

    def __init__(
        self,
        *modules: TensorDictModuleBase | ProbabilisticTensorDictModule,
        partial_tolerant: bool = False,
        return_composite: bool | None = None,
        selected_out_keys: List[NestedKey] | None = None,
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
                selected_out_keys=selected_out_keys,
                inplace=inplace,
            )
        else:
            modules_list = list(modules)
        modules_list = self._convert_modules(modules_list)

        if return_composite is False and not isinstance(
            modules_list[-1],
            (ProbabilisticTensorDictModule, ProbabilisticTensorDictSequential),
        ):
            raise TypeError(
                "The final module passed to ProbabilisticTensorDictSequential must be "
                "an instance of ProbabilisticTensorDictModule or another "
                "ProbabilisticTensorDictSequential (unless return_composite is set to ``True``)."
            )
        if return_composite is None:
            return_composite = not isinstance(
                modules_list[-1],
                (ProbabilisticTensorDictModule, ProbabilisticTensorDictSequential),
            ) or (
                sum(isinstance(m, ProbabilisticTensorDictModule) for m in modules_list)
                > 1
            )

        if not any(
            isinstance(
                m, (ProbabilisticTensorDictModule, ProbabilisticTensorDictSequential)
            )
            for m in modules_list
        ):
            # No probabilistic modules - initialize as a regular TensorDictSequential
            kwargs = {
                "partial_tolerant": partial_tolerant,
                "inplace": inplace,
            }
            if selected_out_keys is not None:
                kwargs["selected_out_keys"] = selected_out_keys
            TensorDictSequential.__init__(self, *modules, **kwargs)
            self._requires_sample = False
            self.__dict__["_det_part"] = None
            # Use return_composite=True so forward() just iterates through modules
            self.return_composite = True
            return

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

        kwargs = {
            "partial_tolerant": partial_tolerant,
            "inplace": inplace,
        }
        if selected_out_keys is not None:
            kwargs["selected_out_keys"] = selected_out_keys
        super().__init__(*modules, **kwargs)
        self.return_composite = return_composite

    # Override: the constructor's checks, for the modules after an edit.
    def _check_modules(self, modules: list[torch.nn.Module]) -> None:
        super()._check_modules(modules)
        if not modules:
            raise ValueError(
                "ProbabilisticTensorDictSequential must consist of zero or more "
                "TensorDictModules followed by a ProbabilisticTensorDictModule"
            )
        if not self.return_composite and not isinstance(
            modules[-1],
            (ProbabilisticTensorDictModule, ProbabilisticTensorDictSequential),
        ):
            raise TypeError(
                "The final module passed to ProbabilisticTensorDictSequential must be "
                "an instance of ProbabilisticTensorDictModule or another "
                "ProbabilisticTensorDictSequential (unless return_composite is set to ``True``)."
            )

    # Override: also rebuilds the deterministic part and the sampling flag,
    # which the constructor caches.
    def _refresh_keys(self) -> None:
        super()._refresh_keys()
        modules = list(self._module_iter())
        if not any(
            isinstance(
                m, (ProbabilisticTensorDictModule, ProbabilisticTensorDictSequential)
            )
            for m in modules
        ):
            self._requires_sample = False
            self.__dict__["_det_part"] = None
            return
        _, out_keys = self._compute_in_and_out_keys(modules[:-1])
        self._requires_sample = any(
            key not in set(out_keys)
            for key in getattr(modules[-1], "dist_sample_keys", [None])
        )
        if self._ordered_dict:
            self.__dict__["_det_part"] = TensorDictSequential(
                collections.OrderedDict(list(self.module.items())[:-1])
            )
        else:
            self.__dict__["_det_part"] = TensorDictSequential(*modules[:-1])

    def __getitem__(self, index: int | slice | str) -> Self | TensorDictModuleBase:
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
    _draw_dist_sample = ProbabilisticTensorDictModule._draw_dist_sample

    @property
    @deprecated(
        "ProbabilisticTensorDictSequential.det_part",
        removal="0.17",
        replacement="ProbabilisticTensorDictSequential.get_dist_params()",
    )
    def det_part(self):
        """The modules that compute the distribution parameters.

        .. deprecated:: 0.15
            This property will be removed in TensorDict 0.17. Use
            :meth:`get_dist_params` to compute the distribution parameters.
        """
        return self._det_part

    def get_dist_params(
        self,
        tensordict: TensorDictBase,
        tensordict_out: TensorDictBase | None = None,
        **kwargs,
    ) -> tuple[D.Distribution, TensorDictBase]:
        """Runs every module except the last one, and returns the tensordict with the distribution parameters.

        This method runs the deterministic part of the :class:`~tensordict.nn.ProbabilisticTensorDictSequential`
        module to obtain the distribution parameters. The interaction type is set to the current global
        interaction type if available, otherwise it defaults to the interaction type of the last module.

        Args:
            tensordict (TensorDictBase): The input tensordict.
            tensordict_out (TensorDictBase, optional): The tensordict where
                the parameters are written. If ``None``, they are written into
                ``tensordict``, which is returned. Pass
                ``tensordict.copy()`` to keep the input unchanged.
                Defaults to ``None``.

        Keyword Args:
            **kwargs: Additional keyword arguments passed to the deterministic part of the module.

        Returns:
            TensorDictBase: the output tensordict containing the distribution parameters.

        .. note:: The interaction type is temporarily set to the specified value during the execution of this method.
        """
        tds = self._det_part
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
                local_num_samples = tdm.num_samples
                if local_num_samples is None:
                    local_num_samples = ()
                num_samples = local_num_samples + num_samples
        return num_samples

    def get_dist(
        self,
        tensordict: TensorDictBase,
        tensordict_out: TensorDictBase | None = None,
        **kwargs,
    ) -> D.Distribution:
        """Returns the distribution resulting from passing the input tensordict through the sequence.

        If `return_composite` is ``False``, this method will only consider the last probabilistic module in the sequence.
        It runs the modules before it with :meth:`get_dist_params`, so the
        intermediate entries are written into ``tensordict``.

        Otherwise, it will return a :class:`~tensordict.nn.distributions.CompositeDistribution` instance containing the distributions of all probabilistic modules.
        It runs the modules on a shallow copy of ``tensordict``, so the input
        does not get new entries.

        Args:
            tensordict (TensorDictBase): The input tensordict.
            tensordict_out (TensorDictBase, optional): Used only when
                ``return_composite`` is ``False``: the tensordict where the
                intermediate entries are written, instead of ``tensordict``.
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
                    if isinstance(tdm, ProbabilisticTensorDictModule):
                        gen, writeback = tdm._resolve_generator(td_copy)
                    else:
                        gen, writeback = None, None
                    sample = tdm._dist_sample(
                        dist, interaction_type=interaction_type(), generator=gen
                    )
                    if writeback is not None:
                        writeback(gen)
                    if tdm.num_samples not in ((), None):
                        td_copy = td_copy.expand(tdm.num_samples + td_copy.shape)
                    if isinstance(tdm, ProbabilisticTensorDictModule):
                        if isinstance(sample, torch.Tensor):
                            sample = [sample]
                        td_copy.update(dict(_zip_strict(tdm.dist_sample_keys, sample)))
                    elif isinstance(sample, torch.Tensor):
                        # A nested sequence with return_composite=False gives the
                        # distribution of its last module, whose keys come first in
                        # dist_sample_keys
                        td_copy.set(tdm.dist_sample_keys[0], sample)
                    else:
                        td_copy.update(sample)
                if isinstance(dist, CompositeDistribution):
                    dists.update(dict(dist))
                else:
                    dists[tdm.dist_sample_keys[0]] = dist
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

        If `self.return_composite` is ``True`` and the distribution is a :class:`~tensordict.nn.distributions.CompositeDistribution`,
        this method will return the log-probability of the entire composite distribution.

        Otherwise, it will only consider the last probabilistic module in the sequence.
        In that case the intermediate entries are written into ``tensordict``,
        as in :meth:`get_dist`.

        Args:
            tensordict (TensorDictBase): The input tensordict, with the
                sampled values.
            tensordict_out (TensorDictBase, optional): If given, the
                distribution is computed on a copy of ``tensordict``, so the
                input does not get new entries. Defaults to ``None``.

        Keyword Args:
            dist (torch.distributions.Distribution, optional): The distribution object. If ``None``, it will be computed using `get_dist`.
                Defaults to ``None``.

        Returns:
            TensorDictBase or torch.Tensor: The log-probability of the input tensordict.

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
        _requires_sample: bool | None = None,
        **kwargs,
    ) -> TensorDictBase:
        # None means this sequence's own flag. A nested sequence receives the
        # caller's value as a real parameter, not through **kwargs, and still
        # keeps a sample its own prefix wrote.
        if _requires_sample is None:
            _requires_sample = self._requires_sample
        else:
            _requires_sample = _requires_sample and self._requires_sample
        if (
            tensordict_out is not None
            or self._select_before_return
            or self.inplace is False
            or self.inplace == "empty"
        ):
            # With inplace=False or "empty", the modules must not write into the input.
            tensordict_exec = tensordict.copy()
        else:
            tensordict_exec = tensordict
        if self.return_composite:
            last_index = len(self) - 1
            for i, m in enumerate(self._module_iter()):
                try:
                    if isinstance(
                        m,
                        (
                            ProbabilisticTensorDictModule,
                            ProbabilisticTensorDictSequential,
                        ),
                    ):
                        # Only the last module may keep a sample the prefix
                        # already wrote. Every earlier one still samples.
                        module_requires_sample = (
                            _requires_sample if i == last_index else True
                        )
                        # A nested sequence also passes **kwargs to its modules.
                        module_kwargs = (
                            kwargs
                            if isinstance(m, ProbabilisticTensorDictSequential)
                            else {}
                        )
                        tensordict_exec = m(
                            tensordict_exec,
                            _requires_sample=module_requires_sample,
                            **module_kwargs,
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
            last_module = self._last_module
            if isinstance(last_module, ProbabilisticTensorDictSequential):
                tensordict_exec = last_module(
                    tensordict_exec, _requires_sample=_requires_sample, **kwargs
                )
            elif isinstance(last_module, ProbabilisticTensorDictModule):
                tensordict_exec = last_module(
                    tensordict_exec, _requires_sample=_requires_sample
                )
            else:
                tensordict_exec = last_module(tensordict_exec, **kwargs)

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
                    keys = list(set(self.out_keys).union(tensordict.keys(True, True)))
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


__getattr__ = deprecated_attributes(
    __name__,
    {
        "DETERMINISTIC_REGISTER": (
            _DETERMINISTIC_REGISTER,
            "a deterministic_sample attribute on the distribution",
        ),
    },
    removal="0.17",
)
