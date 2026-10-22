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
from typing import Any, Callable, Dict, List, overload, TYPE_CHECKING

import torch
from tensordict._deprecation import deprecated, deprecated_attributes, warn_deprecated
from tensordict._nestedkey import NestedKey
from tensordict._td import TensorDict
from tensordict.base import _is_inside, is_tensor_collection
from tensordict.nn.common import dispatch, TensorDictModuleBase
from tensordict.nn.distributions.composite import (
    _add_suffix,
    _resolve_name_map,
    CompositeDistribution,
)
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
from tensordict.utils import (
    _ContextManager,
    _user_stacklevel,
    _zip_strict,
    unravel_key,
)
from torch import distributions as D, Tensor
from torch.compiler import is_compiling
from torch.utils._contextlib import _DecoratorContextManager

try:
    from enum import StrEnum
except ImportError:
    from .utils import _StrEnum as StrEnum

if TYPE_CHECKING:
    from typing_extensions import Self
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

        ``InteractionType(type_str)`` does the same.

        Args:
            type_str (str): the name, for example ``"random"`` or ``"MODE"``.

        Raises:
            ValueError: if no interaction type has this name.
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
    previous value comes back. By default the interaction type applies to
    the whole process (see ``thread_local``).

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


def _rename_entries(
    tensordict: TensorDictBase, renames: list[tuple[NestedKey, NestedKey]]
) -> None:
    # Read every value before writing any, so that keys swapping names do not
    # overwrite each other
    values = [tensordict.pop(old_key) for old_key, _ in renames]
    for (_, new_key), value in zip(renames, values):
        tensordict.set(new_key, value)


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
    log-probability of a value, call :meth:`log_prob`. See
    :ref:`modules-probabilistic` for a guide.

    The value depends on the interaction type: the value of
    :func:`~tensordict.nn.interaction_type` if it is set (see
    :class:`~tensordict.nn.set_interaction_type`), ``default_interaction_type``
    otherwise. See :class:`~tensordict.nn.InteractionType` for what each type
    reads from the distribution.

    The default ``distribution_class`` is
    :class:`~tensordict.nn.distributions.Delta`, which always returns its
    parameter. With it, the module copies its input to ``out_keys``.

    If every key in ``out_keys`` is already in the input and
    :func:`~tensordict.nn.skip_existing` is ``True`` (see
    :class:`~tensordict.nn.set_skip_existing`), the module returns the input
    unchanged. Otherwise it computes a new value and overwrites the entry.

    With ``return_log_prob=True`` and the ``RANDOM`` interaction type, the
    module draws and scores the value in one call if the distribution defines
    the method: ``dist.rsample_and_log_prob(sample_shape)`` if
    ``dist.has_rsample``, ``dist.sample_and_log_prob(sample_shape)``
    otherwise. The method returns a ``(sample, log_prob)`` tuple, with
    ``log_prob`` in the layout of ``dist.log_prob``. Define it when
    ``dist.log_prob(sample)`` cannot recover the draw exactly, for example
    for a tanh-squashed normal that saturates. Otherwise the module calls
    ``dist.log_prob`` on the sample. The :mod:`torch.distributions` classes
    do not define these methods;
    :class:`~tensordict.nn.distributions.CompositeDistribution` defines both.

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
        in_keys (NestedKey, list of NestedKey, or dict): the entries that hold
            the parameters of the distribution. With a key or a list of keys,
            the last element of each key is the keyword argument of
            ``distribution_class`` that it feeds: ``["loc", "scale"]`` for a
            :class:`~torch.distributions.Normal`, and ``("agent", "loc")``
            feeds ``loc``. With a dict, the keys are the keyword arguments and
            the values are the tensordict keys:
            ``{"loc": "mu", "scale": "sigma"}``.
        out_keys (NestedKey or list of NestedKey, optional): the keys where the
            values are written, one per value that the distribution returns.
            If the input already has these keys, the module overwrites them,
            unless :func:`~tensordict.nn.skip_existing` is ``True``. Defaults
            to ``None``: for a
            :class:`~tensordict.nn.distributions.CompositeDistribution`, the
            values of the ``name_map`` of ``distribution_kwargs`` if it has
            one, and the keys of its ``distribution_map`` otherwise. A
            component that ``name_map`` leaves out is written under its own
            name but is not in ``out_keys``, and the module warns. Starting
            with TensorDict 0.17, the default will change to ``out_keys`` that
            include these components. For another distribution, ``None``
            gives the key ``"_"``: the module writes the sample there, and
            warns the first time it does.
            Starting with TensorDict 0.17, the default will change to not
            writing the sample at ``"_"``, which marks an output that is not
            written, as in :class:`~tensordict.nn.TensorDictModule`. Pass
            ``out_keys`` to keep the current keys.

    Keyword Args:
        default_interaction_type (InteractionType or str, optional): the
            interaction type that the module uses when
            :func:`~tensordict.nn.interaction_type` returns ``None``, that is
            outside of a :class:`~tensordict.nn.set_interaction_type` block.
            ``MODE`` and ``MEDIAN`` raise ``NotImplementedError`` if the
            distribution does not have them. ``MEAN`` averages
            ``n_empirical_estimate`` samples if the distribution has no mean.
            TorchRL collectors sample with ``InteractionType.RANDOM`` by
            default (their ``exploration_type`` argument). Defaults to
            ``InteractionType.DETERMINISTIC``.
        distribution_class (type or Callable, optional): the distribution
            class, or a callable that takes the parameters as keyword arguments
            and returns a :class:`torch.distributions.Distribution`. For a
            :class:`~tensordict.nn.distributions.CompositeDistribution`,
            ``distribution_kwargs`` holds its ``distribution_map``, and
            ``out_keys`` can be left out. Defaults to
            :class:`~tensordict.nn.distributions.Delta`.
        distribution_kwargs (dict, optional): extra keyword arguments passed
            to ``distribution_class``, for example constants. To move the
            tensors in it with the module (``module.to(device)``,
            ``module.to(dtype)``), pass a :class:`~tensordict.TensorDictParams`.
            Defaults to ``None``.
        return_log_prob (bool, optional): if ``True``, the module also writes
            the log-probability of the value, at the keys of the table above.
            Defaults to ``False``.
        log_prob_keys (list of NestedKey, optional): the keys of the
            log-probabilities, one per sampled key, in place of the
            ``<key>_log_prob`` defaults, including for a
            :class:`~tensordict.nn.distributions.CompositeDistribution`. If
            :func:`~tensordict.nn.composite_lp_aggregate` is ``True``, passing
            it raises a ``RuntimeError``. Defaults to ``None``.
        log_prob_key (NestedKey, optional): the key of the log-probability:
            of the one sampled key, or, while
            :func:`~tensordict.nn.composite_lp_aggregate` is ``True``, of their
            sum. With several sampled keys while it is ``False``, passing it
            raises a ``RuntimeError``. Defaults to ``None``, which gives the
            key of the table above.

            With ``return_log_prob=True``, a custom ``log_prob_key`` or
            ``log_prob_keys`` entry equal to a sample key, inside one, or
            holding one raises a ``ValueError``, because the log-probability
            would overwrite the sample. The check runs when
            ``distribution_class`` is a class (and, for a
            ``CompositeDistribution``, when ``distribution_kwargs`` holds its
            ``distribution_map``).
        cache_dist (bool, optional): has no effect. Deprecated: passing
            ``True`` warns, and the argument will be removed in TensorDict
            0.17. Defaults to ``False``.
        n_empirical_estimate (int, optional): the number of samples that
            ``MEAN`` averages when the distribution has no mean. Defaults to
            ``1000``.
        num_samples (int or torch.Size, optional): the number of values per
            call. The output tensordict is expanded to
            ``num_samples + batch_size``, and each value gets these leading
            dimensions. With the ``RANDOM`` interaction type, each value is a
            new draw. With another interaction type, the module repeats its
            one deterministic value. Defaults to ``None``: one value, no extra
            dimension.
        generator (torch.Generator, int or NestedKey, optional): the random
            number generator of the sampling, used by the ``RANDOM``
            interaction type and by the sampled estimate of ``MEAN``.

            - ``None`` (default): the global PyTorch generator.
            - a :class:`torch.Generator`: used for every call. Its state
              advances in place, so consecutive calls draw different values.
              It must be on the device of the distribution parameters.
            - an ``int``: the seed of a CPU generator that the constructor
              creates, then used as above.
            - a ``NestedKey``: the key of an entry of the input, read at each
              call. The entry holds a :class:`torch.Generator`, used as above,
              or an integer seed (a scalar tensor or an ``int``). A seed gives
              a new generator, and after sampling the module writes a new seed,
              drawn from that generator, back at the key. If the input has
              no such entry, the global generator is used.

            Sampling with a generator does not change the state of the global
            generator, and works with every distribution. Under
            :func:`torch.compile`, the argument is ignored, without a warning.

    Examples:
        Sample from a normal distribution, and write the log-probability of
        the sample:

        >>> import torch
        >>> from torch import distributions as d
        >>> from tensordict import TensorDict
        >>> from tensordict.nn import InteractionType, ProbabilisticTensorDictModule
        >>> module = ProbabilisticTensorDictModule(
        ...     in_keys=["loc", "scale"],
        ...     out_keys=["action"],
        ...     distribution_class=d.Normal,
        ...     default_interaction_type=InteractionType.RANDOM,
        ...     return_log_prob=True,
        ... )
        >>> _ = torch.manual_seed(0)
        >>> td = module(TensorDict(loc=torch.zeros(3), scale=torch.ones(3)))
        >>> td["action"]
        tensor([ 1.5410, -0.2934, -2.1788])
        >>> td["action_log_prob"]
        tensor([-2.1063, -0.9620, -3.2925])

        Compute the parameters with a network, in a
        :class:`~tensordict.nn.ProbabilisticTensorDictSequential`:

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
        TensorDict(batch_size=[3], fields={
            action: f32[4],
            action_log_prob: f32[],
            hidden: f32[8],
            input: f32[4],
            loc: f32[4],
            params: f32[8],
            scale: f32[4]})
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
        TensorDict(batch_size=[4,3], fields={
            action: f32[4],
            action_log_prob: f32[],
            hidden: f32[8],
            input: f32[4],
            loc: f32[4],
            params: f32[8],
            scale: f32[4]})

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
                    mapped = {unravel_key(name) for name in name_map}
                    if any(
                        unravel_key(name) not in mapped for name in distribution_map
                    ):
                        warnings.warn(
                            "ProbabilisticTensorDictModule got a name_map that leaves "
                            "out components of distribution_map and no out_keys: these "
                            "components are written but are not in out_keys. Starting "
                            "with TensorDict 0.17, the default will change to out_keys "
                            "that include them. Pass out_keys to keep the current keys.",
                            FutureWarning,
                            stacklevel=_user_stacklevel(),
                        )
                else:
                    out_keys = list(distribution_map.keys())
            else:
                out_keys = ["_"]
                self._warn_default_out_key = True
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
        # (default key, custom key) pairs, renamed by
        # ProbabilisticTensorDictSequential.log_prob
        self._custom_log_prob_keys = []
        if log_prob_keys is not None and len(log_prob_keys) == len(self._out_keys):
            for out_key, lp_key in zip(self._out_keys, log_prob_keys):
                lp_key_expected = _add_suffix(out_key, "_log_prob")
                if lp_key != lp_key_expected:
                    self._custom_log_prob_keys.append((lp_key_expected, lp_key))

        self.default_interaction_type = InteractionType(default_interaction_type)

        if isinstance(distribution_class, str):
            from tensordict.nn.distributions import _distributions_maps

            distribution_class = _distributions_maps.get(distribution_class.lower())
        self.distribution_class = distribution_class
        self.distribution_kwargs = distribution_kwargs
        if return_log_prob:
            self._check_log_prob_keys(log_prob_keys, log_prob_key)
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

    def _check_log_prob_keys(self, log_prob_keys, log_prob_key) -> None:
        # A custom log-probability key at a sample key, inside one, or holding
        # one would overwrite the sample. The sample keys are known only for a
        # distribution class: a partial or a factory function is not checked.
        distribution_class = self.distribution_class
        if not isinstance(distribution_class, type):
            return
        if issubclass(distribution_class, CompositeDistribution):
            distribution_map = self.distribution_kwargs.get("distribution_map")
            if distribution_map is None:
                return
            sample_keys = list(
                _resolve_name_map(
                    self.distribution_kwargs.get("name_map"), distribution_map
                ).values()
            )
        else:
            sample_keys = self._out_keys
        # The default keys behave as they always did
        default_keys = {
            _add_suffix(key, "_log_prob") for key in [*self._out_keys, *sample_keys]
        }
        default_keys.add("sample_log_prob")
        lp_keys = log_prob_keys if log_prob_keys is not None else [log_prob_key]
        clashes = [
            (lp_key, sample_key)
            # A key of another type, such as a list, fails at the first call
            for lp_key in map(
                unravel_key, (key for key in lp_keys if isinstance(key, (str, tuple)))
            )
            if lp_key not in default_keys
            for sample_key in sample_keys
            if lp_key == sample_key
            or _is_inside(lp_key, sample_key)
            or _is_inside(sample_key, lp_key)
        ]
        if clashes:
            raise ValueError(
                "A custom log-probability key cannot be a sample key, lie inside one "
                "or hold one, since writing the log-probability would overwrite the "
                "sample. "
                "Got "
                + ", ".join(
                    f"log-probability key {lp_key!r} for sample key {sample_key!r}"
                    for lp_key, sample_key in clashes
                )
                + "."
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
        """The keys that the module writes: the sampled keys, then the log-probability keys if ``return_log_prob=True``.

        See the :ref:`log-probability key table <log-prob-keys>`.
        """
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
        """The keys of the distribution parameters in the input: the ``in_keys``."""
        return list(self.in_keys)

    @property
    def dist_sample_keys(self) -> List[NestedKey]:
        """The keys of the sampled values: the ``out_keys`` without the log-probability keys."""
        return list(self._out_keys)

    def get_dist(self, tensordict: TensorDictBase) -> D.Distribution:
        """Creates a :class:`torch.distributions.Distribution` instance with the parameters provided in the input tensordict.

        Args:
            tensordict (TensorDictBase): the input, with the distribution
                parameters at ``in_keys``.

        Returns:
            A :class:`torch.distributions.Distribution` instance created from the input tensordict.

        Raises:
            TypeError: if the entries of the input do not match the keyword
                arguments of the distribution.

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

    # Set when out_keys defaults to ["_"]; the sample is not written there from 0.17
    _warn_default_out_key = False

    _CHANGE_IN_C_LP_A = "The value returned by composite_lp_aggregate changed between init of {} and its execution ({} -> {}). Make sure the mode matches."

    def log_prob(
        self,
        tensordict,
        *,
        dist: torch.distributions.Distribution | None = None,
    ):
        """Computes the log-probability of the value stored in ``tensordict``.

        Args:
            tensordict (TensorDictBase): the input, with the distribution
                parameters at ``in_keys`` and the value at ``out_keys``.

        Keyword Args:
            dist (torch.distributions.Distribution, optional): the
                distribution. Defaults to ``None``, which builds it from
                ``tensordict`` with :meth:`get_dist`.

        Returns:
            A tensor. For a
            :class:`~tensordict.nn.distributions.CompositeDistribution` while
            :func:`~tensordict.nn.composite_lp_aggregate` is ``False``, a
            tensordict with one ``<key>_log_prob`` entry per sampled key (see
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
        renames = []
        for out_key, lp_key in _zip_strict(self.dist_sample_keys, self.log_prob_keys):
            lp_key_expected = _add_suffix(out_key, "_log_prob")
            if lp_key != lp_key_expected:
                renames.append((lp_key_expected, lp_key))
        if renames:
            _rename_entries(lp, renames)

    @dispatch(auto_batch_size=False)
    @_set_skip_existing_None()
    def forward(
        self,
        tensordict: TensorDictBase,
        tensordict_out: TensorDictBase | None = None,
        _requires_sample: bool = True,
    ) -> TensorDictBase:
        """Builds the distribution from ``tensordict`` and writes a value from it at ``out_keys``.

        With ``return_log_prob=True``, the module also writes the
        log-probability of the value. You can also call the module with the
        parameters as keyword arguments, ``module(loc=..., scale=...)``, and
        get the outputs back (see :func:`~tensordict.nn.dispatch`).

        Args:
            tensordict (TensorDictBase): the input, with the distribution
                parameters at ``in_keys``.
            tensordict_out (TensorDictBase, optional): the tensordict where
                the outputs are written. Defaults to ``None``, which writes
                them into ``tensordict``.

        Returns:
            The tensordict that holds the outputs, ``tensordict_out`` or
            ``tensordict``. With ``num_samples``, a new tensordict, which is
            that tensordict expanded to ``num_samples + batch_size``. The
            input does not get the outputs then.
        """
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
                        self._update_td_lp(log_prob)
                        out_tensors.update(log_prob)
                tensordict_out.update(out_tensors)
            else:
                if isinstance(out_tensors, Tensor):
                    out_tensors = (out_tensors,)
                if (
                    self._warn_default_out_key
                    and not torch.compiler.is_dynamo_compiling()
                ):
                    self._warn_default_out_key = False
                    warnings.warn(
                        "ProbabilisticTensorDictModule got no out_keys and writes "
                        'its sample at "_". Starting with TensorDict 0.17, the '
                        'default will change to not writing the sample at "_". '
                        "Pass out_keys to write the sample at a key of your "
                        "choice and silence this warning.",
                        FutureWarning,
                        stacklevel=_user_stacklevel(),
                    )
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
            if isinstance(dist, CompositeDistribution) and not (
                len(out_tensors) == 1 and is_tensor_collection(out_tensors[0])
            ):
                # A composite takes one tensordict of samples: score the
                # samples in the input, as log_prob() does
                log_prob = self.log_prob(tensordict, dist=dist)
                if is_tensor_collection(log_prob):
                    tensordict_out.update(log_prob)
                else:
                    tensordict_out.set(self.log_prob_key, log_prob)
            else:
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
    """A :class:`~tensordict.nn.TensorDictSequential` with probabilistic modules, which can return their distribution.

    Usually the last module is a
    :class:`~tensordict.nn.ProbabilisticTensorDictModule` and the modules
    before it compute its parameters. A call runs every module and writes the
    sampled values (see :class:`~tensordict.nn.ProbabilisticTensorDictModule`
    for the keys of the log-probabilities). :meth:`get_dist` returns the
    distribution, and :meth:`log_prob` returns the log-probability of a
    sample. See :ref:`modules-probabilistic` for a guide.

    When a module samples from a distribution whose parameters depend on the
    sample of an earlier module, its log-probability is conditional: for a
    value ``z`` drawn from a distribution whose parameters are computed from
    earlier samples ``x`` and ``y``, it is ``log p(z | x, y)``.

    Args:
        *modules (TensorDictModuleBase or Callable): the modules to run, in
            order, usually ending with a
            :class:`~tensordict.nn.ProbabilisticTensorDictModule`. You can
            also pass one list, or one ``OrderedDict`` or ``dict`` that maps
            names to modules. A callable that is not a
            :class:`~tensordict.nn.TensorDictModuleBase` takes and returns a
            tensordict, and its keys do not count in the ``in_keys`` and
            ``out_keys`` of the sequence.

    Keyword Args:
        partial_tolerant (bool, optional): stored as an attribute, but
            :meth:`forward` does not use it: every module runs on the whole
            input, as with ``partial_tolerant=False``. To skip the modules whose
            input keys are missing, or to run them on the members of a lazy
            stack that have the keys, put this module in a
            :class:`~tensordict.nn.TensorDictSequential` with
            ``partial_tolerant=True``. Defaults to ``False``.
        return_composite (bool, optional): what :meth:`get_dist` returns.
            If ``True``, one
            :class:`~tensordict.nn.distributions.CompositeDistribution` that
            holds the distributions of all the probabilistic modules. If
            ``False``, the distribution of the last module: the other
            probabilistic modules only write their values, and the last module
            must be probabilistic. Defaults to ``None``, which gives ``True``
            if there is more than one probabilistic module or if the last
            module is not probabilistic, and ``False`` otherwise.
        selected_out_keys (list of NestedKey, optional): the ``out_keys`` to
            write. Defaults to ``None``, which writes all the ``out_keys``.
        inplace (bool or str, optional): where the sequence writes its
            ``out_keys`` after the last module ran.

            - ``True``: into the input tensordict, which is returned.
            - ``False``: into a new, empty :class:`~tensordict.TensorDict`.
            - ``"empty"``: into ``input.empty()``, which keeps the type, device
              and batch size of the input.
            - ``None`` (default): the sequence returns what the last module
              returns, so the ``inplace`` value of each module decides.

            With ``False`` or ``"empty"``, the modules run on a shallow copy
            of the input, so the input does not change.

    Raises:
        ValueError: if ``modules`` is empty.
        TypeError: if ``return_composite=False`` and the last module is not a
            :class:`~tensordict.nn.ProbabilisticTensorDictModule` or a
            :class:`~tensordict.nn.ProbabilisticTensorDictSequential`.

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
        TensorDict(batch_size=[], fields={loc: f32[3], sample: f32[3], x: f32[3]})
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
        TensorDict(batch_size=[], fields={
            loc2: f32[3],
            loc: f32[3],
            sample0: f32[3],
            sample1: f32[3],
            x: f32[3]})
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
        TensorDict(batch_size=[], fields={
            loc2: f32[3],
            loc: f32[3],
            sample0: f32[3],
            sample1: f32[3],
            x: f32[3]})
        >>> print(module.get_dist(input))
        CompositeDistribution({'sample0': Normal(loc: torch.Size([3]), scale: torch.Size([3])), 'sample1': Normal(loc: torch.Size([3]), scale: torch.Size([3]))})
        >>> print(module.log_prob(td))
        TensorDict(batch_size=[], fields={
            sample0_log_prob: f32[3],
            sample1_log_prob: f32[3]})
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
        TensorDict(batch_size=[], fields={
            loc: f32[3],
            sample0: f32[3],
            x: f32[3],
            y: f32[3]})
        >>> print(module.get_dist(input))
        CompositeDistribution({'sample0': Normal(loc: torch.Size([3]), scale: torch.Size([3]))})
        >>> print(module.log_prob(td))
        TensorDict(batch_size=[], fields={sample0_log_prob: f32[3]})

    """

    @overload
    def __init__(
        self,
        modules: dict[str, TensorDictModuleBase | ProbabilisticTensorDictModule],
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

    @overload
    def __init__(
        self,
        *modules: TensorDictModuleBase | ProbabilisticTensorDictModule,
        partial_tolerant: bool = False,
        return_composite: bool | None = None,
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
                return self._keep_options(
                    type(self)(*mods, return_composite=self.return_composite)
                )
            elif isinstance(
                mods[-1],
                (ProbabilisticTensorDictModule, ProbabilisticTensorDictSequential),
            ):
                return self._keep_options(type(self)(*mods))
            else:
                return self._keep_options(TensorDictSequential(*mods))

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
    ) -> TensorDictBase:
        """Runs every module except the last one, and returns the tensordict with the distribution parameters.

        The modules run with the interaction type of
        :func:`~tensordict.nn.interaction_type` if it is set, and otherwise
        with the ``default_interaction_type`` of the last module that has
        one, so that the probabilistic modules among them sample as in a
        call to the sequence.

        Args:
            tensordict (TensorDictBase): the input, with the entries that the
                first module reads.
            tensordict_out (TensorDictBase, optional): the tensordict where
                the parameters are written. If ``None``, they are written into
                ``tensordict``, which is returned. Pass
                ``tensordict.copy()`` to keep the input unchanged.
                Defaults to ``None``.

        Keyword Args:
            **kwargs: passed to the modules that are not
                :class:`~tensordict.nn.ProbabilisticTensorDictModule`, as in
                :meth:`forward` with ``return_composite=True``.

        Returns:
            The tensordict that holds the distribution parameters.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> from tensordict.nn import (
            ...     ProbabilisticTensorDictModule,
            ...     ProbabilisticTensorDictSequential,
            ...     TensorDictModule,
            ... )
            >>> seq = ProbabilisticTensorDictSequential(
            ...     TensorDictModule(lambda x: x + 1, in_keys=["x"], out_keys=["loc"]),
            ...     ProbabilisticTensorDictModule(
            ...         in_keys=["loc"], out_keys=["y"], distribution_class=torch.distributions.Normal,
            ...         distribution_kwargs={"scale": 1.0},
            ...     ),
            ... )
            >>> params = seq.get_dist_params(TensorDict(x=torch.zeros(2)))
            >>> sorted(params.keys())
            ['loc', 'x']
            >>> seq.build_dist_from_params(params)
            Normal(loc: torch.Size([2]), scale: torch.Size([2]))
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
            if not kwargs:
                return tds(tensordict, tensordict_out)
            # TensorDictSequential.forward raises for keyword arguments, so run
            # the modules here as it does, and pass the keyword arguments on.
            tensordict_exec = (
                tensordict if tensordict_out is None else tensordict.copy()
            )
            for m in tds._module_iter():
                module_kwargs = (
                    {} if isinstance(m, ProbabilisticTensorDictModule) else kwargs
                )
                tensordict_exec = tds._run_module(m, tensordict_exec, **module_kwargs)
            if tensordict_out is None:
                return tensordict_exec
            return tensordict_out.update(tensordict_exec, keys_to_update=tds.out_keys)

    @property
    def log_prob_keys(self):
        """The log-probability keys of all the probabilistic modules, those of the last module first.

        See the :ref:`log-probability key table <log-prob-keys>`.
        """
        lpks = []
        for m in reversed(list(self._module_iter())):
            lpks.extend(getattr(m, "log_prob_keys", []))
        return lpks

    @property
    def log_prob_key(self):
        """The log-probability key of the sequence, when :attr:`log_prob_keys` has exactly one element.

        See the :ref:`log-probability key table <log-prob-keys>`.

        Raises:
            AttributeError: if :attr:`log_prob_keys` does not have exactly one
                element (use :attr:`log_prob_keys`), or if
                :func:`~tensordict.nn.composite_lp_aggregate` changed since a
                module was built. The error message names the inner
                ``ModuleList``.
        """
        # An AttributeError keeps getattr() with a default, hasattr() and
        # inspect.getmembers() working. Python then calls
        # TensorDictModule.__getattr__, which raises its own error.
        try:
            log_prob_keys = self.log_prob_keys
        except RuntimeError as err:
            raise AttributeError("log_prob_key") from err
        if len(log_prob_keys) == 1:
            return log_prob_keys[0]
        raise AttributeError(
            f"log_prob_key needs exactly one log-probability key, got {log_prob_keys}. "
            f"Use log_prob_keys instead."
        )

    @property
    def dist_params_keys(self) -> List[NestedKey]:
        """The keys of the distribution parameters of all the probabilistic modules, those of the last module first."""
        result = []
        for m in reversed(list(self._module_iter())):
            result.extend(getattr(m, "dist_params_keys", []))
        return result

    @property
    def dist_sample_keys(self) -> List[NestedKey]:
        """The keys of the sampled values of all the probabilistic modules, those of the last module first."""
        result = []
        for m in reversed(list(self._module_iter())):
            result.extend(getattr(m, "dist_sample_keys", []))
        return result

    def _log_prob_renames(self) -> tuple[list, bool]:
        # The (default key, custom key) pairs of the modules, and whether a
        # module was built while composite_lp_aggregate() was True: such a
        # module has no custom key, and its log_prob_keys may raise.
        # The loop runs to the end: dynamo cannot close the module iterator.
        renames = []
        built_aggregated = False
        for m in self._module_iter():
            if isinstance(m, ProbabilisticTensorDictSequential):
                sub_renames, sub_aggregated = m._log_prob_renames()
            elif isinstance(m, ProbabilisticTensorDictModule):
                sub_renames = getattr(m, "_custom_log_prob_keys", ())
                sub_aggregated = m._composite_lp_aggreate_at_init
            else:
                continue
            renames.extend(sub_renames)
            built_aggregated = built_aggregated or sub_aggregated
        return renames, built_aggregated

    @property
    def num_samples(self):
        """The leading dimensions that the probabilistic modules add to the output.

        The ``num_samples`` of each probabilistic module, those of the last
        module first, as they appear in the shape of the output. An empty
        tuple if no module sets ``num_samples``.
        """
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
        """Returns the distribution that the sequence samples from, given the input ``tensordict``.

        If ``return_composite`` is ``False``, this is the distribution of the
        last module. The method runs the modules before it with
        :meth:`get_dist_params`, so the intermediate entries are written into
        ``tensordict``.

        If ``return_composite`` is ``True``, this is a
        :class:`~tensordict.nn.distributions.CompositeDistribution` that holds
        the distributions of all the probabilistic modules, at their sample
        keys. The method runs the modules on a shallow copy of
        ``tensordict``, so the input does not get new entries. Each
        probabilistic module draws a value, which the modules after it read,
        so their distributions are conditional on it. Only a probabilistic
        module in the last position does not draw.
        See the examples of :class:`ProbabilisticTensorDictSequential`.

        Args:
            tensordict (TensorDictBase): the input, with the entries that the
                first module reads.
            tensordict_out (TensorDictBase, optional): used only when
                ``return_composite`` is ``False``: the tensordict where the
                intermediate entries are written, instead of ``tensordict``.
                Defaults to ``None``.

        Keyword Args:
            **kwargs: ignored when ``return_composite`` is ``True``. When it
                is ``False``, passed to :meth:`get_dist_params`.

        Returns:
            A :class:`torch.distributions.Distribution`.

        Raises:
            RuntimeError: if the sequence has no probabilistic module.
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
        """The ``default_interaction_type`` of the last module that has one.

        If no module has one, the value of
        :func:`~tensordict.nn.interaction_type`, which is ``None`` outside of a
        :class:`~tensordict.nn.set_interaction_type` block.
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
        """Returns the log-probability of the sampled values in ``tensordict``.

        If ``return_composite`` is ``True``, the distribution is a
        :class:`~tensordict.nn.distributions.CompositeDistribution` and the
        method returns its :meth:`~tensordict.nn.distributions.CompositeDistribution.log_prob`:
        while :func:`~tensordict.nn.composite_lp_aggregate` is ``False``, a
        tensordict with one entry per sampled key, at the key where a call to
        the sequence writes it (see :attr:`log_prob_keys`); while it is
        ``True``, the sum, as a tensor.

        If ``return_composite`` is ``False``, the method returns the
        log-probability of the value of the last module, as
        :meth:`ProbabilisticTensorDictModule.log_prob` does. The intermediate
        entries are then written into ``tensordict``, as in :meth:`get_dist`.

        Args:
            tensordict (TensorDictBase): the input, with the sampled values.
            tensordict_out (TensorDictBase, optional): if given, the
                distribution is computed on a copy of ``tensordict``, so the
                input does not get new entries. Defaults to ``None``.

        Keyword Args:
            dist (torch.distributions.Distribution, optional): the
                distribution. Defaults to ``None``, which builds it with
                :meth:`get_dist`.
            **kwargs: if ``dist`` is ``None``, ``return_composite`` is
                ``False`` and the last module does not override
                :meth:`ProbabilisticTensorDictModule.log_prob`, passed to
                :meth:`get_dist`. Otherwise, passed to the ``log_prob`` method
                of the distribution, or of the last module.

        Returns:
            A tensor, or a tensordict for a composite distribution while
            :func:`~tensordict.nn.composite_lp_aggregate` is ``False``.
            See the examples of :class:`ProbabilisticTensorDictSequential`.
        """
        if tensordict_out is not None:
            tensordict_inp = tensordict.copy()
        else:
            tensordict_inp = tensordict
        return_composite = self.return_composite
        if dist is None:
            reroute = bool(kwargs) and not return_composite
            if reroute:
                # The base log_prob of the last module takes no keyword
                # arguments; a subclass that overrides it keeps getting them.
                leaf = self._last_module
                while type(leaf).log_prob is ProbabilisticTensorDictSequential.log_prob:
                    leaf = leaf._last_module
                reroute = type(leaf).log_prob is ProbabilisticTensorDictModule.log_prob
            if reroute:
                dist = self.get_dist(tensordict_inp, **kwargs)
                kwargs = {}
            else:
                dist = self.get_dist(tensordict_inp)
        if return_composite and isinstance(dist, CompositeDistribution):
            # Check the values within the dist - if not set, choose defaults
            lp = dist.log_prob(
                tensordict,
                **kwargs,
            )
            if is_tensor_collection(lp):
                renames, built_aggregated = self._log_prob_renames()
                if renames and not built_aggregated:
                    # Use the keys a forward call writes. A nested sequence
                    # with return_composite=False adds only its last
                    # distribution.
                    present = set(lp.keys(True, True))
                    _rename_entries(
                        lp, [pair for pair in renames if pair[0] in present]
                    )
            return lp
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
        """Builds the distribution of the last probabilistic module from parameters, without running the other modules.

        The method calls :meth:`ProbabilisticTensorDictModule.get_dist` of the
        last :class:`~tensordict.nn.ProbabilisticTensorDictModule` in the
        sequence, nested sequences included. See the example of
        :meth:`get_dist_params`.

        Args:
            tensordict (TensorDictBase): a tensordict with the distribution
                parameters at the ``in_keys`` of that module.

        Returns:
            A :class:`torch.distributions.Distribution`.

        Raises:
            RuntimeError: if the sequence has no
                :class:`~tensordict.nn.ProbabilisticTensorDictModule`.
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
        """Runs the modules in order, and returns the tensordict with their outputs.

        If the modules before the last one already write the sampled keys of
        the last module, the last module does not sample again.

        Args:
            tensordict (TensorDictBase): the input.
            tensordict_out (TensorDictBase, optional): if given, the modules
                run on a shallow copy of ``tensordict``, and the ``out_keys``
                are written into ``tensordict_out``, which is returned (unless
                ``inplace`` is set). Defaults to ``None``.

        Keyword Args:
            **kwargs: passed to the modules that are not
                :class:`~tensordict.nn.ProbabilisticTensorDictModule`, with
                either value of ``return_composite``.

        Returns:
            The tensordict that holds the outputs (see ``inplace``).
        """
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
