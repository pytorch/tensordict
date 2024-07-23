# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from tensordict.nn.common import (
    dispatch,
    make_tensordict,
    TensorDictModule,
    TensorDictModuleBase,
    TensorDictModuleWrapper,
)
from tensordict.nn.distributions import (
    AddStateIndependentNormalScale,
    CompositeDistribution,
    NormalParamExtractor,
    OneHotCategorical,
    rand_one_hot,
    TruncatedNormal,
)
from tensordict.nn.ensemble import EnsembleModule
from tensordict.nn.functional_modules import (
    get_functional,
    is_functional,
    make_functional,
    repopulate_module,
)
from tensordict.nn.params import TensorDictParams
from tensordict.nn.probabilistic import (
    InteractionType,
    ProbabilisticTensorDictModule,
    ProbabilisticTensorDictSequential,
    set_interaction_mode,
    set_interaction_type,
)
from tensordict.nn.sequence import TensorDictSequential
from tensordict.nn.utils import (
    add_custom_mapping,
    biased_softplus,
    inv_softplus,
    mappings,
    set_skip_existing,
    skip_existing,
)
