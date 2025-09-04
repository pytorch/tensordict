# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from tensordict.nn.distributions import continuous, discrete

from tensordict.nn.distributions.composite import CompositeDistribution
from tensordict.nn.distributions.continuous import (
    AddStateIndependentNormalScale,
    Delta,
    NormalParamExtractor,
)
from tensordict.nn.distributions.discrete import OneHotCategorical, rand_one_hot
from tensordict.nn.distributions.truncated_normal import TruncatedNormal
from tensordict.nn.probabilistic import InteractionType, set_interaction_type
from tensordict.nn.utils import add_custom_mapping, mappings

distributions_maps = {
    distribution_class.lower(): eval(distribution_class)
    for distribution_class in (*continuous.__all__, *discrete.__all__)
}

__all__ = [
    # Distribution classes
    "CompositeDistribution",
    "AddStateIndependentNormalScale",
    "Delta",
    "NormalParamExtractor",
    "OneHotCategorical",
    "rand_one_hot",
    "TruncatedNormal",
    # Interaction types
    "InteractionType",
    "set_interaction_type",
    # Submodules
    "continuous",
    "discrete",
    # Utilities
    "distributions_maps",
    "add_custom_mapping",
    "mappings",
]
