# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from .common import dispatch_kwargs, TensorDictModule, TensorDictModuleWrapper
from .functional_modules import get_functional, make_functional, repopulate_module
from .probabilistic import (
    ProbabilisticTensorDictModule,
    ProbabilisticTensorDictSequential,
    set_interaction_mode,
)
from .sequence import TensorDictSequential
