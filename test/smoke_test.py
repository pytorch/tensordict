# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import sys

import pytest


def test_imports():
    from tensordict import TensorDict  # noqa: F401
    from tensordict.nn import TensorDictModule  # noqa: F401

    # # Check that distributed is not imported
    # v = set(sys.modules.values())
    # try:
    #     from torch import distributed
    # except ImportError:
    #     return
    # assert distributed not in v


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
