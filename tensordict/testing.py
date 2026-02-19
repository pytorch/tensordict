# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Test-related tensorclass definitions for distributed tests.

The classes here live inside the library so that both sender and receiver
processes can import them by fully-qualified name during multiprocessing
(``spawn`` start method) and distributed communication.
"""

import torch

from tensordict import tensorclass


@tensorclass
class MyDistData:
    """Simple tensorclass used to test type recovery over the wire."""

    a: torch.Tensor
    b: torch.Tensor
