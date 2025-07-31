# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import TypeVar

T = TypeVar("T", bound="TensorCollection")


class TensorCollection:
    """A base class for TensorDictBase and TensorClass.

    This is an abstract base class that provides the foundation for tensor collections
    like TensorDict and TensorClass. It serves as a common interface for all tensor
    collection types in the tensordict library.
    """

    def __init__(self, *args, **kwargs):
        """Initialize the TensorCollection.

        This is an abstract base class and should not be instantiated directly.
        """
        raise NotImplementedError(
            "TensorCollection is an abstract base class and cannot be instantiated directly."
        )
