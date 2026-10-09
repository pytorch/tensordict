# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Internal implementation modules for :mod:`tensordict.base`.

``tensordict.base`` imports the mixin modules of this package before it defines
``TensorDictBase``. Importing the package must therefore not import a module
that needs the class, such as :mod:`tensordict._base.factories`.
"""
