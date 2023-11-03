# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


def _unravel_key_to_tuple(key):
    if isinstance(key, str):
        return (key,)
    if not isinstance(key, tuple):
        return None
    result = ()
    for elt in key:
        elt = _unravel_key_to_tuple(elt)
        if elt is None:
            return None
        result = result + elt
    return result


def unravel_key_list(keys):
    return [unravel_key(key) for key in keys]


def unravel_key(key):
    if isinstance(key, str):
        return key
    result = ()
    for elt in key:
        result = result + _unravel_key_to_tuple(elt)
    if len(result) == 1:
        return result[0]
    return result
