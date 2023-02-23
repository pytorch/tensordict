# -*- coding: utf-8 -*-
"""
Manipulating the keys of a TensorDict
=====================================
**Author**: `Tom Begley <https://github.com/tcbegley>`_

In this tutorial you will learn how to work with and manipulate the keys in a
``TensorDict``, including getting and setting keys, iterating over keys, manipulating
nested values, and flattening the keys.
"""

##############################################################################
# Setting and getting keys
# ------------------------
# We can set and get keys using the same syntax as a Python ``dict``

# sphinx_gallery_start_ignore
import warnings

warnings.filterwarnings("ignore")
# sphinx_gallery_end_ignore
import torch
from tensordict.tensordict import TensorDict

tensordict = TensorDict({}, [])

# set a key
a = torch.rand(10)
tensordict["a"] = a

# retrieve the value stored under "a"
assert tensordict["a"] is a

###############################################################################
# .. note::
#
#    Unlike a Python ``dict``, all keys in the ``TensorDict`` must be strings. However
#    as we will see, it is also possible to use tuples of strings to manipulate nested
#    values.
#
# We can also use the methods ``.get()`` and ``.set`` to accomplish the same thing.

tensordict = TensorDict({}, [])

# set a key
a = torch.rand(10)
tensordict.set("a", a)

# retrieve the value stored under "a"
assert tensordict.get("a") is a

###############################################################################
# Like ``dict``, we can provide a default value to ``get`` that should be returned in
# case the requested key is not found.

assert tensordict.get("banana", a) is a

###############################################################################
# Furthermore, when setting keys with ``.set()`` we can use the keyword argument
# ``inplace=True`` to make an inplace update, or equivalently use the ``.set_()``
# method.

tensordict.set("a", torch.zeros(10), inplace=True)

# all the entries of the "a" tensor are now zero
assert (tensordict.get("a") == 0).all()
# but it's still the same tensor as before
assert tensordict.get("a") is a

# we can achieve the same with set_
tensordict.set_("a", torch.ones(10))
assert (tensordict.get("a") == 1).all()
assert tensordict.get("a") is a

##############################################################################
# Nested values
# -------------
# The values of a ``TensorDict`` can themselves be a ``TensorDict``. We can add nested
# values during instantiation, either by adding ``TensorDict`` directly, or using nested
# dictionaries

# creating nested values with a nested dict
nested_tensordict = TensorDict(
    {"a": torch.rand(2, 3), "double_nested": {"a": torch.rand(2, 3)}}, [2, 3]
)
# creating nested values with a TensorDict
tensordict = TensorDict({"a": torch.rand(2), "nested": nested_tensordict}, [2])

print(tensordict)

##############################################################################
# To access these nested values, we can use tuples of strings. For example

double_nested_a = tensordict["nested", "double_nested", "a"]
nested_a = tensordict.get(("nested", "a"))

##############################################################################
# Similarly we can set nested values using tuples of strings

tensordict["nested", "double_nested", "b"] = torch.rand(2, 3)
tensordict.set(("nested", "b"), torch.rand(2, 3))

print(tensordict)

##############################################################################
# Iterating over a TensorDict's contents
# --------------------------------------
# We can iterate over the keys of a ``TensorDict`` using the ``.keys()`` method.

for key in tensordict.keys():
    print(key)

##############################################################################
# By default this will iterate only over the top-level keys in the ``TensorDict``,
# however it is possible to recursively iterate over all of the keys in the
# ``TensorDict`` with the keyword argument ``include_nested=True``. This will iterate
# recursively over all keys in any nested TensorDicts, returning nested keys as tuples
# of strings.

for key in tensordict.keys(include_nested=True):
    print(key)

###############################################################################
# In case you want to only iterate over keys corresponding to ``Tensor`` values, you can
# additionally specify ``leaves_only=True``.

for key in tensordict.keys(include_nested=True, leaves_only=True):
    print(key)

################################################################################
# Much like ``dict``, there are also ``.values`` and ``.items`` methods which accept the
# same keyword arguments.

for key, value in tensordict.items(include_nested=True):
    if isinstance(value, TensorDict):
        print(f"{key} is a TensorDict")
    else:
        print(f"{key} is a Tensor")

###############################################################################
# Checking for existence of a key
# -------------------------------
# To check if a key exists in a ``TensorDict``, use the ``in`` operator in conjunction
# with ``.keys()``.
#
# .. note::
#    Performing ``key in tensordict.keys()`` does efficient ``dict`` lookups of keys
#    (recursively at each level in the nested case), and so performance is not
#    negatively impacted when there is a large number of keys in the ``TensorDict``.

assert "a" in tensordict.keys()
# to check for nested keys, set include_nested=True
assert ("nested", "a") in tensordict.keys(include_nested=True)
assert ("nested", "banana") not in tensordict.keys(include_nested=True)

################################################################################
# Flattening nested keys
# ----------------------
# We can flatten a ``TensorDict`` with nested values using the ``.flatten_keys()``
# method.

print(tensordict, end="\n\n")
print(tensordict.flatten_keys(separator="."))

##############################################################################
# Given a ``TensorDict`` that has been flattened, it is possible to unflatten it again
# with the ``.unflatten_keys()`` method.

flattened_tensordict = tensordict.flatten_keys(separator="-")
print(flattened_tensordict, end="\n\n")
print(flattened_tensordict.unflatten_keys(separator="-"))
