"""
Manipulating the keys of a TensorDict
=====================================
**Author**: `Tom Begley <https://github.com/tcbegley>`_

This tutorial shows how to read, write, rename and delete the keys of a
:class:`~tensordict.TensorDict`, and how to work with nested keys.

What you will learn:

- How to set and get entries, and what happens when a key is missing.
- How to find out if a key exists, and how to iterate over the keys.
- How to read and write nested entries with tuples of strings.
- How to flatten nested keys, and how to select and exclude keys.
"""

##############################################################################
# Setting and getting keys
# ------------------------
# Set and get entries with the same syntax as for a Python ``dict``:

import torch
from tensordict import TensorDict

tensordict = TensorDict()

# set a key
a = torch.rand(10)
tensordict["a"] = a

# retrieve the value stored under "a"
assert tensordict["a"] is a

##############################################################################
# .. note::
#
#    Unlike the keys of a Python ``dict``, all the keys of a ``TensorDict`` must
#    be strings. Tuples of strings refer to nested values, as shown below.
#
# The ``.get()`` and ``.set()`` methods do the same thing:

tensordict = TensorDict()

# set a key
a = torch.rand(10)
tensordict.set("a", a)

# retrieve the value stored under "a"
assert tensordict.get("a") is a

##############################################################################
# Like ``dict.get``, ``get`` returns ``None`` if the key is not in the
# ``TensorDict``. Indexing with ``tensordict[key]`` raises a ``KeyError``
# instead. Use ``tensordict[key]`` when the key must exist, and ``get`` when it
# can be missing:

assert tensordict.get("banana") is None
try:
    tensordict["banana"]
except KeyError:
    print("tensordict['banana'] raises a KeyError")

##############################################################################
# You can also give ``get`` a default value, which it returns if the key is
# not found:

assert tensordict.get("banana", a) is a

##############################################################################
# Like ``dict.setdefault``,
# :meth:`~tensordict.TensorDictBase.setdefault` returns the value of a key. If
# the key is not found, it stores the default value under the key and returns
# it.

assert tensordict.setdefault("banana", a) is a
# a is now stored under "banana"
assert tensordict["banana"] is a

##############################################################################
# To delete a key, use the ``del`` statement, as for a Python ``dict``. The
# :meth:`TensorDict.del_ <tensordict.TensorDict.del_>` method does the same.

del tensordict["banana"]

##############################################################################
# To write new values into the existing tensor of a key, pass
# ``inplace=True`` to ``.set()``, or use the ``.set_()`` method. The tensor
# object stays the same, and only its values change.

tensordict.set("a", torch.zeros(10), inplace=True)

# all the entries of the "a" tensor are now zero
assert (tensordict.get("a") == 0).all()
# but it is still the same tensor as before
assert tensordict.get("a") is a

# we can achieve the same with set_
tensordict.set_("a", torch.ones(10))
assert (tensordict.get("a") == 1).all()
assert tensordict.get("a") is a

##############################################################################
# Renaming keys
# -------------
# To rename a key, use the
# :meth:`TensorDict.rename_key_ <tensordict.TensorDict.rename_key_>` method. The
# value stays in the :class:`~tensordict.TensorDict`, under the new key.

tensordict.rename_key_("a", "b")
assert tensordict.get("b") is a
print(tensordict)

###############################################################################
# Updating multiple values
# ------------------------
# :meth:`TensorDict.update <tensordict.TensorDict.update>` copies the entries of
# another :class:`~tensordict.TensorDict`, or of a ``dict``, into a
# :class:`~tensordict.TensorDict`. It overwrites the keys that exist, and
# creates the keys that do not exist.

tensordict = TensorDict({"a": torch.rand(10), "b": torch.rand(10)}, [10])
tensordict.update(TensorDict({"a": torch.zeros(10), "c": torch.zeros(10)}, [10]))
assert (tensordict["a"] == 0).all()
assert (tensordict["b"] != 0).all()
assert (tensordict["c"] == 0).all()
print(tensordict)

##############################################################################
# Nested values
# -------------
# A value of a ``TensorDict`` can be another ``TensorDict``. To create nested
# values, pass a ``TensorDict`` or a nested ``dict`` to the constructor. A
# nested tensordict has the batch dimensions of its parent, and can have more.

# creating nested values with a nested dict
nested_tensordict = TensorDict(
    {"a": torch.rand(2, 3), "double_nested": {"a": torch.rand(2, 3)}}, [2, 3]
)
# creating nested values with a TensorDict
tensordict = TensorDict({"a": torch.rand(2), "nested": nested_tensordict}, [2])

print(tensordict)

##############################################################################
# To read a nested value, use a tuple of strings:

double_nested_a = tensordict["nested", "double_nested", "a"]
nested_a = tensordict.get(("nested", "a"))

##############################################################################
# To write a nested value, also use a tuple of strings:

tensordict["nested", "double_nested", "b"] = torch.rand(2, 3)
tensordict.set(("nested", "b"), torch.rand(2, 3))

print(tensordict)

##############################################################################
# Iterating over the contents of a TensorDict
# -------------------------------------------
# To iterate over the keys of a ``TensorDict``, use the ``.keys()`` method.
#
# .. note::
#
#    Do not write ``for key in tensordict``. A ``TensorDict`` iterates like a
#    tensor: this loop gives one sub-tensordict for each element of the first
#    batch dimension. If the tensordict has no batch dimensions, the loop
#    raises a ``TypeError``. In the same way, ``len(tensordict)`` is the size of
#    the first batch dimension, not the number of keys.

for key in tensordict.keys():
    print(key)

##############################################################################
# By default, ``.keys()`` gives only the top-level keys of the ``TensorDict``.
# With ``include_nested=True``, it also gives the keys of the nested
# tensordicts, at all levels, as tuples of strings.

for key in tensordict.keys(include_nested=True):
    print(key)

##############################################################################
# To get only the keys of tensor values, and not the keys of nested
# tensordicts, also pass ``leaves_only=True``.

for key in tensordict.keys(include_nested=True, leaves_only=True):
    print(key)

##############################################################################
# As for ``dict``, the ``.values()`` and ``.items()`` methods exist. They take the
# same keyword arguments as ``.keys()``.

for key, value in tensordict.items(include_nested=True):
    if isinstance(value, TensorDict):
        print(f"{key} is a TensorDict")
    else:
        print(f"{key} is a Tensor")

##############################################################################
# Checking for existence of a key
# -------------------------------
# To find out if a key exists in a ``TensorDict``, use the ``in`` operator. It
# also accepts nested keys as tuples of strings. Each ``in`` test is a ``dict``
# lookup at each level of nesting, so its cost does not grow with the number
# of keys.

assert "a" in tensordict
assert ("nested", "a") in tensordict
assert ("nested", "banana") not in tensordict

##############################################################################
# Flattening and unflattening nested keys
# ---------------------------------------
# The ``.flatten_keys()`` method turns nested keys into top-level keys. It
# joins the parts of each nested key with a separator:

print(tensordict, end="\n\n")
print(tensordict.flatten_keys(separator="."))

##############################################################################
# The ``.unflatten_keys()`` method does the inverse:

flattened_tensordict = tensordict.flatten_keys(separator=".")
print(flattened_tensordict, end="\n\n")
print(flattened_tensordict.unflatten_keys(separator="."))

##############################################################################
# ``unflatten_keys`` works on any flat key names with a separator. For
# example, the parameter names of a :class:`torch.nn.Module` use ``"."``. This
# code builds a :class:`~tensordict.TensorDict` whose structure is the structure
# of the module:

import torch.nn as nn

module = nn.Sequential(
    nn.Sequential(nn.Linear(100, 50), nn.Linear(50, 10)),
    nn.Linear(10, 1),
)
params = TensorDict(dict(module.named_parameters()), []).unflatten_keys(".")

print(params)

##############################################################################
# For modules, :meth:`TensorDict.from_module <tensordict.TensorDict.from_module>`
# gives the same result in one call, and also includes the buffers of the
# module. The :ref:`sphx_glr_tutorials_functional.py` tutorial uses it.

params_from_module = TensorDict.from_module(module)
assert set(params_from_module.keys(True, True)) == set(params.keys(True, True))

##############################################################################
# Selecting and excluding keys
# ----------------------------
# Two methods return a new :class:`~tensordict.TensorDict` with a subset of
# the keys:
#
# - :meth:`TensorDict.select <tensordict.TensorDict.select>` keeps only the
#   keys that you name.
# - :meth:`TensorDict.exclude <tensordict.TensorDict.exclude>` keeps all the
#   keys except the keys that you name.

print("Select:")
print(tensordict.select("a", ("nested", "a")), end="\n\n")
print("Exclude:")
print(tensordict.exclude(("nested", "b"), ("nested", "double_nested")))

##############################################################################
# Conclusion
# ----------
#
# In this tutorial, you set, read, renamed and deleted the keys of a
# :class:`~tensordict.TensorDict`. ``get`` returns ``None`` for a missing key,
# and ``tensordict[key]`` raises a ``KeyError``. You checked keys with ``in``,
# iterated over them with ``keys()``, ``values()`` and ``items()``, and used
# tuples of strings for nested keys. Last, you flattened nested keys, and
# selected and excluded keys.
#
# Further reading
# ---------------
#
# - :ref:`keys` explains keys, nested keys and locking.
# - :ref:`non-tensor-data` explains how to store values that are not tensors.
# - :ref:`sphx_glr_tutorials_tensordict_shapes.py` shows the shape operations.
# - :ref:`api-methods-keys` lists all the key methods.
