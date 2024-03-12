.. currentmodule:: tensordict

tensorclass
===========

The ``@tensorclass`` decorator helps you build custom classes that inherit the
behaviour from :class:`~tensordict.TensorDict` while being able to restrict
the possible entries to a predefined set or implement custom methods for your class.

Like :class:`~tensordict.TensorDict`, ``@tensorclass`` supports nesting,
indexing, reshaping, item assignment. It also supports tensor operations like
``clone``, ``squeeze``, ``torch.cat``, ``split`` and many more.
``@tensorclass`` allows non-tensor entries,
however all the tensor operations are strictly restricted to tensor attributes.

One needs to implement their custom methods for non-tensor data.
It is important to note that ``@tensorclass`` does not enforce strict type matching

.. code-block::

  >>> from __future__ import annotations
  >>> from tensordict.prototype import tensorclass
  >>> import torch
  >>> from torch import nn
  >>> from typing import Optional
  >>>
  >>> @tensorclass
  ... class MyData:
  ...     floatdata: torch.Tensor
  ...     intdata: torch.Tensor
  ...     non_tensordata: str
  ...     nested: Optional[MyData] = None
  ...
  ...     def check_nested(self):
  ...         assert self.nested is not None
  >>>
  >>> data = MyData(
  ...   floatdata=torch.randn(3, 4, 5),
  ...   intdata=torch.randint(10, (3, 4, 1)),
  ...   non_tensordata="test",
  ...   batch_size=[3, 4]
  ... )
  >>> print("data:", data)
  data: MyData(
    floatdata=Tensor(shape=torch.Size([3, 4, 5]), device=cpu, dtype=torch.float32, is_shared=False),
    intdata=Tensor(shape=torch.Size([3, 4, 1]), device=cpu, dtype=torch.int64, is_shared=False),
    non_tensordata='test',
    nested=None,
    batch_size=torch.Size([3, 4]),
    device=None,
    is_shared=False)
  >>> data.nested = MyData(
  ...     floatdata = torch.randn(3, 4, 5),
  ...     intdata=torch.randint(10, (3, 4, 1)),
  ...     non_tensordata="nested_test",
  ...     batch_size=[3, 4]
  ... )
  >>> print("nested:", data)
  nested: MyData(
    floatdata=Tensor(shape=torch.Size([3, 4, 5]), device=cpu, dtype=torch.float32, is_shared=False),
    intdata=Tensor(shape=torch.Size([3, 4, 1]), device=cpu, dtype=torch.int64, is_shared=False),
    non_tensordata='test',
    nested=MyData(
        floatdata=Tensor(shape=torch.Size([3, 4, 5]), device=cpu, dtype=torch.float32, is_shared=False),
        intdata=Tensor(shape=torch.Size([3, 4, 1]), device=cpu, dtype=torch.int64, is_shared=False),
        non_tensordata='nested_test',
        nested=None,
        batch_size=torch.Size([3, 4]),
        device=None,
        is_shared=False),
    batch_size=torch.Size([3, 4]),
    device=None,
    is_shared=False)

As it is the case with :class:`~tensordict.TensorDict`, from v0.4 if the batch size
is omitted it is considered as empty.

If a non-empty batch-size is provided, ``@tensorclass`` supports indexing.
Internally the tensor objects gets indexed, however the non-tensor data
remains the same

.. code-block::

  >>> print("indexed:", data[:2])
  indexed: MyData(
     floatdata=Tensor(shape=torch.Size([2, 4, 5]), device=cpu, dtype=torch.float32, is_shared=False),
     intdata=Tensor(shape=torch.Size([2, 4, 1]), device=cpu, dtype=torch.int64, is_shared=False),
     non_tensordata='test',
     nested=MyData(
        floatdata=Tensor(shape=torch.Size([2, 4, 5]), device=cpu, dtype=torch.float32, is_shared=False),
        intdata=Tensor(shape=torch.Size([2, 4, 1]), device=cpu, dtype=torch.int64, is_shared=False),
        non_tensordata='nested_test',
        nested=None,
        batch_size=torch.Size([2, 4]),
        device=None,
        is_shared=False),
     batch_size=torch.Size([2, 4]),
     device=None,
     is_shared=False)

``@tensorclass`` also supports setting and resetting attributes, even for nested objects.

.. code-block::

  >>> data.non_tensordata = "test_changed"
  >>> print("data.non_tensordata: ", repr(data.non_tensordata))
  data.non_tensordata: 'test_changed'

  >>> data.floatdata = torch.ones(3, 4, 5)
  >>> print("data.floatdata:", data.floatdata)
  data.floatdata: tensor([[[1., 1., 1., 1., 1.],
         [1., 1., 1., 1., 1.],
         [1., 1., 1., 1., 1.],
         [1., 1., 1., 1., 1.]],

        [[1., 1., 1., 1., 1.],
         [1., 1., 1., 1., 1.],
         [1., 1., 1., 1., 1.],
         [1., 1., 1., 1., 1.]],

        [[1., 1., 1., 1., 1.],
         [1., 1., 1., 1., 1.],
         [1., 1., 1., 1., 1.],
         [1., 1., 1., 1., 1.]]])

  >>> # Changing nested tensor data
  >>> data.nested.non_tensordata = "nested_test_changed"
  >>> print("data.nested.non_tensordata:", repr(data.nested.non_tensordata))
  data.nested.non_tensordata: 'nested_test_changed'

``@tensorclass`` supports multiple torch operations over the shape and device
of its content, such as `stack`, `cat`, `reshape` or `to(device)`. To get
a full list of the supported operations, check the tensordict documentation.

Here is an example:

.. code-block::

  >>> data2 = data.clone()
  >>> cat_tc = torch.cat([data, data2], 0)
  >>> print("Concatenated data:", catted_tc)
  Concatenated data: MyData(
     floatdata=Tensor(shape=torch.Size([6, 4, 5]), device=cpu, dtype=torch.float32, is_shared=False),
     intdata=Tensor(shape=torch.Size([6, 4, 1]), device=cpu, dtype=torch.int64, is_shared=False),
     non_tensordata='test_changed',
     nested=MyData(
         floatdata=Tensor(shape=torch.Size([6, 4, 5]), device=cpu, dtype=torch.float32, is_shared=False),
         intdata=Tensor(shape=torch.Size([6, 4, 1]), device=cpu, dtype=torch.int64, is_shared=False),
         non_tensordata='nested_test_changed',
         nested=None,
         batch_size=torch.Size([6, 4]),
         device=None,
         is_shared=False),
     batch_size=torch.Size([6, 4]),
     device=None,
     is_shared=False)

Serialization
-------------

Saving a tensorclass instance can be achieved with the `memmap` method.
The saving strategy is as follows: tensor data will be saved using memory-mapped
tensors, and non-tensor data that can be serialized using a json format will
be saved as such. Other data types will be saved using :func:`~torch.save`, which
relies on `pickle`.

Deserializing a `tensorclass` can be done via :meth:`~tensordict.TensorDict.load_memmap`.
The instance created will have the same type as the one saved provided that
the `tensorclass` is available in the working environment:

  >>> data.memmap("path/to/saved/directory")
  >>> data_loaded = TensorDict.load_memmap("path/to/saved/directory")
  >>> assert isinstance(data_loaded, type(data))


Edge cases
----------

``@tensorclass`` supports equality and inequality operators, even for
nested objects. Note that the non-tensor/ meta data is not validated.
This will return a tensor class object with boolean values for
tensor attributes and None for non-tensor attributes

Here is an example:

.. code-block::

  >>> print(data == data2)
  MyData(
     floatdata=Tensor(shape=torch.Size([3, 4, 5]), device=cpu, dtype=torch.bool, is_shared=False),
     intdata=Tensor(shape=torch.Size([3, 4, 1]), device=cpu, dtype=torch.bool, is_shared=False),
     non_tensordata=None,
     nested=MyData(
         floatdata=Tensor(shape=torch.Size([3, 4, 5]), device=cpu, dtype=torch.bool, is_shared=False),
         intdata=Tensor(shape=torch.Size([3, 4, 1]), device=cpu, dtype=torch.bool, is_shared=False),
         non_tensordata=None,
         nested=None,
         batch_size=torch.Size([3, 4]),
         device=None,
         is_shared=False),
     batch_size=torch.Size([3, 4]),
     device=None,
     is_shared=False)

``@tensorclass`` supports setting an item. However, while setting an item
the identity check of non-tensor / meta data is done instead of equality to
avoid performance issues. User needs to make sure that the non-tensor data
of an item matches with the object to avoid discrepancies.

Here is an example:

While setting an item with different ``non_tensor`` data, a :class:`UserWarning` will be
thrown

.. code-block::

  >>> data2.non_tensordata = "test_new"
  >>> data[0] = data2[0]
  UserWarning: Meta data at 'non_tensordata' may or may not be equal, this may result in undefined behaviours

Even though ``@tensorclass`` supports torch functions like :func:`~torch.cat`
and :func:`~torch.stack`, the non-tensor / meta data is not validated.
The torch operation is performed on the tensor data and while returning the
output, the non-tensor / meta data of the first tensor class object is
considered. User needs to make sure that all the list of tensor class objects
have the same non-tensor data to avoid discrepancies

Here is an example:

.. code-block::

  >>> data2.non_tensordata = "test_new"
  >>> stack_tc = torch.cat([data, data2], dim=0)
  >>> print(stack_tc)
  MyData(
      floatdata=Tensor(shape=torch.Size([2, 3, 4, 5]), device=cpu, dtype=torch.float32, is_shared=False),
      intdata=Tensor(shape=torch.Size([2, 3, 4, 1]), device=cpu, dtype=torch.int64, is_shared=False),
      non_tensordata='test',
      nested=MyData(
          floatdata=Tensor(shape=torch.Size([2, 3, 4, 5]), device=cpu, dtype=torch.float32, is_shared=False),
          intdata=Tensor(shape=torch.Size([2, 3, 4, 1]), device=cpu, dtype=torch.int64, is_shared=False),
          non_tensordata='nested_test',
          nested=None,
          batch_size=torch.Size([2, 3, 4]),
          device=None,
          is_shared=False),
      batch_size=torch.Size([2, 3, 4]),
      device=None,
      is_shared=False)

``@tensorclass`` also supports pre-allocation, you can initialize
the object with attributes being None and later set them. Note that while
initializing, internally the ``None`` attributes will be saved as non-tensor / meta data
and while resetting, based on the type of the value of the attribute,
it will be saved as either tensor data or non-tensor / meta  data

Here is an example:

.. code-block::

  >>> @tensorclass
  ... class MyClass:
  ...   X: Any
  ...   y: Any

  >>> data = MyClass(X=None, y=None, batch_size = [3,4])
  >>> data.X = torch.ones(3, 4, 5)
  >>> data.y = "testing"
  >>> print(data)
  MyClass(
     X=Tensor(shape=torch.Size([3, 4, 5]), device=cpu, dtype=torch.float32, is_shared=False),
     y='testing',
     batch_size=torch.Size([3, 4]),
     device=None,
     is_shared=False)

.. autosummary::
    :toctree: generated/
    :template: td_template.rst

    tensorclass
    NonTensorData
    NonTensorStack

Auto-casting
------------

.. warning:: Auto-casting is an experimental feature and subject to changes in
  the future. Compatibility with python<=3.9 is limited.

``@tensorclass`` partially supports auto-casting as an experimental feature.
Methods such as ``__setattr__``, ``update``, ``update_`` and ``from_dict`` will
attempt to cast type-annotated entries to the desired TensorDict / tensorclass
instance (except in cases detailed below). For instance, following code will
cast the `td` dictionary to a :class:`~tensordict.TensorDict` and the `tc`
entry to a :class:`MyClass` instance:

    >>> @tensorclass
    ... class MyClass:
    ...     tensor: torch.Tensor
    ...     td: TensorDict
    ...     tc: MyClass
    ...
    >>> obj = MyClass(
    ...     tensor=torch.randn(()),
    ...     td={"a": torch.randn(())},
    ...     tc={"tensor": torch.randn(()), "td": None, "tc": None})
    >>> assert isinstance(obj.tensor, torch.Tensor)
    >>> assert isinstance(obj.tc, TensorDict)
    >>> assert isinstance(obj.td, MyClass)

.. note:: Type annotated items that include an ``typing.Optional`` or
  ``typing.Union`` will not be compatible with auto-casting, but other items
  in the tensorclass will:

    >>> @tensorclass
    ... class MyClass:
    ...     tensor: torch.Tensor
    ...     tc_autocast: MyClass = None
    ...     tc_not_autocast: Optional[MyClass] = None
    >>> obj = MyClass(
    ...     tensor=torch.randn(()),
    ...     tc_autocast={"tensor": torch.randn(())},
    ...     tc_not_autocast={"tensor": torch.randn(())},
    ... )
    >>> assert isinstance(obj.tc_autocast, MyClass)
    >>> # because the type is Optional or Union, auto-casting is disabled for
    >>> # that variable.
    >>> assert not isinstance(obj.tc_not_autocast, MyClass)

  If at least one item in the class is annotated using the ``type0 | type1``
  semantic, the whole class auto-casting capabilities are deactivated.
  Because ``tensorclass`` supports non-tensor leaves, setting a dictionary in
  these cases will lead to setting it as a plain dictionary instead of a
  tensor collection subclass (``TensorDict`` or ``tensorclass``):

    >>> @tensorclass
    ... class MyClass:
    ...     tensor: torch.Tensor
    ...     td: TensorDict
    ...     tc: MyClass | None
    ...
    >>> obj = MyClass(
    ...     tensor=torch.randn(()),
    ...     td={"a": torch.randn(())},
    ...     tc={"tensor": torch.randn(()), "td": None, "tc": None})
    >>> assert isinstance(obj.tensor, torch.Tensor)
    >>> # tc and td have not been cast
    >>> assert isinstance(obj.tc, dict)
    >>> assert isinstance(obj.td, dict)

.. note:: Auto-casting isn't enabled for leaves (tensors).
  The reason for this is that this feature isn't compatible with type
  annotations that contain the ``type0 | type1`` type hinting semantic, which
  is widespread. Allowing auto-casting would result in very similar codes to
  have drastically different behaviours if the type annotation differs only
  slightly.
