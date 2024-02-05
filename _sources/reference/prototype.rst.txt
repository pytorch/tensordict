.. currentmodule:: tensordict.prototype

tensorclass prototype
=====================

The :obj:`@tensorclass` decorator helps you build custom classes that inherit the
behaviour from :obj:`TensorDict` while being able to restrict the possible entries
to a predefined set or implement custom methods for your class.
Like :obj:`TensorDict`, :obj:`@tensorclass` supports nesting, indexing, reshaping,
item assignment. It also supports tensor operations like clone, squeeze, cat, split and many more.
:obj:`@tensorclass` allows non-tensor entries,
however all the tensor operations are strictly restricted to tensor attributes. One
needs to implement their custom methods for non-tensor data. It is important to note that
:obj:`@tensorclass` does not enforce strict type matching

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
  ...     # sparse_data: Optional[KeyedJaggedTensor] = None
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


:obj:`@tensorclass` supports indexing. Internally the tensor objects gets indexed,
however the non-tensor data remains the same

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

:obj:`@tensorclass` also supports setting and resetting attributes, even for nested objects.

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

:obj:`@tensorclass` supports multiple torch operations over the shape and device
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

Edge cases
~~~~~~~~~~
:obj:`@tensorclass` supports equality and inequality operators, even for
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

:obj:`@tensorclass` supports setting an item. However, while setting an item
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

Even though :obj:`@tensorclass` supports torch functions like cat and stack, the
non-tensor / meta data is not validated. The torch operation is performed on the
tensor data and while returning the output, the non-tensor / meta data of the first
tensor class object is considered. User needs to make sure that all the
list of tensor class objects have the same non-tensor data to avoid discrepancies

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

:obj:`@tensorclass` also supports pre-allocation, you can initialize
the object with attributes being None and later set them. Note that while
initializing, internally the None attributes will be saved as non-tensor / meta data
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

    @tensorclass
