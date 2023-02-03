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
however the non-tensor data remains same

.. code-block::


 >>>print("indexed:", data[:2])
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

:obj:`@tensorclass` also supports and get and set attributes. One can also access
nested tensor class data

.. code-block::

 >>> data.non_tensordata = "test_changed"
 >>> print("data.non_tensordata: ", repr(data.non_tensordata))
 data.non_tensordata: 'test_changed'

 >>> data.floatdata = torch.ones(3, 4, 5)
 >>>print("data.floatdata:", data.floatdata)
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
 >>>print("data.nested.non_tensordata:", repr(data.nested.non_tensordata))
 data.nested.non_tensordata: 'nested_test_changed'

:obj:`@tensorclass` supports torch functions and the behavior is extened to
nested tensor classes as well.Note that the operations are strictly restricted
to tensor data types

Here is an example:

.. code-block::

 >>>data2 = data.clone()
 >>>cat_tc = torch.cat([data, data2], 0)
 >>>print("Concatenated data:", catted_tc)
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

.. autosummary::
    :toctree: generated/
    :template: td_template.rst

    @tensorclass
