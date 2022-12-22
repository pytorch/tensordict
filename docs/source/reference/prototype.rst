.. currentmodule:: tensordict.prototype

tensorclass prototype
=====================

The :obj:`@tensorclass` decorator helps you build custom classes that inherit the
behaviour from :obj:`TensorDict` while being able to restrict the possible entries
to a predefined set or implement custom methods for your class.
Like :obj:`TensorDict`, :obj:`@tensorclass` supports nesting, indexing, reshaping,
item assignment and many more features.

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
  ...     nested: Optional[MyData] = None
  ...     # sparse_data: Optional[KeyedJaggedTensor] = None
  ...
  ...     def check_nested(self):
  ...         assert self.nested is not None
  >>>
  >>> data = MyData(
  ...   floatdata = torch.randn(3, 4, 5),
  ...   intdata=torch.randint(10, (3, 4, 1)),
  ...   batch_size=[3, 4]
  ... )
  >>>
  >>> batch = data[:2]
  >>> print("indexed:", batch)
  indexed: MyData(
      floatdata=Tensor(torch.Size([2, 4, 5]), dtype=torch.float32),
      intdata=Tensor(torch.Size([2, 4, 1]), dtype=torch.int64),
      batch_size=torch.Size([2, 4]),
      device=None,
      is_shared=False)
  >>> data.nested = MyData(
  ...     floatdata = torch.randn(3, 4, 5),
  ...     intdata=torch.randint(10, (3, 4, 1)),
  ...     batch_size=[3, 4]
  ... )
  >>> print("nested:", data)
  nested: MyData(
      floatdata=Tensor(torch.Size([3, 4, 5]), dtype=torch.float32),
      intdata=Tensor(torch.Size([3, 4, 1]), dtype=torch.int64),
      nested=TensorDict(
          fields={
              floatdata: Tensor(torch.Size([3, 4, 5]), dtype=torch.float32),
              intdata: Tensor(torch.Size([3, 4, 1]), dtype=torch.int64)},
          batch_size=torch.Size([3, 4]),
          device=None,
          is_shared=False),
      batch_size=[3, 4],
      device=None,
      is_shared=False)



.. autosummary::
    :toctree: generated/
    :template: td_template.rst

    @tensorclass
