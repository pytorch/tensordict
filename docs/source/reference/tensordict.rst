.. currentmodule:: tensordict

tensordict package
==================

The `TensorDict` class simplifies the process of passing multiple tensors
from module to module by packing them in a dictionary-like object that inherits features from
regular pytorch tensors.


.. autosummary::
    :toctree: generated/
    :template: td_template.rst

    TensorDict
    SubTensorDict
    LazyStackedTensorDict

Utils
-----

.. autosummary::
    :toctree: generated/
    :template: td_template.rst

    utils.expand_as_right
    utils.expand_right
