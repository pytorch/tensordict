.. currentmodule:: {{ module }}


{{ name | underline}}

{% if fullname == "tensordict.TensorDictBase" -%}
The methods of ``TensorDictBase`` are shared by every tensordict class and
by tensorclasses. They are listed by task in :ref:`api-tensordict-methods`,
with one page for each method.

.. autoclass:: {{ name }}
{%- else -%}
``{{ name }}`` has the methods of :class:`~tensordict.TensorDictBase`,
listed by task in :ref:`api-tensordict-methods`. This page lists only the
methods that ``{{ name }}`` adds, and the overrides that have their own
documentation.

.. autoclass:: {{ name }}
    :members:
    :show-inheritance:
{%- endif %}

.. minigallery:: {{ fullname }}
    :add-heading: Tutorials that use {{ name }}
