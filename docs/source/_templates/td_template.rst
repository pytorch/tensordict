.. currentmodule:: {{ module }}


{{ name | underline}}

{% if objtype == "class" -%}
.. autoclass:: {{ name }}
    :members:
    :inherited-members:
{%- else -%}
.. auto{{ objtype }}:: {{ name }}
{%- endif %}
