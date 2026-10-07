.. currentmodule:: {{ module }}


{{ name | underline}}

{% if objtype == "class" -%}
.. autoclass:: {{ name }}
    :members:
{%- else -%}
.. auto{{ objtype }}:: {{ name }}
{%- endif %}
