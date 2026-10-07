.. currentmodule:: {{ module }}


{{ name | underline}}

{% if objtype == "class" -%}
{%- set summary_methods = methods | reject("equalto", "__init__") | reject("in", inherited_members) | list -%}
.. autoclass:: {{ name }}
    :members:
{% if summary_methods %}
    .. rubric:: Methods

    .. autosummary::
{% for item in summary_methods %}
        ~{{ name }}.{{ item }}
{%- endfor %}
{% endif %}
{%- else -%}
.. auto{{ objtype }}:: {{ name }}
{%- endif %}
