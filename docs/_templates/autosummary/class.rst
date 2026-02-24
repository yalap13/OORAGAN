{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

    {% block methods %}
    {% if methods %}
    .. rubric:: Methods

    .. autosummary::
        :toctree: ./
    {% for item in methods %}
    {%- if item not in ['__init__'] %}
        ~{{ name }}.{{ item }}
    {%- endif %}
    {%- endfor %}
    {%- if name in ['Dataset', 'Fitter'] %}
        ~{{ name }}.__getitem__
    {%- endif %}
    {% endif %}
    {% endblock %}

    {% block attributes %}
    {% if attributes %}
    .. autosummary::
        :toctree: ./
    {% for item in attributes %}
    {%- if not item.startswith('_') %}
        ~{{ name }}.{{ item }}
    {%- endif %}
    {%- endfor %}
    {% endif %}
    {% endblock %}
