{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

    {% block methods %}
    {% if methods %}
    .. Autosummary generates the pages but the table wont be generated.
        .. autosummary::
            :toctree: ./
        {% for item in methods %}
            {%- if not item in ['__init__'] %}
                ~{{ name }}.{{ item }}
            {%- endif %}
        {%- endfor %}
    {% endif %}
    {% endblock %}