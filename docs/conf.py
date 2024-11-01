# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

sys.path.insert(0, os.path.abspath("sphinxext"))

project = "OORAGAN"
copyright = "2024, Yannick Lapointe and Gabriel Ouellet"
author = "Yannick Lapointe and Gabriel Ouellet"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.todo",
    "numpydoc",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
    "sphinx_favicon",
    "sphinx_design",
    "sphinxext.opengraph",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_css_files = ["ooragan.css"]
favicons = ["icons/ooragan_favicon.png"]
html_theme_options = {
    "logo": {"text": "OORAGAN", "image_light": "_static/icons/ooragan_logo.svg"},
    "navbar_end": ["navbar-icon-links"],
    "pygment_light_style": "trac",
    "show_prev_next": False,
    "github_url": "https://github.com/yalap13/ooragan/",
    "show_toc_level": 2,
}
html_context = {"default_mode": "light"}
html_show_sourcelink = False
html_sidebars = {"examples": []}

# -- Options for the Sphinx extensions ---------------------------------------

intersphinx_mapping = {
    "Numpy": ("https://numpy.org/doc/stable/", None),
    "GraphingLib": ("https://www.graphinglib.org/latest/", None),
    "Python": ("https://docs.python.org/", None),
}
autodoc_type_aliases = {"ArrayLike": "ArrayLike", "NDArray": "NDArray"}
autosummary_generate = True
numpydoc_show_class_members = True

ogp_site_url = "https://ooragan.readthedocs.io/"
ogp_image = "latest/_static/icons/ooragan_opengraph.png"
ogp_social_cards = {"enable": False}
