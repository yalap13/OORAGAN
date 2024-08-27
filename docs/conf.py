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
    "numpydoc",
    "sphinx_copybutton",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx_favicon",
    "sphinx_design",
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
    "logo": {
        "image_light": "../ooragan_logo.svg",
        "image_dark": "../ooragan_logo.svg",
    },
    "navbar_end": ["navbar-icon-links"],
}
html_context = {"default_mode": "dark"}
