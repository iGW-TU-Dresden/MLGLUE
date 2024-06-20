# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
from datetime import date

sys.path.insert(0, os.path.abspath(".."))

project = 'MLGLUE'
copyright = '2024, Max Rudolph'
author = 'Max Rudolph'
release = "0.0.5" # this should be connected to the versions module
version = "0.0.5" # this should be connected to the versions module
year = date.today().strftime("%Y")

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.ifconfig",
    "sphinx.ext.viewcode",
    "IPython.sphinxext.ipython_console_highlighting",  # lowercase didn't work
    "sphinxcontrib.bibtex",
    "sphinx_design",
    "sphinx.ext.autosectionlabel",
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
autodoc_member_order = 'bysource'



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']
html_logo = "_static/MLGLUE_logo.png"
html_title = "MLGLUE Documentation"
html_short_title = "MLGLUE"
html_show_sphinx = True
html_show_copyright = True

html_theme_options = {
    "use_edit_page_button": True,
    "header_links_before_dropdown": 6,
    "icon_links": [
        {
            "name": "GitHub",  # Label for this link
            "url": "https://github.com/iGW-TU-Dresden/MLGLUE",  # required
            "icon": "fab fa-github-square",
            "type": "fontawesome",  # Default is fontawesome
        }
    ],
}


html_context = {
    "github_user": "iGW-TU-Dresden",
    "github_repo": "MLGLUE",
    "github_version": "main",
    "doc_path": "docs",
}
