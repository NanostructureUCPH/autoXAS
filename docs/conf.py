# Configuration file for the Sphinx documentation builder.

import os
import sys
sys.path.insert(0, os.path.abspath('..'))

import sphinx_rtd_theme
html_theme = "sphinx_rtd_theme"
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

import autoXAS

# -- Project information

project = 'autoXAS'
copyright = '2025, Ulrik Friis-Jensen'
author = 'Ulrik Friis-Jensen'

# -- General configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode',
    'sphinx.ext.apidoc',
    'myst_parser',
]

templates_path = ['_templates']

apidoc_module_dir = '../autoXAS'
apidoc_separate_modules = True

source_suffix = {
    '.rst': 'restructuredtext',
    '.txt': 'markdown',
    '.md': 'markdown',
}

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

todo_include_todos = False

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']