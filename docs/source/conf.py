# Configuration file for the Sphinx documentation builder.

# -- Project information

project = 'autoXAS'
copyright = '2025, Ulrik Friis-Jensen'
author = 'Ulrik Friis-Jensen'

release = '0.4'
version = '0.4.0'

# -- General configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
]

templates_path = ['_templates']

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'