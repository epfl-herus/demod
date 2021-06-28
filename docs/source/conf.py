# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
import datetime
import sphinx_rtd_theme
sys.path.insert(0, os.path.abspath('../..'))
sys.path.insert(0, os.path.abspath('../../demod/simulators'))
# imports simulator classes of interest



# -- Project information -----------------------------------------------------

project = 'demod'
year = datetime.datetime.now().year
copyright = '2020-{}, HERUS-EPFL'.format(year)
author = 'Barsanti Matteo, Constantin Lionel, HERUS-EPFL'

# The full version, including alpha/beta/rc tags
release = 'beta'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',   # Pull in documentation from docstrings in a semi-automatic way
    'sphinx.ext.napoleon',  # Support for google style docstrings
    "sphinx_rtd_theme",     # REad the docs theme https://github.com/readthedocs/sphinx_rtd_theme
    'sphinx.ext.imgmath',   # support for writing maths. use:  :math:`a^2`, requires latex install on computer
    'sphinx_copybutton',    # Makes availability to copy code cells from icon
    # 'sphinx_paramlinks',    # Can make reference to function parameters with :paramref:,
    # removed as buggy with typing at the moment : https://github.com/sqlalchemyorg/sphinx-paramlinks/issues/10
]

# By default, Sphinx expects the master doc to be contents. Read the Docs will set master doc to index instead
master_doc = 'index'

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# -- Options for autodoc ext -------------------------------------------------
autodoc_member_order = 'bysource'
autodoc_typehints = 'description'
autodoc_inherit_docstrings = True
autodoc_type_aliases = {'GetMethod': ':py:data:`simulators.base_simulators.GetMethod`'}



# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# numfig: If true, figures, tables and code-blocks are automatically numbered if they have a caption.
# The numref role is enabled. Obeyed so far only by HTML and LaTeX builders. Default is False.
# https://www.sphinx-doc.org/en/master/usage/configuration.html
numfig = True

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

