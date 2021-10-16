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

sys.path.insert(0, os.path.abspath(".."))

autodoc_mock_imports = [
    "torch",
    "cv2",
    "PIL",
    "mahotas",
    "imutils",
    "matplotlib",
    "simpletransformers",
    "_typeshed",
    "xgboost",
    "ee",
    "catboost",
    "albumentations",
    "skimage",
    "sklearn",
    "rasterio",
    "spacy",
    "geopandas",
    "lightgbm",
    "langdetect",
    "rasterio",
    "osgeo",
    "gdal",
    "vis",
    "seaborn",
    "tensorflow",
    "tensorrt",
    "onnx",
    "onnxruntime",
    "pycuda",
    "rioxarray",
    "numpy",
]

# -- Project information -----------------------------------------------------

project = "OmdenaLore"
copyright = "2021, Kaushal"
author = "Kaushal"

# The full version, including alpha/beta/rc tags
release = "0.2.8"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.ifconfig",
    "sphinx.ext.autosectionlabel",
    "sphinx_rtd_theme",
    "sphinx.ext.napoleon",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
source_suffix = [".rst", ".md"]

# The master toctree document.
master_doc = "index"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
# html_theme_options = {}

# -- Options for todo extension ----------------------------------------------

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True
