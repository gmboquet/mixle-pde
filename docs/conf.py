"""Sphinx configuration for the mixle-pde documentation."""

from __future__ import annotations

import sys
from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_CORE = ROOT.parent / "mixle"
for path in (ROOT, WORKSPACE_CORE):
    if path.exists():
        sys.path.insert(0, str(path))

project = "mixle-pde"
author = "Grant Boquet"
copyright = "2026, Grant Boquet"

pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
release = pyproject["project"]["version"]
version = ".".join(release.split(".")[:2])

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

source_suffix = {".rst": "restructuredtext"}
master_doc = "index"
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

autosummary_generate = False
autodoc_default_options = {
    "members": True,
    "no-index": True,
    "show-inheritance": True,
}
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_typehints_format = "short"
autodoc_preserve_defaults = True
add_module_names = False
napoleon_google_docstring = True
napoleon_numpy_docstring = True

autodoc_mock_imports = []

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

html_theme = "furo"
html_title = "mixle-pde"
html_static_path = []
todo_include_todos = False

# Furo ships no host-agnostic version switcher (its built-in one only activates under Read the Docs
# hosting) -- this repo's own _templates/sidebar/version-switcher.html reads the version list from
# switcher.json, rendered once at the site root by sphinx-polyversion (see docs/poly.py). Only takes
# effect on builds run through `sphinx-polyversion`; a plain `sphinx-build` (single-version, e.g. local
# `make html`) still renders the partial, but its fetch of `../switcher.json` 404s harmlessly -- the
# button just shows an empty menu.
html_sidebars = {
    "**": [
        "sidebar/brand.html",
        "sidebar/version-switcher.html",
        "sidebar/search.html",
        "sidebar/scroll-start.html",
        "sidebar/navigation.html",
        "sidebar/ethical-ads.html",
        "sidebar/scroll-end.html",
        "sidebar/variant-selector.html",
    ]
}
