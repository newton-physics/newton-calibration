# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Sphinx configuration.

The settings follow Newton's ``docs/conf.py``. All calibration pages are in
``calibration/``, and ``index.rst`` includes ``calibration/_toctree.rst``.
"""

import datetime
import os
import sys
import tomllib
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

github_version = os.environ.get("GITHUB_REF_NAME") or "main"

# -- Project information ------------------------------------------------------

project = "Newton Calibration"
author = "The Newton Developers"
copyright = f"{datetime.date.today().year}, The Newton Developers. Documentation licensed under CC-BY-4.0"
with open(project_root / "pyproject.toml", "rb") as f:
    release = tomllib.load(f)["project"]["version"]

# -- General configuration ----------------------------------------------------

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "sphinx.ext.extlinks",
    "sphinx.ext.githubpages",
    "sphinx.ext.mathjax",
    "sphinx_copybutton",
]

templates_path = ["_templates"]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# The Markdown files in docs/ itself are not part of the site.
exclude_patterns = ["_build", "*.md", "calibration/_toctree.rst"]

extlinks = {
    "github": (
        f"https://github.com/newton-physics/newton-calibration/blob/{github_version}/%s",
        "%s",
    ),
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "newton": ("https://newton-physics.github.io/newton/latest", None),
    "warp": ("https://nvidia.github.io/warp/stable", None),
}

# -- Autodoc configuration ----------------------------------------------------

autodoc_typehints = "description"
autodoc_preserve_defaults = True
autodoc_typehints_description_target = "documented"
toc_object_entries_show_parents = "hide"

autodoc_default_options = {
    "members": True,
    "member-order": "groupwise",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
    "imported-members": True,
}

autodoc_inherit_docstrings = False
autodoc_mock_imports = ["cma", "pyglet", "scipy"]

autosummary_generate = True
autosummary_ignore_module_all = False
autosummary_imported_members = True

# -- HTML output --------------------------------------------------------------

html_title = "Newton Calibration"
html_theme = "pydata_sphinx_theme"
html_show_sourcelink = False

html_theme_options = {
    "navigation_depth": 1,
    "show_nav_level": 1,
    "show_toc_level": 2,
    "collapse_navigation": False,
    "show_prev_next": False,
    "use_edit_page_button": False,
    "logo": {
        "text": f"Newton Calibration <span style='font-size: 0.8em; color: #888;'>({release})</span>",
    },
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/newton-physics/newton-calibration",
            "icon": "fa-brands fa-github",
        },
    ],
}

html_sidebars = {"**": ["sidebar-nav-bs.html"]}
