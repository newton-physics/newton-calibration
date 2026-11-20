# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Sphinx configuration.

The settings follow Newton's ``docs/conf.py``. All calibration pages are in
``calibration/``, and ``index.rst`` includes ``calibration/_toctree.rst``.
"""

import datetime
import os
import re
import sys
import tomllib
from fnmatch import fnmatch
from pathlib import Path

docs_root = Path(__file__).resolve().parent
project_root = docs_root.parent
sys.path.insert(0, str(project_root))

github_url = "https://github.com/newton-physics/newton-calibration"
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
    "sphinxcontrib.mermaid",
]

templates_path = ["_templates"]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# The design notes in docs/ are not part of the site.
exclude_patterns = [
    "_build",
    "calibration/_toctree.rst",
    "generic-calibration-workflow-design.md",
    "tuning-class-diagram.md",
    "tuning-skills-implementation.md",
    "tuning-skills-module-design.md",
]

# The Markdown pages are also read on GitHub, so accept GitHub's syntax:
# heading anchors, $...$ math, and ```math and ```mermaid blocks.
myst_heading_anchors = 4
myst_enable_extensions = ["dollarmath", "attrs_inline"]
myst_fence_as_directive = ["math", "mermaid"]

extlinks = {
    "github": (f"{github_url}/blob/{github_version}/%s", "%s"),
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

# -- Markdown links -----------------------------------------------------------

# Relative link targets, such as ../examples/cable_calibration/README.md#setup.
_relative_link = re.compile(r"\]\((?!https?:|mailto:|#)([^)\s]+)\)")
# GitHub anchors, such as <a id="ref-r1"></a>.
_html_anchor = re.compile(r'<a id="([^"]+)"></a>')


_image_suffixes = {".gif", ".jpeg", ".jpg", ".png", ".svg"}


def _is_site_page(path: Path) -> bool:
    """Return whether a file is a page or an image of this site."""
    if not path.is_relative_to(docs_root):
        return False
    if path.suffix in _image_suffixes:
        return True
    if path.suffix not in source_suffix:
        return False
    name = path.relative_to(docs_root).as_posix()
    return not any(fnmatch(name, pattern) for pattern in exclude_patterns)


def _rewrite_markdown(app, docname: str, source: list[str]) -> None:
    """Make links and anchors written for GitHub work on the site.

    A link to a repository file that is not a page or an image of the site
    becomes a link to the file on GitHub. An HTML anchor becomes a MyST anchor.
    """
    if not str(app.env.doc2path(docname)).endswith(".md"):
        return
    page_dir = (docs_root / docname).parent

    def replace(match: re.Match) -> str:
        target, _, anchor = match.group(1).partition("#")
        path = (page_dir / target).resolve()
        if _is_site_page(path) or not path.is_relative_to(project_root):
            return match.group(0)
        kind = "tree" if path.is_dir() else "blob"
        url = f"{github_url}/{kind}/{github_version}/{path.relative_to(project_root).as_posix()}"
        return f"]({url}#{anchor})" if anchor else f"]({url})"

    source[0] = _relative_link.sub(replace, _html_anchor.sub(r"[]{#\1}", source[0]))


def setup(app):
    app.connect("source-read", _rewrite_markdown)
