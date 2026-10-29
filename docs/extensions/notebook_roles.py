"""Resolve MyST-style roles such as ``{func}`cassiopeia.solver.nj``` in notebooks.

nbsphinx converts Markdown cells with pandoc, which does not know MyST roles, so they
would be shown verbatim. Each role is turned into a pandoc raw reStructuredText span,
which Sphinx then resolves like any other cross-reference (including intersphinx).
"""

import json
import re

from sphinx.application import Sphinx

ROLE = re.compile(r"\{(func|class|meth|mod|attr|obj|data)\}`([^`]+)`")


def _to_rst(text: str) -> str:
    return ROLE.sub(lambda m: f"``:{m[1]}:`{m[2]}` ``{{=rst}}", text)


def convert_notebook_roles(app: Sphinx, docname: str, source: list[str]) -> None:
    """Rewrite the roles in the Markdown cells of a notebook source."""
    if not str(app.env.doc2path(docname)).endswith(".ipynb") or not ROLE.search(source[0]):
        return
    notebook = json.loads(source[0])
    for cell in notebook["cells"]:
        if cell["cell_type"] != "markdown":
            continue
        text = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
        cell["source"] = _to_rst(text)
    source[0] = json.dumps(notebook)


def setup(app: Sphinx):
    """Register the source-read hook with Sphinx."""
    app.connect("source-read", convert_notebook_roles)
    return {"parallel_read_safe": True}
