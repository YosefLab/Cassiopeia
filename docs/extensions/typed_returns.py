# code from https://github.com/theislab/scanpy/blob/master/docs/extensions/typed_returns.py
# with some minor adjustment
import re

from sphinx.application import Sphinx
from sphinx.ext.napoleon import NumpyDocstring


def process_return(lines):
    """Yield reStructuredText for return items with linked types."""
    for line in lines:
        m = re.fullmatch(r"(?P<param>\w+)\s+:\s+(?P<type>[\w.]+)", line)
        if m:
            # Once this is in scanpydoc, we can use the fancy hover stuff
            yield f"-{m['param']} (:class:`~{m['type']}`)"
        else:
            yield line


def scanpy_parse_returns_section(self, section):
    """Parse numpydoc return sections and format them with linked types."""
    lines_raw = list(process_return(self._dedent(self._consume_to_next_section())))
    lines = self._format_block(":returns: ", lines_raw)
    if lines and lines[-1]:
        lines.append("")
    return lines


def setup(app: Sphinx):
    """Register the typed return section parser with Sphinx."""
    NumpyDocstring._parse_returns_section = scanpy_parse_returns_section
