# -*- coding: utf-8 -*-

"""Top-level for Cassiopeia development."""

from . import preprocess as pp
from . import solver
from . import plotting as pl
from . import data
from . import critique
from . import simulator as sim
from . import tools as tl

# https://github.com/python-poetry/poetry/pull/2366#issuecomment-652418094
# https://github.com/python-poetry/poetry/issues/144#issuecomment-623927302
try:
    import importlib.metadata as importlib_metadata
except ModuleNotFoundError:
    import importlib_metadata
package_name = "cassiopeia"
__version__ = importlib_metadata.version(package_name)

import sys

sys.modules.update({f"{__name__}.{m}": globals()[m] for m in ["tl", "pp", "pl"]})
