"""Top-level for Cassiopeia development."""

import sys

from . import critique, data, solver, utils
from . import plotting as pl
from . import preprocess as pp
from . import simulator as sim
from . import spatial as sp
from . import tools as tl

# https://github.com/python-poetry/poetry/pull/2366#issuecomment-652418094
# https://github.com/python-poetry/poetry/issues/144#issuecomment-623927302
try:
    import importlib.metadata as importlib_metadata
except ModuleNotFoundError:
    import importlib_metadata
package_name = "cassiopeia-lineage"
__version__ = importlib_metadata.version(package_name)

sys.modules.update({f"{__name__}.{m}": globals()[m] for m in ["tl", "pp", "pl", "sim", "sp", "utils"]})
del sys

__all__ = ["pp", "solver", "pl", "data", "critique", "sim", "sp", "tl", "utils"]
