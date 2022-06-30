"""
This file defines the ClonalSpatialDataSimulator, which is a subclass of
the SpatialDataSimulator. The ClonalSpatialDataSimulator simulates spatial
coordinates with a spatial constraints that clonal populations (i.e. subclones)
are spatially localized together.
"""

import numpy as np
import pandas as pd

from cassiopeia.data import CassiopeiaTree
from cassiopeia.mixins import DataSimulatorError
from cassiopeia.simulator.SpatialDataSimulator import SpatialDataSimulator


class ClonalSpatialDataSimulator(SpatialDataSimulator):
    """
    Simulate spatial data with a clonal spatial autocorrelation constraint.

    This subclass of `SpatialDataSimulator` simulates the spatial coordinates of
    each cell in the provided `CassiopeiaTree` with spatial constraints such that
    subclones are more likely to be spatially autocorrelated.

    The simulation procedure is as follows.
    1. N coordinates are randomly sampled in space, where N is the number of leaves.
        Note that there is no mapping between leaves and coordinates (yet).
        All N coordinates are assigned to the root of the tree.
    2. The tree is traversed from the root to the leaves. At each node, the coordinates
        assigned to that node are split according to the number of leaves in each
        child. A spatial constraint is applied to this step by iteratively assigning
        each coordinate to the spatially closest child.

    Args:
        dim: Number of spatial dimensions. For instance, a value of 2 indicates
            a 2D slice. Only `2` is currently supported.
        space: Numpy array mask representing the space that cells may be placed.
            For example, to place cells on a 2D circlular surface, this argument
            will be a boolean Numpy array where the circular surface is indicated
            with True. The `dim` argument is ignored if this is provided. Otherwise,
            a circular.


        diffusion_coeficient: The diffusion coefficient to use in the Brownian
            motion process. Specifically, 2 * `diffusion_coefficient` * (branch
            length) is the variance of the Normal distribution.
        scale_unit_area: Whether or not the space should be scaled to
            have unit length in all dimensions. Defaults to `True`.

    Raises:
        DataSimulatorError if `dim` is less than equal to zero, or the diffusion
            coefficient is negative.
    """
