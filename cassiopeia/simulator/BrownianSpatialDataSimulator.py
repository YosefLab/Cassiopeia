"""
This file defines the BrownianSpatialDataSimulator, which is a subclass of
the SpatialDataSimulator. The BrownianSpatialDataSimulator simulates spatial
coordinates by simulating Brownian motion of each cell.
"""
from typing import Optional
import numpy as np
import pandas as pd

from cassiopeia.data import CassiopeiaTree
from cassiopeia.mixins import DataSimulatorError
from cassiopeia.simulator.SpatialDataSimulator import SpatialDataSimulator


class BrownianSpatialDataSimulator(SpatialDataSimulator):
    """
    Simulate spatial data with a Brownian motion process.

    This subclass of `SpatialDataSimulator` simulates the spatial coordinates of
    each cell in the provided `CassiopeiaTree` through a Brownian motion process.

    The simulation procedure is as follows. The tree is traversed from the root
    to the leaves. The the root cell is placed at the origin. At each split
    (i.e. when a cell divides), two new cells are placed at new coordinate X
    relative to the position of the parent X' (so, the absolute coordinate is
    X' + X). X is a n-dimensional vector with x_i ~ Normal(0, 2*D*t), where D is
    the diffusion coefficient and t is the time since the last cell division. X
    is sampled independently for each dimension for each cell, so the two new
    cells will be placed at different coordinates. Note that this process is
    dependent on the scale of the branch lengths.

    Args:
        dim: Number of spatial dimensions. For instance, a value of 2 indicates
            a 2D slice.
        diffusion_coeficient: The diffusion coefficient to use in the Brownian
            motion process. Specifically, 2 * `diffusion_coefficient` * (branch
            length) is the variance of the Normal distribution.
        scale_unit_area: Whether or not the space should be scaled to
            have unit length in all dimensions. Defaults to `True`.
        random_seed: A seed for reproducibility

    Raises:
        DataSimulatorError if `dim` is less than equal to zero, or the diffusion
            coefficient is negative.
    """

    def __init__(
        self,
        dim: int,
        diffusion_coefficient: float,
        scale_unit_area: bool = True,
        random_seed: Optional[int] = None,
    ):
        if dim <= 0:
            raise DataSimulatorError("Number of dimensions must be positive.")
        if diffusion_coefficient < 0:
            raise DataSimulatorError(
                "Diffusion coefficient must be non-negative."
            )

        self.dim = dim
        self.diffusion_coefficient = diffusion_coefficient
        self.scale_unit_area = scale_unit_area
        self.random_seed = random_seed

    def overlay_data(
        self,
        tree: CassiopeiaTree,
        attribute_key: str = "spatial",
    ):
        """Overlays spatial data onto the CassiopeiaTree via Brownian motion.

        Args:
            tree: The CassiopeiaTree to overlay spatial data on to.
            attribute_key: The name of the attribute to save the coordinates as.
                This also serves as the prefix of the coordinates saved into
                the `cell_meta` attribute as `{attribute_key}_i` where i is
                an integer from 0...`dim-1`.
        """
        if self.random_seed:
            np.random.seed(self.random_seed)

        # Using numpy arrays instead of tuples for easy vector operations
        locations = {tree.root: np.zeros(self.dim)}
        for parent, child in tree.depth_first_traverse_edges(source=tree.root):
            parent_location = locations[parent]
            branch_length = tree.get_branch_length(parent, child)

            locations[child] = parent_location + np.random.normal(
                scale=np.sqrt(2 * self.diffusion_coefficient * branch_length),
                size=self.dim,
            )

        # Scale if desired
        # Note that Python dictionaries preserve order since 3.6
        if self.scale_unit_area:
            all_coordinates = np.array(list(locations.values()))

            # Shift each dimension so that the smallest value is at 0.
            all_coordinates -= all_coordinates.min(axis=0)

            # Scale all dimensions (by the same value) so that all values are
            # between [0, 1]. We don't scale each dimension separately because
            # we want to retain the shape of the distribution.
            all_coordinates /= all_coordinates.max()
            locations = {
                node: coordinates
                for node, coordinates in zip(locations.keys(), all_coordinates)
            }

        # Set node attributes
        for node, loc in locations.items():
            tree.set_attribute(node, attribute_key, tuple(loc))

        # Set cell meta
        cell_meta = (
            tree.cell_meta.copy()
            if tree.cell_meta is not None
            else pd.DataFrame(index=tree.leaves)
        )
        columns = [f"{attribute_key}_{i}" for i in range(self.dim)]
        cell_meta[columns] = np.nan
        for leaf in tree.leaves:
            cell_meta.loc[leaf, columns] = locations[leaf]
        tree.cell_meta = cell_meta
