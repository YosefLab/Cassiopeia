"""
A subclass of LeafSubsampler, the SpatialSampler.

The SpatialSampler is a sampler that is used to simulate non-single-cell
spatial assays by overlaying an 2-dimensional grid on the space the leaves are
located on.
"""
import copy
from typing import Optional

import numpy as np
from shapely.geometry import Point, Polygon

from cassiopeia.data.CassiopeiaTree import CassiopeiaTree, CassiopeiaTreeError
from cassiopeia.simulator.LeafSubsampler import (
    LeafSubsampler,
    LeafSubsamplerError,
)


class SpatialSampler(LeafSubsampler):
    """
    Simulate a non-single-cell 2D spatial assay.

    At a high level, this sampler overlays a grid on the spatial coordinates,
    representing beads/spots. A cell (not a biological cell, but a cell of the
    grid) then captures characters of leaves that are contained within its
    boundaries. For simplicity, spots and leaves are represented as squares.

    Args:
        grid_size: The size of each dimension of the unit cell (of the grid).
        cell_size: The size of each dimension of a cell (i.e. leaf). This can
            be set to zero to constrain the entirety of each cell be
            captured by a single spot (i.e. a "point" cell).
        capture_rate_function: A function that takes the proportion of a cell
            that is captured by a spot and returns the probability that any
            cassette is captured on that spot. By default, the proportion is
            used as the probability (i.e. if a spot captures 0.5 of a cell,
            then the probability that a cassette is captured on that spot is
            0.5).
    """
    def __init__(
        self,
        grid_size: float,
        cell_size: float = 0.0,
        capture_rate_function: Callable[[float], float] = lambda p: p,
    ):
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.capture_rate_function = capture_rate_function

    def create_grid_polygons(
        self,
        bounds_x: Tuple[float, float],
        bounds_y: Tuple[float, float],
    ) -> List[Polygon]:
        """Create a 2D grid.

        The first grid cell always has `(bounds_x[0], bounds_y[0])` as a vertex.

        Args:
            bounds_x: Tuple representing the lower and upper bounds in the
                x dimension.
            bounds_y: Tuple representing the lower and upper bounds in the
                y dimension.

        Raises:
            LeafSubsamplerError: if any of the bounds are invalid

        Returns:
            A list of :class:`Polygon` instances, which define each grid
                cell.
        """
        min_x, max_x = bounds_x
        min_y, max_y = bounds_y

        if min_x >= max_x:
            raise LeafSubsamplerError(f"Invalid bounds for x: {bounds_x}")
        if min_y >= max_y:
            raise LeafSubsamplerError(f"Invalid bounds for y: {bounds_y}")

        return [
            Polygon([(x, y), (x + self.grid_size, y), (x + self.grid_size, y + self.grid_size)])
            for x in np.arange(min_x, max_x, self.grid_size)
            for y in np.arange(min_y, max_y, self.grid_size)
        ]

    def create_cell_polygon(self, x: float, y: float) -> Union[Point, Polygon]:
        """Create a polygon representing a cell (leaf).

        Args:
            x: x coordinate
            y: x coordinate

        Returns:
            A :class:`Point` (if `cell_size==0.0`) or :class:`Polygon`
            representing the cell.
        """
        if self.cell_size > 0:
            return Polygon([
                (x - self.cell_size / 2, y - self.cell_size / 2),
                (x + self.cell_size / 2, )
            ])


    def subsample_leaves(
        self,
        tree: CassiopeiaTree,
        attribute_key: str = 'spatial',
        collapse_source: Optional[str] = None,
        collapse_duplicates: bool = True,
    ) -> CassiopeiaTree:
        """Construct a new CassiopeiaTree by overlaying a 2D grid.

        Args:
            tree: The CassiopeiaTree for which to subsample leaves. All the
                leaves must have the `attribute_key` attribute that contains
                2D spatial coordinates.
            attribute_key: Node attribute that contains the spatial coordinates.
                Defaults to `spatial`.
            collapse_source: The source node from which to collapse unifurcations
            collapse_duplicates: Whether or not to collapse duplicated character
                states, so that only unique character states are present in each
                ambiguous state. Defaults to True.

        Raises:
            CassiopeiaTreeError if any of the leaves does not have the
                `attribute_key` attribute
            LeafSubsamplerError if not all spatial coordinates are 2-dimensional
        """
        locations = {}
        for leaf in tree.leaves:
            location = tree.get_attribute(leaf, attribute_key)
            if len(location) != 2:
                raise LeafSubsamplerError(
                    f"All coordinates must be 2-dimensional, but leaf {leaf} "
                    f"has {len(location)}-dimensional coordinates."
                )
