"""
A subclass of LeafSubsampler, the SpatialSampler.

The SpatialSampler is a sampler that is used to simulate non-single-cell
spatial assays by overlaying an 2-dimensional grid on the space the leaves are
located on.
"""
import copy
from typing import Callable, Dict, List, Optional, Tuple, Union
import uuid

import numpy as np
import pandas as pd
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

    def create_spot_polygons(
        self,
        bounds_x: Tuple[float, float],
        bounds_y: Tuple[float, float],
    ) -> List[Polygon]:
        """Create a 2D grid, representing capture spots/beads.

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
            Polygon(
                [
                    (x, y),
                    (x + self.grid_size, y),
                    (x + self.grid_size, y + self.grid_size),
                    (x, y + self.grid_size),
                ]
            )
            for x in np.arange(min_x, max_x, self.grid_size)
            for y in np.arange(min_y, max_y, self.grid_size)
        ]

    def create_cell_polygon(self, x: float, y: float) -> Union[Point, Polygon]:
        """Create a polygon representing a cell (leaf).

        A cell is represented as a square with `cell_size` as the length of
        each dimension.

        Args:
            x: Bottom-left x coordinate
            y: Bottom-left y coordinate

        Returns:
            A :class:`Point` (if `cell_size==0.0`) or :class:`Polygon`
            representing the cell.
        """
        if self.cell_size > 0:
            return Polygon(
                [
                    (x, y),
                    (x + self.cell_size, y),
                    (x + self.cell_size, y + self.cell_size),
                    (x, y + self.cell_size),
                ]
            )
        else:
            return Point(x, y)

    def capture_cells_on_spot(
        self, spot: Polygon, cells: Dict[str, Union[Polygon, Point]]
    ) -> Dict[str, float]:
        """Find all cells captured by the given spot.

        Args:
            spot: :class:`Polygon` representing a spot.
            cells: Dictionary of cell names to their :class:`Polygon` or
                :class:`Point` definitions, from which to detect what cells are
                captured by the provided spot.

        Returns:
            A dictionary of cells that overlap the given spot. The keys are
            the cell names, and the values are the fraction of each cell that
            is captured by the spot.
        """
        captured = {}
        for cell, cell_geometry in cells.items():
            if spot.intersects(cell_geometry):
                if self.cell_size > 0:
                    captured[cell] = (
                        spot.intersection(cell_geometry).area
                        / cell_geometry.area
                    )
                else:
                    captured[cell] = 1.0
        return captured

    def sample_spot_states(
        self, tree: CassiopeiaTree, captured: Dict[str, float]
    ) -> List[Union[int, Tuple[int, ...]]]:
        """Sample states from captured cells to generate spot states.

        Args:
            tree: The CassiopeiaTree that contains the cell states.
            captured: Dictionary containing leaf name to captured fraction
                mappings, as returned by :func:`capture_cells_on_spot`.

        Returns:
            Spot states
        """
        spot_states = [[] for _ in range(tree.n_character)]
        for leaf, fraction in captured.items():
            capture_rate = self.capture_rate_function(fraction)
            states = tree.get_character_states(leaf)
            for i, state in enumerate(states):
                if (
                    np.random.random() < capture_rate
                    and state != tree.missing_state_indicator
                ):
                    spot_states[i].append(state)

        # Convert to tuples
        cleaned_spot_states = []
        for state in spot_states:
            if not state:
                cleaned_spot_states.append(tree.missing_state_indicator)
            elif len(state) == 1:
                cleaned_spot_states.append(state[0])
            else:
                cleaned_spot_states.append(tuple(state))

        return cleaned_spot_states

    def subsample_leaves(
        self,
        tree: CassiopeiaTree,
        attribute_key: str = "spatial",
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
                or if any of the leaves have ambiguous states
        """
        locations = {}
        min_x, max_x = np.inf, -np.inf
        min_y, max_y = np.inf, -np.inf

        for leaf in tree.leaves:
            if tree.is_ambiguous(leaf):
                raise LeafSubsamplerError(f"Leaf {leaf} is ambiguous.")
            location = tree.get_attribute(leaf, attribute_key)
            if len(location) != 2:
                raise LeafSubsamplerError(
                    f"All coordinates must be 2-dimensional, but leaf {leaf} "
                    f"has {len(location)}-dimensional coordinates."
                )
            locations[leaf] = location
            min_x = min(min_x, location[0])
            max_x = max(max_x, location[0])
            min_y = min(min_y, location[1])
            max_y = max(max_y, location[1])

        # Construct grid and cells
        spots = self.create_spot_polygons(
            (min_x, max_x + self.cell_size),
            (min_y, max_y + self.cell_size),
        )
        cells = {
            leaf: self.create_cell_polygon(location[0], location[1])
            for leaf, location in locations.items()
        }

        # Capture cells in each spot.
        # This dictionary will contain the index of each spot (in the spots
        # list) as the keys and a dictionary of captured cells (as returned by
        # `capture_cells_on_spot`) as values.
        captured_spots = {}
        for i, spot in enumerate(spots):
            captured = self.capture_cells_on_spot(spot, cells)
            if captured:
                captured_spots[i] = captured
        print(captured_spots)

        # Construct a new network and character matrix
        sampled_tree = tree.copy()
        spot_leaves = {}
        for spot_i, captured in captured_spots.items():
            spot_uuid = str(uuid.uuid4())
            spot_leaves[spot_uuid] = spot_i
            spot_states = self.sample_spot_states(tree, captured)

            # Add new leaf
            captured_leaves = list(captured.keys())
            new_time = sum(
                tree.get_time(leaf) for leaf in captured_leaves
            ) / len(captured_leaves)
            sampled_tree.add_leaf(
                tree.find_lca(*captured_leaves)
                if len(captured_leaves) > 1
                else tree.parent(captured_leaves[0]),
                spot_uuid,
                states=spot_states,
                time=new_time,
            )

        # Remove all other leaves
        # Construct new character matrix and cell meta.
        relabel_map = {}
        character_matrix = pd.DataFrame(
            columns=tree.character_matrix.columns, dtype=object
        )
        meta_columns = [f"{attribute_key}_0", f"{attribute_key}_1"]
        cell_meta = pd.DataFrame(columns=meta_columns)
        for leaf in sampled_tree.leaves:
            if leaf not in spot_leaves:
                sampled_tree.remove_leaf_and_prune_lineage(leaf)
            else:
                # Set attribute in network
                spot_center = spots[spot_leaves[leaf]].centroid
                location = (spot_center.x, spot_center.y)
                sampled_tree.set_attribute(leaf, attribute_key, location)

                # Add rows to character_matrix and cell_meta
                new_leaf = f"s{len(relabel_map)}"
                states = sampled_tree.get_character_states(leaf)
                states_arr = np.empty(len(states), dtype=object)
                states_arr[:] = states
                character_matrix.loc[new_leaf] = states_arr
                cell_meta.loc[new_leaf, meta_columns] = location

                relabel_map[leaf] = new_leaf

        # Relabel spots to be an integer prefixed with "s" (indicating "spot")
        sampled_tree.relabel_nodes(relabel_map)
        sampled_tree.character_matrix = character_matrix
        sampled_tree.cell_meta = cell_meta

        if collapse_source is None:
            collapse_source = sampled_tree.root
        sampled_tree.collapse_unifurcations(source=collapse_source)

        if collapse_duplicates:
            sampled_tree.collapse_ambiguous_characters()

        return sampled_tree
