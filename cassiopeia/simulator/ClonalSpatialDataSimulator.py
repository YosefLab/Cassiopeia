"""
This file defines the ClonalSpatialDataSimulator, which is a subclass of
the SpatialDataSimulator. The ClonalSpatialDataSimulator simulates spatial
coordinates with a spatial constraints that clonal populations (i.e. subclones)
are spatially localized together.
"""
from typing import List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from scipy import spatial

from cassiopeia.data import CassiopeiaTree
from cassiopeia.mixins import DataSimulatorError, try_import
from cassiopeia.simulator.SpatialDataSimulator import SpatialDataSimulator

cv2 = try_import("cv2")
disc = try_import("poisson_disc")
neighbors = try_import("sklearn.neighbors")


class ClonalSpatialDataSimulator(SpatialDataSimulator):
    """
    Simulate spatial data with a clonal spatial autocorrelation constraint.

    This subclass of `SpatialDataSimulator` simulates the spatial coordinates of
    each cell in the provided `CassiopeiaTree` with spatial constraints such
    that subclones are more likely to be spatially autocorrelated.

    The simulation procedure is as follows.
    1. N coordinates are randomly sampled in space, where N is the number of
        leaves. Note that there is no mapping between leaves and coordinates
        (yet). All N coordinates are assigned to the root of the tree.
    2. The tree is traversed from the root to the leaves. At each node, the
        coordinates assigned to that node are split according to the number of
        leaves in each child. A spatial constraint is applied to this step by
        iteratively assigning each coordinate to the spatially closest child.

    Args:
        shape: Shape of the space to place cells on. For instance, (100, 100)
            means a 2D surface of 100x100 pixels. By default in two dimensions
            (length is 2), an elliptical surface is used. In higher dimensions,
            the entire (hyper)cuboid is used.
        space: Numpy array mask representing the space that cells may be placed.
            For example, to place cells on a 2D circlular surface, this argument
            will be a boolean Numpy array where the circular surface is
            indicated with True.
        random_seed: A seed for reproducibility

    Raises:
        DataSimulatorError if neither `shape` nor `space` are provided
    """

    def __init__(
        self,
        shape: Optional[Tuple[int, ...]] = None,
        space: Optional[np.ndarray] = None,
        random_seed: Optional[int] = None,
    ):
        if None in (cv2, disc, neighbors):
            raise DataSimulatorError(
                "Some required modules were not found. Make sure you installed "
                "Cassiopeia with the `spatial` extras, or run `pip install "
                "cassiopeia-lineage[spatial]`."
            )

        if (shape is None) == (space is None):
            raise DataSimulatorError(
                "Exactly one of `shape` or `space` must be provided."
            )
        self.random_seed = random_seed

        if shape is not None:
            self.dim = len(shape)
            if self.dim == 2:
                center_x = shape[1] // 2
                center_y = shape[0] // 2
                self.space = cv2.ellipse(
                    np.zeros(shape, dtype=np.uint8),
                    (center_x, center_y),
                    (center_x, center_y),
                    0,
                    0,
                    360,
                    1,
                    -1,
                ).astype(bool)
            else:
                self.space = np.ones(shape, dtype=bool)

        if space is not None:
            self.dim = space.ndim
            self.space = space

    @staticmethod
    def __triangulation_graph(points: np.ndarray) -> nx.Graph:
        """Compute a fully-connected Delaunay triangulation graph from a set of points.

        Args:
            points: Points to triangulate

        Returns:
            Networkx graph
        """
        tri = spatial.Delaunay(points)
        G = nx.Graph()
        for path in tri.simplices:
            nx.add_path(G, path)

        for n1, n2 in G.edges:
            G[n1][n2]["weight"] = spatial.distance.euclidean(
                points[n1], points[n2]
            )
        return G

    @staticmethod
    def __nearest_neighbors_graph(points: np.ndarray, k: int) -> nx.Graph:
        """Compute the nearest neighbors graph from a set of points.

        Args:
            points: Point coordinates
            k: Number of nearest neighbors

        Returns:
            Networkx graph
        """
        distances = neighbors.kneighbors_graph(points, k, mode="distance")
        G = nx.from_scipy_sparse_array(distances)
        return G

    @classmethod
    def __points_to_graph(cls, points: np.ndarray) -> nx.Graph:
        """Construct a connected graph from a set of points.

        This function uses Delauney triangulation if the number of points is
        greater than five by calling `__triangulation_graph`. Otherwise, a
        nearest-neighbors graph is constructed with `__nearest_neighbors_graph`.

        Delauney triangulation is much faster than nearest neighbors for many
        nodes, but Delaunay triangulation only works with more than a certain
        number of nodes.

        Args:
            points: Points

        Returns:
            Networkx graph
        """
        return (
            cls.__triangulation_graph(points)
            if len(points) > 5
            else cls.__nearest_neighbors_graph(points, min(5, len(points) - 1))
        )

    @staticmethod
    def __split_graph(
        G: nx.Graph, sizes: Tuple[int, ...]
    ) -> Tuple[List[int], ...]:
        """Generate a node partition of exact sizes.

        A set of seed nodes, the same size as the number of elements in `sizes`,
        is randomly selected among all nodes. Using these seed nodes, each
        non-seed node is assigned to a partition one at a time by iterating
        through a sorted list of all distances of the form seed->non-seed.

        Args:
            G: Graph to partition
            sizes: Tuple of integers indicating the partition sizes

        Returns:
            Obtained node partition as a tuple of lists of integers
        """
        if not nx.is_connected(G):
            raise DataSimulatorError("Graph is not connected.")
        if sum(sizes) != len(G.nodes):
            raise DataSimulatorError(
                f"Can not obtain node partition {sizes} for graph of "
                "{len(G.nodes)} nodes."
            )

        # Find seeds
        seeds = dict(
            zip(np.random.choice(G.nodes, len(sizes), replace=False), sizes)
        )

        # Find minimum weighted paths from each seed to every node
        seed_distances = {
            seed: nx.single_source_dijkstra_path_length(G, seed)
            for seed in seeds
        }

        # Assign each non-seed point.
        # We assign exactly the desired number of nodes to each seed by
        # iterating through a sorted list of all distances and assigning each
        # node one at a time.
        distance_seed_nodes = sorted(
            (distance, seed, node)
            for seed, distances in seed_distances.items()
            for node, distance in distances.items()
        )
        assigned = set()
        assignments = {}
        for _, seed, node in distance_seed_nodes:
            if (
                node in assigned
                or len(assignments.get(seed, [])) == seeds[seed]
            ):
                continue

            assignments.setdefault(seed, []).append(node)
            assigned.add(node)
        return tuple(assignments[seed] for seed in seeds)

    def sample_points(self, n: int) -> np.ndarray:
        """Sample the given number of points within the `shape` of this object.

        Points are sampled using Poisson-Disc sampling to generate approximately
        equally-spaced points. The Bridson algorithm is used, which is
        implemented in the poisson_disc package. 
        https://www.cs.ubc.ca/~rbridson/docs/bridson-siggraph07-poissondisk.pdf

        Args:
            n: Number of points to sample.

        Returns:
            `n` sampled points within `shape`
        """
        shape = self.space.shape
        radius = (min(shape) / (n ** (1 / self.dim))) / 2
        while True:
            points = disc.Bridson_sampling(dims=np.array(shape), radius=radius)
            radius /= 2

            points = points[self.space[tuple(points.T.astype(int))]]
            if len(points) >= n:
                break

        np.random.shuffle(points)
        return points[:n]

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

        # Sample point coordinates.
        points = self.sample_points(tree.n_cell)

        # Assign each of these coordinates to leaves.
        # Work from top to bottom of tree, splitting leaves
        # Initially, all points are assigned to the root.
        locations = {}
        point_assignments = [tree.root] * len(points)
        for node in tree.depth_first_traverse_nodes(postorder=False):
            if tree.is_leaf(node):
                continue

            children = tree.children(node)

            node_idx = np.array(
                [
                    i
                    for i, assign in enumerate(point_assignments)
                    if assign == node
                ]
            )
            node_points = points[node_idx]
            locations[node] = node_points.mean(axis=0)
            # The only requirement for this graph is that it must be connected.
            G = self.__points_to_graph(node_points)

            assignments = self.__split_graph(
                G,
                tuple(len(tree.leaves_in_subtree(child)) for child in children),
            )
            for child, nodes in zip(children, assignments):
                for i in node_idx[nodes]:
                    point_assignments[i] = child
        # Add leaf locations
        locations.update(
            {leaf: points[i] for i, leaf in enumerate(point_assignments)}
        )

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
