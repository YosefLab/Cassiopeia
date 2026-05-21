"""Functional spatial data simulation for Cassiopeia."""

import networkx as nx
import numpy as np
import pandas as pd
import treedata as td
from scipy import spatial as scipy_spatial

from cassiopeia.mixins import DataSimulatorError, try_import

cv2 = try_import("cv2")
disc = try_import("poisson_disc")
neighbors = try_import("sklearn.neighbors")


def brownian_spatial(
    tdata: td.TreeData,
    dim: int,
    diffusion_coefficient: float,
    scale_unit_area: bool = True,
    random_seed: int | None = None,
    tree_key: str = "tree",
    spatial_key: str = "spatial",
) -> td.TreeData:
    """Overlay spatial coordinates via Brownian motion along the tree.

    The root is placed at the origin. Each other node's position is the parent
    position plus a Normal(0, sqrt(2 * D * t)) displacement per dimension,
    where D is the diffusion coefficient and t is the branch length. Displacements
    are independent across dimensions and nodes.

    Leaf coordinates are stored in ``tdata.obsm[spatial_key]`` as a DataFrame
    with columns ``dim_0, dim_1, ...``. All node coordinates (including internal
    nodes) are stored in ``tdata.obst[tree_key].nodes[node][spatial_key]``.

    Args:
        tdata: TreeData with a tree in ``obst[tree_key]``. Nodes must have a
            ``"time"`` attribute.
        dim: Number of spatial dimensions.
        diffusion_coefficient: Diffusion coefficient D (>= 0). Variance per
            unit time is 2 * D.
        scale_unit_area: If True (default), shift and scale all coordinates so
            they lie in [0, 1] (all dimensions scaled by the same factor).
        random_seed: NumPy random seed for reproducibility.
        tree_key: Key in ``tdata.obst`` for the tree.
        spatial_key: Key for storing coordinates in ``tdata.obsm`` and as node
            attributes.

    Returns:
        tdata modified in-place with spatial coordinates added.

    Raises:
        DataSimulatorError: If ``dim`` <= 0 or ``diffusion_coefficient`` < 0.
    """
    if dim <= 0:
        raise DataSimulatorError("Number of dimensions must be positive.")
    if diffusion_coefficient < 0:
        raise DataSimulatorError("Diffusion coefficient must be non-negative.")

    if random_seed is not None:
        np.random.seed(random_seed)

    tree = tdata.obst[tree_key]
    root = next(n for n in tree if tree.in_degree(n) == 0)

    locations: dict = {root: np.zeros(dim)}
    for node in nx.topological_sort(tree):
        if node == root:
            continue
        parent = next(iter(tree.predecessors(node)))
        branch_len = tree.nodes[node]["time"] - tree.nodes[parent]["time"]
        locations[node] = locations[parent] + np.random.normal(
            scale=np.sqrt(2 * diffusion_coefficient * branch_len),
            size=dim,
        )

    if scale_unit_area:
        all_coords = np.array(list(locations.values()))
        all_coords -= all_coords.min(axis=0)
        max_val = all_coords.max()
        if max_val > 0:
            all_coords /= max_val
        locations = dict(zip(locations.keys(), all_coords))

    for node, loc in locations.items():
        tree.nodes[node][spatial_key] = loc

    leaves = list(tdata.obs_names)
    columns = [f"dim_{i}" for i in range(dim)]
    coords = np.array([locations[leaf] for leaf in leaves])
    tdata.obsm[spatial_key] = pd.DataFrame(coords, index=leaves, columns=columns)

    return tdata


def clonal_spatial(
    tdata: td.TreeData,
    shape: tuple[int, ...] | None = None,
    space: np.ndarray | None = None,
    random_seed: int | None = None,
    tree_key: str = "tree",
    spatial_key: str = "spatial",
) -> td.TreeData:
    """Overlay spatial coordinates with clonal spatial autocorrelation.

    N coordinates (N = number of leaves) are sampled in the given space via
    Poisson-disc sampling. These are then assigned to leaves by traversing the
    tree top-down: at each internal node, its assigned points are split among
    children by spatial proximity (closest-seed Dijkstra assignment on a
    Delaunay/kNN graph), so that leaves sharing recent ancestry tend to be
    spatially clustered.

    Leaf coordinates are stored in ``tdata.obsm[spatial_key]`` as a DataFrame
    with columns ``dim_0, dim_1, ...``. Internal node coordinates (centroid of
    their assigned points) are stored in ``tdata.obst[tree_key].nodes[node][spatial_key]``.

    Requires the ``spatial`` extras: ``pip install cassiopeia-lineage[spatial]``.

    Args:
        tdata: TreeData with a tree in ``obst[tree_key]``.
        shape: Shape of the spatial region (e.g. ``(100, 100)``). For 2D, an
            elliptical region is used; for higher dimensions, the entire
            hypercuboid. Mutually exclusive with ``space``.
        space: Boolean mask defining the spatial region. Mutually exclusive
            with ``shape``.
        random_seed: NumPy random seed for reproducibility.
        tree_key: Key in ``tdata.obst`` for the tree.
        spatial_key: Key for storing coordinates in ``tdata.obsm`` and as node
            attributes.

    Returns:
        tdata modified in-place with spatial coordinates added.

    Raises:
        DataSimulatorError: If spatial extras are missing, or if neither/both
            of ``shape`` and ``space`` are provided.
    """
    if cv2 is None or disc is None or neighbors is None:
        raise DataSimulatorError(
            "Some required modules were not found. Install cassiopeia with "
            "the `spatial` extras: pip install cassiopeia-lineage[spatial]"
        )

    if (shape is None) == (space is None):
        raise DataSimulatorError("Specify exactly one of `shape` or `space`.")

    if random_seed is not None:
        np.random.seed(random_seed)

    if shape is not None:
        dim = len(shape)
        if dim == 2:
            center_x = shape[1] // 2
            center_y = shape[0] // 2
            actual_space = cv2.ellipse(
                np.zeros(shape, dtype=np.uint8),
                (center_x, center_y),
                (center_x, center_y),
                0, 0, 360, 1, -1,
            ).astype(bool)
        else:
            actual_space = np.ones(shape, dtype=bool)
    else:
        dim = space.ndim
        actual_space = space

    tree = tdata.obst[tree_key]
    root = next(n for n in tree if tree.in_degree(n) == 0)
    n_leaves = len(tdata.obs_names)

    points = _sample_points(actual_space, dim, n_leaves)

    point_assignments = [root] * len(points)
    locations: dict = {}

    for node in nx.topological_sort(tree):
        if tree.out_degree(node) == 0:
            continue

        children = list(tree.successors(node))
        node_idx = np.array(
            [i for i, assign in enumerate(point_assignments) if assign == node]
        )
        node_points = points[node_idx]
        locations[node] = node_points.mean(axis=0)

        G = _points_to_graph(node_points)
        sizes = tuple(
            _n_leaves_in_subtree(tree, child) for child in children
        )
        assignments = _split_graph(G, sizes)

        for child, partition in zip(children, assignments):
            for i in node_idx[partition]:
                point_assignments[i] = child

    locations.update({node: points[i] for i, node in enumerate(point_assignments)})

    for node, loc in locations.items():
        tree.nodes[node][spatial_key] = loc

    leaves = list(tdata.obs_names)
    columns = [f"dim_{i}" for i in range(dim)]
    coords = np.array([locations[leaf] for leaf in leaves])
    tdata.obsm[spatial_key] = pd.DataFrame(coords, index=leaves, columns=columns)

    return tdata


# --- Private helpers ---


def _n_leaves_in_subtree(tree: nx.DiGraph, node: str) -> int:
    """Count leaf descendants of node (inclusive if node is a leaf)."""
    if tree.out_degree(node) == 0:
        return 1
    return sum(1 for d in nx.descendants(tree, node) if tree.out_degree(d) == 0)


def _sample_points(space: np.ndarray, dim: int, n: int) -> np.ndarray:
    """Sample n points in space using Poisson-disc (Bridson) sampling."""
    shape = space.shape
    radius = (min(shape) / (n ** (1 / dim))) / 2
    while True:
        points = disc.Bridson_sampling(dims=np.array(shape), radius=radius)
        radius /= 2
        points = points[space[tuple(points.T.astype(int))]]
        if len(points) >= n:
            break
    np.random.shuffle(points)
    return points[:n]


def _triangulation_graph(points: np.ndarray) -> nx.Graph:
    """Fully-connected Delaunay triangulation graph with Euclidean edge weights."""
    tri = scipy_spatial.Delaunay(points)
    G = nx.Graph()
    for path in tri.simplices:
        nx.add_path(G, path)
    for n1, n2 in G.edges:
        G[n1][n2]["weight"] = scipy_spatial.distance.euclidean(points[n1], points[n2])
    return G


def _nearest_neighbors_graph(points: np.ndarray, k: int) -> nx.Graph:
    """k-nearest-neighbors graph with distance edge weights."""
    distances = neighbors.kneighbors_graph(points, k, mode="distance")
    return nx.from_scipy_sparse_array(distances)


def _points_to_graph(points: np.ndarray) -> nx.Graph:
    """Build a connected graph (Delaunay for >5 points, kNN otherwise)."""
    if len(points) > 5:
        return _triangulation_graph(points)
    return _nearest_neighbors_graph(points, min(5, len(points) - 1))


def _split_graph(G: nx.Graph, sizes: tuple[int, ...]) -> tuple[list[int], ...]:
    """Partition G nodes into groups of given sizes by closest-seed assignment.

    Seeds are chosen randomly; each non-seed node is assigned to the seed
    with the shortest Dijkstra path, respecting the target partition size.
    """
    if not nx.is_connected(G):
        raise DataSimulatorError("Graph is not connected.")
    if sum(sizes) != len(G.nodes):
        raise DataSimulatorError(
            f"Cannot partition {len(G.nodes)} nodes into sizes {sizes}."
        )

    seeds = dict(zip(np.random.choice(list(G.nodes), len(sizes), replace=False), sizes))
    seed_distances = {
        seed: nx.single_source_dijkstra_path_length(G, seed) for seed in seeds
    }

    distance_seed_nodes = sorted(
        (dist, seed, node)
        for seed, dists in seed_distances.items()
        for node, dist in dists.items()
    )

    assigned: set = set()
    assignments: dict = {}
    for _, seed, node in distance_seed_nodes:
        if node in assigned or len(assignments.get(seed, [])) == seeds[seed]:
            continue
        assignments.setdefault(seed, []).append(node)
        assigned.add(node)

    return tuple(assignments[seed] for seed in seeds)
