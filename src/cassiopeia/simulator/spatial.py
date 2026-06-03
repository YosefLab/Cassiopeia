"""Functional spatial data simulation for Cassiopeia."""

import networkx as nx
import numpy as np
import treedata as td
from scipy import spatial as scipy_spatial
from scipy.stats.qmc import PoissonDisk

from cassiopeia.mixins import DataSimulatorError


def brownian_spatial(
    tdata: td.TreeData,
    dim: int = 2,
    diffusion_coefficient: float = 1.0,
    scale_unit_area: bool = False,
    random_seed: int | None = None,
    tree_key: str = "simulated",
    time_key: str = "time",
    key_added: str = "spatial",
    copy: bool = False,
) -> td.TreeData:
    """Overlay spatial coordinates via Brownian motion along the tree.

    The root is placed at the origin. Each other node's position is the parent
    position plus a Normal(0, sqrt(2 * D * t)) displacement per dimension,
    where D is the diffusion coefficient and t is the branch length. Displacements
    are independent across dimensions and nodes.

    Leaf coordinates are stored in ``tdata.obsm[key_added]`` as a numpy array.
    All node coordinates (including internal nodes) are stored in ``tdata.obst[tree_key].nodes[node][key_added]``.

    Args:
        tdata: TreeData with a tree in ``obst[tree_key]``.
        dim: Number of spatial dimensions.
        diffusion_coefficient: Diffusion coefficient D (>= 0). Variance per
            unit time is 2 * D.
        scale_unit_area: If True, shift and scale all coordinates so
            they lie in [0, 1] (all dimensions scaled by the same factor).
        random_seed: NumPy random seed for reproducibility.
        tree_key: Key in ``tdata.obst`` for the tree.
        time_key: Node attribute key for branch lengths (default ``"time"``). Must be present on all nodes.
        key_added: Key for storing coordinates in ``tdata.obsm`` and as node
            attributes.
        copy: If ``True``, operate on a copy of ``tdata`` and return the copy.
            If ``False`` (default), modify ``tdata`` in-place.

    Returns:
        Modified ``tdata``. If ``copy=True``, a new TreeData; otherwise the
        input modified in-place.

    Raises:
        DataSimulatorError: If ``dim`` <= 0 or ``diffusion_coefficient`` < 0.
    """
    if dim <= 0:
        raise DataSimulatorError("Number of dimensions must be positive.")
    if diffusion_coefficient < 0:
        raise DataSimulatorError("Diffusion coefficient must be non-negative.")

    if copy:
        tdata = tdata.copy()

    if random_seed is not None:
        np.random.seed(random_seed)

    tree = tdata.obst[tree_key]
    root = next(n for n in tree if tree.in_degree(n) == 0)

    locations: dict = {root: np.zeros(dim)}
    for node in nx.topological_sort(tree):
        if node == root:
            continue
        parent = next(iter(tree.predecessors(node)))
        branch_len = tree.nodes[node][time_key] - tree.nodes[parent][time_key]
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
        locations = dict(zip(locations.keys(), all_coords, strict=False))

    for node, loc in locations.items():
        tree.nodes[node][key_added] = loc

    leaves = list(tdata.obs_names)
    coords = np.array([locations[leaf] for leaf in leaves])
    tdata.obsm[key_added] = coords

    if copy:
        return tdata


def clonal_spatial(
    tdata: td.TreeData,
    shape: tuple[int, ...] | None = None,
    space: np.ndarray | None = None,
    random_seed: int | None = None,
    tree_key: str = "simulated",
    key_added: str = "spatial",
    copy: bool = False,
) -> td.TreeData:
    """Overlay spatial coordinates with clonal spatial autocorrelation.

    N coordinates (N = number of leaves) are sampled in the given space via
    Poisson-disc sampling. These are then assigned to leaves by traversing the
    tree top-down: at each internal node, its assigned points are split among
    children by spatial proximity (closest-seed Dijkstra assignment on a
    Delaunay/kNN graph), so that leaves sharing recent ancestry tend to be
    spatially clustered.

    Leaf coordinates are stored in ``tdata.obsm[key_added]`` as a numpy array.
    Internal node coordinates (centroid of their assigned points) are stored in ``tdata.obst[tree_key].nodes[node][key_added]``.

    Args:
        tdata: TreeData with a tree in ``obst[tree_key]``.
        shape: Shape of the spatial region (e.g. ``(100, 100)``). For 2D, an
            elliptical region is used; for higher dimensions, the entire
            hypercuboid. Mutually exclusive with ``space``.
        space: Boolean mask defining the spatial region. Mutually exclusive
            with ``shape``.
        random_seed: NumPy random seed for reproducibility.
        tree_key: Key in ``tdata.obst`` for the tree.
        key_added: Key for storing coordinates in ``tdata.obsm`` and as node
            attributes.
        copy: If ``True``, operate on a copy of ``tdata`` and return the copy.
            If ``False`` (default), modify ``tdata`` in-place.

    Returns:
        Modified ``tdata``. If ``copy=True``, a new TreeData; otherwise the
        input modified in-place.

    Raises:
        DataSimulatorError: If neither/both of ``shape`` and ``space`` are provided.
    """
    if (shape is None) == (space is None):
        raise DataSimulatorError("Specify exactly one of `shape` or `space`.")

    if copy:
        tdata = tdata.copy()

    if random_seed is not None:
        np.random.seed(random_seed)

    if shape is not None:
        dim = len(shape)
        if dim == 2:
            center_x = shape[1] // 2
            center_y = shape[0] // 2
            if center_x > 0 and center_y > 0:
                y, x = np.ogrid[: shape[0], : shape[1]]
                actual_space = ((x - center_x) / center_x) ** 2 + (
                    (y - center_y) / center_y
                ) ** 2 <= 1
            else:
                actual_space = np.ones(shape, dtype=bool)
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
        node_idx = np.array([i for i, assign in enumerate(point_assignments) if assign == node])
        node_points = points[node_idx]
        locations[node] = node_points.mean(axis=0)

        G = _points_to_graph(node_points)
        sizes = tuple(_n_leaves_in_subtree(tree, child) for child in children)
        assignments = _split_graph(G, sizes)

        for child, partition in zip(children, assignments, strict=False):
            for i in node_idx[partition]:
                point_assignments[i] = child

    locations.update({node: points[i] for i, node in enumerate(point_assignments)})

    for node, loc in locations.items():
        tree.nodes[node][key_added] = loc

    leaves = list(tdata.obs_names)
    coords = np.array([locations[leaf] for leaf in leaves])
    tdata.obsm[key_added] = coords

    if copy:
        return tdata


# --- Private helpers ---


def _n_leaves_in_subtree(tree: nx.DiGraph, node: str) -> int:
    """Count leaf descendants of node (inclusive if node is a leaf)."""
    if tree.out_degree(node) == 0:
        return 1
    return sum(1 for d in nx.descendants(tree, node) if tree.out_degree(d) == 0)


def _sample_points(space: np.ndarray, dim: int, n: int) -> np.ndarray:
    """Sample n points in space using Poisson-disc sampling."""
    shape = space.shape
    # PoissonDisk works in the unit hypercube; convert radius to normalized units
    radius = 1.0 / (2 * n ** (1 / dim))
    while True:
        engine = PoissonDisk(d=dim, radius=radius, seed=np.random.randint(2**31))
        unit_points = engine.fill_space()
        # Scale to pixel coordinates and filter by the boolean mask
        points = unit_points * np.array(shape)
        idx = tuple(points.astype(int).clip(0, np.array(shape) - 1).T)
        points = points[space[idx]]
        if len(points) >= n:
            break
        radius /= 2
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
    tree = scipy_spatial.KDTree(points)
    distances, indices = tree.query(points, k=k + 1)  # +1 to exclude self
    G = nx.Graph()
    G.add_nodes_from(range(len(points)))
    for i, (dists, idxs) in enumerate(zip(distances[:, 1:], indices[:, 1:], strict=False)):
        for d, j in zip(dists, idxs, strict=False):
            G.add_edge(i, int(j), weight=float(d))
    return G


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
        raise DataSimulatorError(f"Cannot partition {len(G.nodes)} nodes into sizes {sizes}.")

    seeds = dict(
        zip(np.random.choice(list(G.nodes), len(sizes), replace=False), sizes, strict=False)
    )
    seed_distances = {seed: nx.single_source_dijkstra_path_length(G, seed) for seed in seeds}

    distance_seed_nodes = sorted(
        (dist, seed, node) for seed, dists in seed_distances.items() for node, dist in dists.items()
    )

    assigned: set = set()
    assignments: dict = {}
    for _, seed, node in distance_seed_nodes:
        if node in assigned or len(assignments.get(seed, [])) == seeds[seed]:
            continue
        assignments.setdefault(seed, []).append(node)
        assigned.add(node)

    return tuple(assignments[seed] for seed in seeds)
