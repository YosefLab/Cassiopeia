import networkx as nx
import numpy as np
import pandas as pd
import treedata as td

from cassiopeia.mixins import DataSimulatorError
from cassiopeia.utils import _check_tree_has_key, _get_digraph, get_root


def _add_expression_to_tdata(tdata, expression, layer=None):
    """Update TreeData with expression data."""
    new_tdata = td.TreeData(
        obs=tdata.obs.copy(),
        var=pd.DataFrame(index=[f"Gene_{i}" for i in range(expression.shape[1])]),
        obsm=tdata.obsm.copy(),
        varm=tdata.varm.copy(),
        layers=tdata.layers.copy(),
        uns=tdata.uns.copy(),
        obst=tdata.obst.copy(),
        alignment=tdata.alignment,
        label=tdata.label,
    )
    if layer is not None:
        new_tdata.layers[layer] = expression
    else:
        new_tdata.X = expression
    return new_tdata


def _expression_from_latent(X_latent, latent_dim, n_genes, rng):
    """Get expression matrix from latent matrix and loading matrix."""
    # Scale by 1/sqrt(latent_dim) so gene-level variance is ~O(1)
    gene_loading_matrix = rng.normal(
        loc=0.0,
        scale=1.0 / np.sqrt(latent_dim),
        size=(latent_dim, n_genes),
    )
    X = X_latent @ gene_loading_matrix
    return X


def brownian_expression(
    tree: td.TreeData,
    latent_dim: int = 20,
    n_genes: int = 5000,
    diffusion: float = 1.0,
    momentum: float = 0.5,
    random_seed: int | None = None,
    latent_key: str = "X_latent",
    depth_key: str = "time",
    tree_key: str | None = None,
    layer_added: str | None = None,
) -> td.TreeData:
    r"""Simulate expression data along a tree via Brownian motion in latent space.

    This function simulates a latent Brownian-motion trajectory for each cell
    along a rooted tree with AR(1)-style momentum in the latent
    displacements. The latent coordinates are then linearly projected into
    gene space using a random (or user-supplied) loading matrix.

    The function returns a copy of  ``tree`` with the following updates:

    * Latent coordinates for each node are stored in ``tree.obsm[latent_key]``
      and as node attributes under ``latent_key``.
    * The reconstructed expression matrix is stored either in
      ``tree.layers[layer_added]`` (if provided) or in ``tree.X``.

    Parameters
    ----------
    tree:
        The :class:`td.TreeData` object.
    latent_dim
        Dimensionality of the latent Brownian space.
    n_genes
        Number of genes to simulate in the observed expression matrix.
    diffusion
        Diffusion coefficient :math:`D` for the Brownian process. For an
        edge of length :math:`\\Delta t`, the latent variance along that edge
        is :math:`2 D \\Delta t`.
    momentum
        Momentum coefficient :math:`m` in [0, 1). When non-zero, the latent
        displacement on a child edge is modeled as an AR(1) process:

        ``disp_child = m * disp_parent + innovation``

        where the innovation variance is chosen so that the marginal
        displacement variance along each edge matches the Brownian variance.
        When ``momentum == 0.0``, displacements are i.i.d. Gaussian for each
        edge.
    random_seed
        Optional random seed for reproducibility.
    latent_key
        Key under which to store latent positions in node attributes and in
        ``tree.obsm[latent_key]``.
    depth_key
        Name of the node attribute giving the time/depth from the root.
    tree_key
        Specifies which tree to use if ``tree`` contains multiple trees.
    layer_added
        If specified, the reconstructed expression matrix is stored in
        ``tree.layers[layer_added]``. Otherwise it is stored in ``tree.X``.

    Returns:
    -------
    tree : :class:`td.TreeData`
        The modified TreeData object with simulated data overlaid.

    Raises:
    ------
    DataSimulatorError
        If ``latent_dim <= 0``, ``diffusion_coefficient < 0``,
        ``momentum`` is not in ``[0, 1)``, or if the depth/time is decreasing
        along any edge.

    Notes:
    -----
    * The root node is placed at the origin in latent space.
    * Branches with zero length (or zero implied variance) simply inherit the
      parent's latent position with zero displacement.
    """
    # Setup
    if latent_dim <= 0:
        raise DataSimulatorError("Number of dimensions must be positive.")
    if diffusion < 0:
        raise DataSimulatorError("Diffusion coefficient must be non-negative.")
    if not (0.0 <= momentum < 1.0):
        raise DataSimulatorError("Momentum must be in [0, 1).")
    rng = np.random.default_rng(random_seed)

    # Extract directed tree and root
    t, _ = _get_digraph(tree, tree_key)
    root = get_root(t)
    _check_tree_has_key(t, depth_key)
    last_disp_key = "_last_displacement"

    # Simulate latent space trajectories
    for node in nx.topological_sort(t):
        if node == root:
            # Root at origin in latent space; zero displacement
            t.nodes[node][latent_key] = np.zeros(latent_dim)
            t.nodes[node][last_disp_key] = np.zeros(latent_dim)
            continue

        parent = next(t.predecessors(node))
        t_parent = t.nodes[parent][depth_key]
        t_node = t.nodes[node][depth_key]
        branch_length = t_node - t_parent

        if branch_length < 0:
            raise DataSimulatorError(
                f"Depth/time must be non-decreasing along edges, "
                f"but {parent}->{node} has branch_length={branch_length}."
            )

        # Brownian variance along this edge
        variance = 2.0 * diffusion * branch_length

        if branch_length == 0.0 or variance == 0.0:
            # Zero-length edge: no new displacement
            disp = np.zeros(latent_dim)
        else:
            parent_disp = t.nodes[parent][last_disp_key]

            if momentum == 0.0:
                # Pure Brownian: displacements are i.i.d. Gaussian
                innovation_std = np.sqrt(variance)
                disp = rng.normal(
                    loc=0.0,
                    scale=innovation_std,
                    size=latent_dim,
                )
            else:
                # AR(1) displacement: disp_child = m * disp_parent + innovation
                # Choose innovation variance such that Var(disp_child) matches
                # the Brownian variance along the edge:
                # Var(innovation) = (1 - m^2) * variance
                innovation_var = (1.0 - momentum**2) * variance
                innovation_std = np.sqrt(innovation_var)
                innovation = rng.normal(
                    loc=0.0,
                    scale=innovation_std,
                    size=latent_dim,
                )
                disp = momentum * parent_disp + innovation

        # Update latent position and last displacement for this node
        t.nodes[node][latent_key] = t.nodes[parent][latent_key] + disp
        t.nodes[node][last_disp_key] = disp

    # Build latent matrix (cells x latent_dim)
    X_latent = np.vstack([t.nodes[node][latent_key] for node in tree.obs_names])
    tree.obsm[latent_key] = X_latent

    # Get gene expression via linear projection
    X = _expression_from_latent(X_latent, latent_dim, n_genes, rng)

    # Clean up temporary keys
    for node in t.nodes:
        t.nodes[node].pop(last_disp_key, None)

    return _add_expression_to_tdata(tree, X, layer_added)


def _rescale_node_times(
    tree: nx.DiGraph,
    time_key: str,
    scaled_time_key: str,
    min_time: float = 0.0,
    max_time: float = 1.0,
):
    """Linearly rescale node times so that they lie in [min_time, max_time]."""
    times = np.array([float(tree.nodes[n][time_key]) for n in tree.nodes], dtype=float)
    t_min, t_max = times.min(), times.max()

    if t_max == t_min:
        for n in tree.nodes:
            tree.nodes[n][scaled_time_key] = float(min_time)
        return

    scale = (max_time - min_time) / (t_max - t_min)
    for n in tree.nodes:
        original_t = float(tree.nodes[n][time_key])
        tree.nodes[n][scaled_time_key] = min_time + (original_t - t_min) * scale


def fate_tree_expression(
    lineage_tree: td.TreeData,
    fate_tree: td.TreeData | nx.DiGraph,
    n_genes: int = 5000,
    noise: float = 0.0,
    factor_key: str = "X_latent",
    prob_key: str | None = None,
    depth_key: str = "time",
    random_state: int | None = None,
    layer_added: str | None = None,
) -> td.TreeData:
    """Annotate a lineage tree with continuous latent factors using a factor (fate) tree.

    For each lineage node:
      - follow a single path in the factor tree, consistent with the parent's assignment,
      - use rescaled factor-tree time to find a factor-tree edge covering the node's time,
      - set the node's latent vector to a linear interpolation of the edge's endpoints.

    Parameters
    ----------
    lineage_tree
        The lineage :class:`td.TreeData`to annotate.
    fate_tree
        Fate/factor tree with latent vectors on nodes.
        Node attributes:
          - depth_key : float
          - factor_key : np.ndarray, shape (latent_dim,)
        Edge attributes (optional):
          - prob_key : float, used as transition weight when choosing children.
    n_genes
        Number of genes to simulate in the observed expression matrix.
    noise
        Standard deviation of Gaussian noise added to latent vectors.
    factor_key
        Node attribute on fate_tree containing latent vectors.
    prob_key
        If provided, use this edge attribute on fate_tree[u][v][prob_key]
        as transition weights when choosing a child of u. If None, choose uniformly.
    depth_key
        Node attribute containing times on lineage_tree and fate_tree.
    random_state
        Seed for reproducibility.
    layer_added
        If specified, the reconstructed expression matrix is stored in
        ``lineage_tree.layers[layer_added]``. Otherwise it is stored in ``lineage_tree.X``.

    Returns:
    -------
    lineage_tdata : :class:`td.TreeData`
        Copy of lineage_tree with latent factors and expression data overlaid.
    """
    # Setup
    lineage_tdata = lineage_tree.copy()
    rng = np.random.default_rng(random_state)
    scaled_time_key = "scaled_time"
    factor_edge_key = "factor_edge"
    fate_tree, _ = _get_digraph(fate_tree, copy=True)
    lineage_tree, _ = _get_digraph(lineage_tree, copy=True)
    factor_root = get_root(fate_tree)
    lineage_root = get_root(lineage_tree)
    factor_topo = list(nx.topological_sort(fate_tree))
    lineage_topo = list(nx.topological_sort(lineage_tree))

    # infer latent_dim from factor tree
    example = factor_topo[0]
    z_example = np.asarray(fate_tree.nodes[example][factor_key], dtype=float)
    latent_dim = z_example.shape[0]

    # Rescale node times to [0, 1]
    for tree in [fate_tree, lineage_tree]:
        _rescale_node_times(tree, time_key=depth_key, scaled_time_key=scaled_time_key)

    # Helper: choose a child of u in fate_tree
    def choose_child(u):
        """Choose a child of u in fate_tree using prob_key if provided."""
        children = list(fate_tree.successors(u))
        if not children:
            return None
        if prob_key is None:
            return rng.choice(children)
        # Use edge attribute prob_key as weight, default 1.0 if missing
        weights = np.array(
            [fate_tree[u][v].get(prob_key, 1.0) for v in children],
            dtype=float,
        )
        if weights.sum() <= 0:
            probs = np.ones_like(weights) / len(weights)
        else:
            probs = weights / weights.sum()
        return rng.choice(children, p=probs)

    # Helper: advance along factor tree so that time t lies on an edge
    def advance_edge(edge, t: float):
        u, v = edge
        while True:
            t_v = float(fate_tree.nodes[v][scaled_time_key])
            # if t is within [t_u, t_v] or we're clamped at a leaf
            if t <= t_v or u == v:
                return (u, v)
            # otherwise step forward from v to one of its children
            next_child = choose_child(v)
            if next_child is None:
                # at a leaf; clamp
                return (u, v)
            u, v = v, next_child

    # Assign a starting factor edge for the lineage root
    start_child = choose_child(factor_root)
    if start_child is None:
        root_edge = (factor_root, factor_root)
    else:
        root_edge = (factor_root, start_child)

    factor_edge_for_lineage = {lineage_root: root_edge}

    # Annotate lineage tree
    embeddings = []
    for node in lineage_topo:
        t_node = float(lineage_tree.nodes[node][scaled_time_key])
        if node == lineage_root:
            edge = factor_edge_for_lineage[node]
        else:
            parent = next(lineage_tree.predecessors(node))
            parent_edge = factor_edge_for_lineage[parent]
            edge = advance_edge(parent_edge, t_node)
            factor_edge_for_lineage[node] = edge

        u, v = edge
        t_u = float(fate_tree.nodes[u][scaled_time_key])
        t_v = float(fate_tree.nodes[v][scaled_time_key])

        z_u = np.asarray(fate_tree.nodes[u][factor_key], dtype=float)

        if u == v or t_v == t_u:
            z = z_u.copy()
        else:
            z_v = np.asarray(fate_tree.nodes[v][factor_key], dtype=float)
            alpha = (t_node - t_u) / (t_v - t_u)
            alpha = np.clip(alpha, 0.0, 1.0)
            z = (1.0 - alpha) * z_u + alpha * z_v

        lineage_tree.nodes[node][factor_key] = z
        if factor_edge_key is not None:
            lineage_tree.nodes[node][factor_edge_key] = (u, v)
        embeddings.append(z)

    embeddings = np.vstack(embeddings)
    if noise > 0.0:
        embeddings += rng.normal(loc=0.0, scale=noise, size=embeddings.shape)
    columns = [f"{i}" for i in range(latent_dim)]
    embeddings = pd.DataFrame(embeddings, index=lineage_topo, columns=columns)
    lineage_tdata.obsm[factor_key] = embeddings.loc[lineage_tdata.obs_names]

    # Get gene expression via linear projection
    X = _expression_from_latent(embeddings, latent_dim, n_genes, rng)

    return _add_expression_to_tdata(lineage_tdata, X, layer_added)
