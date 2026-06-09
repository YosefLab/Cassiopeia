from typing import Literal

import networkx as nx
import numpy as np
import pandas as pd
import treedata as td

from cassiopeia.mixins import DataSimulatorError
from cassiopeia.utils import _check_tree_has_key, _get_digraph, _get_root


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


def _sample_counts(
    X: np.ndarray,
    distribution: Literal["gaussian", "poisson", "negative_binomial"],
    library_size: float,
    dispersion: float,
    rng: np.random.Generator,
) -> np.ndarray:
    r"""Apply an observation model to a continuous expression projection.

    For ``"gaussian"`` the continuous projection is returned unchanged. For the
    count models the projection is mapped to a non-negative per-cell rate via a
    row-wise softmax scaled by ``library_size`` (so each cell's expected total
    counts is ``library_size``), then sampled:

    * ``"poisson"``: ``counts ~ Poisson(lam)``.
    * ``"negative_binomial"``: ``counts ~ NB(n, p)`` with ``n = 1/dispersion``
      and ``p = n / (n + lam)``, i.e. mean ``lam`` and variance
      ``lam + dispersion * lam**2``.

    Returns:
        ``X`` unchanged for ``"gaussian"``; an ``int64`` count matrix otherwise.
    """
    if distribution == "gaussian":
        return X

    # Row-wise softmax (max-subtracted for numerical stability) -> per-cell
    # composition summing to 1, scaled to the expected library size.
    shifted = X - X.max(axis=1, keepdims=True)
    weights = np.exp(shifted)
    probs = weights / weights.sum(axis=1, keepdims=True)
    lam = library_size * probs

    if distribution == "poisson":
        return rng.poisson(lam).astype(np.int64)
    if distribution == "negative_binomial":
        if dispersion <= 0:
            raise DataSimulatorError(
                "dispersion must be positive for the negative_binomial distribution."
            )
        n = 1.0 / dispersion
        p = n / (n + lam)
        return rng.negative_binomial(n, p).astype(np.int64)
    raise DataSimulatorError(
        f"Unknown distribution {distribution!r}. Expected one of "
        "'gaussian', 'poisson', 'negative_binomial'."
    )


def brownian_expression(
    tdata: td.TreeData,
    latent_dim: int = 20,
    n_genes: int = 5000,
    diffusion: float = 1.0,
    momentum: float = 0.5,
    distribution: Literal["gaussian", "poisson", "negative_binomial"] = "gaussian",
    library_size: float = 10000,
    dispersion: float = 0.1,
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

    The function returns a copy of ``tdata`` with the following updates:

    * Latent coordinates for each node are stored in ``tdata.obsm[latent_key]``
      and as node attributes under ``latent_key``.
    * The reconstructed expression matrix is stored either in
      ``tdata.layers[layer_added]`` (if provided) or in ``tdata.X``.

    Parameters
    ----------
    tdata:
        The :class:`td.TreeData` object.
    latent_dim
        Dimensionality of the latent Brownian space.
    n_genes
        Number of genes to simulate in the observed expression matrix.
    distribution
        Observation model used to generate the expression matrix from the
        latent projection. ``"gaussian"`` (default) returns the continuous
        projection unchanged; ``"poisson"`` and ``"negative_binomial"`` sample
        integer counts (see :func:`_sample_counts`).
    library_size
        Expected total counts per cell, used to scale the per-cell rate for the
        count distributions. Ignored when ``distribution == "gaussian"``.
    dispersion
        Overdispersion (inverse size) of the negative binomial; the count
        variance is ``lam + dispersion * lam**2``. Ignored for the other
        distributions.
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
        ``tdata.obsm[latent_key]``.
    depth_key
        Name of the node attribute giving the time/depth from the root.
    tree_key
        Specifies which tree to use if ``tdata`` contains multiple trees.
    layer_added
        If specified, the reconstructed expression matrix is stored in
        ``tdata.layers[layer_added]``. Otherwise it is stored in ``tdata.X``.

    Returns:
    -------
    tdata : :class:`td.TreeData`
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
    t, _ = _get_digraph(tdata, tree_key)
    root = _get_root(t)
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
    X_latent = np.vstack([t.nodes[node][latent_key] for node in tdata.obs_names])
    tdata.obsm[latent_key] = X_latent

    # Get gene expression via linear projection, then apply the observation model
    X = _expression_from_latent(X_latent, latent_dim, n_genes, rng)
    X = _sample_counts(X, distribution, library_size, dispersion, rng)

    # Clean up temporary keys
    for node in t.nodes:
        t.nodes[node].pop(last_disp_key, None)

    return _add_expression_to_tdata(tdata, X, layer_added)


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


def trajectory_expression(
    tdata: td.TreeData,
    trajectory: td.TreeData | nx.DiGraph,
    n_genes: int = 5000,
    distribution: Literal["gaussian", "poisson", "negative_binomial"] = "gaussian",
    library_size: float = 10000,
    dispersion: float = 0.1,
    latent_noise: float = 0.0,
    latent_key: str = "X_latent",
    prob_key: str | None = None,
    depth_key: str = "time",
    random_seed: int | None = None,
    layer_added: str | None = None,
) -> td.TreeData:
    r"""Simulate expression along a lineage by mapping cells onto a cell-state trajectory.

    Unlike :func:`brownian_expression`, which lets each cell drift freely in
    latent space, this function constrains cells to follow a predefined
    cell-state ``trajectory``. The trajectory is a rooted tree whose nodes carry
    latent vectors (under ``latent_key``) and times (under ``depth_key``); its
    edges are continuous transitions between states and its branch points are
    fate decisions. The lineage tree (``tdata``) supplies *when* each cell exists
    (its node times), while the trajectory supplies *what state* a cell occupies
    at a given time. The result is expression data in which clonally related
    cells share state up to the point their ancestors committed to different
    trajectory branches.

    Algorithm
    ---------
    1. **Time rescaling.** The node times of the lineage tree and of the
       trajectory are each independently rescaled to ``[0, 1]`` (see
       :func:`_rescale_node_times`), so the two trees are compared on a common
       pseudotime axis regardless of their absolute time units.
    2. **Walking the trajectory.** Cells are visited in topological (root → leaf)
       order. Each cell inherits the trajectory edge its parent ended on, then
       *advances* forward along the trajectory until it reaches the edge whose
       time interval contains the cell's rescaled time. Whenever the walk passes
       a branch point, one outgoing edge is chosen at random — uniformly, or in
       proportion to the ``prob_key`` edge weights if provided. Because the
       choice is made per cell, sibling lineages can commit to different
       branches, reproducing divergent fate decisions. A walk that reaches a
       trajectory leaf before exhausting the cell's time is clamped there.
    3. **Latent interpolation.** Within the resolved edge ``(u, v)`` the cell's
       latent vector is the linear interpolation
       ``z = (1 - alpha) * z_u + alpha * z_v``, where
       ``alpha = (t_cell - t_u) / (t_v - t_u)`` is clamped to ``[0, 1]``. Cells
       clamped at a leaf (or on a zero-length edge) take that node's latent
       vector directly.
    4. **Noise and projection.** Optional Gaussian noise of standard deviation
       ``latent_noise`` is added to every latent vector. The leaf latent vectors
       are then linearly projected into ``n_genes`` gene-space dimensions with a
       random loading matrix and passed through the ``distribution`` observation
       model (see :func:`_sample_counts`).

    The function returns a copy of ``tdata`` with:

    * leaf latent vectors in ``tdata.obsm[latent_key]`` (n_leaves × latent_dim),
      where ``latent_dim`` is inferred from the trajectory's latent vectors;
    * the expression matrix in ``tdata.layers[layer_added]`` (if given) or
      ``tdata.X``.

    Parameters
    ----------
    tdata
        The lineage :class:`td.TreeData` to annotate. Its tree must carry node
        times under ``depth_key``.
    trajectory
        The cell-state trajectory, as a :class:`td.TreeData` or
        :class:`networkx.DiGraph`. Node attributes:

          - ``depth_key`` : float — pseudotime of the state.
          - ``latent_key`` : np.ndarray of shape ``(latent_dim,)`` — the state's
            latent coordinates.

        Optional edge attribute ``prob_key`` : float — relative weight used when
        choosing among the children of a branch point.
    n_genes
        Number of genes to simulate in the observed expression matrix.
    distribution
        Observation model used to generate the expression matrix from the
        latent projection. ``"gaussian"`` (default) returns the continuous
        projection unchanged; ``"poisson"`` and ``"negative_binomial"`` sample
        integer counts (see :func:`_sample_counts`).
    library_size
        Expected total counts per cell, used to scale the per-cell rate for the
        count distributions. Ignored when ``distribution == "gaussian"``.
    dispersion
        Overdispersion (inverse size) of the negative binomial; the count
        variance is ``lam + dispersion * lam**2``. Ignored for the other
        distributions.
    latent_noise
        Standard deviation of Gaussian noise added to each cell's latent vector
        before projection into gene space.
    latent_key
        Node attribute on ``trajectory`` holding the latent vectors; also the
        key under which leaf latent vectors are written to ``tdata.obsm``.
    prob_key
        Edge attribute on ``trajectory`` used as the transition weight when
        choosing a child at a branch point. If ``None``, children are chosen
        uniformly at random.
    depth_key
        Node attribute holding times on both ``tdata`` and ``trajectory``.
    random_seed
        Seed for reproducibility.
    layer_added
        If specified, the expression matrix is stored in
        ``tdata.layers[layer_added]``. Otherwise it is stored in ``tdata.X``.

    Returns:
    -------
    tdata : :class:`td.TreeData`
        Copy of ``tdata`` with leaf latent vectors in ``obsm[latent_key]`` and
        the simulated expression matrix overlaid.
    """
    # Setup
    lineage_tdata = tdata.copy()
    rng = np.random.default_rng(random_seed)
    scaled_time_key = "scaled_time"
    factor_edge_key = "factor_edge"
    trajectory, _ = _get_digraph(trajectory, copy=True)
    lineage_tree, _ = _get_digraph(tdata, copy=True)
    factor_root = _get_root(trajectory)
    lineage_root = _get_root(lineage_tree)
    factor_topo = list(nx.topological_sort(trajectory))
    lineage_topo = list(nx.topological_sort(lineage_tree))

    # infer latent_dim from factor tree
    example = factor_topo[0]
    z_example = np.asarray(trajectory.nodes[example][latent_key], dtype=float)
    latent_dim = z_example.shape[0]

    # Rescale node times to [0, 1]
    for tree in [trajectory, lineage_tree]:
        _rescale_node_times(tree, time_key=depth_key, scaled_time_key=scaled_time_key)

    # Helper: choose a child of u in trajectory
    def choose_child(u):
        """Choose a child of u in trajectory using prob_key if provided."""
        children = list(trajectory.successors(u))
        if not children:
            return None
        if prob_key is None:
            return rng.choice(children)
        # Use edge attribute prob_key as weight, default 1.0 if missing
        weights = np.array(
            [trajectory[u][v].get(prob_key, 1.0) for v in children],
            dtype=float,
        )
        if weights.sum() <= 0:
            probs = np.ones_like(weights) / len(weights)
        else:
            probs = weights / weights.sum()
        return rng.choice(children, p=probs)

    # Helper: advance along the trajectory so that time t lies on an edge
    def advance_edge(edge, t: float):
        u, v = edge
        while True:
            t_v = float(trajectory.nodes[v][scaled_time_key])
            # if t is within [t_u, t_v] or we're clamped at a leaf
            if t <= t_v or u == v:
                return (u, v)
            # otherwise step forward from v to one of its children
            next_child = choose_child(v)
            if next_child is None:
                # at a leaf; clamp
                return (u, v)
            u, v = v, next_child

    # Assign a starting trajectory edge for the lineage root
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
        t_u = float(trajectory.nodes[u][scaled_time_key])
        t_v = float(trajectory.nodes[v][scaled_time_key])

        z_u = np.asarray(trajectory.nodes[u][latent_key], dtype=float)

        if u == v or t_v == t_u:
            z = z_u.copy()
        else:
            z_v = np.asarray(trajectory.nodes[v][latent_key], dtype=float)
            alpha = (t_node - t_u) / (t_v - t_u)
            alpha = np.clip(alpha, 0.0, 1.0)
            z = (1.0 - alpha) * z_u + alpha * z_v

        lineage_tree.nodes[node][latent_key] = z
        if factor_edge_key is not None:
            lineage_tree.nodes[node][factor_edge_key] = (u, v)
        embeddings.append(z)

    embeddings = np.vstack(embeddings)
    if latent_noise > 0.0:
        embeddings += rng.normal(loc=0.0, scale=latent_noise, size=embeddings.shape)
    columns = [f"{i}" for i in range(latent_dim)]
    embeddings = pd.DataFrame(embeddings, index=lineage_topo, columns=columns)
    # Restrict to observed cells (leaves) so the latent and expression matrices
    # are aligned with ``tdata.obs_names``.
    embeddings = embeddings.loc[lineage_tdata.obs_names]
    lineage_tdata.obsm[latent_key] = embeddings

    # Get gene expression via linear projection, then apply the observation model
    X = _expression_from_latent(embeddings.to_numpy(), latent_dim, n_genes, rng)
    X = _sample_counts(X, distribution, library_size, dispersion, rng)

    return _add_expression_to_tdata(lineage_tdata, X, layer_added)
