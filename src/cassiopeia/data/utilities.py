"""General utilities for the datasets encountered in Cassiopeia."""

import collections
import copy
from collections.abc import Callable

import networkx as nx
import numba
import numpy as np
import pandas as pd
from treedata import TreeData

from cassiopeia.data import CassiopeiaTree
from cassiopeia.mixins import CassiopeiaTreeWarning, is_ambiguous_state  # noqa: F401
from cassiopeia.mixins.errors import CassiopeiaError, CassiopeiaTreeError
from cassiopeia.preprocess import utilities as preprocessing_utilities


def get_lca_characters(
    vecs: list[list[int] | list[tuple[int, ...]]],
    missing_state_indicator: int,
) -> list[int]:
    """Builds the character vector of the LCA of a list of character vectors, obeying Camin-Sokal Parsimony.

    For each index in the reconstructed vector, imputes the non-missing
    character if only one of the constituent vectors has a missing value at that
    index, and imputes missing value if all have a missing value at that index.

    Importantly, this method will infer ancestral characters for an ambiguous
    state. If the intersection between two states (even ambiguous) is non-zero
    and not the missing state, and has length exactly 1, we assign the ancestral
    state this value. Else, if the intersection length is greater than 1, the
    value '0' is assigned.

    Args:
        vecs: A list of character vectors to generate an LCA for
        missing_state_indicator: The character representing missing values

    Returns:
            A list representing the character vector of the LCA

    """
    k = len(vecs[0])
    for i in vecs:
        assert len(i) == k
    lca_vec = [0] * len(vecs[0])
    for i in range(k):
        if np.all(np.array([vec[i] for vec in vecs], dtype=object) == missing_state_indicator):
            lca_vec[i] = missing_state_indicator
        else:
            all_states = [vec[i] for vec in vecs if vec[i] != missing_state_indicator]

            # this check is specifically if all_states consists of a single
            # ambiguous state.
            if len(list(set(all_states))) == 1:
                state = all_states[0]
                # lca_vec[i] = state
                if is_ambiguous_state(state) and len(state) == 1:
                    lca_vec[i] = state[0]
                else:
                    lca_vec[i] = all_states[0]
            else:
                all_ambiguous = np.all([is_ambiguous_state(s) for s in all_states])
                chars = set.intersection(
                    *map(
                        set,
                        [state if is_ambiguous_state(state) else [state] for state in all_states],
                    )
                )
                if len(chars) == 1:
                    lca_vec[i] = list(chars)[0]
                if all_ambiguous:
                    # if we only have ambiguous states, we set the LCA state
                    # to be the intersection.
                    lca_vec[i] = tuple(chars)
    return lca_vec


def newick_to_networkx(newick_string: str) -> nx.DiGraph:
    """Converts a newick string to a networkx DiGraph.

    Parses a newick string (supporting named internal nodes and branch lengths)
    into a directed tree.  Unnamed internal nodes are assigned unique names of
    the form ``cassiopeia_internal_node{i}``; branches without an explicit length
    default to a length of ``1.0``.

    Args:
        newick_string: A newick string.

    Returns:
            A networkx DiGraph.
    """
    g = nx.DiGraph()
    s = newick_string.strip()
    if s.endswith(";"):
        s = s[:-1]

    pos = 0  # current parse position
    internal_node_iter = 0

    def _parse_label() -> str:
        nonlocal pos
        start = pos
        while pos < len(s) and s[pos] not in ",():":
            pos += 1
        return s[start:pos].strip()

    def _parse_clade() -> tuple[str, float | None]:
        """Parse a clade at ``pos`` and return its ``(name, branch_length)``."""
        nonlocal pos, internal_node_iter
        children = []
        if pos < len(s) and s[pos] == "(":
            pos += 1  # consume "("
            while True:
                children.append(_parse_clade())
                if pos < len(s) and s[pos] == ",":
                    pos += 1
                    continue
                if pos < len(s) and s[pos] == ")":
                    pos += 1
                break

        name = _parse_label()
        length = None
        if pos < len(s) and s[pos] == ":":
            pos += 1
            length = float(_parse_label())

        if not name and children:
            name = f"cassiopeia_internal_node{internal_node_iter}"
            internal_node_iter += 1

        for child_name, child_length in children:
            g.add_edge(name, child_name, length=1.0 if child_length is None else child_length)

        return name, length

    _parse_clade()
    return g


def ete3_to_networkx(tree: "ete3.Tree") -> nx.DiGraph:  # noqa: F821
    """Converts an ete3 Tree to a networkx DiGraph.

    ``ete3`` is an optional dependency; this helper only operates on an
    already-constructed ete3 ``Tree`` passed by the caller and does not import
    ete3 itself.

    Args:
        tree: an ete3 Tree object

    Returns:
            a networkx DiGraph
    """
    g = nx.DiGraph()
    internal_node_iter = 0
    for n in tree.traverse():
        if n.name == "":
            n.name = f"cassiopeia_internal_node{internal_node_iter}"
            internal_node_iter += 1

        if n.is_root():
            continue

        g.add_edge(n.up.name, n.name, length=n.dist)

    return g


def to_newick(
    tree: nx.DiGraph,
    record_branch_lengths: bool = False,
    record_node_names: bool = False,
) -> str:
    """Converts a networkx graph to a newick string.

    Args:
        tree: A networkx tree
        record_branch_lengths: Whether to record branch lengths on the tree in
            the newick string
        record_node_names: Whether to record internal node names on the tree in
            the newick string

    Returns:
            A newick string representing the topology of the tree
    """

    def _to_newick_str(g, node):
        is_leaf = g.out_degree(node) == 0
        weight_string = ""

        if record_branch_lengths and g.in_degree(node) > 0:
            parent = list(g.predecessors(node))[0]
            weight_string = ":" + str(g[parent][node]["length"])

        _name = str(node)

        name_string = ""
        if record_node_names:
            name_string = f"{_name}"

        return (
            f"{_name}" + weight_string
            if is_leaf
            else (
                "("
                + ",".join(_to_newick_str(g, child) for child in g.successors(node))
                + ")"
                + name_string
                + weight_string
            )
        )

    root = [node for node in tree if tree.in_degree(node) == 0][0]
    return _to_newick_str(tree, root) + ";"


def sample_bootstrap_character_matrices(
    character_matrix: pd.DataFrame,
    prior_probabilities: dict[int, dict[int, float]] | None = None,
    num_bootstraps: int = 10,
    random_state: np.random.RandomState | None = None,
) -> list[tuple[pd.DataFrame, dict[int, dict[int, float]]]]:
    """Generates bootstrapped character matrices from a character matrix.

    Ingests a character matrix and randomly creates bootstrap samples by
    sampling characters with replacement. Each bootstrapped character matrix,
    then, retains the same number of characters but some will be repeated and
    some will be ignored. If a prior proability dictionary is also passed in,
    then a new priors dictionary will be created for each bootstrapped character
    matrix.

    Args:
        character_matrix: Character matrix
        prior_probabilities: Probabilities of each (character, state) pair.
        num_bootstraps: Number of bootstrap samples to create.
        random_state: A numpy random state to from which to draw samples

    Returns:
            A list of bootstrap samples in the form
            (bootstrap_character_matrix, bootstrap_priors).
    """
    bootstrap_samples = []
    M = character_matrix.shape[1]
    for _ in range(num_bootstraps):
        if random_state:
            sampled_cut_sites = random_state.choice(M, M, replace=True)
        else:
            sampled_cut_sites = np.random.choice(M, M, replace=True)

        bootstrapped_character_matrix = character_matrix.iloc[:, sampled_cut_sites]
        bootstrapped_character_matrix.columns = [f"random_character{i}" for i in range(M)]

        new_priors = {}
        if prior_probabilities:
            for i, cut_site in zip(range(M), sampled_cut_sites, strict=False):
                new_priors[i] = prior_probabilities[cut_site]

        bootstrap_samples.append((bootstrapped_character_matrix, new_priors))

    return bootstrap_samples


def sample_bootstrap_allele_tables(
    allele_table: pd.DataFrame,
    indel_priors: pd.DataFrame | None = None,
    num_bootstraps: int = 10,
    random_state: np.random.RandomState | None = None,
    cut_sites: list[str] | None = None,
) -> list[
    tuple[
        pd.DataFrame,
        dict[int, dict[int, float]],
        dict[int, dict[int, str]],
        list[str],
    ]
]:
    """Generates bootstrap character matrices from an allele table.

    This function will take in an allele table, generated with the Cassiopeia
    preprocess pipeline and produce several bootstrap character matrices with
    respect to intBCs rather than individual cut-sites as in
    `sample_bootstrap_character_matrices`. This is useful because oftentimes
    there are dependencies between cut-sites on the same intBC TargetSite.

    Args:
        allele_table: AlleleTable from the Cassiopeia preprocessing pipeline
        indel_priors: A dataframe mapping indel identities to prior
            probabilities
        num_bootstraps: number of bootstrap samples to create
        random_state: A numpy random state for reproducibility.
        cut_sites: Columns in the AlleleTable to treat as cut sites. If None,
            we assume that the cut-sites are denoted by columns of the form
            "r{int}" (e.g. "r1")

    Returns:
            A list of bootstrap samples in the form of tuples
            (bootstrapped character matrix, prior dictionary,
            state to indel mapping, bootstrapped intBC set)
    """
    if cut_sites is None:
        cut_sites = preprocessing_utilities.get_default_cut_site_columns(allele_table)

    lineage_profile = preprocessing_utilities.convert_alleletable_to_lineage_profile(
        allele_table, cut_sites
    )

    intbcs = allele_table["intBC"].unique()
    M = len(intbcs)

    bootstrap_samples = []

    for _ in range(num_bootstraps):
        if random_state:
            sampled_intbcs = random_state.choice(intbcs, M, replace=True)
        else:
            sampled_intbcs = np.random.choice(intbcs, M, replace=True)

        bootstrap_intbcs = sum(
            [[intbc + f"_{cut_site}" for cut_site in cut_sites] for intbc in sampled_intbcs],
            [],
        )
        b_sample = lineage_profile[bootstrap_intbcs]

        (
            bootstrapped_character_matrix,
            priors,
            state_to_indel,
        ) = preprocessing_utilities.convert_lineage_profile_to_character_matrix(
            b_sample, indel_priors=indel_priors
        )

        bootstrap_samples.append(
            (
                bootstrapped_character_matrix,
                priors,
                state_to_indel,
                bootstrap_intbcs,
            )
        )

    return bootstrap_samples


def resolve_most_abundant(state: tuple[int, ...]) -> int:
    """Resolve an ambiguous character by selecting the most abundant.

    This function is designed to be used with
    :func:`CassiopeiaTree.resolve_ambiguous_characters`. It resolves an ambiguous
    character, represented as a tuple of integers, by selecting the most abundant,
    where ties are resolved randomly.

    Args:
        state: Ambiguous state as a tuple of integers

    Returns:
            Selected state as a single integer
    """
    most_common = collections.Counter(state).most_common()
    return np.random.choice([state for state, count in most_common if count == most_common[0][1]])


def compute_phylogenetic_weight_matrix(
    tree: CassiopeiaTree,
    inverse: bool = False,
    inverse_fn: Callable[[int | float], float] = lambda x: 1 / x,
) -> pd.DataFrame:
    """Computes the phylogenetic weight matrix.

    Computes the distances between all leaves in a tree. The user has the option
    to return the inverse matrix, (i.e., transform distances to proximities) and
    specify an appropriate inverse function.

    This function computes the phylogenetic weight matrix in O(n^2 logn) time.

    An NxN weight matrix is returned.

    Args:
        tree: CassiopeiaTree
        inverse: Convert distances to proximities
        inverse_fn: Inverse function (default = 1 / x)

    Returns:
            An NxN phylogenetic weight matrix
    """
    N = tree.n_cell
    W = pd.DataFrame(np.zeros((N, N)), index=tree.leaves, columns=tree.leaves)

    for leaf1 in tree.leaves:
        distances = tree.get_distances(leaf1, leaves_only=True)
        for leaf2, _d in distances.items():
            if inverse:
                _d = inverse_fn(_d) if _d > 0 else np.inf

            W.loc[leaf1, leaf2] = W.loc[leaf2, leaf1] = _d

    np.fill_diagonal(W.values, 0)

    return W


@numba.jit(nopython=True)
def net_relatedness_index(
    dissimilarity_map: np.array, indices_1: np.array, indices_2: np.array
) -> float:
    """Computes the net relatedness index between indices.

    Using the dissimilarity map specified and the indices of samples, compute
    the net relatedness index, defined as:

    sum(distances over i,j in indices_1,indices_2) / (|indices_1| x |indices_2|)

    Args:
        dissimilarity_map: Dissimilarity map between all samples.
        indices_1: Indices corresponding to the first group.
        indices_2: Indices corresponding to the second group.

    Returns:
            The Net Relatedness Index (NRI)
    """
    nri = 0
    for i in indices_1:
        for j in indices_2:
            nri += dissimilarity_map[i, j]

    return nri / (len(indices_1) * len(indices_2))


def compute_inter_cluster_distances(
    tree: CassiopeiaTree,
    meta_item: str | None = None,
    meta_data: pd.DataFrame | None = None,
    dissimilarity_map: pd.DataFrame | None = None,
    distance_function: Callable = net_relatedness_index,
    **kwargs,
) -> pd.DataFrame:
    """Computes mean distance between clusters.

    Compute the mean distance between categories in a categorical variable. By
    default, the phylogenetic weight matrix will be computed and used for this
    distance calculation, but a user can optionally provide a dissimilarity
    map instead.

    This function performs the computation in O(K^2)*O(distance_function) time
    for a variable with K categories.

    Args:
        tree: CassiopeiaTree
        meta_item: Column in the cell meta data of the tree. If `meta_data` is
            specified, this is ignored.
        meta_data: Meta data to use for this calculation. This argument takes
            priority over meta_item.
        dissimilarity_map: Dissimilarity map to use for distances. If this is
            specified, the phylogenetic weight matrix is not computed.
        distance_function: Function to compute distance between two clusters.
        **kwargs: Arguments to pass to the distance function.

    Returns:
            A K x K distance matrix.
    """
    meta_data = tree.cell_meta[meta_item] if (meta_data is None) else meta_data

    # ensure that the meta data is categorical
    if not pd.api.types.is_string_dtype(meta_data):
        raise CassiopeiaError("Meta data must be categorical or a string.")

    D = (
        compute_phylogenetic_weight_matrix(tree)
        if (dissimilarity_map is None)
        else dissimilarity_map
    )

    unique_states = meta_data.unique()
    K = len(unique_states)
    inter_cluster_distances = pd.DataFrame(
        np.zeros((K, K)), index=unique_states, columns=unique_states
    )

    # align distance matrix and meta_data
    D = D.loc[meta_data.index.values, meta_data.index.values]

    for state1 in unique_states:
        indices_1 = np.where(np.array(meta_data) == state1)[0]
        for state2 in unique_states:
            indices_2 = np.where(np.array(meta_data) == state2)[0]

            distance = distance_function(D.values, indices_1, indices_2, **kwargs)
            inter_cluster_distances.loc[state1, state2] = distance

    return inter_cluster_distances


def cassiopeia_to_treedata(
    cassiopeia_tree: CassiopeiaTree,
    tree_key: str = "tree",
    characters_key: str = "characters",
    dissimilarity_map_key: str | None = "distances",
    preserve_layers: bool = True,
    preserve_metadata: bool = True,
) -> TreeData:
    """Convert a CassiopeiaTree object to TreeData format.

    Args:
    cassiopeia_tree: CassiopeiaTree
        Source CassiopeiaTree object to convert
    tree_key: str, default "tree"
        Key name for the tree in TreeData.obst
    characters_key: str, default "characters"
        Key name for character matrix in TreeData.obsm
    dissimilarity_map_key: str, default "distances"
        Key name for dissimilarity map in TreeData.obsp
    preserve_layers: bool, default True
        Whether to preserve character matrix layers in obsm
    preserve_metadata: bool, default True
        Whether to preserve cell and character metadata

    Returns:
    TreeData
        Converted TreeData object with:
        - X = None
        - obsm[characters_key] = character matrix (if present)
        - obst[tree_key] = tree topology (if present)
        - obsp[dissimilarity_map_key] = dissimilarity map (if present)
    """
    # Extract character matrix into obsm
    obsm = {}
    if cassiopeia_tree.character_matrix is not None:
        obsm[characters_key] = cassiopeia_tree.character_matrix
        obs = pd.DataFrame(index=cassiopeia_tree.character_matrix.index)

    # Extract tree topology into obst
    obst = {}
    try:
        tree_topology = cassiopeia_tree.get_tree_topology()
        if tree_topology is not None:
            obst[tree_key] = tree_topology
            obs = pd.DataFrame(index=cassiopeia_tree.leaves)
    except CassiopeiaTreeError as err:
        if cassiopeia_tree.character_matrix is None:
            raise CassiopeiaError(
                "CassiopeiaTree must have either a character matrix or a tree to convert to TreeData."
            ) from err

    # Extract observation metadata (obs)
    if preserve_metadata and cassiopeia_tree.cell_meta is not None:
        obs = cassiopeia_tree.cell_meta.copy()

    # Extract character matrix layers into obsm
    if preserve_layers and hasattr(cassiopeia_tree, "layers"):
        for layer_name, layer_data in cassiopeia_tree.layers.items():
            if layer_data is not None:
                obsm[f"layer_{layer_name}"] = layer_data

    # Store dissimilarity map if it exists
    obsp = {}
    dissim_map = cassiopeia_tree.get_dissimilarity_map()
    if dissim_map is not None:
        obsp[dissimilarity_map_key] = dissim_map

    # Extract unstructured annotations (uns)
    uns = {}
    name_mapping = {"missing_state_indicator": "missing_state", "root_sample_name": "root_name"}
    if preserve_metadata:
        # Store CassiopeiaTree-specific data in uns
        for key, value in cassiopeia_tree.parameters.items():
            uns[key] = copy.deepcopy(value)
        for key in ["priors", "missing_state_indicator", "root_sample_name", "character_meta"]:
            if hasattr(cassiopeia_tree, key):
                value = getattr(cassiopeia_tree, key)
                if value is not None:
                    if key in name_mapping:
                        uns[name_mapping[key]] = copy.deepcopy(value)
                    else:
                        uns[key] = copy.deepcopy(value)
        uns["converted_from"] = "CassiopeiaTree"

    # Create TreeData object
    treedata_obj = TreeData(
        X=None,
        obs=obs,
        obst=obst if obst else None,
        obsm=obsm if obsm else None,
        obsp=obsp if obsp else None,
        uns=uns if uns else None,
        label="tree",
    )

    return treedata_obj
