"""Cassiopeia-ILP solver: functional API and ILPSolver shim.

Infers the maximum-parsimony tree by solving for a Steiner Tree over an inferred
potential graph of evolutionary intermediates (Jones et al, Genome Biol 2020).
The ILP optimization is performed using Gurobi, which is an optional dependency.
"""

from __future__ import annotations

import datetime
import hashlib
import itertools
import logging
import time
import warnings
from typing import TYPE_CHECKING

import networkx as nx
import numpy as np
import pandas as pd

from cassiopeia import dissimilarity
from cassiopeia.data import utilities as data_utilities
from cassiopeia.dissimilarity._pairwise import _encode_integer_matrix
from cassiopeia.mixins import ILPSolverError, is_ambiguous_state, logger
from cassiopeia.solver import ilp_solver_utilities
from cassiopeia.utils import (
    _get_characters,
    _get_parameter,
    _node_name_generator,
    _set_tree,
    _transform_priors,
)

if TYPE_CHECKING:
    from treedata import TreeData

    from cassiopeia.data import CassiopeiaTree


# ── Potential graph inference ─────────────────────────────────────────────────


def _add_edge_weights(
    potential_graph: nx.DiGraph,
    weights: dict[int, dict[int, float]] | None = None,
    missing_state_indicator: int = -1,
) -> nx.DiGraph:
    """Annotate each edge with the (weighted) Hamming distance between its nodes."""
    weighted_graph = potential_graph.copy()
    for u, v in weighted_graph.edges():
        weighted_graph[u][v]["weight"] = dissimilarity.weighted_hamming(
            list(u), list(v), missing_state_indicator, weights
        )
    return weighted_graph


def _infer_potential_graph(
    character_matrix: pd.DataFrame,
    pid: str,
    lca_height: int,
    maximum_potential_graph_layer_size: int,
    weights: dict[int, dict[int, float]] | None = None,
    missing_state_indicator: int = -1,
) -> nx.DiGraph:
    """Infer a potential graph of evolutionary intermediates.

    Invokes ``ilp_solver_utilities.infer_potential_graph_cython`` (Cython), which
    operates on the integer character matrix and returns integer
    ``(parent_tuple, child_tuple)`` edges directly.
    """
    potential_graph_edges = ilp_solver_utilities.infer_potential_graph_cython(
        character_matrix.to_numpy(),
        pid,
        lca_height,
        maximum_potential_graph_layer_size,
        missing_state_indicator,
    )

    if len(potential_graph_edges) == 0:
        raise ILPSolverError(
            "Potential Graph could not be found with solver parameters. Try "
            "increasing `maximum_potential_graph_layer_size` or using another solver."
        )

    potential_graph = nx.DiGraph()
    potential_graph.add_edges_from(potential_graph_edges)

    return _add_edge_weights(potential_graph, weights, missing_state_indicator)


# ── Steiner tree ILP ──────────────────────────────────────────────────────────


def _generate_steiner_model(
    potential_graph: nx.DiGraph,
    root,
    targets: list,
):
    """Build a Gurobi mixed-integer model for the Steiner Tree problem."""
    try:
        import gurobipy
    except ModuleNotFoundError as err:
        raise ILPSolverError(
            "Gurobi not found. You must install Gurobi & gurobipy from source."
        ) from err

    source_flow = dict.fromkeys(potential_graph.nodes(), 0)

    if root not in potential_graph.nodes:
        raise ILPSolverError("Root node not in potential graph.")
    for t in targets:
        if t not in potential_graph.nodes:
            raise ILPSolverError("Target node not in potential graph.")

    # remove source from targets if it exists there
    targets = [t for t in targets if t != root]

    source_flow[root] = len(targets)
    for target in targets:
        source_flow[target] = -1

    model = gurobipy.Model("steiner")

    # add flow for edges
    edge_variables = {}
    for u, v in potential_graph.edges():
        edge_variables[u, v] = model.addVar(
            vtype=gurobipy.GRB.INTEGER,
            lb=0,
            ub=len(targets),
            name=f"edge_{u}_{v}",
        )

    # add edge-usage indicator variable
    edge_variables_binary = {}
    for u, v in potential_graph.edges():
        edge_variables_binary[u, v] = model.addVar(
            vtype=gurobipy.GRB.BINARY, name=f"edge_binary_{u}_{v}"
        )

    model.update()

    # check if edge is used
    for u, v in potential_graph.edges():
        model.addConstr(edge_variables_binary[u, v] >= (edge_variables[u, v] / len(targets)))

    # flow conservation constraints
    for v in potential_graph.nodes():
        model.addConstr(
            (
                gurobipy.quicksum(edge_variables[u, v] for u in potential_graph.predecessors(v))
                + source_flow[v]
            )
            == (gurobipy.quicksum(edge_variables[v, w] for w in potential_graph.successors(v)))
        )

    objective_expression = gurobipy.quicksum(
        edge_variables_binary[u, v] * potential_graph[u][v]["weight"]
        for u, v in potential_graph.edges()
    )
    model.setObjective(objective_expression, gurobipy.GRB.MINIMIZE)

    return model, edge_variables


def _solve_steiner_instance(
    model,
    edge_variables,
    potential_graph: nx.DiGraph,
    pid: str,
    logfile: str | None,
    *,
    mip_gap: float,
    seed: int | None,
    convergence_iteration_limit: int,
    convergence_time_limit: int,
) -> list[nx.DiGraph]:
    """Optimize the Steiner Tree model and return proposed solution subgraphs."""
    try:
        import gurobipy
    except ModuleNotFoundError as err:
        raise ILPSolverError(
            "Gurobi not found. You must install Gurobi & gurobipy from source."
        ) from err

    model.params.LogToConsole = 0

    model.params.THREADS = 1
    model.params.Presolve = 2
    model.params.MIPFocus = 1
    model.params.Cuts = 1
    model.params.Method = 4

    model.params.MIPGAP = mip_gap
    if logfile is not None:
        model.params.LogFile = logfile

    if seed is not None:
        model.params.Seed = seed

    if convergence_iteration_limit > 0:
        model.params.IterationLimit = convergence_iteration_limit

    if convergence_time_limit > 0:
        model.params.TimeLimit = convergence_time_limit

    start_time = time.time()

    model.optimize()

    # recover subgraphs
    solutions = []
    for i in range(model.SolCount):
        model.params.SolutionNumber = i
        subgraph = nx.DiGraph()
        value_for_edge = model.getAttr("xn", edge_variables)
        for u, v in potential_graph.edges():
            if value_for_edge[u, v] > 0:
                subgraph.add_edge(u, v, weight=potential_graph[u][v]["weight"])
        solutions.append(subgraph)

    end_time = time.time()

    execution_delta = datetime.timedelta(seconds=(end_time - start_time))
    days = execution_delta.days
    hours = execution_delta.seconds // 3600
    minutes = execution_delta.seconds // 60
    seconds = execution_delta.seconds % 60

    logger.info(
        f"(Process {pid}) Steiner tree solving tool {days} days, "
        f"{hours} hours, {minutes} minutes, and {seconds} seconds."
    )
    if model.status != gurobipy.GRB.status.OPTIMAL:
        logger.info(
            f"(Process {pid}) Warning: Steiner tree solving did not result in an optimal model."
        )

    return solutions


def _post_process_steiner_solution(
    solution: nx.DiGraph,
    root,
) -> nx.DiGraph:
    """Remove self-loops, spurious roots, and enforce a single parent per node."""
    processed_solution = solution.copy()
    for edge in nx.selfloop_edges(processed_solution):
        processed_solution.remove_edge(edge[0], edge[1])

    # remove spurious roots
    spurious_roots = [n for n in processed_solution if processed_solution.in_degree(n) == 0]
    while len(spurious_roots) > 1:
        for r in spurious_roots:
            if r != root:
                processed_solution.remove_node(r)
        spurious_roots = [n for n in processed_solution if processed_solution.in_degree(n) == 0]

    # impose that each node only has one parent
    non_tree_nodes = [n for n in processed_solution.nodes() if processed_solution.in_degree(n) > 1]
    for node in non_tree_nodes:
        parents = processed_solution.predecessors(node)
        parents = sorted(
            parents,
            key=lambda k: processed_solution[k][node]["weight"],
            reverse=True,
        )

        if len(parents) == 2 and (
            parents[1] in nx.ancestors(processed_solution, parents[0])
            or (parents[0] in nx.ancestors(processed_solution, parents[1]))
        ):
            if parents[1] in nx.ancestors(processed_solution, parents[0]):
                processed_solution.remove_edge(parents[1], node)
            else:
                processed_solution.remove_edge(parents[0], node)
        else:
            for parent in parents[1:]:
                processed_solution.remove_edge(parent, node)

    return processed_solution


def _append_sample_names_and_remove_spurious_leaves(
    solution: nx.DiGraph, character_matrix: pd.DataFrame
) -> nx.DiGraph:
    """Attach sample names to their deepest matching state node and prune.

    Adds each sample as a leaf under the deepest node whose character state it
    matches (each sample added once), then removes any extant nodes that do not
    correspond to samples, pruning the resulting spurious lineages.
    """
    root = [n for n in solution if solution.in_degree(n) == 0][0]

    sample_lookup = character_matrix.apply(lambda x: tuple(x.values), axis=1)

    states_added = []
    for node in nx.dfs_postorder_nodes(solution, source=root):
        if node in states_added:
            continue

        samples = sample_lookup[sample_lookup == node].index
        if len(samples) > 0:
            solution.add_edges_from([(node, sample) for sample in samples])
            states_added.append(node)

    # remove extant lineages that don't correspond to leaves
    leaves = [n for n in solution if solution.out_degree(n) == 0]
    for leaf in leaves:
        if leaf not in character_matrix.index:
            curr_parent = list(solution.predecessors(leaf))[0]
            solution.remove_node(leaf)
            while len(list(solution.successors(curr_parent))) < 1 and curr_parent != root:
                next_parent = list(solution.predecessors(curr_parent))[0]
                solution.remove_node(curr_parent)
                curr_parent = next_parent

    return solution


def _finalize(
    tdata: CassiopeiaTree | TreeData,
    solution: nx.DiGraph,
    character_matrix: pd.DataFrame,
    characters_key: str | None,
    key_added: str,
) -> None:
    """Rename internal (tuple) nodes to unique names and store the tree."""
    node_name_generator = _node_name_generator()
    sample_set = set(character_matrix.index)
    rename = {}
    seen = set()
    for node in solution.nodes():
        if node in sample_set:
            continue
        new_name = next(node_name_generator)
        while new_name in seen:
            new_name = next(node_name_generator)
        rename[node] = new_name
        seen.add(new_name)
    tree = nx.relabel_nodes(solution, rename)

    _set_tree(tdata, tree, key_added)


@logger.namespaced("ILPSolver")
def ilp(
    tdata: TreeData,
    convergence_time_limit: int = 12600,
    convergence_iteration_limit: int = 0,
    maximum_potential_graph_layer_size: int = 10000,
    maximum_potential_graph_lca_distance: int | None = None,
    weighted: bool = False,
    seed: int | None = None,
    mip_gap: float = 0.01,
    logfile: str | None = None,
    characters_key: str | None = None,
    key_added: str = "ilp",
    prior_transformation: str = "negative_log",
    missing_state: int | str | None = None,
    unmodified_state: int | str | None = None,
    priors: dict[int, dict[int, float]] | None = None,
    copy: bool = False,
) -> TreeData | None:
    """Reconstruct a tree with Cassiopeia-ILP (maximum parsimony).

    Infers a potential graph of evolutionary intermediates and solves a Steiner
    Tree over it with Gurobi. The character matrix is read from ``tdata.obsm``
    and the result is stored as an ``nx.DiGraph`` in ``tdata.obst[key_added]``.

    Args:
        tdata: TreeData to operate on.
        convergence_time_limit: ILP convergence time limit (seconds). Ignored if 0.
        convergence_iteration_limit: ILP iteration limit. Ignored if 0.
        maximum_potential_graph_layer_size: Maximum potential-graph layer size.
        maximum_potential_graph_lca_distance: Maximum LCA height to add to the
            potential graph. If ``None`` or 0, the maximum pairwise distance is used.
        weighted: Weight potential-graph edges by mutation negative-log-likelihood.
            Requires priors.
        seed: Random seed for ILP optimization.
        mip_gap: Objective gap for the MILP.
        logfile: File to log progress to, or ``None``.
        characters_key: Key in ``tdata.obsm`` for the character matrix
            (default ``'characters'``).
        key_added: Key in ``tdata.obst`` for the resulting tree.
        prior_transformation: Transformation applied to priors to form weights.
        missing_state: Missing-state value (read from ``tdata.uns`` if ``None``).
        unmodified_state: Unmodified/uncut state value (read from ``tdata.uns``
            if ``None``).
        priors: Priors for character states, as a dict mapping character index
            to dicts mapping state to prior probability (read from ``tdata.uns``
            if ``None``).
        copy: If ``True``, return a copy of *tdata*; otherwise modify in-place
            and return ``None``.

    Returns:
        A modified copy of *tdata* if ``copy=True``, else ``None``.

    Raises:
        ILPSolverError: On missing character matrix, ambiguous states, or
            ``weighted=True`` without priors.
    """
    tdata = tdata.copy() if copy else tdata
    character_matrix = _get_characters(tdata, characters_key).copy()
    missing_state_indicator = _get_parameter(tdata, "missing_state", value=missing_state)
    unmodified_state = _get_parameter(tdata, "unmodified_state", value=unmodified_state)
    priors = _get_parameter(tdata, "priors", value=priors)

    if weighted and not priors:
        raise ILPSolverError("Specify prior probabilities for weighted analysis.")

    # configure logger
    file_handler = None
    if logfile is not None:
        file_handler = logging.FileHandler(logfile)
        file_handler.setLevel(logging.INFO)
        logger.addHandler(file_handler)
    logger.ch.setLevel(logging.getLogger().level)

    logger.info("Solving tree with the following parameters.")
    logger.info(f"Convergence time limit: {convergence_time_limit}")
    logger.info(f"Convergence iteration limit: {convergence_iteration_limit}")
    logger.info(f"Max potential graph layer size: {maximum_potential_graph_layer_size}")
    logger.info(f"Max potential graph lca distance: {maximum_potential_graph_lca_distance}")
    logger.info(f"MIP gap: {mip_gap}")

    if any(is_ambiguous_state(state) for state in character_matrix.values.flatten()):
        raise ILPSolverError("Solver does not support ambiguous states.")

    # The potential-graph / Steiner-tree machinery operates on integer states, so
    # encode string/categorical matrices (unmodified -> 0, missing -> -1, other
    # states -> distinct positive integers).
    character_matrix, missing_state_indicator = _encode_integer_matrix(
        character_matrix, missing_state_indicator, unmodified_state
    )

    unique_character_matrix = character_matrix.drop_duplicates()

    weights = None
    if priors:
        weights = _transform_priors(priors, prior_transformation)

    # find the root of the tree & generate process ID
    root = tuple(
        data_utilities.get_lca_characters(
            unique_character_matrix.values.tolist(), missing_state_indicator
        )
    )
    logger.info(f"Phylogenetic root: {root}")
    pid = hashlib.md5("|".join([str(r) for r in root]).encode("utf-8")).hexdigest()

    targets = [tuple(t) for t in unique_character_matrix.values.tolist()]

    if unique_character_matrix.shape[0] == 1:
        optimal_solution = nx.DiGraph()
        optimal_solution.add_node(root)
        optimal_solution = _append_sample_names_and_remove_spurious_leaves(
            optimal_solution, character_matrix
        )
        _finalize(tdata, optimal_solution, character_matrix, characters_key, key_added)
        if file_handler is not None:
            logger.removeHandler(file_handler)
            file_handler.close()
        return tdata if copy else None

    # determine the maximum LCA distance to consider
    if (maximum_potential_graph_lca_distance is not None) and (
        maximum_potential_graph_lca_distance > 0
    ):
        max_lca_distance = maximum_potential_graph_lca_distance
    else:
        max_lca_distance = 0
        lca_distances = [
            dissimilarity.nonmissing_hamming(
                root,
                np.array(u),
                missing_state_indicator=missing_state_indicator,
            )
            for u in targets
        ]
        for i, j in itertools.combinations(range(len(lca_distances)), 2):
            max_lca_distance = max(max_lca_distance, lca_distances[i] + lca_distances[j] + 1)

    # infer the potential graph
    potential_graph = _infer_potential_graph(
        unique_character_matrix,
        pid,
        max_lca_distance,
        maximum_potential_graph_layer_size,
        weights,
        missing_state_indicator,
    )

    # relabel nodes to integers for the ILP, then solve
    nodes = list(potential_graph.nodes())
    encoder = dict(zip(nodes, list(range(len(nodes))), strict=False))
    decoder = {v: k for k, v in encoder.items()}

    _potential_graph = nx.relabel_nodes(potential_graph, encoder)
    _targets = [encoder[x] for x in targets]
    _root = encoder[root]

    model, edge_variables = _generate_steiner_model(_potential_graph, _root, _targets)
    proposed_solutions = _solve_steiner_instance(
        model,
        edge_variables,
        _potential_graph,
        pid,
        logfile,
        mip_gap=mip_gap,
        seed=seed,
        convergence_iteration_limit=convergence_iteration_limit,
        convergence_time_limit=convergence_time_limit,
    )

    optimal_solution = proposed_solutions[0]
    optimal_solution = nx.relabel_nodes(optimal_solution, decoder)
    optimal_solution = _post_process_steiner_solution(optimal_solution, root)
    optimal_solution = _append_sample_names_and_remove_spurious_leaves(
        optimal_solution, character_matrix
    )

    _finalize(tdata, optimal_solution, character_matrix, characters_key, key_added)

    if file_handler is not None:
        logger.removeHandler(file_handler)
        file_handler.close()

    return tdata if copy else None


# ── Backward-compat shim ─────────────────────────────────────────────────────


class ILPSolver:
    """Cassiopeia-ILP maximum-parsimony solver.

    Thin shim around :func:`cassiopeia.solver.ilp` for backward compatibility.
    For new code, prefer calling :func:`cassiopeia.solver.ilp` directly.
    """

    def __init__(
        self,
        convergence_time_limit: int = 12600,
        convergence_iteration_limit: int = 0,
        maximum_potential_graph_layer_size: int = 10000,
        maximum_potential_graph_lca_distance: int | None = None,
        weighted: bool = False,
        seed: int | None = None,
        mip_gap: float = 0.01,
        prior_transformation: str = "negative_log",
    ):
        warnings.warn(
            "ILPSolver is deprecated and will be removed in a future release. "
            "Use cassiopeia.solver.ilp() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.convergence_time_limit = convergence_time_limit
        self.convergence_iteration_limit = convergence_iteration_limit
        self.maximum_potential_graph_layer_size = maximum_potential_graph_layer_size
        self.maximum_potential_graph_lca_distance = maximum_potential_graph_lca_distance
        self.weighted = weighted
        self.seed = seed
        self.mip_gap = mip_gap
        self.prior_transformation = prior_transformation

    def solve(
        self,
        cassiopeia_tree: CassiopeiaTree,
        layer: str | None = None,
        collapse_mutationless_edges: bool = False,
        logfile: str = "stdout.log",
    ) -> None:
        """Infer a tree with Cassiopeia-ILP in-place.

        Args:
            cassiopeia_tree: CassiopeiaTree to solve in-place.
            layer: Character matrix layer to use.
            collapse_mutationless_edges: Collapse edges with no inferred
                mutations after solving.
            logfile: Location to log progress.
        """
        ilp(
            cassiopeia_tree,
            characters_key=layer,
            convergence_time_limit=self.convergence_time_limit,
            convergence_iteration_limit=self.convergence_iteration_limit,
            maximum_potential_graph_layer_size=self.maximum_potential_graph_layer_size,
            maximum_potential_graph_lca_distance=self.maximum_potential_graph_lca_distance,
            weighted=self.weighted,
            seed=self.seed,
            mip_gap=self.mip_gap,
            prior_transformation=self.prior_transformation,
            logfile=logfile,
        )
        if collapse_mutationless_edges:
            cassiopeia_tree.collapse_mutationless_edges(infer_ancestral_characters=True)
