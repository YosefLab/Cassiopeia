"""Functional tree topology simulators for Cassiopeia."""

import warnings
from collections.abc import Callable, Generator
from queue import PriorityQueue, Queue

import networkx as nx
import numpy as np
import treedata as td

from cassiopeia.mixins import TreeSimulatorError
from cassiopeia.utils import _get_leaf_data
from cassiopeia.utils import collapse_unifurcations as _collapse_unifurcations


def complete_binary(
    num_cells: int | None = None,
    depth: int | None = None,
    key_added: str = "simulated",
) -> td.TreeData:
    """Simulate a complete binary tree.

    Uses :func:`nx.balanced_tree` to generate a perfectly balanced binary tree.
    Exactly one of ``num_cells`` or ``depth`` must be provided. All branches
    have equal length normalized by tree height (height = 1).

    Args:
        num_cells: Number of leaf cells. Must be a power of 2.
        depth: Depth of the tree. Number of cells will be ``2^depth``.
        key_added: Key under which the tree is stored in ``obst``.

    Returns:
        A TreeData with the simulated tree in ``obst[key_added]``. Each node
        has ``"time"`` (0 to 1, normalized) and ``"depth"`` (int) attributes.

    Raises:
        TreeSimulatorError: If neither or both of ``num_cells``/``depth`` are
            given, if ``num_cells`` is not a power of 2, or if depth <= 0.
    """
    if (num_cells is None) == (depth is None):
        raise TreeSimulatorError("One of `num_cells` or `depth` must be provided.")
    if num_cells is not None:
        log2_num_cells = np.log2(num_cells)
        if log2_num_cells != int(log2_num_cells):
            raise TreeSimulatorError("`num_cells` must be a power of 2.")
        depth = int(log2_num_cells)
    if depth <= 0:
        raise TreeSimulatorError("`depth` must be greater than 0.")

    def _name_gen() -> Generator[str, None, None]:
        i = 1
        while True:
            yield str(i)
            i += 1

    names = _name_gen()
    tree = nx.balanced_tree(2, depth, create_using=nx.DiGraph)
    tree.add_edge("root", 0)
    nx.relabel_nodes(tree, {node: next(names) for node in tree.nodes if node != "root"}, copy=False)
    depths = nx.single_source_shortest_path_length(tree, "root")
    nx.set_node_attributes(tree, depths, "depth")
    max_depth = max(depths.values())
    times = {node: d / max_depth for node, d in depths.items()}
    nx.set_node_attributes(tree, times, "time")

    return td.TreeData(obst={key_added: tree})


def birth_death_process(
    birth_waiting_distribution: Callable[[float], float] = lambda scale: np.random.lognormal(
        mean=np.log(scale), sigma=0.5
    ),
    initial_birth_scale: float = 1.0,
    death_waiting_distribution: Callable[[], float] = lambda: np.inf,
    mutation_distribution: Callable[[], int] | None = None,
    fitness_distribution: Callable[[], float] | None = None,
    fitness_base: float = np.e,
    num_extant: int | None = None,
    experiment_time: float | None = None,
    collapse_unifurcations: bool = True,
    random_seed: int | None = None,
    initial_tree: nx.DiGraph | None = None,
    key_added: str = "simulated",
) -> td.TreeData:
    """Simulate a phylogenetic tree via a forward birth-death process with fitness.

    Starting from an initial lineage (or from the leaves of ``initial_tree``),
    births represent lineage branching and deaths represent cessation. Branch
    lengths represent lifetimes. Fitness mutations can alter per-lineage birth
    rates.

    At least one stopping condition (``num_extant`` or ``experiment_time``) must
    be provided. Both may be provided, and the simulation stops at whichever is
    reached first.

    Args:
        birth_waiting_distribution: Samples birth waiting times; takes a scale
            parameter (float) as input.
        initial_birth_scale: Initial scale parameter for birth distribution.
        death_waiting_distribution: Samples death waiting times (no args).
            Defaults to no death (returns ``inf``).
        mutation_distribution: Samples number of fitness mutations at division.
            If ``None``, no fitness mutations occur.
        fitness_distribution: Samples the exponent for each fitness mutation.
            Required when ``mutation_distribution`` is provided.
        fitness_base: Base raised by the fitness exponent to compute the
            multiplicative fitness coefficient. Default is ``e``.
        num_extant: Stop when this many lineages exist simultaneously.
        experiment_time: Stop when total elapsed time reaches this value.
        collapse_unifurcations: Whether to collapse unifurcations after pruning.
        random_seed: NumPy random seed for reproducibility.
        initial_tree: Optional ``nx.DiGraph`` from which to resume simulation.
            Leaf nodes of this tree become the starting lineages. Nodes should
            have ``"birth_scale"`` and ``"time"`` attributes (defaults to
            ``initial_birth_scale`` and 0 if absent).
        key_added: Key under which the result tree is stored in ``obst``.

    Returns:
        A TreeData with the simulated tree in ``obst[key_added]``. Each node
        has ``"time"`` (cumulative age) and ``"birth_scale"`` attributes.

    Raises:
        TreeSimulatorError: For invalid parameters or if all lineages die before
            a stopping condition is reached.
    """
    if num_extant is None and experiment_time is None:
        raise TreeSimulatorError("Please specify at least one stopping condition")
    if mutation_distribution is not None and fitness_distribution is None:
        raise TreeSimulatorError("Please specify a fitness strength distribution")
    if num_extant is not None and num_extant <= 0:
        raise TreeSimulatorError("Please specify number of extant lineages greater than 0")
    if num_extant is not None and type(num_extant) is not int:
        raise TreeSimulatorError("Please specify an integer number of extant tips")
    if experiment_time is not None and experiment_time <= 0:
        raise TreeSimulatorError("Please specify an experiment time greater than 0")

    starting_index = 0
    if initial_tree is not None:
        leaves = [n for n in initial_tree if initial_tree.out_degree(n) == 0]
        starting_index = max(int(l) for l in leaves) + 1

    def _name_gen(start: int = 0) -> Generator[str, None, None]:
        i = start
        while True:
            yield str(i)
            i += 1

    if random_seed is not None:
        np.random.seed(random_seed)

    _max_attempts = 10
    for _attempt in range(_max_attempts):
        names = _name_gen(starting_index)
        tree = _initialize_bd_tree(initial_tree, initial_birth_scale, names)
        current_lineages: PriorityQueue = PriorityQueue()
        observed_nodes: list[str] = []

        starting_lineage = _make_initial_lineages(tree)

        try:
            if len(tree.nodes) == 1:
                _sample_lineage_event(
                    starting_lineage,
                    current_lineages,
                    tree,
                    names,
                    observed_nodes,
                    birth_waiting_distribution,
                    death_waiting_distribution,
                    experiment_time,
                    mutation_distribution,
                    fitness_distribution,
                    fitness_base,
                )
            else:
                current_lineages = starting_lineage

            while not current_lineages.empty():
                if num_extant and current_lineages.qsize() == num_extant:
                    remaining = []
                    while not current_lineages.empty():
                        _, _, lineage = current_lineages.get()
                        remaining.append(lineage)
                    min_time = remaining[0]["total_time"]
                    for lineage in remaining:
                        parent = list(tree.predecessors(lineage["id"]))[0]
                        tree.nodes[lineage["id"]]["time"] += min_time - lineage["total_time"]
                        tree.nodes[lineage["id"]]["birth_scale"] = tree.nodes[parent]["birth_scale"]
                        observed_nodes.append(lineage["id"])
                    break

                _, _, lineage = current_lineages.get()
                if lineage["active"]:
                    for _ in range(2):
                        _sample_lineage_event(
                            lineage,
                            current_lineages,
                            tree,
                            names,
                            observed_nodes,
                            birth_waiting_distribution,
                            death_waiting_distribution,
                            experiment_time,
                            mutation_distribution,
                            fitness_distribution,
                            fitness_base,
                        )

            result = _build_tree(tree, observed_nodes, collapse_unifurcations)
            tdata = td.TreeData(obst={key_added: result}, uns={"default_depth": "time"})
            tdata.obs["time"] = _get_leaf_data(result, "time")
            tdata.obs["birth_scale"] = _get_leaf_data(result, "birth_scale")
            return tdata

        except TreeSimulatorError as e:
            if "All lineages died" not in str(e) or _attempt == _max_attempts - 1:
                raise
            if _attempt == 0:
                warnings.warn(
                    "All lineages died before stopping condition; retrying simulation.",
                    UserWarning,
                    stacklevel=2,
                )


def _initialize_bd_tree(
    initial_tree: nx.DiGraph | None,
    initial_birth_scale: float,
    names: Generator,
) -> nx.DiGraph:
    if initial_tree is not None:
        tree = initial_tree.copy()
        for node in tree.nodes:
            if "birth_scale" not in tree.nodes[node]:
                tree.nodes[node]["birth_scale"] = initial_birth_scale
            if "time" not in tree.nodes[node]:
                tree.nodes[node]["time"] = 0
        return tree

    tree = nx.DiGraph()
    root = next(names)
    tree.add_node(root)
    tree.nodes[root]["birth_scale"] = initial_birth_scale
    tree.nodes[root]["time"] = 0
    return tree


def _make_lineage_dict(
    id_value: str,
    birth_scale: float,
    total_time: float,
    active_flag: bool,
) -> dict:
    return {
        "id": id_value,
        "birth_scale": birth_scale,
        "total_time": total_time,
        "active": active_flag,
    }


def _make_initial_lineages(tree: nx.DiGraph):
    leaves = [node for node in tree if tree.out_degree(node) == 0]
    current_lineages: PriorityQueue = PriorityQueue()
    for leaf in leaves:
        lineage_dict = _make_lineage_dict(
            leaf,
            tree.nodes[leaf]["birth_scale"],
            tree.nodes[leaf]["time"],
            True,
        )
        if len(tree.nodes) == 1:
            return lineage_dict
        current_lineages.put((tree.nodes[leaf]["time"], leaf, lineage_dict))
    return current_lineages


def _update_fitness(
    birth_scale: float,
    mutation_distribution: Callable[[], int] | None,
    fitness_distribution: Callable[[], float] | None,
    fitness_base: float,
) -> float:
    coefficient = 1.0
    if mutation_distribution is not None:
        num_mutations = int(mutation_distribution())
        if num_mutations < 0:
            raise TreeSimulatorError("Negative number of mutations detected")
        for _ in range(num_mutations):
            coefficient *= fitness_base ** fitness_distribution()
    return birth_scale * coefficient


def _sample_lineage_event(
    lineage: dict,
    current_lineages: PriorityQueue,
    tree: nx.DiGraph,
    names: Generator,
    observed_nodes: list[str],
    birth_waiting_distribution: Callable[[float], float],
    death_waiting_distribution: Callable[[], float],
    experiment_time: float | None,
    mutation_distribution: Callable[[], int] | None,
    fitness_distribution: Callable[[], float] | None,
    fitness_base: float,
) -> None:
    if not lineage["active"]:
        raise TreeSimulatorError("Cannot sample event for non-active lineage")

    unique_id = next(names)
    birth_wait = birth_waiting_distribution(lineage["birth_scale"])
    death_wait = death_waiting_distribution()

    if birth_wait <= 0 or death_wait <= 0:
        raise TreeSimulatorError("0 or negative waiting time detected")

    tree.add_node(unique_id)
    tree.add_edge(lineage["id"], unique_id)

    if (
        experiment_time
        and lineage["total_time"] + birth_wait >= experiment_time
        and lineage["total_time"] + death_wait >= experiment_time
    ):
        tree.nodes[unique_id]["birth_scale"] = lineage["birth_scale"]
        tree.nodes[unique_id]["time"] = experiment_time
        current_lineages.put(
            (
                experiment_time,
                unique_id,
                _make_lineage_dict(unique_id, lineage["birth_scale"], experiment_time, False),
            )
        )
        observed_nodes.append(unique_id)

    elif birth_wait < death_wait:
        updated_scale = _update_fitness(
            lineage["birth_scale"], mutation_distribution, fitness_distribution, fitness_base
        )
        new_time = birth_wait + lineage["total_time"]
        tree.nodes[unique_id]["birth_scale"] = updated_scale
        tree.nodes[unique_id]["time"] = new_time
        current_lineages.put(
            (
                new_time,
                unique_id,
                _make_lineage_dict(unique_id, updated_scale, new_time, True),
            )
        )

    else:
        new_time = death_wait + lineage["total_time"]
        tree.nodes[unique_id]["birth_scale"] = lineage["birth_scale"]
        tree.nodes[unique_id]["time"] = new_time
        current_lineages.put(
            (
                new_time,
                unique_id,
                _make_lineage_dict(unique_id, lineage["birth_scale"], new_time, False),
            )
        )


def _prune_dead_lineages(tree: nx.DiGraph, observed_nodes: list[str]) -> None:
    observed = set(observed_nodes)
    root = next(n for n in tree if tree.in_degree(n) == 0)
    to_remove = {n for n in tree if tree.out_degree(n) == 0} - observed - {root}
    while to_remove:
        tree.remove_nodes_from(to_remove)
        to_remove = {n for n in tree if tree.out_degree(n) == 0} - observed - {root}


def _build_tree(
    tree: nx.DiGraph,
    observed_nodes: list[str],
    do_collapse: bool,
) -> nx.DiGraph:
    tree = tree.copy()
    _prune_dead_lineages(tree, observed_nodes)

    if do_collapse and len(tree.nodes) > 1:
        tree = _collapse_unifurcations(tree, collapse_root=False)

    if len(tree.nodes) == 1:
        raise TreeSimulatorError("All lineages died before stopping condition")

    return tree


def simple_fit_subclone(
    branch_length_neutral: float | Callable[[], float] = 1.0,
    branch_length_fit: float | Callable[[], float] = 0.5,
    experiment_duration: float = 10.0,
    generations_until_fit_subclone: int = 5,
    key_added: str = "simulated",
) -> td.TreeData:
    """Simulate a clonal population that develops one fit subclone.

    All cells evolve neutrally until generation ``generations_until_fit_subclone``,
    at which point exactly one lineage gains fitness and begins dividing at the
    faster ``branch_length_fit`` rate. The tree grows via BFS until
    ``experiment_duration`` is reached.

    Leaf nodes are named ``"{id}_neutral"`` or ``"{id}_fit"`` depending on which
    population they belong to. All leaf times equal ``experiment_duration``.

    Args:
        branch_length_neutral: Branch length for neutrally evolving cells.
            May be a scalar or a zero-argument callable for stochastic lengths.
        branch_length_fit: Branch length for the fit subclone. May be a scalar
            or a zero-argument callable.
        experiment_duration: Total experiment length; cells that have not yet
            divided by this time become leaves.
        generations_until_fit_subclone: Generation at which one lineage gains
            fitness.
        key_added: Key under which the tree is stored in ``obst``.

    Returns:
        A TreeData with the simulated tree in ``obst[key_added]``. All nodes
        have a ``"time"`` attribute.
    """
    if isinstance(branch_length_neutral, (int, float)):
        _bl_neutral: Callable[[], float] = lambda: branch_length_neutral  # type: ignore[arg-type]
    else:
        _bl_neutral = branch_length_neutral
    if isinstance(branch_length_fit, (int, float)):
        _bl_fit: Callable[[], float] = lambda: branch_length_fit  # type: ignore[arg-type]
    else:
        _bl_fit = branch_length_fit

    def _name_gen() -> Generator[str, None, None]:
        i = 0
        while True:
            yield str(i)
            i += 1

    tree: nx.DiGraph = nx.DiGraph()
    names = _name_gen()
    q: Queue = Queue()
    times: dict[str, float] = {}
    fits: dict[str, bool] = {}

    root = next(names) + "_neutral"
    tree.add_node(root)
    times[root] = 0.0
    fits[root] = False

    root_child = next(names) + "_neutral"
    tree.add_edge(root, root_child)
    q.put((root_child, 0.0, "neutral", 0))

    subclone_started = False
    while not q.empty():
        node, time, fitness, generation = q.get()
        fits[node] = fitness == "fit"
        bl = _bl_neutral() if fitness == "neutral" else _bl_fit()
        division_time = time + bl
        if division_time >= experiment_duration:
            times[node] = experiment_duration
            continue
        times[node] = division_time
        left_fitness = fitness
        right_fitness = fitness
        if not subclone_started and generation + 1 == generations_until_fit_subclone:
            subclone_started = True
            left_fitness = "fit"
        left_child = next(names) + "_" + left_fitness
        right_child = next(names) + "_" + right_fitness
        tree.add_nodes_from([left_child, right_child])
        tree.add_edges_from([(node, left_child), (node, right_child)])
        q.put((left_child, division_time, left_fitness, generation + 1))
        q.put((right_child, division_time, right_fitness, generation + 1))

    nx.set_node_attributes(tree, times, "time")
    nx.set_node_attributes(tree, fits, "fit")
    tdata = td.TreeData(obst={key_added: tree}, uns={"default_depth": "time"})
    tdata.obs["time"] = _get_leaf_data(tree, "time")
    tdata.obs["fit"] = _get_leaf_data(tree, "fit")
    return tdata
