"""
This file stores a subclass of TreeSolver, the SimpleFitSubcloneSimulator. The
SimpleFitSubcloneSimulator simulates a clonal population which develops one
fit subclone.
"""

from collections.abc import Callable, Generator
from queue import Queue

import networkx as nx

from cassiopeia.data import CassiopeiaTree
from cassiopeia.simulator.TreeSimulator import TreeSimulator


class SimpleFitSubcloneSimulator(TreeSimulator):
    r"""
    Simulates a clonal population which develops one fit subclone.

    This TreeSimulator simulates a (binary) clonal population that evolves
    neutrally with constant or generated branch length 'branch_length_neutral',
    until in generation 'generations_until_fit_subclone' one of the lineages
    develops fitness and starts to expand with the new constant or generated
    branch length 'branch_length_fit'. The total length of the experiment is
    given by 'experiment_duration'. This TreeSimulator is useful because (1) it
    generates both deterministic and stochastic phylogenies and (2) the
    phylogenies are as simple as possible while being non-neutral. The ability
    to control the branch lengths of the neutrally evolving population and of
    the fit subpopulation allows us to simulate different degrees of fitness
    as well as stochasticity.

    Note that the branches for the leaves of the tree need not have length
    exactly equal to 'branch_length_neutral' nor 'branch_length_fit', because
    the population gets assayed at the arbitrary time specified by
    'experiment_duration', which is not necessarily the time at which a
    division event happens.

    The names of the leaf nodes will be given by a unique integer ID, followed
    by "_neutral" for the neutrally evolving individuals, and by "_fit" for the
    fit individuals.

    Finally, note that it is possible to make the subclone less fit, i.e.
    expand slower, for there is no constraint in the branch lengths, although
    the most typical use case is the simulation of a more fit subpopulation.

    Args:
        branch_length_neutral: Branch length of the neutrally evolving
            individuals. All individuals are neutrally evolving until generation
            'generations_until_fit_subclone', when exactly one of the lineages
            develops fitness. A callable can be provided instead of a constant,
            which is useful for simulating random branch lengths.
        branch_length_fit: The branch length of the fit subclone, which appears
            exactly at generation 'generations_until_fit_subclone'. A callable
            can be provided instead of a constant, which is useful for
            simulating random branch lengths.
        experiment_duration: The total length of the experiment.
        generations_until_fit_subclone: The generation at which one lineage
            develops fitness, giving rise to a fit subclone.
    """

    def __init__(
        self,
        branch_length_neutral: float | Callable[[], float],
        branch_length_fit: float | Callable[[], float],
        experiment_duration: float,
        generations_until_fit_subclone: int,
    ):
        self.branch_length_neutral = self._create_callable(branch_length_neutral)
        self.branch_length_fit = self._create_callable(branch_length_fit)
        self.experiment_duration = experiment_duration
        self.generations_until_fit_subclone = generations_until_fit_subclone

    def _create_callable(self, x: float | Callable[[], float]) -> Callable[[], float]:
        # In case the user provides an int, we still hold their back...
        if type(x) in [int, float]:

            def constant_branch_length_callable() -> float:
                return x

            return constant_branch_length_callable
        else:
            return x

    def simulate_tree(self) -> CassiopeiaTree:
        r"""See base class."""
        branch_length_neutral = self.branch_length_neutral
        branch_length_fit = self.branch_length_fit
        experiment_duration = self.experiment_duration
        generations_until_fit_subclone = self.generations_until_fit_subclone

        def node_name_generator() -> Generator[str, None, None]:
            i = 0
            while True:
                yield str(i)
                i += 1

        tree = nx.DiGraph()  # This is what will get populated.

        names = node_name_generator()
        # Contains: (node, time, fitness, generation)
        q = Queue()  # type: Queue[Tuple[str, float, str, int]]
        times = {}

        root = next(names) + "_neutral"
        tree.add_node(root)
        times[root] = 0.0

        root_child = next(names) + "_neutral"
        tree.add_edge(root, root_child)
        q.put((root_child, 0.0, "neutral", 0))
        subclone_started = False
        while not q.empty():
            # Pop next node
            (node, time, node_fitness, generation) = q.get()
            time_till_division = branch_length_neutral() if node_fitness == "neutral" else branch_length_fit()
            time_of_division = time + time_till_division
            if time_of_division >= experiment_duration:
                # Not enough time left for the individual to divide.
                times[node] = experiment_duration
                continue
            # Create children, add edges to them, and push children to the
            # queue.
            times[node] = time_of_division
            left_child_fitness = node_fitness
            right_child_fitness = node_fitness
            if not subclone_started and generation + 1 == generations_until_fit_subclone:
                # Start the subclone
                subclone_started = True
                left_child_fitness = "fit"
            left_child = next(names) + "_" + left_child_fitness
            right_child = next(names) + "_" + right_child_fitness
            tree.add_nodes_from([left_child, right_child])
            tree.add_edges_from([(node, left_child), (node, right_child)])
            q.put(
                (
                    left_child,
                    time_of_division,
                    left_child_fitness,
                    generation + 1,
                )
            )
            q.put(
                (
                    right_child,
                    time_of_division,
                    right_child_fitness,
                    generation + 1,
                )
            )
        res = CassiopeiaTree(tree=tree)
        res.set_times(times)
        return res
