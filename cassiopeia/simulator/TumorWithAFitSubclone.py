"""
This file stores a subclass of TreeSolver, the TumorWithAFitSubclone. The
TumorWithAFitSubclone simulates a tumor phylogeny which develops one fit
subclone.
"""
import networkx as nx
from queue import Queue

from cassiopeia.data import CassiopeiaTree
from .TreeSimulator import TreeSimulator


class TumorWithAFitSubclone(TreeSimulator):
    r"""
    Simulates a tumor phylogeny which develops one fit subclone.

    This TreeSimulator simulates a (binary) tumor phylogeny that evolves
    neutrally with constant branch length 'branch_length_neutral', until in
    generation 'generations_until_fit_subclone' one of the lineages develops
    fitness and starts to expand with the new branch length 'branch_length_fit'.
    The total length of the experiment is given by 'experiment_duration'.
    This TreeSimulator is useful because (1) it generates deterministic
    phylogenies and (2) the phylogenies are as simple as possible while being
    non-neutral. The ability to control the branch lengths of the neutrally
    evolving cells and of the fit cells allows us to simulate different degrees
    of fitness.

    Note that the branches for the leaves of the tree need not have length
    exactly equal to branch_length_neutral nor branch_length_fit, because the
    cells get assayed at the arbitrary time specified by 'experiment_duration',
    which is not necessarily the time at which they divide.

    The names of the leaf nodes will be given by a unique integer ID, followed
    by "_neutral" for the neutrally evolving cells, and by "_fit" for the fit
    cells.

    Finally, note that we do not enforce the condition branch_length_neutral >
    branch_length_fit, therefore it is possible to make the subclone less fit,
    i.e. expand slower.

    Args:
        branch_length_neutral: Branch length of the neutrally evolving cells.
            All cells are neutrally evolving until generation
            'generations_until_fit_subclone', when exactly one of the lineages
            develops fitness.
        branch_length_fit: The branch length of the fit subclone, which appears
            exactly at generation 'generations_until_fit_subclone'.
        experiment_duration: The total length of the experiment.
        generations_until_fit_subclone: The generation at which one lineage
            develops fitness, giving rise to a fit subclone.
    """

    def __init__(
        self,
        branch_length_neutral: float,
        branch_length_fit: float,
        experiment_duration: float,
        generations_until_fit_subclone: int,
    ):
        self.branch_length_neutral = branch_length_neutral
        self.branch_length_fit = branch_length_fit
        self.experiment_duration = experiment_duration
        self.generations_until_fit_subclone = generations_until_fit_subclone

    def simulate_tree(self) -> CassiopeiaTree:
        r"""
        See base class.
        """
        branch_length_neutral = self.branch_length_neutral
        branch_length_fit = self.branch_length_fit
        experiment_duration = self.experiment_duration
        generations_until_fit_subclone = self.generations_until_fit_subclone

        def node_name_generator():
            i = 0
            while True:
                yield str(i)
                i += 1

        tree = nx.DiGraph()  # This is what will get populated.

        names = node_name_generator()
        q = Queue()  # Contains: (node, time, fitness, generation)
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
            time_till_division = (
                branch_length_neutral
                if node_fitness == "neutral"
                else branch_length_fit
            )
            time_of_division = time + time_till_division
            if time_of_division >= experiment_duration:
                # Not enough time left for the cell to divide.
                times[node] = experiment_duration
                continue
            # Create children, add edges to them, and push children to the
            # queue.
            times[node] = time_of_division
            left_child_fitness = node_fitness
            right_child_fitness = node_fitness
            if (
                not subclone_started
                and generation + 1 == generations_until_fit_subclone
            ):
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