"""
A general simulator for a birth-death tree simulation process, including 
fitness. Takes any given distribution on waiting times and fitness 
mutations.
"""

import networkx as nx
import numpy as np
from queue import PriorityQueue

from typing import Callable, Dict, Generator, Optional, Union

from cassiopeia.data.CassiopeiaTree import CassiopeiaTree
import cassiopeia.data.utilities as utilities
from cassiopeia.simulator.TreeSimulator import TreeSimulator, TreeSimulatorError


class BirthDeathFitnessSimulator(TreeSimulator):

    def __init__(
        self,
        birth_waiting_distribution: Callable[[float], float],
        initial_birth_scale: float,
        death_waiting_distribution: Optional[Callable[[], float]] = None,
        mutation_distribution: Optional[Callable[[], int]] = None,
        fitness_distribution: Optional[Callable[[], float]] = None,
        num_extant: Optional[int] = None,
        experiment_time: Optional[float] = None,
    ):
        if num_extant is None and experiment_time is None:
            raise TreeSimulatorError("Please specify a stopping condition")
        if num_extant and experiment_time:
            raise TreeSimulatorError(
                "Please choose only one stopping condition"
            )
        if mutation_distribution is not None and fitness_distribution is None:
            raise TreeSimulatorError(
                "Please specify a fitness strength distribution"
            )
        if num_extant is not None and num_extant <= 0:
            raise TreeSimulatorError(
                "Please specify number of extant lineages greater than 0"
            )
        if experiment_time is not None and experiment_time <= 0:
            raise TreeSimulatorError(
                "Please specify an experiment time greater than 0"
            )

        if death_waiting_distribution is None:
            death_waiting_distribution = lambda: np.inf

        self.birth_waiting_distribution = birth_waiting_distribution
        self.initial_birth_scale = initial_birth_scale
        self.death_waiting_distribution = death_waiting_distribution
        self.mutation_distribution = mutation_distribution
        self.fitness_distribution = fitness_distribution
        self.num_extant = num_extant
        self.experiment_time = experiment_time

    def simulate_tree(
        self,
    ) -> CassiopeiaTree:
        """Simulates trees from a general birth/death process with fitness.

        A forward-time birth/death process is simulated by maintaining a queue
        of lineages and updating them with birth and death events. At each
        live currently extant node in the tree representing a division event,
        the lifespan of two potential descendants is simulated. For each
        descendant, waiting times are sampled from the birth and death
        distributions, and the smaller time is used to represent the next
        event. If a death occurs, the lineage exists until death occurs, and
        if death occurs before the end of the experiment the lineage is removed.
        If a birth event is sampled, then the lineage exists until it divides,
        at which point a new internal node representing a division event is
        added. The process is represented as a tree, with internal nodes
        representing division events, branch lengths representing the lifetime
        lengths, and leaves representing samples observed at the end of the
        experiment. If no death waiting time distribution is provided, the
        process reduces to a Yule birth process.

        Fitness is represented by each lineage maintaining its own birth scale
        parameter. This parameter determines the shape of the distribution
        from which birth waiting times are sampled and thus affects how
        quickly cells divide. At each division event, the fitness is updated
        by sampling from a distribution determining the number of mutations,
        and the multiplicative strength of each mutation is determined by
        another distribution. The birth scale parameter of the lineage is
        then updated by the total multiplicative strength factor across all
        mutations and passed on to the descendant nodes.

        There are two stopping conditions for the simulation. The first is
        "number of extant nodes", which specifies the simulation to stop the
        first moment the specified number of extant nodes exist. The second is
        "experiment time", which specifies the time at which to end the
        experiment, i.e. the experiment ends when all living lineages reach
        this time in their total lived time. At least one of these two stopping
        criteria must be provided.

        Example use snippet:
            # note that numpy uses a different parameterization of the
            # exponential distribution with the scale parameter, which is 1/rate

            birth_waiting_distribution = lambda scale: np.random.exponential(scale)
            death_waiting_distribution = np.random.exponential(1.5)
            initial_birth_scale = 0.5
            mutation_distribution = lambda: 1 if np.random.uniform() > 0.5 else 0
            fitness_distribution = lambda: 2 ** np.random.uniform(-1,1)

            tree = generate_birth_death(
                birth_waiting_distribution,
                initial_birth_scale,
                death_waiting_distribution=death_waiting_distribution,
                mutation_distribution=mutation_distribution,
                fitness_distribution=fitness_distribution,
                num_extant=8
            )

        Args:
            birth_waiting_distribution: A function that samples waiting times from the
                birth distribution. Must take a scale parameter as the input
            initial_birth_scale: The global scale parameter that is used at the
                start of the experiment
            death_waiting_distribution: A function that samples waiting times from the
                death distribution
            mutation_distribution: A function that samples the number of mutations
                that occur at a division event. If None, then no mutations are
                sampled
            fitness_distribution: A function that samples the multiplicative
                update to the scale parameter of the current lineage at a
                division event. Must not be None if mutation_distribution provided
            num_extant: Specifies the number of extant lineages living at one
                time as a stopping condition for the experiment
            experiment_time: Specifies the time that the experiment runs as a
                stopping condition for the experiment

        Returns:
            A CassiopeiaTree with the tree topology initialized
        """

        # Samples whether birth, death, or the end of the experiment comes next
        # for a given lineage, and any fitness changes
        def sample_lineage_event(lineage: Dict[str, Union[int, float]]
        ) -> None:
            """A helper function that samples an event for a lineage.

            Takes a lineage and determines the next event in that lineage's
            future. Simulates the lifespan of a new descendant. Birth and
            death waiting times are sampled, representing how long the
            descendant lived. If a death event occurs first, then the lineage
            with the new descendant is added to the queue of currently alive,
            but its status is marked as inactive and will be removed at the
            time the lineage dies. If a birth event occurs first, then the
            lineage with the new descendant is added to the queue, but with its
            status marked as active, and further events will be sampled at the
            time the lineage divides. Additionally, its fitness will be updated.
            The descendant node is added to the tree object, with the edge
            weight between the current node and the descendant representing the
            lifespan of the descendant. In the case the descendant would live
            past the end of the experiment (both birth and death times exceed
            past the end of the experiment), then the lifespan is cut off at
            the experiment time and a final observed sample is added to the
            tree. In this case the lineage is marked as inactive as well.

            Args:
                unique_id: The unique ID number to be used to name a new node
                    added to the tree
                lineage: The current extant lineage to extend. Contains the ID
                    of the internal node to attach the descendant to, the
                    current birth scale parameter of the lineage, the current
                    total lived time of the lineage, and the status of whether
                    the lineage is still dividing

            Returns:
                The new unique ID to use for the next event, incremented only if
                a new descendant is added to the tree
            """
            assert(lineage["active"])

            unique_id = next(names)

            birth_waiting_time = self.birth_waiting_distribution(lineage["birth_scale"])
            death_waiting_time = self.death_waiting_distribution()
            if birth_waiting_time <= 0 or death_waiting_time <= 0:
                raise TreeSimulatorError(
                    "0 or negative waiting time detected"
                )

            # If birth and death would happen after the total experiment time,
            # just cut off the living branch length at the experiment time
            if (
                self.experiment_time
                and lineage["total_time"] + birth_waiting_time
                >= self.experiment_time
                and lineage["total_time"] + death_waiting_time
                >= self.experiment_time
            ):
                tree.add_node(unique_id)
                tree.nodes[unique_id]["birth_scale"] = lineage["birth_scale"]
                tree.add_edge(
                    lineage["id"],
                    unique_id,
                    weight=self.experiment_time - lineage["total_time"],
                )
                tree.nodes[unique_id]["total_time"] = self.experiment_time

                current_lineages.put(
                    (
                        self.experiment_time,
                        unique_id,
                        {
                            "id": unique_id,
                            "birth_scale": lineage["birth_scale"],
                            "total_time": self.experiment_time,
                            "active": False,
                        },
                    )
                )
                observed_nodes.append(unique_id)

            else:
                if birth_waiting_time < death_waiting_time:
                    # Update fitness
                    base_selection_coefficient = 1
                    if self.mutation_distribution:
                        num_mutations = int(self.mutation_distribution())
                        if num_mutations < 0:
                            raise TreeSimulatorError(
                                "Negative number of mutations detected"
                            )
                        for _ in range(num_mutations):
                            base_selection_coefficient *= self.fitness_distribution()
                            if base_selection_coefficient < 0:
                                raise TreeSimulatorError(
                                    "Negative mutation strength detected"
                                )

                    # Annotate parameters for a given node in the tree
                    tree.add_node(unique_id)
                    tree.nodes[unique_id]["birth_scale"] = (
                        lineage["birth_scale"] * base_selection_coefficient
                    )
                    tree.add_edge(
                        lineage["id"], unique_id, weight=birth_waiting_time
                    )
                    tree.nodes[unique_id]["total_time"] = (
                        birth_waiting_time + lineage["total_time"]
                    )
                    # Add the newly generated cell to the list of living lineages
                    current_lineages.put(
                        (
                            birth_waiting_time + lineage["total_time"],
                            unique_id,
                            {
                                "id": unique_id,
                                "birth_scale": lineage["birth_scale"]
                                * base_selection_coefficient,
                                "total_time": birth_waiting_time
                                + lineage["total_time"],
                                "active": True,
                            },
                        )
                    )

                else:
                    tree.add_node(unique_id)
                    tree.nodes[unique_id]["birth_scale"] = lineage["birth_scale"]
                    tree.add_edge(
                        lineage["id"], unique_id, weight=death_waiting_time
                    )
                    tree.nodes[unique_id]["total_time"] = (
                        death_waiting_time + lineage["total_time"]
                    )
                    current_lineages.put(
                        (
                            death_waiting_time + lineage["total_time"],
                            unique_id,
                            {
                                "id": unique_id,
                                "birth_scale": lineage["birth_scale"],
                                "total_time": death_waiting_time
                                + lineage["total_time"],
                                "active": False,
                            },
                        )
                    )

        def node_name_generator() -> Generator[str, None, None]:
            i = 0
            while True:
                yield str(i)
                i += 1
        names = node_name_generator()

        # Instantiate the implicit root
        tree = nx.DiGraph()
        root = next(names)
        tree.add_node(root)
        tree.nodes[root]["birth_scale"] = self.initial_birth_scale
        tree.nodes[root]["total_time"] = 0
        current_lineages = PriorityQueue()
        # Records the nodes that are observed at the end of the experiment
        observed_nodes = []
        starting_lineage = {
            "id": root,
            "birth_scale": self.initial_birth_scale,
            "total_time": 0,
            "active": True,
        }

        # Sample the waiting time until the first division
        sample_lineage_event(starting_lineage)

        # Perform the process until there are no active extant lineages left
        while not current_lineages.empty():
            # If number of extant lineages is the stopping criterion, at the
            # first instance of having n extant tips, stop the experiment
            # and set the total lineage time for each lineage to be equal to
            # the minimum, to produce ultrametric trees. Also, the birth_scale
            # parameter of each leaf is rolled back to equal its parent's.
            if self.num_extant:
                if current_lineages.qsize() == self.num_extant:
                    remaining_lineages = []
                    while not current_lineages.empty():
                        _, _, lineage = current_lineages.get()
                        remaining_lineages.append(lineage)
                    min_total_time = remaining_lineages[0]["total_time"]
                    for lineage in remaining_lineages:
                        parent = list(tree.predecessors(lineage["id"]))[0]
                        tree.edges[parent, lineage["id"]]["weight"] += (
                            min_total_time - lineage["total_time"]
                        )
                        tree.nodes[lineage["id"]]["birth_scale"] = tree.nodes[
                            parent
                        ]["birth_scale"]
                        observed_nodes.append(lineage["id"])
            if not current_lineages.empty():
                # Pop the minimum age lineage to simulate forward time
                _, _, lineage = current_lineages.get()
                # If the lineage is no longer active, just remove it from the queue.
                # This represents the time at which the lineage dies.
                if lineage["active"]:
                    for _ in range(2):
                        sample_lineage_event(lineage)

        # Prune dead lineages and collapse resulting unifurcations
        if len(tree.nodes) > 1:
            for i in list(tree.nodes):
                if tree.out_degree(i) == 0 and i not in observed_nodes:
                    utilities.remove_and_prune_lineage(i, tree)
            if len(tree.nodes) > 1:
                utilities.collapse_unifurcations(tree, source="1")

        # If only implicit root remains after pruning dead lineages, error
        if len(tree.nodes) == 1:
            raise TreeSimulatorError(
                "All lineages died before stopping condition"
            )

        return CassiopeiaTree(tree=tree)


