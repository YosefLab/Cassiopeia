"""
This file stores a general phylogenetic tree simulator using forward birth-death
process, including differing fitness on lineages on the tree. Allows for a 
variety of division and fitness regimes to be specified by the user.
"""
from typing import Callable, Dict, Generator, List, Optional, Union

import networkx as nx
import numpy as np
from queue import PriorityQueue

from cassiopeia.data.CassiopeiaTree import CassiopeiaTree
from cassiopeia.simulator.TreeSimulator import TreeSimulator, TreeSimulatorError


class BirthDeathFitnessSimulator(TreeSimulator):
    """Simulator class for a general forward birth-death process with fitness.

    Implements a flexible phylogenetic tree simulator using a forward birth-
    death process. In this process starting from an initial root lineage,
    births represent the branching of a new lineage and death represents the
    cessation of an existing lineage. The process is represented as a tree,
    with internal nodes representing division events, branch lengths
    representing the lifetimes of individuals, and leaves representing samples
    observed at the end of the experiment.

    Allows any distribution on birth and death waiting times to
    be specified, including constant, exponential, weibull, etc. If no death
    waiting time distribution is provided, the process reduces to a Yule birth
    process. Also robustly simulates differing fitness on lineages within a
    simulated tree. Fitness in this context represents potential mutations that
    may be acquired on a lineage that change the rate at which new members are
    born. Each lineage maintains its own birth scale parameter, altered from an
    initial specified experiment-wide birth scale parameter by accrued
    mutations. Different fitness regimes can be specified based on user
    provided distributions on how often fitness mutations occur and their
    respective strengths.

    There are two stopping conditions for the simulation. The first is "number
    of extant nodes", which specifies the simulation to run until the first
    moment a number of extant nodes exist. The second is "experiment time",
    which specifies the time at which lineages are sampled. At least one of
    these two stopping criteria must be provided. Both can be provided in which
    case the simulation is run until one of the stopping conditions is reached.

    Example use snippet:
        # note that numpy uses a different parameterization of the
        # exponential distribution with the scale parameter, which is 1/rate

        birth_waiting_distribution = lambda scale: np.random.exponential(scale)
        death_waiting_distribution = np.random.exponential(1.5)
        initial_birth_scale = 0.5
        mutation_distribution = lambda: 1 if np.random.uniform() > 0.5 else 0
        fitness_distribution = lambda: np.random.uniform(-1,1)
        fitness_base = 2

        bd_sim = BirthDeathFitnessSimulator(
            birth_waiting_distribution,
            initial_birth_scale,
            death_waiting_distribution=death_waiting_distribution,
            mutation_distribution=mutation_distribution,
            fitness_distribution=fitness_distribution,
            fitness_base=fitness_base,
            num_extant=8
        )
        tree = bd_sim.simulate_tree()

    Args:
        birth_waiting_distribution: A function that samples waiting times
            from the birth distribution. Determines how often births occur.
            Must take a scale parameter as the input
        initial_birth_scale: The initial scale parameter that is used at the
            start of the experiment
        death_waiting_distribution: A function that samples waiting times
            from the death distribution. Determines how often deaths occur
        mutation_distribution: A function that samples the number of
            mutations that occur at a division event. If None, then no
            mutations are sampled
        fitness_distribution: One of the two elements in determining the
            multiplicative coefficient of a fitness mutation. A function that
            samples the exponential that the fitness base is raised by.
            Determines the distribution of fitness mutation strengths. Must not
            be None if mutation_distribution provided
        fitness_base: One of the two elements in determining the
            multiplicative strength of a fitness mutation. The base that is
            raised by the value given by the fitness distribution. Determines
            the base strength of fitness mutations. By default is e, Euler's
            Constant
        num_extant: Specifies the number of extant lineages existing at the
            same time as a stopping condition for the experiment
        experiment_time: Specifies the total time that the experiment runs as a
            stopping condition for the experiment
        collapse_unifurcations: Specifies whether to collapse unifurcations in
            the tree resulting from pruning dead lineages
        random_seed: A seed for reproducibility

    Raises:
        TreeSimulatorError if invalid stopping conditions are provided or if a
        fitness distribution is not provided when a mutation distribution isn't
    """

    def __init__(
        self,
        birth_waiting_distribution: Callable[[float], float],
        initial_birth_scale: float,
        death_waiting_distribution: Optional[
            Callable[[], float]
        ] = lambda: np.inf,
        mutation_distribution: Optional[Callable[[], int]] = None,
        fitness_distribution: Optional[Callable[[], float]] = None,
        fitness_base: float = np.e,
        num_extant: Optional[int] = None,
        experiment_time: Optional[float] = None,
        collapse_unifurcations: bool = True,
        random_seed: int = None,
    ):
        if num_extant is None and experiment_time is None:
            raise TreeSimulatorError(
                "Please specify at least one stopping condition"
            )
        if mutation_distribution is not None and fitness_distribution is None:
            raise TreeSimulatorError(
                "Please specify a fitness strength distribution"
            )
        if num_extant is not None and num_extant <= 0:
            raise TreeSimulatorError(
                "Please specify number of extant lineages greater than 0"
            )
        if num_extant is not None and type(num_extant) is not int:
            raise TreeSimulatorError(
                "Please specify an integer number of extant tips"
            )
        if experiment_time is not None and experiment_time <= 0:
            raise TreeSimulatorError(
                "Please specify an experiment time greater than 0"
            )

        self.birth_waiting_distribution = birth_waiting_distribution
        self.initial_birth_scale = initial_birth_scale
        self.death_waiting_distribution = death_waiting_distribution
        self.mutation_distribution = mutation_distribution
        self.fitness_distribution = fitness_distribution
        self.fitness_base = fitness_base
        self.num_extant = num_extant
        self.experiment_time = experiment_time
        self.collapse_unifurcations = collapse_unifurcations
        self.random_seed = random_seed

    def simulate_tree(self,) -> CassiopeiaTree:
        """Simulates trees from a general birth/death process with fitness.

        A forward-time birth/death process is simulated by tracking a series of
        lineages and sampling event waiting times for each lineage. Each
        lineage draws death waiting times from the same distribution, but
        maintains its own birth scale parameter that determines the shape of
        its birth waiting time distribution. At each division event, fitness
        mutation events are sampled, and the birth scale parameter is scaled by
        their multiplicative coefficients. This updated birth scale passed
        onto successors.

        Returns:
            A CassiopeiaTree with the tree topology initialized with the
            simulated tree

        Raises:
            TreeSimulatorError if all lineages die before a stopping condition
        """

        def node_name_generator() -> Generator[str, None, None]:
            """Generates unique node names for the tree."""
            i = 0
            while True:
                yield str(i)
                i += 1

        names = node_name_generator()

        # Set the seed
        if self.random_seed:
            np.random.seed(self.random_seed)

        # Instantiate the implicit root
        tree = nx.DiGraph()
        root = next(names)
        tree.add_node(root)
        tree.nodes[root]["birth_scale"] = self.initial_birth_scale
        tree.nodes[root]["time"] = 0
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
        self.sample_lineage_event(
            starting_lineage, current_lineages, tree, names, observed_nodes
        )

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
                        tree.nodes[lineage["id"]]["time"] += (
                            min_total_time - lineage["total_time"]
                        )
                        tree.nodes[lineage["id"]]["birth_scale"] = tree.nodes[
                            parent
                        ]["birth_scale"]
                        observed_nodes.append(lineage["id"])
                    break
            # Pop the minimum age lineage to simulate forward time
            _, _, lineage = current_lineages.get()
            # If the lineage is no longer active, just remove it from the queue.
            # This represents the time at which the lineage dies.
            if lineage["active"]:
                for _ in range(2):
                    self.sample_lineage_event(
                        lineage, current_lineages, tree, names, observed_nodes
                    )

        cassiopeia_tree = CassiopeiaTree(tree=tree)
        time_dictionary = {}
        for i in tree.nodes:
            time_dictionary[i] = tree.nodes[i]["time"]
        cassiopeia_tree.set_times(time_dictionary)

        # Prune dead lineages and collapse resulting unifurcations
        for i in cassiopeia_tree.nodes:
            if cassiopeia_tree.is_leaf(i) and i not in observed_nodes:
                cassiopeia_tree.remove_and_prune_lineage(i)
        if self.collapse_unifurcations and len(cassiopeia_tree.nodes) > 1:
            cassiopeia_tree.collapse_unifurcations(source="1")

        # If only implicit root remains after pruning dead lineages, error
        if len(cassiopeia_tree.nodes) == 1:
            raise TreeSimulatorError(
                "All lineages died before stopping condition"
            )

        return cassiopeia_tree

    def sample_lineage_event(
        self,
        lineage: Dict[str, Union[int, float]],
        current_lineages: PriorityQueue,
        tree: nx.DiGraph,
        names: Generator,
        observed_nodes: List[str],
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
        time the lineage divides. Additionally, its fitness will be updated
        by altering its birth rate. The descendant node is added to the
        tree object, with the edge weight between the current node and the
        descendant representing the lifespan of the descendant. In the
        case the descendant would live past the end of the experiment (both
        birth and death times exceed past the end of the experiment), then
        the lifespan is cut off at the experiment time and a final observed
        sample is added to the tree. In this case the lineage is marked as
        inactive as well.

        Args:
            unique_id: The unique ID number to be used to name a new node
                added to the tree
            lineage: The current extant lineage to extend. Contains the ID
                of the internal node to attach the descendant to, the
                current birth scale parameter of the lineage, the current
                total lived time of the lineage, and the status of whether
                the lineage is still dividing
            current_lineages: The queue containing currently alive lineages
            tree: The tree object being constructed by the simulator
                representing the birth death process
            names: A generator providing unique names for tree nodes
            observed_nodes: A list of nodes that are observed at the end of
                the experiment

        Raises:
            TreeSimulatorError if a negative waiting time is sampled or a
            non-active lineage is passed in
        """
        if not lineage["active"]:
            raise TreeSimulatorError(
                "Cannot sample event for non-active lineage"
            )

        unique_id = next(names)

        birth_waiting_time = self.birth_waiting_distribution(
            lineage["birth_scale"]
        )
        death_waiting_time = self.death_waiting_distribution()
        if birth_waiting_time <= 0 or death_waiting_time <= 0:
            raise TreeSimulatorError("0 or negative waiting time detected")

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
            tree.add_edge(lineage["id"], unique_id)
            tree.nodes[unique_id]["time"] = self.experiment_time

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

            # Indicate this node is observed at the end of experiment
            observed_nodes.append(unique_id)

        else:
            if birth_waiting_time < death_waiting_time:
                # Update birth rate
                updated_birth_scale = self.update_fitness(
                    lineage["birth_scale"]
                )

                # Annotate parameters for a given node in the tree
                tree.add_node(unique_id)
                tree.nodes[unique_id]["birth_scale"] = updated_birth_scale
                tree.add_edge(lineage["id"], unique_id)
                tree.nodes[unique_id]["time"] = (
                    birth_waiting_time + lineage["total_time"]
                )
                # Add the newly generated cell to the list of living lineages
                current_lineages.put(
                    (
                        birth_waiting_time + lineage["total_time"],
                        unique_id,
                        {
                            "id": unique_id,
                            "birth_scale": updated_birth_scale,
                            "total_time": birth_waiting_time
                            + lineage["total_time"],
                            "active": True,
                        },
                    )
                )

            else:
                tree.add_node(unique_id)
                tree.nodes[unique_id]["birth_scale"] = lineage["birth_scale"]
                tree.add_edge(lineage["id"], unique_id)
                tree.nodes[unique_id]["time"] = (
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

    def update_fitness(self, birth_scale: float) -> float:
        """Updates a lineage birth scale, which represents its fitness.

        At each division event, the fitness is updated by sampling from a
        distribution determining the number of mutations. The birth scale
        parameter of the lineage is then scaled by the total multiplicative
        coefficient across all mutations and passed on to the descendant
        nodes. The multiplicative factor of each mutation is determined by
        exponentiating a base parameter by a value drawn from another
        'fitness' distribution. Therefore, negative values from the fitness
        distribution are valid and down-scale the birth scale parameter.
        The base determines the base strength of the mutations in either
        direction and the fitness distribution determines how the mutations
        are distributed.

        Args:
            birth_scale: The birth_scale to be updated

        Returns:
            The updated birth_scale

        Raises:
            TreeSimulatorError if a negative number of mutations is sampled
        """
        base_selection_coefficient = 1
        if self.mutation_distribution:
            num_mutations = int(self.mutation_distribution())
            if num_mutations < 0:
                raise TreeSimulatorError(
                    "Negative number of mutations detected"
                )
            for _ in range(num_mutations):
                base_selection_coefficient *= (
                    self.fitness_base ** self.fitness_distribution()
                )
        return birth_scale * base_selection_coefficient
