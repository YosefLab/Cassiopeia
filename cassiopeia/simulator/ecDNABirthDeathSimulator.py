"""
This file stores a general phylogenetic tree simulator using forward birth-death
process, including differing fitness on lineages on the tree. Allows for a
variety of division and fitness regimes to be specified by the user.
"""
from random import random
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd

# from queue import PriorityQueue
import heapq

from cassiopeia.data.CassiopeiaTree import CassiopeiaTree
from cassiopeia.mixins import ecDNABirthDeathSimulatorError, TreeSimulatorError
from cassiopeia.simulator.BirthDeathFitnessSimulator import (
    BirthDeathFitnessSimulator,
)


class ecDNABirthDeathSimulator(BirthDeathFitnessSimulator):
    """Simulator class for a forward birth-death process with fitness in a population with ecDNA.

        "Implements a flexible phylogenetic tree simulator using a forward birth-
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
        case the simulation is run until one of the stopping conditions is reached."

    -(excerpted from from BirthDeathFitnessSimulator, update / check for accuracy)

    Example use snippet: (excerpted from from BirthDeathFitnessSimulator, update / check for accuracy)
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

    Args: (excerpted from from BirthDeathFitnessSimulator, update / check for accuracy)
        birth_waiting_distribution: A function that samples waiting times
            from the birth distribution. Determines how often births occur.
            Must take a scale parameter as the input
        initial_birth_scale: The initial scale parameter that is used at the
            start of the experiment
        death_waiting_distribution: A function that samples waiting times
            from the death distribution. Determines how often deaths occur. Default is no-death.
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
        initial_copy_number: Initial copy number for parental lineage.
        cosegregation_coefficient: A coefficient describing how likely it is for one species to be co-inherited
            with one specific species (currently modeled as the first in the array).
            TODO: how do we make this generalizable to multiple species each with different pairwise covariances?
        splitting_function: As implemented, the function that describes segregation of each species at cell division.
            TO DO: fix this implementation to allow for non-independent segregation.
        fitness_array: Fitnesses with respect to copy number of each species in a cell. This should be a matrix in
            R^e (where e is the number of ecDNA species being modelled).
        capture_efficiency: Probability of observing an ecDNA species. Used as the the probability of a binomial process.

    Raises: (excerpted from from BirthDeathFitnessSimulator, update / check for accuracy)
        TreeSimulatorError if invalid stopping conditions are provided or if a
        fitness distribution is not provided when a mutation distribution isn't
    """

    # update for ecDNA-specific initial conditions / simulation parameters.
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
        initial_copy_number: np.array = np.array([1]),
        cosegregation_coefficient: float = 0.0,
        splitting_function: Callable[[int], int] = lambda c, x: c
        + np.random.binomial(x, p=0.5),
        fitness_array: np.array = np.array([0, 1]),
        capture_efficiency: float = 1.0,
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
        self.initial_copy_number = initial_copy_number
        self.cosegregation_coefficient = cosegregation_coefficient
        self.splitting_function = splitting_function
        self.fitness_array = fitness_array
        self.capture_efficiency = capture_efficiency

    # update to store cn_array in node. (tree.nodes[root]["cn_array"])
    def initialize_tree(self, names) -> nx.DiGraph:
        """initializes a tree (nx.DiGraph() object with one node)"""
        tree = nx.DiGraph()
        root = next(names)
        tree.add_node(root)
        tree.nodes[root]["birth_scale"] = self.initial_birth_scale
        tree.nodes[root]["time"] = 0
        tree.nodes[root]["ecdna_array"] = self.initial_copy_number

        return tree, root

    # update to compute fitness using lineage cn_array
    def update_fitness(self, ecdna_array: np.array) -> float:
        """Updates a lineage birth scale, which represents its (Malthusian) fitness.

        Fitness is computed as a function of copy number, using the fitness_array
        (which defines fitness for CN=0 or CN >0 for each species, with epistasis).

        Args:
            ecdna_array: The birth_scale to be updated

        Returns:
            The updated birth_scale

        Raises:
            TreeSimulatorError if a negative number of mutations is sampled
        """

        return self.initial_birth_scale * (
            1.0 + self.fitness_array[tuple((ecdna_array > 0).astype(int))]
        )

    def sample_lineage_event(
        self,
        lineage: Dict[str, Union[int, float]],
        # current_lineages: PriorityQueue,
        current_lineages: List[Tuple[int, Any]],
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

        # TO DO: this is a really hacky fix b/c it bypasses the length checks of whether
        # the first birth_waiting_time exceeds self.experiment_time.
        # Also, it just assumes the first event is a birth.  we could also WOLOG that the
        # first birth_waiting_time of the experiment is 0
        # (but that requires shifting times elsewhere in order to permit correct model
        # comparison to non-ecDNA simulators).
        if lineage["total_time"] == 0:
            # Update birth rate
            updated_birth_scale = self.update_fitness(
                tree.nodes[lineage["id"]]["ecdna_array"]
            )

            # Annotate parameters for a given node in the tree
            tree.add_node(unique_id)
            tree.nodes[unique_id]["birth_scale"] = updated_birth_scale
            tree.add_edge(lineage["id"], unique_id)
            tree.nodes[unique_id]["time"] = (
                birth_waiting_time + lineage["total_time"]
            )
            tree.nodes[unique_id]["ecdna_array"] = tree.nodes[lineage["id"]][
                "ecdna_array"
            ]  # child_ecdna_array

            # Add the newly generated cell to the list of living lineages
            heapq.heappush(
                current_lineages,
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
                ),
            )
            # current_lineages.put(
            #     (
            #         birth_waiting_time + lineage["total_time"],
            #         unique_id,
            #         {
            #             "id": unique_id,
            #             "birth_scale": updated_birth_scale,
            #             "total_time": birth_waiting_time
            #             + lineage["total_time"],
            #             "active": True,
            #         },
            #     )
            # )
            return

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
            tree.nodes[unique_id]["ecdna_array"] = tree.nodes[lineage["id"]][
                "ecdna_array"
            ]
            heapq.heappush(
                current_lineages,
                (
                    self.experiment_time,
                    unique_id,
                    {
                        "id": unique_id,
                        "birth_scale": lineage["birth_scale"],
                        "total_time": self.experiment_time,
                        "active": False,
                    },
                ),
            )
            # current_lineages.put(
            #     (
            #         self.experiment_time,
            #         unique_id,
            #         {
            #             "id": unique_id,
            #             "birth_scale": lineage["birth_scale"],
            #             "total_time": self.experiment_time,
            #             "active": False,
            #         },
            #     )
            # )

            # Indicate this node is observed at the end of experiment
            observed_nodes.append(unique_id)

        else:
            if birth_waiting_time < death_waiting_time:
                # Update birth rate
                updated_birth_scale = self.update_fitness(
                    tree.nodes[lineage["id"]]["ecdna_array"]
                )

                child_ecdna_array = self.get_ecdna_array(lineage["id"], tree)

                # Annotate parameters for a given node in the tree
                tree.add_node(unique_id)
                tree.nodes[unique_id]["birth_scale"] = updated_birth_scale
                tree.add_edge(lineage["id"], unique_id)
                tree.nodes[unique_id]["time"] = (
                    birth_waiting_time + lineage["total_time"]
                )
                tree.nodes[unique_id]["ecdna_array"] = child_ecdna_array
                # Add the newly generated cell to the list of living lineages
                heapq.heappush(
                    current_lineages,
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
                    ),
                )
                # current_lineages.put(
                #     (
                #         birth_waiting_time + lineage["total_time"],
                #         unique_id,
                #         {
                #             "id": unique_id,
                #             "birth_scale": updated_birth_scale,
                #             "total_time": birth_waiting_time
                #             + lineage["total_time"],
                #             "active": True,
                #         },
                #     )
                # )

            else:
                tree.add_node(unique_id)
                tree.nodes[unique_id]["birth_scale"] = lineage["birth_scale"]
                tree.add_edge(lineage["id"], unique_id)
                tree.nodes[unique_id]["time"] = (
                    death_waiting_time + lineage["total_time"]
                )
                tree.nodes[unique_id]["ecdna_array"] = tree.nodes[
                    lineage["id"]
                ]["ecdna_array"]
                heapq.heappush(
                    current_lineages,
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
                    ),
                )
                # current_lineages.put(
                #     (
                #         death_waiting_time + lineage["total_time"],
                #         unique_id,
                #         {
                #             "id": unique_id,
                #             "birth_scale": lineage["birth_scale"],
                #             "total_time": death_waiting_time
                #             + lineage["total_time"],
                #             "active": False,
                #         },
                #     )
                # )

    def get_ecdna_array(self, parent_id: str, tree: nx.DiGraph) -> np.array:
        """Generates an ecDNA array for a child given its parent and sister information.

        Args:
            parent_id: ID of parent in the generated tree.
            tree: The in-progress tree.

        Returns:
            Numpy array corresponding to the ecDNA copy numbers for the new child.
        """

        parental_ecdna_array = 2 * tree.nodes[parent_id]["ecdna_array"]

        has_child = tree.out_degree(parent_id) > 0

        new_ecdna_array = parental_ecdna_array.copy()

        if has_child:
            child_id = [n for n in tree.successors(parent_id)][0]
            child_ecdna_array = tree.nodes[child_id]["ecdna_array"]

            new_ecdna_array = parental_ecdna_array - child_ecdna_array
        else:

            new_ecdna_array = np.array([0] * len(parental_ecdna_array))

            new_ecdna_array[0] = self.splitting_function(
                0, parental_ecdna_array[0]
            )

            for species in range(1, len(parental_ecdna_array)):

                cosegregating_compartment = int(
                    self.cosegregation_coefficient
                    * (new_ecdna_array[0] / max(1, parental_ecdna_array[0]))
                    * parental_ecdna_array[species]
                )
                sister_cell_cosegregating = int(
                    self.cosegregation_coefficient
                    * (
                        (parental_ecdna_array[0] - new_ecdna_array[0])
                        / max(1, parental_ecdna_array[0])
                    )
                    * parental_ecdna_array[species]
                )

                random_compartment = (
                    parental_ecdna_array[species]
                    - cosegregating_compartment
                    - sister_cell_cosegregating
                )

                inherited_fraction = self.splitting_function(
                    cosegregating_compartment,
                    random_compartment,
                )

                new_ecdna_array[species] = inherited_fraction

        # check that new ecdnay array entries do not exceed parental entries
        if np.any(new_ecdna_array > parental_ecdna_array):
            raise ecDNABirthDeathSimulatorError(
                "Child ecDNA entries exceed parental entries."
            )

        return new_ecdna_array

    def populate_tree_from_simulation(
        self, tree: nx.DiGraph, observed_nodes: List[str]
    ) -> CassiopeiaTree:
        """Populates tree with appropriate meta data.

        Args:
            tree: The tree simulated with ecDNA and fitness values populated as attributes.
            observed_nodes: The observed leaves of the tree.

        Returns:
            A CassiopeiaTree with relevant node attributes filled in.
        """

        cassiopeia_tree = CassiopeiaTree(tree=tree)

        # Prune dead lineages and collapse resulting unifurcations
        to_remove = list(set(cassiopeia_tree.leaves) - set(observed_nodes))
        cassiopeia_tree.remove_leaves_and_prune_lineages(to_remove)
        if self.collapse_unifurcations and len(cassiopeia_tree.nodes) > 1:
            cassiopeia_tree.collapse_unifurcations(source="1")

        # If only implicit root remains after pruning dead lineages, error
        if len(cassiopeia_tree.nodes) == 1:
            raise ecDNABirthDeathSimulatorError(
                "All lineages died before stopping condition"
            )

        time_dictionary = {}
        for node in tree.nodes:
            time_dictionary[node] = tree.nodes[node]["time"]

            cassiopeia_tree.set_attribute(
                node, "ecdna_array", tree.nodes[node]["ecdna_array"]
            )
            cassiopeia_tree.set_attribute(
                node, "fitness", tree.nodes[node]["birth_scale"]
            )
        cassiopeia_tree.set_times(time_dictionary)

        leaf_ecdna_arrays = [
            tree.nodes[node]["ecdna_array"] for node in cassiopeia_tree.leaves
        ]
        cell_metadata = pd.DataFrame(
            leaf_ecdna_arrays,
            columns=[
                f"ecDNA_{i}" for i in range(len(self.initial_copy_number))
            ],
            index=cassiopeia_tree.leaves,
        )

        # apply noise model
        for i in range(len(self.initial_copy_number)):
            cell_metadata[f"Observed_ecDNA_{i}"] = cell_metadata.apply(
                lambda x: np.random.binomial(
                    x[f"ecDNA_{i}"], self.capture_efficiency
                ),
                axis=1,
            )

        cassiopeia_tree.cell_meta = cell_metadata.astype(int).copy()

        return cassiopeia_tree
