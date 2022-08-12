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
from cassiopeia.mixins import TreeSimulatorError
# from cassiopeia.simulator.TreeSimulator import TreeSimulator
from cassiopeia.simulator.BirthDeathFitnessSimulator import BirthDeathFitnessSimulator

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
        initial_CN: np.array = np.array([1]),
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
        self.initial_CN = initial_CN

    # update to store cn_array in node. (tree.nodes[root]["cn_array"])
    def initialize_tree(
            self, 
            names
        ) -> nx.DiGraph:
            ''' initializes a tree (nx.DiGraph() object with one node)
            '''
            tree = nx.DiGraph()
            root = next(names)
            tree.add_node(root)
            tree.nodes[root]["birth_scale"] = self.initial_birth_scale
            tree.nodes[root]["time"] = 0
            tree.nodes[root]["cn_array"] = self.initial_CN
            
            return tree, root

    # update to include self.initial_CN_array 
    def make_initial_lineage_dict(
        self,
        id_value
    ):
        lineage = self.make_lineage_dict(id_value, self.initial_birth_scale, 0, True, self.initial_CN)
        return lineage
    

    # update to compute fitness using lineage cn_array 
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

    # update to handle CN_array of self (in lineage and tree) and daughters (in lineage only) (see comments within function)
    def make_daughter(
        self,
        unique_id, 
        birth_scale,
        updated_time, 
        active_flag,
        lineage: Dict[str, Union[int, float]],
        current_lineages: PriorityQueue,
        tree: nx.DiGraph,
        daughter_ind: int 
    ):
        """Updates the tree and current_lineages with a new daughter.

        Tree has a new  node unique_id, with data  
        """
        # 1. compute any values to be updated by grabbing current values from the lineage Dict and performing operations on them as needed. 
        # (Grab my own CN array from the lineage)


        # 2. add daughter node to tree with appropriate parameters (my own CN array will go here into the new node)
        tree.add_node(unique_id)
        tree.nodes[unique_id]["birth_scale"] = birth_scale
        tree.add_edge(lineage["id"], unique_id)
        tree.nodes[unique_id]["time"] = (
            updated_time
        )

        # 3. put new lineage into current_lineages (CNs of my daughters will go here as cn_array_daughter_1 and cn_array_daughter_2) 
        current_lineages.put(
            (
                updated_time,
                unique_id,
                self.make_lineage_dict(
                unique_id, 
                birth_scale,
                updated_time,
                active_flag,
                ),
            )
        )


    #  should this be a static method?
    # update to include ecDNA copy number 
    @staticmethod
    def make_lineage_dict(
        id_value, 
        birth_scale,
        total_time,
        active_flag,
        cn_array,
    ):
        lineage = {
            "id": id_value,
            "birth_scale": birth_scale,
            "total_time": total_time,
            "active": active_flag,
            "cn_array":cn_array,
        }
        return lineage
    