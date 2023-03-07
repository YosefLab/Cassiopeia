"""
Abstract class FitnessEstimator, for the fitness estimation module.

All algorithms are derived classes of this abstract class, and at a minimum
implement a method called `estimate_fitness`. Fitness will be stored as
the attribute 'fitness' of each node.
"""
import abc

from cassiopeia.data import CassiopeiaTree


class FitnessEstimatorError(Exception):
    """An Exception class for the FitnessEstimator class."""

    pass


class FitnessEstimator(abc.ABC):
    """
    FitnessEstimator is an abstract class that all fitness
    estimation algorithms derive from. At minimum, all FitnessEstimator
    subclasses will implement a method called `estimate_fitness`.
    Fitness will be stored as the attribute 'fitness' of each node.
    """

    @abc.abstractmethod
    def estimate_fitness(self, tree: CassiopeiaTree) -> None:
        """Estimates fitness for each node in the tree.

        Fitness will be stored as the attribute 'fitness' of each node.

        Args:
            cassiopeia_tree: CassiopeiaTree storing an initialized
            tree topology with estimated branch lengths.
        """
