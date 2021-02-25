"""
Abstract class TreeSimulator, for tree simulation module.

All tree simulators are derived classes of this abstract class, and at a minimum
implement a method called `simulate_tree`.
"""
import abc
from cassiopeia.data import CassiopeiaTree


class TreeSimulator(abc.ABC):
    """
    TreeSimulator is an abstract class that all tree simulators derive from.
    """

    @abc.abstractmethod
    def simulate_tree(self) -> CassiopeiaTree:
        """Simulate a CassiopeiaTree"""
        pass
