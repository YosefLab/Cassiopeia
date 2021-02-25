"""
Abstract class DataSimulator, for overlaying data onto a CassiopeiaTree.

All data simulators are derived classes of this abstract class, and at a minimum
implement a method called `overlay_data`.
"""
import abc
from cassiopeia.data import CassiopeiaTree


class DataSimulator(abc.ABC):
    """
    DataSimulator is an abstract class that all data overlayers derive from.
    """

    @abc.abstractmethod
    def overlay_data(self, tree: CassiopeiaTree) -> None:
        """Overlay data on a CassiopeiaTree (in-place)"""
        pass
