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

    A DataSimulator is very generic and meant to give users the flexibility to
    overlay any kind of data onto the tree using this single API. The prime
    example of data a user might want to overlay on a tree is lineage tracing
    data, for which there is a specific subclass LineageTracingDataSimulator.
    Other data of interest might include: transcriptomes, proteomes, etc.
    """

    @abc.abstractmethod
    def overlay_data(self, tree: CassiopeiaTree) -> None:
        """
        Overlay data on a CassiopeiaTree (in-place).

        The tree topology must be initialized.

        Args:
            tree: the CassiopeiaTree to overlay the data on. The tree topology
                must be initialized.
        """
        pass
