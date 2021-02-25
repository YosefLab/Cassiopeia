"""
This file stores an abstract subclass of DataSimulator, the
LineageTracingDataSimulator. A LineageTracingDataSimulator overlays lineage
tracing data onto a CassiopeiaTree, i.e. it sets the character states of a
CassiopeiaTree.
"""
import abc
from cassiopeia.data import CassiopeiaTree
from cassiopeia.simulator.DataSimulator import DataSimulator


class LineageTracingDataSimulator(DataSimulator):
    """
    LineageTracingDataSimulator is an abstract class that all lineage tracing
    data simulators derive from.
    """

    @abc.abstractmethod
    def overlay_data(self, tree: CassiopeiaTree) -> None:
        """
        Overlay lineage tracing data onto the CassiopeiaTree (in-place).
        This sets the character states of all nodes in the tree.
        """
        pass
