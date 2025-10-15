"""Abstract DataSimulator subclass for overlaying lineage tracing data."""

import abc

from cassiopeia.data import CassiopeiaTree
from cassiopeia.simulator.DataSimulator import DataSimulator


class LineageTracingDataSimulator(DataSimulator):
    """LineageTracingDataSimulator is an abstract class that all lineage tracing

    data simulators derive from.

    A LineageTracingDataSimulator is useful for simulating lineage tracing
    assays in silico, allowing us to explore the utility of lineage tracing
    technologies such as base editors, GESTALT, etc. for recovering the ground
    truth cell phylogeny. In a typical simulation pipeline, a
    LineageTracingDataSimulator is used to overlay lineage tracing data on a
    CassiopeiaTree, and then a CassiopeiaSolver is used to reconstruct the tree
    topology.

    As a result, LineageTracingDataSimulators allow us to study the impact of
    different aspects of the lineage tracing assay - such as number of
    barcodes, mutation rates, etc. - on our ability to recover the ground
    truth phylogeny.
    """

    @abc.abstractmethod
    def overlay_data(self, tree: CassiopeiaTree) -> None:
        """Overlay lineage tracing data onto the CassiopeiaTree (in-place).

        This sets the character states of all nodes in the tree, as well
        as the character matrix. The tree is expected to have its topology
        initialized, as well as meaningful branch lengths.

        Args:
            tree: the CassiopeiaTree to overlay the lineage tracing data on.
                The tree topology must be initialized.
        """
