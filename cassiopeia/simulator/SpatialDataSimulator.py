"""
This file stores an abstract subclass of DataSimulator, the
SpatialDataSimulator. A SpatialDataSimulator overlays spatial data onto a
Cassiopeiatree, i.e. it sets the spatial coordinates of a CassiopeiaTree
(in particular, as attributes of the nodes of the tree and the cell meta).
"""
import abc

from cassiopeia.data import CassiopeiaTree
from cassiopeia.simulator.DataSimulator import DataSimulator


class SpatialDataSimulator(DataSimulator):
    """
    SpatialDataSimulator is an abstract class that all spatial data simulators
    derive from.

    A SpatialDataSimulator is useful for simulating spatial assays in silico.
    In a typical simulation pipeline, a SpatialDataSimulator is used to overlay
    spatial coordinates on a CassiopeiaTree, and then a CassiopeiaSolver is used
    to reconstruct the tree topology (to simulate single-cell-resolution spatial
    assays) or a SpatialLeafSubsampler is used (to simulate
    non-single-cell-resoultion spatial assays).
    """

    @abc.abstractmethod
    def overlay_data(self, tree: CassiopeiaTree) -> None:
        """
        Overlay spatial data onto the CassiopeiaTree (in-place).

        This sets the spatial coordinates of all nodes in the tree. These
        coordinates are stored as the `spatial` node attribute. For leaves,
        these exact coordinates are saved as columns in the `cell_meta` attribute
        of the CassiopeiaTree.

        Args:
            tree: the CassiopeiaTree to overlay the lineage tracing data on.
                The tree topology must be initialized.
        """
