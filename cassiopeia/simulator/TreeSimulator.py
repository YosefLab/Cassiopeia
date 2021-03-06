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

    A TreeSimulator returns a CassiopeiaTree with at least its tree topology
    initialized. The character matrix need not be initialized (this is
    accomplished instead using a LineageTracingDataSimulator object). The
    branch lengths may be interpretable or not depending on the specific
    TreeSimulator.

    The purpose of the TreeSimulator is to allow users to perform in silico
    simulations of single-cell phylogenies, such as tumor phylogenies, organism
    development, etc., providing a ground truth phylogeny and thus a means to
    evaluate methodologies for reconstructing and analyzing single-cell
    phylogenies.
    """

    @abc.abstractmethod
    def simulate_tree(self) -> CassiopeiaTree:
        """
        Simulate a CassiopeiaTree.

        The returned tree will have at least its tree topology initialized.
        """
