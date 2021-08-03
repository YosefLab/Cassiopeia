"""
Abstract class BranchLengthEstimator, for the branch length estimation module.

All algorithms are derived classes of this abstract class, and at a minimum
implement a method called `estimate_branch_lengths`.
"""
import abc

from cassiopeia.data import CassiopeiaTree


class BranchLengthEstimator(abc.ABC):
    """
    BranchLengthEstimator is an abstract class that all branch length
    estimation algorithms derive from. At minimum, all BranchLengthEstimator
    subclasses will implement a method called `estimate_branch_lengths`.
    """

    @abc.abstractmethod
    def estimate_branch_lengths(self, tree: CassiopeiaTree) -> None:
        """Estimates branch lengths for the given tree.

        Args:
            cassiopeia_tree: CassiopeiaTree storing character information for
                phylogenetic inference; the tree topology must have been
                initialized (for example by means of a CassiopeiaSolver).
        """
        pass
