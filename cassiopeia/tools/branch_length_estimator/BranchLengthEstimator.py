import abc

from cassiopeia.data import CassiopeiaTree


class BranchLengthEstimatorError(Exception):
    """An Exception class for the CassiopeiaTree class."""

    pass


class BranchLengthEstimator(abc.ABC):
    r"""
    Abstract base class for all branch length estimators.

    A BranchLengthEstimator implements a method estimate_branch_lengths which,
    given a Tree with lineage tracing character vectors at the leaves (and
    possibly at the internal nodes too), estimates the branch lengths of the
    tree.
    """

    @abc.abstractmethod
    def estimate_branch_lengths(self, tree: CassiopeiaTree) -> None:
        r"""
        Estimates the branch lengths of the tree.

        Annotates the tree's nodes with their estimated age, and
        the tree's branches with their estimated lengths. Operates on the tree
        in-place.

        Args:
            tree: The tree for which to estimate branch lengths.
        """
