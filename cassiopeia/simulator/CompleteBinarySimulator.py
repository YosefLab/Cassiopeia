"""
This file defines the CompleteBinarySimulator, which inherits TreeSimulator,
that simulates complte binary trees. In this sense, this is the simplest tree
simulator.
"""
from typing import Generator, Optional

import networkx as nx
import numpy as np

from cassiopeia.data.CassiopeiaTree import CassiopeiaTree
from cassiopeia.simulator.TreeSimulator import TreeSimulator, TreeSimulatorError


class CompleteBinarySimulator(TreeSimulator):
    """Simulate a complete binary tree.

    Internally, this class uses :func:`nx.balanced_tree` to generate a
    perfectly balanced binary tree of specified size. Only one of ``num_cells``
    or ``depth`` should be provided.

    Args:
        num_cells: Number of cells to simulate. Needs to be a power of 2. The
            depth of the tree will be `log2(num_cells)`.
        depth: Depth of the tree. The number of cells will be `2^depth`.

    Raises:
        TreeSimulatorError if neither or both ``num_cells`` or ``depth`` are
            provided, if ``num_cells`` is not a power of 2, or if the calculated
            depth is not greater than 0.
    """

    def __init__(
        self,
        num_cells: Optional[int] = None,
        depth: Optional[int] = None,
    ):
        if (num_cells is None) == (depth is None):
            raise TreeSimulatorError(
                "One of `num_cells` or `depth` must be provided."
            )
        if num_cells is not None:
            log2_num_cells = np.log2(num_cells)
            if log2_num_cells != int(log2_num_cells):
                raise TreeSimulatorError("`num_cells` must be a power of 2.")
            depth = int(log2_num_cells)
        if depth <= 0:
            raise TreeSimulatorError("`depth` must be grater than 0.")
        self.depth = depth

    def simulate_tree(
        self,
    ) -> CassiopeiaTree:
        """Simulates a complete binary tree.

        Returns:
            A CassiopeiaTree with the tree topology initialized with the
            simulated tree
        """

        def node_name_generator() -> Generator[str, None, None]:
            """Generates unique node names for the tree."""
            i = 0
            while True:
                yield str(i)
                i += 1

        names = node_name_generator()

        tree = nx.balanced_tree(2, self.depth, create_using=nx.DiGraph)
        mapping = {"root": next(names)}
        mapping.update({node: next(names) for node in tree.nodes})
        # Add root, which indicates the initiating cell
        tree.add_edge("root", 0)
        nx.relabel_nodes(tree, mapping, copy=False)
        return CassiopeiaTree(tree=tree)
