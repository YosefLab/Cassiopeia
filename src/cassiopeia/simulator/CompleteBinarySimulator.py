"""Module defining the CompleteBinarySimulator, which inherits TreeSimulator.

The CompleteBinarySimulator simulates complete binary trees. In this sense, this is the simplest tree
simulator.
"""

from collections.abc import Generator

import networkx as nx
import numpy as np
import treedata as td

from cassiopeia.mixins import TreeSimulatorError
from cassiopeia.simulator.TreeSimulator import TreeSimulator


class CompleteBinarySimulator(TreeSimulator):
    """Simulate a complete binary tree.

    Internally, this class uses :func:`nx.balanced_tree` to generate a
    perfectly balanced binary tree of specified size. Only one of ``num_cells``
    or ``depth`` should be provided. All branches have equal length that is
    normalized by the height of the tree (i.e. the tree has height 1).

    Args:
        num_cells: Number of cells to simulate. Needs to be a power of 2. The
            depth of the tree will be `log2(num_cells)`.
        depth: Depth of the tree. The number of cells will be `2^depth`.

    Raises:
            TreeSimulatorError if neither or both ``num_cells`` or ``depth`` are
            provided, if ``num_cells`` is not a power of 2, or if the calculated
            depth is not greater than 0.
    """

    def __init__(self, num_cells: int | None = None, depth: int | None = None):
        if (num_cells is None) == (depth is None):
            raise TreeSimulatorError("One of `num_cells` or `depth` must be provided.")
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
        tree_key: str = "tree",
    ) -> td.TreeData:
        """Simulates a complete binary tree.

        Returns:
            A TreeData object with .obst[`tree_key`] containing the simulated tree.
        """

        def node_name_generator() -> Generator[str, None, None]:
            """Generates unique node names for the tree."""
            i = 1
            while True:
                yield str(i)
                i += 1

        names = node_name_generator()

        tree = nx.balanced_tree(2, self.depth, create_using=nx.DiGraph)
        tree.add_edge("root", 0)
        nx.relabel_nodes(
            tree, {node: next(names) for node in tree.nodes if node != "root"}, copy=False
        )
        depths = nx.single_source_shortest_path_length(tree, "root")
        nx.set_node_attributes(tree, depths, "depth")
        max_depth = max(depths.values())
        times = {node: depth / max_depth for node, depth in depths.items()}
        nx.set_node_attributes(tree, times, "time")

        return td.TreeData(obst={tree_key: tree})
