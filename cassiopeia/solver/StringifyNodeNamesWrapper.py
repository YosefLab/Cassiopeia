import copy
from cassiopeia.data import CassiopeiaTree
from cassiopeia.solver import CassiopeiaSolver


# https://stackoverflow.com/questions/1443129/completely-wrap-an-object-in-python
# permalink: https://stackoverflow.com/a/1445289
class StringifyNodeNamesWrapper(CassiopeiaSolver.CassiopeiaSolver):
    r"""
    Wraps a CassiopeiaSolver.
    The wrapped solver is used to solve for the tree topology, after which
    the node names are cast to string. This is because some solvers create
    nodes with integer IDs, and this can break downstream code.
    """
    def __init__(self, solver: CassiopeiaSolver.CassiopeiaSolver):
        solver = copy.deepcopy(solver)
        self.__class__ = type(
            solver.__class__.__name__,
            (self.__class__, solver.__class__),
            {},
        )
        self.__dict__ = solver.__dict__
        self.__solver = solver

    def solve(self, tree: CassiopeiaTree) -> None:
        self.__solver.solve(tree)
        relabel_map = {node: 'internal-' + str(node) for node in
                       tree.internal_nodes}
        num_nodes_before = len(tree.nodes)
        tree.relabel_nodes(relabel_map)
        num_nodes_after = len(tree.nodes)
        if num_nodes_before != num_nodes_after:
            raise RuntimeError("There was a colision stringifying node names.")
