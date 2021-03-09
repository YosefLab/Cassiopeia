import copy
from cassiopeia.data import CassiopeiaTree, resolve_multifurcations
from cassiopeia.solver import CassiopeiaSolver


# https://stackoverflow.com/questions/1443129/completely-wrap-an-object-in-python
# permalink: https://stackoverflow.com/a/1445289
class ResolveMultifurcationsWrapper(CassiopeiaSolver.CassiopeiaSolver):
    r"""
    Wraps a CassiopeiaSolver.
    The wrapped solver is used to solve for the tree topology, after which
    the multifurcations are resolved.
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
        resolve_multifurcations(tree)
