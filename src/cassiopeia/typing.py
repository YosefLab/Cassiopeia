from networkx import DiGraph
from treedata import TreeData

from cassiopeia.data import CassiopeiaTree

TreeLike = CassiopeiaTree | TreeData | DiGraph
