from typing import Callable
import dendropy
import numpy as np
import ete3
import spectraltree
from typing import Callable

from cassiopeia.data import CassiopeiaTree
from cassiopeia.data import utilities as data_utilities
from cassiopeia.solver import CassiopeiaSolver

class STDRSolver(CassiopeiaSolver.CassiopeiaSolver):

    def __init__(
        self,
        similarity_function: Callable[[np.array], np.array],
        threshold: int=64,
        min_split: int=1,
        merge_method: str="least_square",
        verbose: bool = False
    ):

        self.similarity_function = similarity_function
        self.threshold = threshold
        self.min_split = min_split
        self.merge_method = merge_method
        self.verbose = verbose

    def solve(self, cassiopeia_tree: CassiopeiaTree) -> None:
        character_matrix = cassiopeia_tree.get_current_character_matrix()
        taxon_namespace = dendropy.TaxonNamespace(list(character_matrix.index))
        metadata = spectraltree.utils.TaxaMetadata(taxon_namespace, list(character_matrix.index), alphabet=None)

        stdr_nj = spectraltree.STDR(spectraltree.NeighborJoining, self.similarity_function)   

        tree_stdr_nj = stdr_nj(character_matrix.values, 
                taxa_metadata= metadata, 
                threshold = self.threshold,
                min_split = self.min_split,
                merge_method = self.merge_method, 
                verbose=self.verbose)
        newick_str = tree_stdr_nj.as_string(schema="newick")
        cassiopeia_tree.populate_tree(newick_str)