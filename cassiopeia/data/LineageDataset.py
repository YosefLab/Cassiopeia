"""
This file stores the basic data structure for Cassiopeia - the
LineageDataset. At a minimum, this data structure will contain a character
matrix containing that character state information for all the cells in a given
clonal population. Other important data is also stored here, like the priors
for given character states as well any meta data associated with this clonal 
population.

When a solver has been called on this object, a tree 
will be added to the data structure at which point basic properties can be 
queried like the average tree depth or agreement between character states and
phylogeny.

This object can be passed to any CassiopeiaSolver subclass as well as any
analysis module, like a branch length estimator or rate matrix estimator
"""
import ete3
import networkx as nx
import pandas as pd
from typing import Dict, Optional, Union


class LineageDataset:

    def __init__(
        self,
        character_matrix: pd.DataFrame,
        cell_meta: Optional[pd.DataFrame] = None,
        character_meta: Optional[pd.DataFrame] = None,
        priors: Optional[Dict[int, Dict[int, float]]] = None,
        tree: Optional[Union[str, ete3.Tree, nx.DiGraph]] = None,
    ):
        
        self.character_matrix = character_matrix
        self.cell_meta = cell_meta
        self.character_meta = character_meta
        self.priors = priors
        self.tree = tree

    @property
    def n_cells(self) -> int:
        """Returns number of cells in character matrix.
        """
        return self.character_matrix.shape[0]

    @property
    def n_character(self) -> int:
        """Returns number of characters in character matrix.
        """
        return self.character_matrix.shape[1]

    def get_newick(self) -> str:
        """Returns newick format of tree.
        """
        pass

    def get_mean_depth_of_tree(self) -> float:
        """Computes mean depth of tree.
        """
        pass

    def cophenetic_correlation(self) -> float:
        """Computes cophenetic correlation.

        Compares the character-state distances with the tree tree distances
        to compute the cophenetic correlation as a measure of agreement between
        the two distances.

        Returns
            The cophenetic correlation
        """
        pass

