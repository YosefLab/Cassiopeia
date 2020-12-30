"""
This file stores the basic data structure for Cassiopeia - the
CassiopeiaTree. At a minimum, this data structure will contain a character
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
from typing import Dict, List, Optional, Union

from cassiopeia.data import utilities


class CassiopeiaTree:
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
        self.__network = None

        if isinstance(tree, nx.DiGraph):
            self.__network = tree
        if isinstance(tree, str):
            self.__network = utilities.newick_to_networkx(tree)
        if isinstance(tree, ete3.Tree):
            self.__network = utilities.ete3_to_networkx(tree)

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

    @property
    def root(self) -> str:
        """Returns root of tree.
        """
        if self.__network is None:
            return None
        return [n for n in self.__network if self.__network.in_degree(n) == 0][
            0
        ]

    @property
    def leaves(self) -> List[str]:
        """Returns leaves of tree.
        """
        if self.__network is None:
            return None
        return [n for n in self.__network if self.__network.out_degree(n) == 0]

    @property
    def internal_nodes(self) -> List[str]:
        """Returns internal nodes in tree.
        """
        if self.__network is None:
            return None
        return [
            n
            for n in self.__network
            if self.__network.out_degree(n) > 1
            and self.__network.in_degree(n) > 0
        ]

    @property
    def nodes(self) -> List[str]:
        """Retruns all nodes in tree.
        """
        if self.__network is None:
            return None
        return [n for n in self.__network]

    def reconstruct_ancestral_characters(self):
        """Reconstruct ancestral character states
        """
        pass

    def children(self, node: str):
        """Gets the children of a given node.
        """
        pass

    def set_age(self, node: str, age: float) -> None:
        """Sets the age of a node.
        """
        pass

    def get_age(self, node: str) -> float:
        """Gets the age of a node.
        """
        pass

    def set_branch_length(self, parent: str, child: str, length: float) -> None:
        """Sets the length of a branch.
        """
        pass

    def get_branch_length(self, parent: str, child: str) -> float:
        """Gets the length of a branch.
        """
        pass

    def set_state(self, node: str, character: int, state: int) -> None:
        """Sets the state of a single character for a node.
        """
        pass

    def set_states(self, node: str, states: List[int]) -> None:
        """Sets all the states for a particular node.
        """
        pass

    def get_state(self, node: str, character: int) -> int:
        """Gets the state of a single character for a particular node.
        """
        pass

    def get_states(self, node: str) -> List[int]:
        """Gets all the character states for a particular node.
        """
        pass

    def depth_first_traverse_nodes(self, postorder: bool = True):
        """Depth first traversal of the tree.

        Returns the nodes from a DFS on the tree.

        Args:
            postorder: Return the nodes in postorder. If False, returns in 
                preorder.
        """
        pass

    def leaves_in_subtree(self, node) -> List[str]:
        """Get leaves in subtree below a given node.
        """
        pass

    def get_newick(self) -> str:
        """Returns newick format of tree.
        """
        return utilities.to_newick(self.__network)

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
