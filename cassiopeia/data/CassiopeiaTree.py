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
    """Basic tree object for Cassiopeia.

    This object stores the key attributes and functionalities a user might want
    for working with lineage tracing experiments. At its core, it stores
    three main items - a tree, a character matrix, and meta data associated
    with the data.

    The tree can be fed into the object via Ete3, Networkx, or can be inferred
    using one of the CassiopeiaSolver algorithms in the `solver` module.

    A character matrix can be stored in the object, containing the states
    observed for each cell. In typical lineage tracing experiments, these are
    integer representations of the indels observed at each unique cut site.

    Meta data for cells or characters can also be stored in this object. These
    items can be categorical or numerical in nature. Common examples of cell
    meta data are the cluster identity, tissue identity, or number of target-site
    UMIs per cell. These items can be used in downstream analyses, for example
    the FitchCount algorithm which infers the number of transitions between
    categorical variables (e.g., tissues). Common examples of character meta
    data are the proportion of missing data for each character or the entropy
    of states. These are good statistics to have for feature selection.

    TODO(mattjones315): Add experimental meta data as arguments.
    TODO(mattjones315): Add functionality that mutates the underlying tree
        structure: pruning lineages, collapsing mutationless edges, and
        collapsing unifurcations
    TODO(mattjones315): Add utility methods to compute the colless index
        and the cophenetic correlation wrt to some cell meta item

    Args:
        character_matrix: The character matrix for the lineage.
        cell_meta: Per-cell meta data
        character_meta: Per-character meta data
        priors: A dictionary storing the probability of a character mutating
            to a particular state.
        tree: A tree for the lineage. 
    """

    def __init__(
        self,
        character_matrix: Optional[pd.DataFrame] = None,
        cell_meta: Optional[pd.DataFrame] = None,
        character_meta: Optional[pd.DataFrame] = None,
        priors: Optional[Dict[int, Dict[int, float]]] = None,
        tree: Optional[Union[str, ete3.Tree, nx.DiGraph]] = None
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
        """Returns all nodes in tree.
        """
        if self.__network is None:
            return None
        return [n for n in self.__network]

    @property
    def edges(self) -> List[Tuple[str, str]]:
        """Returns all edges in the tree.
        """
        if self.__network is None:
            return None
        return [(u, v) for (u,v) in self.__network.edges]

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

    def get_max_depth_of_tree(self) -> float:
        """Computes the max depth of the tree.
        """
        pass

    def get_mutations_along_edge(self, parent: str, child: str) -> List[Tuple[int, int]]:
        """Gets the mutations along an edge of interest.

        Returns a list of tuples (character, state) of mutations that occur
        along an edge.

        Args:
            parent: parent in tree
            child: child in tree

        Returns:
            A list of (character, state) tuples indicating which character
                mutated and to which state.
        """
        pass

    def relabel_nodes(self, relabel_map: Dict[str, str]):
        """Relabels the nodes in the tree.

        Renames the nodes in the tree according to the relabeling map. Modifies
        the tree inplace.

        Args:
            relabel_map: A mapping of old names to new names.
        """
        pass
