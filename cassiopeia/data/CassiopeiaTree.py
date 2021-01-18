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
from typing import Dict, Iterator, List, Optional, Tuple, Union

from cassiopeia.data import utilities


class CassiopeiaTreeError(Exception):
    """An Exception class for the CassiopeiaTree class."""
    pass

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
        character_matrix: pd.DataFrame,
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

        self.populate_tree(tree)

    def populate_tree(self, tree: Union[str, ete3.Tree, nx.DiGraph]):

        if isinstance(tree, nx.DiGraph):
            self.__network = tree
        if isinstance(tree, str):
            self.__network = utilities.newick_to_networkx(tree)
        if isinstance(tree, ete3.Tree):
            self.__network = utilities.ete3_to_networkx(tree)

        if self.__network is not None:

            # add character states
            for n in self.nodes:
                if n in self.character_matrix.index.values:
                    self.__network.nodes[n]['character_states'] = self.character_matrix.loc[n].to_list()
                else:
                    self.__network.nodes[n]['character_states'] = []
            
            # instantiate branch lengths
            for u,v in self.edges:
                self.__network[u][v]['length'] = 1

            # instantiate node ages and edge depth
            self.__network.nodes[self.root]['age'] = 0
            self.__network.nodes[self.root]['edge_depth'] = 0
            for u,v in self.depth_first_traverse_edges(source=self.root):
                self.__network.nodes[v]['age'] = self.__network.nodes[u]['age'] + self.__network[u][v]['length']
                self.__network.nodes[v]['edge_depth'] = self.__network.nodes[v]['age']

    @property
    def n_cell(self) -> int:
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

        Returns:
            The root.
        
        Raises:
            CassiopeiaTreeError if the tree has not been initialized.
        """
        if self.__network is None:
            raise CassiopeiaTreeError("Tree is not initialized.")
        return [n for n in self.__network if self.__network.in_degree(n) == 0][
            0
        ]

    @property
    def leaves(self) -> List[str]:
        """Returns leaves of tree.

        Returns:
            The leaves of the tree.
        
        Raises:
            CassiopeiaTreeError if the tree has not been initialized.
        """
        if self.__network is None:
            raise CassiopeiaTreeError("Tree is not initialized.")
        return [n for n in self.__network if self.__network.out_degree(n) == 0]

    @property
    def internal_nodes(self) -> List[str]:
        """Returns internal nodes in tree (including the root).

        Returns:
            The internal nodes of the tree (i.e. all nodes not at the leaves)
        
        Raises:
            CassiopeiaTreeError if the tree has not been initialized.
        """
        if self.__network is None:
            raise CassiopeiaTreeError("Tree is not initialized.")
        return [
            n
            for n in self.__network
            if self.__network.out_degree(n) > 1
        ]

    @property
    def nodes(self) -> List[str]:
        """Returns all nodes in tree.

        Returns:
            All nodes of the tree (internal + leaves)
        
        Raises:
            CassiopeiaTreeError if the tree has not been initialized.
        """
        if self.__network is None:
            raise CassiopeiaTreeError("Tree is not initialized.")
        return [n for n in self.__network]

    @property
    def edges(self) -> List[Tuple[str, str]]:
        """Returns all edges in the tree.

        Returns:
            All edges of the tree.
        
        Raises:
            CassiopeiaTreeError if the tree has not been initialized.
        """
        if self.__network is None:
            raise CassiopeiaTreeError("Tree is not initialized.")
        return [(u, v) for (u,v) in self.__network.edges]

    def is_leaf(self, node: str) -> bool:
        """Returns whether or not the node is a leaf.

        Returns:
            Whether or not the node is a leaf.
        
        Raises:
            CassiopeiaTreeError if the tree has not been initialized.
        """
        if self.__network is None:
            raise CassiopeiaTreeError("Tree is not initialized.")
        return (self.__network.out_degree(node) == 0)

    def is_root(self, node: str) -> bool:
        """Returns whether or not the node is the root.

        Returns:
            Whether or not the node is the root.
        
        Raises:
            CassiopeiaTreeError if the tree has not been initialized.
        """
        if self.__network is None:
            raise CassiopeiaTreeError("Tree is not initialized.")
        return (node == self.root)

    def reconstruct_ancestral_characters(self):
        """Reconstruct ancestral character states.

        Reconstructs ancestral states (i.e., those character states in the
        internal nodes) using the Camin-Sokal parsimony criterion (i.e.,
        irreversibility). Operates on the tree in place.
        """
        pass

    def children(self, node: str) -> List[str]:
        """Gets the children of a given node.

        Args:
            node: A node in the tree.

        Returns:
            A list of nodes that are direct children of the input node.
        """
        return [v for v in self.__network.successors(node)]

    def set_age(self, node: str, age: float) -> None:
        """Sets the age of a node.
        """
        pass

    def get_age(self, node: str) -> float:
        """Gets the age of a node.
        """
        pass

    def set_branch_length(self, parent: str, child: str, length: float):
        """Sets the length of a branch.
        """
        pass

    def get_branch_length(self, parent: str, child: str) -> float:
        """Gets the length of a branch.
        """
        pass

    def set_state(self, node: str, character: int, state: int):
        """Sets the state of a single character for a node.
        """
        self.__network.nodes[node]['character_states'][character] = state

    def set_states(self, node: str, states: List[int]):
        """Sets all the states for a particular node.

        Args:
            node: Node in the tree
            states: A list of states to add to the node.

        Raises:
            CassiopeiaTreeError if the character vector is the incorrect length.
        """
        if len(states) != self.n_character:
            raise CassiopeiaTreeError("Input character vector is not the right length.")

        self.__network.nodes[node]['character_states'] = states

    def get_state(self, node: str, character: int) -> int:
        """Gets the state of a single character for a particular node.

        Args:
            node: Node in the tree.
            character: Index of character, 0 indexed.

        Returns:
            The character state at the specified position.
        """
        return self.__network.nodes[node]['character_states'][character]

    def get_states(self, node: str) -> List[int]:
        """Gets all the character states for a particular node.

        Args:
            node: Node in the tree.

        Returns:
            The full character state array of the specified node.
        """
        return self.__network.nodes[node]['character_states']

    def depth_first_traverse_nodes(self, source: Optional[int] = None, postorder: bool = True) -> Iterator[str]:
        """Nodes from depth first traversal of the tree.

        Returns the nodes from a DFS on the tree.

        Args:
            source: Where to begin the depth first traversal.
            postorder: Return the nodes in postorder. If False, returns in 
                preorder.

        Returns:
            A list of nodes from the depth first traversal.
        """

        if source is None:
            source = self.root

        if postorder:
            return nx.dfs_postorder_nodes(self.__network, source=source)
        else:
            return nx.dfs_preorder_nodes(self.__network, source=source)

    def depth_first_traverse_edges(self, source: Optional[int] = None) -> Iterator[Tuple[str, str]]:
        """Edges from depth first traversal of the tree.

        Returns the edges from a DFS on the tree.

        Args:
            source: Where to begin the depth first traversal.

        Returns:
            A list of edges from the depth first traversal.
        """

        if source is None:
            source = self.root

        return nx.dfs_edges(self.__network, source=source)

    def leaves_in_subtree(self, node) -> List[str]:
        """Get leaves in subtree below a given node.

        Args:
            node: Root of the subtree.

        Returns:
            A list of the leaves in the subtree rooted at the specified node.
        """
        
        return [n for n in self.depth_first_traverse_nodes(source = node) if self.__network.out_degree(n) == 0]

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
