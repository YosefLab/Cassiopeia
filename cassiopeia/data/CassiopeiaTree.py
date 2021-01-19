"""
This file stores the basic data structure for Cassiopeia - the
CassiopeiaTree. This data structure will typically contain a character
matrix containing that character state information for all the cells in a given
clonal population (though this is not required). Other important data is also
stored here, like the priors for given character states as well any meta data
associated with this clonal  population.

When a solver has been called on this object, a tree 
will be added to the data structure at which point basic properties can be 
queried like the average tree depth or agreement between character states and
phylogeny.

This object can be passed to any CassiopeiaSolver subclass as well as any
analysis module, like a branch length estimator or rate matrix estimator
"""
import ete3
import networkx as nx
import numpy as np
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
        missing_state_indicator: An indicator for missing states in the
            character matrix.
        cell_meta: Per-cell meta data
        character_meta: Per-character meta data
        priors: A dictionary storing the probability of a character mutating
            to a particular state.
        tree: A tree for the lineage. 
    """

    def __init__(
        self,
        character_matrix: Optional[pd.DataFrame] = None,
        missing_state_indicator: int = -1,
        cell_meta: Optional[pd.DataFrame] = None,
        character_meta: Optional[pd.DataFrame] = None,
        priors: Optional[Dict[int, Dict[int, float]]] = None,
        tree: Optional[Union[str, ete3.Tree, nx.DiGraph]] = None,
    ):

        self.__original_character_matrix = None
        self.__current_character_matrix = None
        if character_matrix is not None:
            self.__original_character_matrix = character_matrix.copy()
            self.__current_character_matrix = character_matrix.copy()
        
        self.missing_state_indicator = missing_state_indicator
        self.cell_meta = cell_meta
        self.character_meta = character_meta
        self.priors = priors
        self.__network = None

        if tree is not None:
            self.populate_tree(tree)

    def populate_tree(self, tree: Union[str, ete3.Tree, nx.DiGraph]):

        if isinstance(tree, nx.DiGraph):
            self.__network = tree
        elif isinstance(tree, str):
            self.__network = utilities.newick_to_networkx(tree)
        elif isinstance(tree, ete3.Tree):
            self.__network = utilities.ete3_to_networkx(tree)
        else:
            raise CassiopeiaTreeError(
                "Please pass an ete3 Tree, a newick string, or a Networkx object."
            )

        # add character states
        for n in self.nodes:
            if (
                self.__original_character_matrix is not None
                and n in self.__original_character_matrix.index.values
            ):
                self.__network.nodes[n][
                    "character_states"
                ] = self.__original_character_matrix.loc[n].to_list()
            else:
                self.__network.nodes[n]["character_states"] = []

        # instantiate branch lengths
        for u, v in self.edges:
            self.__network[u][v]["length"] = 1

        # instantiate node time
        self.__network.nodes[self.root]["time"] = 0
        for u, v in self.depth_first_traverse_edges(source=self.root):
            self.__network.nodes[v]["time"] = (
                self.__network.nodes[u]["time"] + self.__network[u][v]["length"]
            )

    def initialize_character_states_at_leaves(
        self, character_matrix: Union[pd.DataFrame, Dict]
    ):
        """Populates character states at leaves.

        Assigns character states to the leaves of the tree. This function
        must have a character state assignment to all leaves of the tree.

        Args:
            character_matrix: A pandas dataframe or dictionary for mapping
                character states to the leaves of the tree.
        
        Raises:
            CassiopeiaTreeError if not all leaves are accounted for or if the
                tree has not been initialized.
        """
        if self.__network is None:
            raise CassiopeiaTreeError("Tree has not been initialized.")

        if isinstance(character_matrix, dict):
            character_matrix = pd.DataFrame.from_dict(
                character_matrix, orient="index"
            )

        if len(np.setdiff1d(self.leaves, character_matrix.index.values)) > 0:
            raise CassiopeiaTreeError(
                "Character matrix does not account for all the leaves."
            )

        for n in self.leaves:
            self.__set_character_states(n, character_matrix.loc[n].tolist())

        self.__original_character_matrix = character_matrix.copy()
        self.__current_character_matrix = character_matrix.copy()

    def initialize_all_character_states(self, mapping: Dict):
        """Populates character states across the tree.

        Assigns character states to all of the nodes in the tree. The mapping
        must have an entry for every node in the tree.

        Args:
            mapping: A mapping containing character state assignments for every
                node
        
        Raises:
            CassiopeiaTreeError if the tree is not initialized or if the
                mapping does not contain assignments for every node.
        """
        if self.__network is None:
            raise CassiopeiaTreeError("Tree has not been initialized.")

        if len(np.setdiff1d(self.nodes, character_matrix.keys())) > 0:
            raise CassiopeiaTreeError(
                "Mapping does not account for all the nodes."
            )

        character_matrix = {}
        for n in self.nodes:
            if self.is_leaf():
                character_matrix[n] = mapping[n]
            self.__set_character_states(n, mapping[n])

        character_matrix = pd.DataFrame.from_dict(
            character_matrix, orient="index"
        )
        self.__original_character_matrix = character_matrix.copy()
        self.__current_character_matrix = character_matrix.copy()

    def get_original_character_matrix(self) -> pd.DataFrame:
        """Gets the original character matrix.

        The returned character matrix is the original character matrix of
        observations. Downstream operations might change the character state
        observations for the cells and if this happens, the changes will
        not be reflected here. Instead, the changes will be reflected in the 
        character matrix obtained with `get_current_character_matrix`.

        Returns:
            A copy of the original, unmodified character matrix.

        Raises:
            CassiopeiaTreeError if the character matrix does not exist. 
        """
        if self.__original_character_matrix is None:
            raise CassiopeiaTreeError("Character matrix does not exist.")
        return self.__original_character_matrix.copy()

    def get_current_character_matrix(self) -> pd.DataFrame:
        """Gets the original character matrix.

        The returned character matrix is the modified character matrix of
        observations. When downstream operations are used to change the
        character state observations in the leaves of the tree, these changes
        will be reflected here. A "raw" version of the character matrix can
        be found in the `get_original_character_matrix` method.

        Returns:
            A copy of the modified character matrix.

        Raises:
            CassiopeiaTreeError if the character matrix does not exist. 
        """
        if self.__current_character_matrix is None:
            raise CassiopeiaTreeError("Character matrix does not exist.")
        return self.__current_character_matrix.copy()

    @property
    def n_cell(self) -> int:
        """Returns number of cells in tree.
        """
        if self.__original_character_matrix is None:
            if self.__network is None:
                return 0
            return len(self.leaves)
        return self.__original_character_matrix.shape[0]

    @property
    def n_character(self) -> int:
        """Returns number of characters in character matrix.
        """
        if self.__original_character_matrix is None:
            if self.__network is None:
                return 0
            if "character_states" in self.__network.nodes[self.leaves[0]]:
                return len(self.get_character_states(self.leaves[0]))
            return 0
        return self.__original_character_matrix.shape[1]

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
        return [n for n in self.__network if self.__network.out_degree(n) > 1]

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
        return [(u, v) for (u, v) in self.__network.edges]

    def is_leaf(self, node: str) -> bool:
        """Returns whether or not the node is a leaf.

        Returns:
            Whether or not the node is a leaf.
        
        Raises:
            CassiopeiaTreeError if the tree has not been initialized.
        """
        if self.__network is None:
            raise CassiopeiaTreeError("Tree is not initialized.")
        return self.__network.out_degree(node) == 0

    def is_root(self, node: str) -> bool:
        """Returns whether or not the node is the root.

        Returns:
            Whether or not the node is the root.
        
        Raises:
            CassiopeiaTreeError if the tree has not been initialized.
        """
        if self.__network is None:
            raise CassiopeiaTreeError("Tree is not initialized.")
        return node == self.root

    def reconstruct_ancestral_characters(self):
        """Reconstruct ancestral character states.

        Reconstructs ancestral states (i.e., those character states in the
        internal nodes) using the Camin-Sokal parsimony criterion (i.e.,
        irreversibility). Operates on the tree in place.
        """
        if self.__network is None:
            raise CassiopeiaTreeError("Tree is not initialized.")

        for n in self.depth_first_traverse_nodes(postorder=True):
            if self.is_leaf(n):
                continue
            children = self.children(n)
            character_states = [self.get_character_states(c) for c in children]
            reconstructed = utilities.get_lca_characters(
                character_states, self.missing_state_indicator
            )
            self.__set_character_states(n, reconstructed)

    def parent(self, node: str) -> str:
        """Gets the parent of a node.

        Args:
            node: A node in the tree

        Returns:
            The parent of the node.

        Raises:
            CassiopeiaTreeError if the tree is not initialized.
        """
        if self.__network is None:
            raise CassiopeiaTreeError("Tree is not initialized.")

        return [u for u in self.__network.predecessors(node)][0]

    def children(self, node: str) -> List[str]:
        """Gets the children of a given node.

        Args:
            node: A node in the tree.

        Returns:
            A list of nodes that are direct children of the input node.

        Raises:
            CassiopeiaTreeError if the tree is not initialized.
        """
        if self.__network is None:
            raise CassiopeiaTreeError("Tree is not initialized.")
        return [v for v in self.__network.successors(node)]

    def set_time(self, node: str, new_time: float) -> None:
        """Sets the time of a node.

        Importantly, this maintains consistency with the rest of the tree. In
        other words, setting the time of a particular node will change the 
        length of the edge leading into the node and the edge leading out. This
        function assumes monotonicity of times are maintained (i.e. no negative
        branch lengths).

        Args:
            node: Node in the tree
            new_time: New time for the node.

        Raises:
            CassiopeiaTreeError if the tree is not initialized, if the new
                time is less than the time of the parent, or if monotonicity
                is not maintained.
        """
        if self.__network is None:
            raise CassiopeiaTreeError("Tree is not initialized")

        parent = self.parent(node)
        if new_time < self.get_time(parent):
            raise CassiopeiaTreeError(
                "New age is less than the age of the parent."
            )

        for child in self.children(node):
            if new_time > self.get_time(child):
                raise CassiopeiaTreeError(
                "New age is greater than than a child."
            )

        self.__network.nodes[node]["time"] = new_time

        self.__network[parent][node]['length'] = new_time - self.get_time(parent)
        for child in self.children(node):
            self.__network[node][child]['length'] = self.get_time(child) - new_time

    def get_time(self, node: str) -> float:
        """Gets the time of a node.

        Returns the time of a node, defined as the sum of edge lengths from the
        root to the node.

        Raises:
            CassiopeiaTreeError if the tree has not been initialized.
        """
        if self.__network is None:
            raise CassiopeiaTreeError("Tree is not initialized.")

        return self.__network.nodes[node]["time"]

    def set_branch_length(self, parent: str, child: str, length: float):
        """Sets the length of a branch.

        Adjusts the branch length of the specified parent-child relationship.
        This procedure maintains the consistency with the rest of the times in
        the tree. Namely, by changing the branch length here, it will change
        the times of all the nodes below the parent of interest, relative to the
        difference between the old and new branch length.

        Args:
            parent: Parent node of the edge
            child: Child node of the edge 
            length: New edge length

        Raises:
            CassiopeiaTreeError if the tree is not initialized, if the edge
                does not exist, or if the edge length is negative.
        """
        if self.__network is None:
            raise CassiopeiaTreeError("Tree is not initialized.")

        if child not in self.children(parent):
            raise CassiopeiaTreeError("Edge does not exist.")

        if length < 0:
            raise CassiopeiaTreeError("Edge length must be positive.")

        self.__network[parent][child]["length"] = length

        for (u, v) in self.depth_first_traverse_edges(source=parent):
            self.__network.nodes[v]["time"] = (
                self.__network.nodes[u]["time"] + self.__network[u][v]["length"]
            )

    def get_branch_length(self, parent: str, child: str) -> float:
        """Gets the length of a branch.

        Raises:
            CassiopeiaTreeError if the tree has not been initialized or if the
                branch does not exist in the tree.
        """
        if self.__network is None:
            raise CassiopeiaTreeError("Tree is not initialized.")

        if child not in self.children(parent):
            raise CassiopeiaTreeError("Edge does not exist.")

        return self.__network[parent][child]["length"]

    def set_character_states(self, node: str, states=List[int]):
        """Sets the character states for a particular node.
        
        Args:
            node: Node in the tree
            states: A list of states to add to the node.

        Raises:
            CassiopeiaTreeError if the character vector is the incorrect length,
                or if the node of interest is a leaf that has not been
                instantiated.
        """
        if len(states) != self.n_character:
            raise CassiopeiaTreeError(
                "Input character vector is not the right length."
            )

        if self.is_leaf(node):
            if self.get_character_states(node) == []:
                raise CassiopeiaTreeError(
                    "Leaf node character states have not been instantiated with initialize_character_states_at_leaves"
                )
        self.__set_character_states(node, states)

        if self.is_leaf(node):
            self.__current_character_matrix.loc[node] = states

    def __set_character_states(self, node: str, states: List[int]):
        """A private method for setting states.

        A private method for setting states of nodes with no checks. Useful
        for the internal CassiopeiaTree API.

        Args:
            node: Node in the tree
            states: A list of states to add to the node.
        """
        self.__network.nodes[node]["character_states"] = states

    def get_character_states(self, node: str) -> List[int]:
        """Gets all the character states for a particular node.

        Args:
            node: Node in the tree.

        Returns:
            The full character state array of the specified node.
        """
        return self.__network.nodes[node]["character_states"]

    def depth_first_traverse_nodes(
        self, source: Optional[int] = None, postorder: bool = True
    ) -> Iterator[str]:
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

    def depth_first_traverse_edges(
        self, source: Optional[int] = None
    ) -> Iterator[Tuple[str, str]]:
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

        return [
            n
            for n in self.depth_first_traverse_nodes(source=node)
            if self.__network.out_degree(n) == 0
        ]

    def get_newick(self) -> str:
        """Returns newick format of tree.
        """
        return utilities.to_newick(self.__network)

    def get_mean_depth_of_tree(self) -> float:
        """Computes mean depth of tree.

        Returns the mean depth of the tree. If branch lengths have not been
        estimated, depth is by default the number of edges in the tree.

        Raises:
            CassiopeiaTreeError if the tree has not been initialized.
        """
        if self.__network is None:
            raise CassiopeiaTreeError("Tree is not initialized.")

        depths = [self.get_time(l) for l in self.leaves]
        return np.mean(depths)

    def get_max_depth_of_tree(self) -> float:
        """Computes the max depth of the tree.

        Returns the maximum depth of the tree. If branch lengths have not been
        estimated, depth is by default the number of edges in the tree.

        Raises:
            CassiopeiaTreeError if the tree has not been initialized.
        """
        if self.__network is None:
            raise CassiopeiaTreeError("Tree is not initialized.")

        depths = [self.get_time(l) for l in self.leaves]
        return np.max(depths)

    def get_mutations_along_edge(
        self, parent: str, child: str
    ) -> List[Tuple[int, int]]:
        """Gets the mutations along an edge of interest.

        Returns a list of tuples (character, state) of mutations that occur
        along an edge. Characters are 1-indexed.

        Args:
            parent: parent in tree
            child: child in tree

        Returns:
            A list of (character, state) tuples indicating which character
                mutated and to which state.

        Raises:
            CassipeiaTreeError if the edge does not exist or if the tree is 
                not initialized.
        """
        if self.__network is None:
            raise CassiopeiaTreeError("Tree is not initialized.")

        if child not in self.children(parent):
            raise CassiopeiaTreeError("Edge does not exist.")

        parent_states = self.get_character_states(parent)
        child_states = self.get_character_states(child)

        mutations = []
        for i in range(self.n_character):
            if parent_states[i] == 0 and child_states[i] != 0:
                mutations.append((i + 1, child_states[i]))

        return mutations

    def relabel_nodes(self, relabel_map: Dict[str, str]):
        """Relabels the nodes in the tree.

        Renames the nodes in the tree according to the relabeling map. Modifies
        the tree inplace.

        Args:
            relabel_map: A mapping of old names to new names.

        Raises:
            CassiopeiaTreeError if the tree is not initialized.
        """
        if self.__network is None:
            raise CassiopeiaTreeError("Tree is not initalized.")

        self.__network = nx.relabel_nodes(self.__network, relabel_map)
