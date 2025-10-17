"""Module describing the basic data structure for Cassiopeia - the CassiopeiaTree.

This data structure will typically contain a character
matrix containing that character state information for all the cells in a given
clonal population (though this is not required). Other important data is also
stored here, like the priors for given character states as well any meta data
associated with this clonal population.

When a solver has been called on this object, a tree will be added to the data
structure at which point basic properties can be queried like the average tree
depth or agreement between character states and phylogeny.

This object can be passed to any CassiopeiaSolver subclass as well as any
analysis module, like a branch length estimator or rate matrix estimator
"""

import collections
import copy
import heapq
import warnings
from collections.abc import Callable, Iterator
from typing import Any, Optional

import ete3
import networkx as nx
import numpy as np
import pandas as pd
import scipy

from cassiopeia.data import utilities
from cassiopeia.data.Layers import Layers
from cassiopeia.mixins import (
    CassiopeiaTreeError,
    CassiopeiaTreeWarning,
    is_ambiguous_state,
)
from cassiopeia.solver import solver_utilities


class CassiopeiaTree:
    """Basic tree object for Cassiopeia.

    This object stores the key attributes and functionalities a user might want
    for working with lineage tracing experiments. At its core, it stores
    three main items — a tree, a character matrix, and meta data associated
    with the data.

    The tree can be fed into the object via Ete3, Networkx, or can be inferred
    using one of the CassiopeiaSolver algorithms in the `solver` module. The
    tree here is only used for obtaining the *topology* of the tree.

    A character matrix can be stored in the object, containing the states
    observed for each cell. In typical lineage tracing experiments, these are
    integer representations of the indels observed at each unique cut site. We
    track both an unmodified version of the character matrix (obtainable via
    the `get_original_character_matrix` method) that does not maintain
    consistency with the character states of the leaves, and a working character
    matrix (obtainable via the `get_current_character_matrix` method) that
    is updated when the character states of leaves are changed.

    Some reconstruction algorithms will make use of dissimilarities between
    cells. To this end, we store these *dissimilarity maps* in the
    CassiopeiaTree class itself. For users trying to diagnose the reconstruction
    accuracy with a known ground truth, they can compare this dissimilarity
    map to the phylogenetic distance on the tree.

    Meta data for cells or characters can also be stored in this object. These
    items can be categorical or numerical in nature. Common examples of cell
    meta data are the cluster identity, tissue identity, or number of target-site
    UMIs per cell. These items can be used in downstream analyses, for example
    the FitchCount algorithm which infers the number of transitions between
    categorical variables (e.g., tissues). Common examples of character meta
    data are the proportion of missing data for each character or the entropy
    of states. These are good statistics to have for feature selection.

    Args:
        character_matrix: The character matrix for the lineage.
        missing_state_indicator: An indicator for missing states in the
            character matrix.
        cell_meta: Per-cell meta data.
        character_meta: Per-character meta data.
        priors: A dictionary storing the probability of each character mutating
            to a particular state.
        tree: A tree for the lineage.
        dissimilarity_map: An NxN dataframe storing the pairwise dissimilarities
            between samples.
        root_sample_name: The name of the sample to treat as the root. This
            is not always used, but will be added if needed during tree
            reconstruction. If the user already has a sample in the character
            matrix or dissimilarity map that they would like to use as the
            phylogenetic root, they can specify it here.
    """

    def __init__(
        self,
        character_matrix: pd.DataFrame | None = None,
        missing_state_indicator: int = -1,
        cell_meta: pd.DataFrame | None = None,
        character_meta: pd.DataFrame | None = None,
        priors: dict[int, dict[int, float]] | None = None,
        tree: str | ete3.Tree | nx.DiGraph | None = None,
        dissimilarity_map: pd.DataFrame | None = None,
        parameters: dict[str, Any] | None = None,
        root_sample_name: str | None = None,
    ) -> None:
        self.missing_state_indicator = missing_state_indicator
        self.cell_meta = cell_meta
        self.character_meta = character_meta
        self.priors = priors
        self.__network = None
        self.__cache = {}

        self._layers = Layers(self, None)

        self._character_matrix = None
        if character_matrix is not None:
            self.character_matrix = character_matrix

        self._parameters = {}
        if parameters is not None:
            self.parameters = parameters

        if tree is not None:
            tree = copy.deepcopy(tree)
            self.populate_tree(tree)

        # these attributes are helpful for distance based solvers
        self.__dissimilarity_map = None
        if dissimilarity_map is not None:
            self.set_dissimilarity_map(dissimilarity_map)
        self.root_sample_name = root_sample_name

    def populate_tree(
        self,
        tree: str | ete3.Tree | nx.DiGraph,
        layer: str | None = None,
    ) -> None:
        """Populates a tree object in CassiopeiaTree.

        Accepts a tree topology and introduces the topology into the object. In
        doing so, this function will also append character states to the leaves
        of the tree topology, if possible, and instantiate edge lengths by
        default to be length 1. Node times will be instantiated as well,
        corresponding to their tree depth.

        Args:
            tree: A tree topology specified as a networkx DiGraph, a newick
                string, or an ete3 Tree.
            layer: Layer to use for character matrix. If this is None,
                then the current `character_matrix` variable will be used.
        """
        if isinstance(tree, nx.DiGraph):
            self.__network = tree
        elif isinstance(tree, str):
            self.__network = utilities.newick_to_networkx(tree)
        elif isinstance(tree, ete3.Tree):
            self.__network = utilities.ete3_to_networkx(tree)
        else:
            raise CassiopeiaTreeError(
                "Please pass an ete3 Tree, a newick string, or a Networkx DiGraph object."
            )

        # enforce all names to be strings
        rename_dictionary = {}
        for n in self.__network.nodes:
            rename_dictionary[n] = str(n)

        self.__network = nx.relabel_nodes(self.__network, rename_dictionary)

        # clear cache if we're changing the topology of the tree
        self.__cache = {}

        if layer:
            character_matrix = self.layers[layer]
        else:
            character_matrix = self.character_matrix

        for n in self.nodes:
            self.__network.nodes[n]["character_states"] = []

        if character_matrix is not None:
            self.set_character_states_at_leaves(layer=layer)

        # instantiate branch lengths
        for u, v in self.edges:
            # only set branch lengths that have not already been set
            if "length" not in self.__network[u][v].keys():
                self.__network[u][v]["length"] = 1

        # instantiate node time
        self.__network.nodes[self.root]["time"] = 0
        for u, v in self.depth_first_traverse_edges(source=self.root):
            self.__network.nodes[v]["time"] = (
                self.__network.nodes[u]["time"] + self.__network[u][v]["length"]
            )

    def __check_network_initialized(self) -> None:
        """Checks that topology has been initialized."""
        if self.__network is None:
            raise CassiopeiaTreeError("Tree has not been initialized.")

    @property
    def layers(self) -> Layers:
        """Dictionary object storing versions of character matrices.

        Layers in this CassiopeiaTree object are inspired by AnnData's
        layers functionality. All layers have the same number of samples
        as the tree and :attr:`character_matrix`.

        Return the layer named `"missing"`::
            cas_tree.layers["missing"]

        Create or replace the `"merged"` layer::
            cas_tree.layers["merged"] = ...

        Delete the `"missing"` layer::
            del cas_tree.layers["missing"]
        """
        return self._layers

    @property
    def character_matrix(self) -> pd.DataFrame:
        """Return the character matrix currently associated with the tree."""
        return self._character_matrix

    @character_matrix.setter
    def character_matrix(self, character_matrix: pd.DataFrame):
        """Initializes a character matrix in the object.

        Args:
            character_matrix: Character matrix of mutation observations.
        """
        if not all(isinstance(i, str) for i in character_matrix.index):
            raise CassiopeiaTreeError("Index of character matrix must consist of strings.")

        self._character_matrix = character_matrix.copy()

    def set_character_states_at_leaves(
        self,
        character_matrix: pd.DataFrame | dict | None = None,
        layer: str | None = None,
    ) -> None:
        """Populates character states at leaves.

        Assigns character states to the leaves of the tree. This function
        must have a character state assignment to all leaves of the tree. If no
        argument is passed, the default :attr:`character_matrix` is used to
        set character states at the leaves.

        Args:
            character_matrix: A separate character matrix to use for setting
                character states at the leaves. This character matrix is not
                stored in the object afterwards. If this is None, uses the
                default character matrix stored in :attr:`character_matrix`
            layer: Layer to use for resetting character information at leaves.
                If this is None, uses the default character matrix stored in
                :attr:`character_matrix`.

        Raises:
            CassiopeiaTreeError if not all leaves are accounted for or if the
                tree has not been initialized.
        """
        self.__check_network_initialized()

        if character_matrix is not None and layer is not None:
            raise CassiopeiaTreeError(
                "You may only specify one of the following: character_matrix or layer."
            )

        if character_matrix is not None:
            if isinstance(character_matrix, dict):
                character_matrix = pd.DataFrame.from_dict(character_matrix, orient="index")
        else:
            if layer:
                character_matrix = self.layers[layer]
            else:
                character_matrix = self.character_matrix

        if set(self.leaves) != set(character_matrix.index.values):
            raise CassiopeiaTreeError("Character matrix index does not match set of leaves.")

        for n in self.leaves:
            self.__set_character_states(n, character_matrix.loc[n].tolist())

    def set_all_character_states(
        self,
        character_state_mapping: dict[str, list[int]],
        add_layer: str | None = None,
    ) -> None:
        """Populates character states across the tree.

        Assigns character states to all of the nodes in the tree. The mapping
        must have an entry for every node in the tree.

        Args:
            character_state_mapping: A mapping containing character state
                assignments for every node
            add_layer: Layer to which to add the new character matrix. If this
                is None, then the specified character matrix will be set to
                the default matrix in this object.

        Raises:
            CassiopeiaTreeError if the tree is not initialized or if the
                character_state_mapping does not contain assignments for every
                node.
        """
        self.__check_network_initialized()

        if set(character_state_mapping.keys()) != set(self.nodes):
            raise CassiopeiaTreeError("Mapping does not account for all the nodes.")

        character_matrix = {}
        for n in self.nodes:
            if self.is_leaf(n):
                character_matrix[n] = character_state_mapping[n]
            self.__set_character_states(n, character_state_mapping[n])

        character_matrix = pd.DataFrame.from_dict(character_matrix, orient="index")

        if add_layer:
            self._layers[add_layer] = character_matrix.copy()
        else:
            self._character_matrix = character_matrix.copy()

    def freeze_character_matrix(self, add_layer: str):
        """Freezes character matrix in specified layer.

        Adds a new layer the CassiopeiaTree object corresponding to the current
        version of the character matrix.

        Args:
            add_layer: Layer to create.

        Raises:
            CassiopeiaTreeError if no character matrix exists to freeze.
            CassiopeiaTreeWarning if layer already exists.
        """
        character_matrix = self.character_matrix

        if character_matrix is None:
            raise CassiopeiaTreeError("No character matrix found in this CassiopeiaTree object.")

        if add_layer in self.layers:
            raise CassiopeiaTreeWarning(
                f"Layer {add_layer} already exists, overwriting character matrix in layer."
            )

        self.layers[add_layer] = character_matrix.copy()

    @property
    def parameters(self) -> pd.DataFrame:
        """Return the tree-level parameters associated with the object."""
        return self._parameters

    def reset_parameters(self):
        """Resets the parameters attribute on the tree."""
        self._parameters = {}

    @property
    def n_cell(self) -> int:
        """Returns number of cells in tree.

        Raises:
            CassiopeiaTreeError if the object is empty (i.e. no tree or
            character matrix).
        """
        if self.character_matrix is None:
            if self.__network is None:
                raise CassiopeiaTreeError(
                    "This is an empty object with no tree or character matrix."
                )
            return len(self.leaves)
        return self.character_matrix.shape[0]

    @property
    def n_character(self) -> int:
        """Returns number of characters in character matrix.

        Raises:
            CassiopeiaTreeError if the object is empty (i.e. no tree or
            character matrix) or if the character states have not been
            initialized.
        """
        if self.character_matrix is None:
            if self.__network is None:
                raise CassiopeiaTreeError(
                    "This is an empty object with no tree or character matrix."
                )
            if self.get_character_states(self.leaves[0]) == []:
                raise CassiopeiaTreeError(
                    "Character states have not been initialized at leaves."
                    " Use set_character_states_at_leaves or populate_tree"
                    " with the character matrix that specifies the leaf"
                    " character states."
                )
            return len(self.get_character_states(self.leaves[0]))
        return self.character_matrix.shape[1]

    @property
    def root(self) -> str:
        """Returns root of tree.

        Returns:
            The root.

        Raises:
            CassiopeiaTreeError if the tree has not been initialized.
        """
        self.__check_network_initialized()

        if "root" not in self.__cache:
            self.__cache["root"] = [n for n in self.__network if self.is_root(n)][0]
        return self.__cache["root"]

    @property
    def leaves(self) -> list[str]:
        """Returns leaves of tree.

        Returns:
            The leaves of the tree.

        Raises:
            CassiopeiaTreeError if the tree has not been initialized.
        """
        self.__check_network_initialized()

        if "leaves" not in self.__cache:
            self.__cache["leaves"] = [n for n in self.__network if self.is_leaf(n)]
        return self.__cache["leaves"][:]

    @property
    def internal_nodes(self) -> list[str]:
        """Returns internal nodes in tree (including the root).

        Returns:
            The internal nodes of the tree (i.e. all nodes not at the leaves)

        Raises:
            CassiopeiaTreeError if the tree has not been initialized.
        """
        self.__check_network_initialized()

        if "internal_nodes" not in self.__cache:
            self.__cache["internal_nodes"] = [n for n in self.__network if self.is_internal_node(n)]
        return self.__cache["internal_nodes"][:]

    @property
    def nodes(self) -> list[str]:
        """Returns all nodes in tree.

        Returns:
            All nodes of the tree (internal + leaves)

        Raises:
            CassiopeiaTreeError if the tree has not been initialized.
        """
        self.__check_network_initialized()

        if "nodes" not in self.__cache:
            self.__cache["nodes"] = list(self.__network)
        return self.__cache["nodes"][:]

    @property
    def edges(self) -> list[tuple[str, str]]:
        """Returns all edges in the tree.

        Returns:
            All edges of the tree.

        Raises:
            CassiopeiaTreeError if the tree has not been initialized.
        """
        self.__check_network_initialized()

        if "edges" not in self.__cache:
            self.__cache["edges"] = [(u, v) for (u, v) in self.__network.edges]
        return self.__cache["edges"][:]

    def is_leaf(self, node: str) -> bool:
        """Returns whether or not the node is a leaf.

        Returns:
            Whether or not the node is a leaf.

        Raises:
            CassiopeiaTreeError if the tree has not been initialized.
        """
        self.__check_network_initialized()
        return self.__network.out_degree(node) == 0

    def is_root(self, node: str) -> bool:
        """Returns whether or not the node is the root.

        Returns:
            Whether or not the node is the root.

        Raises:
            CassiopeiaTreeError if the tree has not been initialized.
        """
        self.__check_network_initialized()
        return self.__network.in_degree(node) == 0

    def is_internal_node(self, node: str) -> bool:
        """Returns whether or not the node is an internal node.

        Returns:
            Whether or not the node is an internal node (i.e. out degree is
            greater than 0). In this case, the root is considered an internal
            node.

        Raises:
            CassiopeiaTreeError if the tree has not been initialized.
        """
        self.__check_network_initialized()
        return self.__network.out_degree(node) > 0

    def is_ambiguous(self, node: str) -> bool:
        """Returns whether the node is ambiguous.

        This is detected by checking if any of the characters are tuples.

        Args:
            node: Name of the node to check if it has ambiguous character(s)

        Returns:
            True if the node is ambiguous, False otherwise.

        Raises:
            CassiopeiaTreeError if the tree has not been initialized.
        """
        self.__check_network_initialized()
        states = self.get_character_states(node)
        return any(is_ambiguous_state(state) for state in states)

    def collapse_ambiguous_characters(self) -> None:
        """Only retain unique characters for ambiguous nodes.

        Ambiguous nodes have character strings represented as a list of tuples
        of integers. The inner list may contain multiple of the same states, encoding
        the relative abundance of a certain state in the ambiguous state distribution.
        Calling this function removes such duplicates and only retains unique
        characters. This function is idempotent and does nothing for trees that
        have no ambiguous characters.

        Raises:
            CassiopeiaTreeError if the tree has not been initialized.
        """
        self.__check_network_initialized()

        for node in self.nodes:
            if self.is_ambiguous(node):
                states = self.get_character_states(node)
                # Modify states in place. This is okay because get_character_states
                # returns a copy.
                modified = False
                for i in range(len(states)):
                    if is_ambiguous_state(states[i]):
                        new_states = tuple(set(states[i]))
                        if states != new_states:
                            states[i] = new_states
                            modified = True
                if modified:
                    self.set_character_states(node, states)

    def resolve_ambiguous_characters(
        self,
        resolve_function: Callable[[tuple[int, ...]], int] = utilities.resolve_most_abundant,
    ) -> None:
        """Resolve all nodes with ambiguous characters.

        A custom ``resolve_function`` may be provided to perform the resolution.
        By default, the most abundant state is selected.
        One is randomly selected on ties. Modifies the tree in-place.

        Args:
            resolve_function: Function that performs character resolution. This
                function is called once per ambiguous character state, and thus
                takes a single integer tuple as its argument and returns the
                resolved character state.

        Raises:
            CassiopeiaTreeError if the tree has not been initialized.
        """
        self.__check_network_initialized()

        for node in self.nodes:
            if self.is_ambiguous(node):
                states = self.get_character_states(node)

                # Modify states in place. This is okay because get_character_states
                # returns a copy.
                modified = False
                for i in range(len(states)):
                    if is_ambiguous_state(states[i]):
                        states[i] = resolve_function(states[i])
                        modified = True
                if modified:
                    self.set_character_states(node, states)

        # All nodes have a single character string, so we cast the character matrix
        # back to an integer.
        self._character_matrix = self._character_matrix.astype(int, copy=False)

    def reconstruct_ancestral_characters(self) -> None:
        """Reconstruct ancestral character states.

        Reconstructs ancestral states (i.e., those character states in the
        internal nodes) using the Camin-Sokal parsimony criterion (i.e.,
        irreversibility). Operates on the tree in place.

        Raises:
            CassiopeiaTreeError if the tree has not been initialized.
        """
        self.__check_network_initialized()

        for n in self.depth_first_traverse_nodes(postorder=True):
            if self.is_leaf(n):
                if self.get_character_states(n) == []:
                    raise CassiopeiaTreeError(
                        "Character states have not been initialized at leaves."
                        " Use set_character_states_at_leaves or populate_tree"
                        " with the character matrix that specifies the leaf"
                        " character states."
                    )
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
        self.__check_network_initialized()

        return list(self.__network.predecessors(node))[0]

    def children(self, node: str) -> list[str]:
        """Gets the children of a given node.

        Args:
            node: A node in the tree.

        Returns:
            A list of nodes that are direct children of the input node.

        Raises:
            CassiopeiaTreeError if the tree is not initialized.
        """
        self.__check_network_initialized()
        return list(self.__network.successors(node))

    def reorder_children(self, node: str, child_order: list[str]) -> None:
        """Reorders the children of a particular node.

        Args:
            node: Node in the tree
            child_order: A list with the new order of children for the node.

        Raises:
            CassiopeiaTreeError if the node of interest is a leaf that has not
                been instantiated, or if the new order of children is not a
                permutation of the original children.
        """
        self.__check_network_initialized()

        if self.is_leaf(node):
            raise CassiopeiaTreeError("Cannot reorder children of a leaf node.")

        if set(child_order) != set(self.children(node)):
            raise CassiopeiaTreeError(
                "New order of children is not apermutation of the original children."
            )

        self.__network.remove_edges_from([(node, child) for child in self.children(node)])
        self.__network.add_edges_from([(node, child) for child in child_order])

    def __remove_node(self, node) -> None:
        """Private method to remove node from tree.

        Args:
            node: A node in the tree to be removed

        Raises:
            CassiopeiaTreeError if the tree is not initialized.
        """
        self.__check_network_initialized()

        self.__network.remove_node(node)

        # this will change the topology of the tree, so reset the cache
        self.__cache = {}

    def __add_node(self, node) -> None:
        """Private method to add node to tree.

        Args:
            node: A node to be added to the tree.

        Raises:
            CassiopeiaTreeError if the tree is not initialized.
        """
        self.__check_network_initialized()

        self.__network.add_node(node)

        # this will change the topology of the tree, so reset the cache
        self.__cache = {}

    def __remove_edge(self, u, v) -> None:
        """Private method to remove edge from tree.

        Args:
            u: The source node of the directed edge to be removed
            v: The sink node of the directed edge to be removed

        Raises:
            CassiopeiaTreeError if the tree is not initialized.
        """
        self.__check_network_initialized()

        self.__network.remove_edge(u, v)

        # this will change the topology of the tree, so reset the cache
        self.__cache = {}

    def __add_edge(self, u, v) -> None:
        """Private method to add edge to tree.

        Args:
            u: The source node of the directed edge to be added
            v: The sink node of the directed edge to be added

        Raises:
            CassiopeiaTreeError if the tree is not initialized.
        """
        self.__check_network_initialized()

        self.__network.add_edge(u, v)

        # this will change the topology of the tree, so reset the cache
        self.__cache = {}

    def set_time(self, node: str, new_time: float) -> None:
        """Sets the time of a node.

        Importantly, this maintains consistency with the rest of the tree. In
        other words, setting the time of a particular node will change the
        length of the edge leading into the node and the edges leading out. This
        function requires monotonicity of times are maintained (i.e. no negative
        branch lengths).

        Args:
            node: Node in the tree
            new_time: New time for the node.

        Raises:
            CassiopeiaTreeError if the tree is not initialized, if the new
                time is less than the time of the parent, or if monotonicity
                is not maintained.
        """
        self.__check_network_initialized()

        if not self.is_root(node):
            parent = self.parent(node)
            if new_time < self.get_time(parent):
                raise CassiopeiaTreeError("New age is less than the age of the parent.")

        for child in self.children(node):
            if new_time > self.get_time(child):
                raise CassiopeiaTreeError("New age is greater than than a child.")

        self.__network.nodes[node]["time"] = new_time

        self.__network[parent][node]["length"] = new_time - self.get_time(parent)
        for child in self.children(node):
            self.__network[node][child]["length"] = self.get_time(child) - new_time

        if "distances" in self.__cache:
            del self.__cache["distances"]

    def set_times(self, time_dict: dict[str, float]) -> None:
        """Sets the time of all nodes in the tree.

        Importantly, this maintains consistency with the rest of the tree. In
        other words, setting the time of all nodes will change the length of
        the edges too. This function requires monotonicity of times are
        maintained (i.e. no negative branch lengths).

        Args:
            time_dict: Dictionary mapping nodes to their time.

        Raises:
            CassiopeiaTreeError if the tree is not initialized, or if the time
            of any parent is greater than that of a child.
        """
        self.__check_network_initialized()

        # TODO: Check that the keys of time_dict match exactly the nodes in the
        # tree and raise otherwise?
        # Currently, if nodes are missing in time_dict, code below blows up. If
        # extra nodes are present, they are ignored.

        for parent, child in self.edges:
            time_parent = time_dict[parent]
            time_child = time_dict[child]
            if time_parent > time_child:
                raise CassiopeiaTreeError(
                    f"Time of parent greater than that of child: {time_parent} > {time_child}"
                )
            self.__network[parent][child]["length"] = time_child - time_parent
        for node, time in time_dict.items():
            self.__network.nodes[node]["time"] = time

        if "distances" in self.__cache:
            del self.__cache["distances"]

    def get_time(self, node: str) -> float:
        """Gets the time of a node.

        Returns the time of a node, defined as the sum of edge lengths from the
        root to the node.

        Raises:
            CassiopeiaTreeError if the tree has not been initialized.
        """
        self.__check_network_initialized()

        return self.__network.nodes[node]["time"]

    def get_times(self) -> dict[str, float]:
        """Gets the times of all nodes.

        Returns the times of all nodes, defined as the sum of edge lengths from
        the root to that node.

        Raises:
            CassiopeiaTreeError if the tree has not been initialized.
        """
        self.__check_network_initialized()

        return {node: self.get_time(node) for node in self.nodes}

    def __set_branch_length(self, parent: str, child: str, length: float) -> None:
        """A private method for setting branch lengths.

        A private method for setting branch lengths with no checks. Useful
        for the internal CassiopeiaTree API.

        Args:
            parent: Parent node of the edge
            child: Child node of the edge
            length: New edge length
        """
        self.__network[parent][child]["length"] = length

    def set_branch_length(self, parent: str, child: str, length: float) -> None:
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
        self.__check_network_initialized()

        if child not in self.children(parent):
            raise CassiopeiaTreeError("Edge does not exist.")

        if length < 0:
            raise CassiopeiaTreeError("Edge length must be positive.")

        self.__set_branch_length(parent, child, length)

        for u, v in self.depth_first_traverse_edges(source=parent):
            self.__network.nodes[v]["time"] = (
                self.__network.nodes[u]["time"] + self.__network[u][v]["length"]
            )

        if "distances" in self.__cache:
            del self.__cache["distances"]

    def set_branch_lengths(self, branch_length_dict: dict[tuple[str, str], float]) -> None:
        """Sets the length of multiple branches on a tree.

        Adjusts the branch length of specified parent-child relationships.
        This procedure maintains the consistency with the rest of the times in
        the tree. Namely, by changing branch lengths here, it will change
        the times of all the nodes in the tree such that the times are
        representative of the new branch lengths.

        Args:
            branch_length_dict: A dictionary of edges to updated branch lengths

        Raises:
            CassiopeiaTreeError if the tree has not been initialized.
        """
        self.__check_network_initialized()

        for edge, length in branch_length_dict.items():
            u, v = edge[0], edge[1]
            if v not in self.children(u):
                raise CassiopeiaTreeError("Edge does not exist.")
            if length < 0:
                raise CassiopeiaTreeError("Edge length must be positive.")
            self.__set_branch_length(u, v, length)

        for u, v in self.depth_first_traverse_edges():
            self.__network.nodes[v]["time"] = (
                self.__network.nodes[u]["time"] + self.__network[u][v]["length"]
            )

        if "distances" in self.__cache:
            del self.__cache["distances"]

    def get_branch_length(self, parent: str, child: str) -> float:
        """Gets the length of a branch.

        Raises:
            CassiopeiaTreeError if the tree has not been initialized or if the
                branch does not exist in the tree.
        """
        self.__check_network_initialized()

        if child not in self.children(parent):
            raise CassiopeiaTreeError("Edge does not exist.")

        return self.__network[parent][child]["length"]

    def set_character_states(
        self,
        node: str,
        states: list[int] | list[tuple[int, ...]],
        layer: str | None = None,
    ) -> None:
        """Sets the character states for a particular node.

        Args:
            node: Node in the tree
            states: A list of states to add to the node.
            layer: Name of layer to modify. If the layer is not specified,
                :attr:`character_matrix` is modifed directly.

        Raises:
            CassiopeiaTreeError if the character vector is the incorrect length,
                or if the node of interest is a leaf that has not been
                instantiated.
        """
        self.__check_network_initialized()

        if len(states) != self.n_character:
            raise CassiopeiaTreeError("Input character vector is not the right length.")

        if self.is_leaf(node):
            if self.get_character_states(node) == []:
                raise CassiopeiaTreeError(
                    "Character states have not been initialized at leaves."
                    " Use set_character_states_at_leaves or populate_tree"
                    " with the character matrix that specifies the leaf"
                    " character states."
                )

        self.__set_character_states(node, states)

        if self.is_leaf(node):
            # Cast entire character matrix to object dtype if we get an ambiguous
            # character string.
            if self.is_ambiguous(node):
                if layer is not None:
                    self.layers[layer] = self.layers[layer].astype(object, copy=False)
                else:
                    self._character_matrix = self._character_matrix.astype(object, copy=False)

            # Pass in as numpy array of tuples to bypass the VisibleDeprecationWarning
            # from creating an ndarray from ragged nested sequences
            states_arr = np.empty(len(states), dtype=object)
            states_arr[:] = states
            if layer is not None:
                if layer in self.layers:
                    self.layers[layer].loc[node] = states_arr
                else:
                    raise ValueError(f"Layer {layer} does not exist.")
            else:
                self._character_matrix.loc[node] = states_arr

    def __set_character_states(self, node: str, states: list[int] | list[tuple[int, ...]]) -> None:
        """A private method for setting states.

        A private method for setting states of nodes with no checks. Useful
        for the internal CassiopeiaTree API.

        Args:
            node: Node in the tree
            states: A list of states to add to the node.
        """
        self.__network.nodes[node]["character_states"] = states

    def get_character_states(self, node: str) -> list[int] | list[tuple[int, ...]]:
        """Gets all the character states for a particular node.

        Args:
            node: Node in the tree.

        Returns:
            The full character state array of the specified node.

        Raises:
            CassiopeiaTreeError if the tree has not been initialized.
        """
        self.__check_network_initialized()

        return self.__network.nodes[node]["character_states"][:]

    def get_all_ancestors(self, node: str, include_node: bool = False) -> list[str]:
        """Gets all the ancestors of a particular node.

        Nodes that are closest to the given node appear first in the list.
        The ancestors, including those of all intermediate nodes, are cached.

        Args:
            node: Node in the tree
            include_node: Whether or not to include the node itself in the list
                of ancestors. If True, the first element of the list is the node
                itself.

        Returns:
            The list of nodes along the path from the root to the node.
        """
        self.__check_network_initialized()

        if "ancestors" not in self.__cache:
            self.__cache["ancestors"] = {}

        if self.is_root(node):
            return []

        if node not in self.__cache["ancestors"]:
            parent = self.parent(node)
            self.__cache["ancestors"][node] = [parent] + self.get_all_ancestors(parent)

        # Note that the cache never includes the node itself.
        ancestors = self.__cache["ancestors"][node]
        if include_node:
            ancestors = [node] + ancestors
        return ancestors

    def depth_first_traverse_nodes(
        self, source: str | None = None, postorder: bool = True
    ) -> Iterator[str]:
        """Nodes from depth first traversal of the tree.

        Returns the nodes from a DFS on the tree.

        Args:
            source: Where to begin the depth first traversal.
            postorder: Return the nodes in postorder. If False, returns in
                preorder.

        Returns:
            A list of nodes from the depth first traversal.

        Raises:
            CassiopeiaTreeError if the tree has not been initialized.
        """
        self.__check_network_initialized()

        if source is None:
            source = self.root

        if postorder:
            return nx.dfs_postorder_nodes(self.__network, source=source)
        else:
            return nx.dfs_preorder_nodes(self.__network, source=source)

    def depth_first_traverse_edges(self, source: str | None = None) -> Iterator[tuple[str, str]]:
        """Edges from depth first traversal of the tree.

        Returns the edges from a DFS on the tree.

        Args:
            source: Where to begin the depth first traversal.

        Returns:
            A list of edges from the depth first traversal.

        Raises:
            CassiopeiaTreeError if the tree has not been initialized.
        """
        self.__check_network_initialized()

        if source is None:
            source = self.root

        return nx.dfs_edges(self.__network, source=source)

    def breadth_first_traverse_edges(self, source: str | None = None) -> Iterator[tuple[str, str]]:
        """Edges from breadth first traversal of the tree.

        Returns the edges from a BFS on the tree.

        Args:
            source: Where to begin the breadth first traversal.

        Returns:
            A list of edges from the breadth first traversal.

        Raises:
            CassiopeiaTreeError if the tree has not been initialized.
        """
        self.__check_network_initialized()

        if source is None:
            source = self.root

        return nx.bfs_edges(self.__network, source=source)

    def leaves_in_subtree(self, node) -> list[str]:
        """Get leaves in subtree below a given node.

        Args:
            node: Root of the subtree.

        Returns:
            A list of the leaves in the subtree rooted at the specified node.

        Raises:
            CassiopeiaTreeError if the tree has not been initialized.
        """
        self.__check_network_initialized()

        if "subtree" not in self.__cache:
            self.__cache["subtree"] = {}

            for n in self.depth_first_traverse_nodes(postorder=True):
                if self.is_leaf(n):
                    self.__cache["subtree"][n] = [n]
                else:
                    leaves = []
                    for child in self.children(n):
                        leaves += self.leaves_in_subtree(child)
                    self.__cache["subtree"][n] = leaves

        return self.__cache["subtree"][node]

    def subset_clade(self, node: str, copy: bool = False) -> Optional["CassiopeiaTree"]:
        """Subset CassiopeiaTree object at node.

        Given a node in the CassiopeiaTree, subset the entire tree object
        to only the nodes that fall below that node.

        Args:
            node: Node identifier in the object.
            copy: Return a copy or operate in place.

        Returns:
            A subset CassiopeiaTree object if copy is set to true, else None.
        """
        if copy:
            new_tree = self.copy()
            new_tree.subset_clade(node)
            return new_tree

        nodes_in_subtree = self.depth_first_traverse_nodes(source=node)
        to_remove = list(set(self.nodes) - set(nodes_in_subtree))
        for node in to_remove:
            self.__remove_node(node)

        # Remove all removed nodes from data fields
        # This function will also clear the cache
        self.__register_data_with_tree()

    def get_newick(self, record_branch_lengths=False, record_node_names=False) -> str:
        """Returns newick format of tree.

        Args:
            record_branch_lengths: Whether to record branch lengths on the tree
            in the newick string
            record_node_names: Whether to record internal node names on the tree
            in the newick string

        Returns:
            The tree in the form of a newick string

        Raises:
            CassiopeiaTreeError if the tree has not been initialized.
        """
        self.__check_network_initialized()

        # No nodes should have a comma in the name. Otherwise, they will be
        # displayed as disconnected nodes. For instance, if there is a node with
        # name "1,2,3", then three separate nodes will be output. We decide to
        # raise an exception instead of a warning because the returned newick string
        # will be a WRONG tree.
        if any("," in node for node in self.nodes):
            raise CassiopeiaTreeError("No nodes may have the comma (,) character in its name.")

        return utilities.to_newick(self.__network, record_branch_lengths, record_node_names)

    def get_tree_topology(self) -> nx.DiGraph:
        """Returns the tree in Networkx format.

        Raises:
            CassiopeiaTreeError if the tree has not been initialized.
        """
        self.__check_network_initialized()

        if self.__network:
            return self.__network.copy()
        else:
            return None

    def _get_node_depths(self) -> list[float]:
        """Computes the depth of each node in the tree.

        Returns:
            List with depth of each node in the tree

        Raises:
            CassiopeiaTreeError if the tree has not been initialized.
        """
        self.__check_network_initialized()

        root_time = self.get_time(self.root)
        depths = [self.get_time(l) - root_time for l in self.leaves]
        return depths

    def get_mean_depth_of_tree(self) -> float:
        """Computes mean depth of tree.

        Returns the mean depth of the tree. If branch lengths have not been
        estimated, depth is by default the number of edges in the tree.

        Returns:
            Mean depth of the tree.

        Raises:
            CassiopeiaTreeError if the tree has not been initialized.
        """
        self.__check_network_initialized()

        return np.mean(self._get_node_depths())

    def get_max_depth_of_tree(self) -> float:
        """Computes the maximum depth of the tree (in terms of time).

        The maximum depth of the tree (in terms of time) is defined as the
        greatest time distance of any leaf of the tree from the root. Because
        branch lengths are by default equal to 1, if branch lengths have not
        yet been estimated (say with the
        cassiopeia.tools.branch_length_estimator module), then the max depth
        will be the number of edges in the tree from root to the furthest leaf.

        Returns:
            Maximum depth of the tree.

        Raises:
            CassiopeiaTreeError if the tree has not been initialized.
        """
        self.__check_network_initialized()

        return np.max(self._get_node_depths())

    def get_mutations_along_edge(
        self,
        parent: str,
        child: str,
        treat_missing_as_mutations: bool = False,
    ) -> list[tuple[int, int]]:
        """Gets the mutations along an edge of interest.

        Returns a list of tuples (character, state) of mutations that occur
        along an edge. Characters are 0-indexed.

        Note that parent states can be ambiguous if all child states have the
        same ambiguous state. If this is the case, there will not be a mutation
        detected along an edge, but we handle this case so as to not throw
        an error handling ambiguous states.

        Args:
            parent: parent in tree
            child: child in tree
            treat_missing_as_mutations: Whether to treat missing states as
                mutations.

        Returns:
            A list of (character, state) tuples indicating which character
                mutated and to which state.

        Raises:
            CassiopeiaTreeError if the edge does not exist or if the tree is
                not initialized.
        """
        self.__check_network_initialized()

        if child not in self.children(parent):
            raise CassiopeiaTreeError("Edge does not exist.")

        parent_states = self.get_character_states(parent)
        child_states = self.get_character_states(child)

        mutations = []
        for i in range(self.n_character):
            parent_state = (
                list(parent_states[i])
                if is_ambiguous_state(parent_states[i])
                else [parent_states[i]]
            )
            child_state = (
                list(child_states[i]) if is_ambiguous_state(child_states[i]) else [child_states[i]]
            )

            if len(np.intersect1d(parent_state, child_state)) < 1:
                if treat_missing_as_mutations:
                    mutations.append((i, child_states[i]))
                elif (
                    parent_states[i] != self.missing_state_indicator
                    and child_states[i] != self.missing_state_indicator
                ):
                    mutations.append((i, child_states[i]))

        return mutations

    def relabel_nodes(self, relabel_map: dict[str, str]) -> None:
        """Relabels the nodes in the tree.

        Renames the nodes in the tree according to the relabeling map. Modifies
        the tree in-place.

        Args:
            relabel_map: A mapping of old names to new names.

        Raises:
            CassiopeiaTreeError if the tree is not initialized.
        """
        self.__check_network_initialized()

        self.__network = nx.relabel_nodes(self.__network, relabel_map)

        # reset cache because we've changed names
        self.__cache = {}

    def __register_data_with_tree(self) -> None:
        """Makes the leaf data consistent with the leaves in the tree.

        Removes any leaves from the character matrix (all layers), cell
        metadata, and dissimilarity maps that do not appear in the tree.
        Additionally, adds any leaves that appear in the tree but not in the
        character matrix, cell metadata, or dissimilarity map with default
        values.

        The default values for each table is as follows:
        * character matrix: all states are missing values
            (``missing_state_indicator``)
        * cell metadata: None
        * dissimilarity map: ``np.inf`` distance from the leaf to all other
            leaves
        """
        # this will change the topology of the tree, so reset the cache
        self.__cache = {}

        leaves_set = set(self.leaves)
        if self.character_matrix is not None:
            remove_from_character_matrix = set(self.character_matrix.index) - leaves_set

            self.character_matrix = self.character_matrix.drop(index=remove_from_character_matrix)

            layer_copies = self.layers.copy()
            for name, layer_copy in layer_copies:
                layer_copies[name] = layer_copy.drop(index=remove_from_character_matrix)

            add_to_character_matrix = leaves_set - set(self.character_matrix.index)

            for to_add in add_to_character_matrix:
                # Note that we initialize state each iteration so that each state
                # is a new object.
                char_vector = [self.missing_state_indicator] * self.n_character
                self.__set_character_states(to_add, char_vector)
                self.character_matrix.loc[to_add] = char_vector

                for name, layer_copy in layer_copies:
                    layer_copy.loc[to_add] = char_vector
                    self.layers[name] = layer_copy

        if self.cell_meta is not None:
            remove_from_cell_meta = set(self.cell_meta.index) - leaves_set
            self.cell_meta = self.cell_meta.drop(index=remove_from_cell_meta)

            add_to_cell_meta = leaves_set - set(self.cell_meta.index)
            for to_add in add_to_cell_meta:
                self.cell_meta.loc[to_add] = None

        if self.__dissimilarity_map is not None:
            remove_from_dissimilarity_map = set(self.__dissimilarity_map.index) - leaves_set
            self.__dissimilarity_map = self.__dissimilarity_map.drop(
                index=remove_from_dissimilarity_map,
                columns=remove_from_dissimilarity_map,
            )

            add_to_dissimilarity_map = leaves_set - set(self.__dissimilarity_map.index)
            for to_add in add_to_dissimilarity_map:
                self.__dissimilarity_map.loc[to_add] = np.inf
                self.__dissimilarity_map[to_add] = np.inf

    def add_leaf(
        self,
        parent: str,
        node: str,
        states: list[int] | list[tuple[int, ...]] | None = None,
        time: float | None = None,
        branch_length: float | None = None,
        dissimilarity: dict[str, float] | None = None,
    ) -> None:
        """Add a leaf to the given parent node.

        The parent node may NOT also be a leaf, as this makes bookkeeping the
        character, dissimilarity and cell meta quite complicated. Also, adding a
        leaf to an existing leaf is probably not a common use case.

        Optional arguments ``states``, ``time``, ``branch_length``,
        ``dissimilarity`` may be provided to initialize the character
        string, time, branch length, dissimilarity, and cell meta for this leaf
        respectively. See :func:`__register_data_with_tree`
        for default values that are used when any of these are not provided.

        Note that only one of ``time`` and ``branch_length`` may be provided, and
        when neither are provided, the new leaf is added at the same time as the
        ``parent``.

        Args:
            parent: Parent node, to which to connect the leaf
            node: Name of the leaf to add
            states: Character states to initialize the new leaf with
            time: Time to place the new leaf
            branch_length: Branch length to place the new leaf
            dissimilarity: Indel dissimilarity between the new leaf and all
                other leaves

        Raises:
            CassiopeiaTreeError if ``parent`` does not exist or ``node``
                already exists, if ``parent`` is a leaf, if the tree has not
                been initialized, or if both ``time`` and ``branch_length``
                are provided.
        """
        self.__check_network_initialized()

        # Make sure the node is not already in the tree and the parent exists
        if node in self.nodes:
            raise CassiopeiaTreeError(f"Node {node} already exists.")
        if parent not in self.nodes:
            raise CassiopeiaTreeError(f"Node {parent} does not exist.")
        if parent in self.leaves:
            raise CassiopeiaTreeError("Can not add a leaf to a leaf.")
        if time and branch_length:
            raise CassiopeiaTreeError("Only one of `time` and `branch_length` may be provided.")

        self.__add_edge(parent, node)
        if states:
            self.__set_character_states(node, states)
            self.character_matrix.loc[node] = states
        if time:
            self.set_time(node, time)
        elif branch_length:
            self.set_branch_length(parent, node, branch_length)
        else:
            # By default, add leaf at the same time as parent.
            self.set_branch_length(parent, node, 0)
        if dissimilarity:
            self.set_dissimilarity(node, dissimilarity)

        # Update new leaf data with defaults
        # This function will also clear the cache
        self.__register_data_with_tree()

    def remove_leaves_and_prune_lineages(self, nodes: str | list[str]) -> None:
        """Removes a leaf (leaves) from the tree and prunes the lineage(s).

        Removes the specified leaves and all ancestors of those leaves that are
        no longer the ancestor of any of the remaining leaves. In the context
        of a phylogeny, this prunes the lineage of all nodes no longer relevant
        to observed samples. Additionally, maintains consistency with the
        updated tree by removing the node from all leaf data.

        Args:
            nodes: The leaf (leaves) to be removed. Can be a single string or
                a list of strings

        Raises:
            CassiopeiaTreeError if the tree is not initialized or any of the
                input nodes are not leaves
        """
        self.__check_network_initialized()

        if isinstance(nodes, str):
            nodes = [nodes]

        for n in nodes:
            if not self.is_leaf(n):
                raise CassiopeiaTreeError("A specified node is not a leaf.")

        # Keep track of nodes to check and their depths
        nodes_to_check = set()
        nodes_depth_queue = []

        # Remove leaves from the tree
        for n in nodes:
            parent = next(self.__network.predecessors(n))
            if parent not in nodes_to_check:
                parent_time = self.__network.nodes[parent]["time"]
                heapq.heappush(nodes_depth_queue, (parent_time, parent))
                nodes_to_check.add(parent)
            self.__network.remove_node(n)

        # Check nodes with a children removed from bottom to top
        while len(nodes_to_check) > 0:
            _, n = heapq.heappop(nodes_depth_queue)
            nodes_to_check.remove(n)
            if n == self.root:
                continue
            children = list(self.__network.successors(n))
            # Remove nodes with no children
            if len(children) == 0:
                parent = next(self.__network.predecessors(n))
                parent_time = self.__network.nodes[parent]["time"]
                self.__network.remove_node(n)
                if parent not in nodes_to_check:
                    heapq.heappush(nodes_depth_queue, (parent_time, parent))
                    nodes_to_check.add(parent)

        # Remove all removed nodes from data fields
        # This function will also clear the cache
        self.__register_data_with_tree()

    def collapse_unifurcations(self, source: int | None = None) -> None:
        """Collapses unifurcations on the tree.

        Removes all internal nodes that have in degree and out degree of 1,
        connecting their parent and children nodes by branchs with lengths
        equal to the total time elapsed from parent to each child. Therefore
        preserves the times of nodes that are not removed.

        Args:
            source: The node at which to begin the tree traversal

        Raises:
            CassiopeiaTreeError if the tree has not been initialized.
        """
        self.__check_network_initialized()

        from cassiopeia import utils

        start = source if source is not None else self.root
        if start not in self.__network:
            raise CassiopeiaTreeError(f"Node {start} does not exist in the tree.")

        subtree_nodes = {start}
        subtree_nodes.update(nx.descendants(self.__network, start))
        if len(subtree_nodes) <= 1:
            return

        collapsed_subtree = utils.collapse_unifurcations(
            self.__network.subgraph(subtree_nodes).copy()
        )

        nodes_to_remove = subtree_nodes - set(collapsed_subtree.nodes)
        for node in nodes_to_remove:
            self.__remove_node(node)

        remaining_nodes = subtree_nodes & set(collapsed_subtree.nodes)
        for u, v in list(self.__network.edges()):
            if u in remaining_nodes and v in remaining_nodes:
                self.__remove_edge(u, v)

        for node, data in collapsed_subtree.nodes(data=True):
            if node not in self.__network:
                self.__add_node(node)
            self.__network.nodes[node].update(data)

        for u, v, data in collapsed_subtree.edges(data=True):
            if not self.__network.has_edge(u, v):
                self.__add_edge(u, v)
            else:
                self.__network[u][v].clear()
            self.__network[u][v].update(data)

        # reset cache because we've changed the tree topology
        self.__cache = {}

    def collapse_mutationless_edges(self, infer_ancestral_characters: bool) -> None:
        """Collapses mutationless edges in the tree in-place.

        Uses the internal node annotations of a tree to collapse edges with no
        mutations. The introduction of a missing data event is considered a
        mutation in this context. Either takes the existing character states on
        the tree or infers the annotations bottom-up from the samples obeying
        Camin-Sokal Parsimony. Preserves the times of nodes that are not removed
        by connecting the parent and children of removed nodes by branches with
        lengths equal to the total time elapsed from parent to each child.

        Args:
            tree: A networkx DiGraph object representing the tree
            infer_ancestral_characters: Whether to infer the ancestral characters
                states of the tree

        Raises:
            CassiopeiaTreeError if the tree has not been initialized or if
                a node does not have character states initialized
        """
        self.__check_network_initialized()

        if infer_ancestral_characters:
            self.reconstruct_ancestral_characters()

        for n in list(self.depth_first_traverse_nodes(postorder=True)):
            if self.is_leaf(n):
                if self.get_character_states(n) == []:
                    raise CassiopeiaTreeError(
                        "Character states have not been initialized at leaves."
                        " Use set_character_states_at_leaves or populate_tree"
                        " with the character matrix that specifies the leaf"
                        " character states."
                    )
                continue

            if self.get_character_states(n) == []:
                raise CassiopeiaTreeError(
                    "Character states empty at internal node. Annotate"
                    " character states or infer ancestral characters by"
                    " setting infer_ancestral_characters=True."
                )

            for child in self.children(n):
                if not self.is_leaf(child):
                    t = self.get_branch_length(n, child)
                    if self.get_character_states(n) == self.get_character_states(child):
                        for grandchild in self.children(child):
                            t_ = self.get_branch_length(child, grandchild)
                            self.__add_edge(n, grandchild)
                            self.__set_branch_length(n, grandchild, t + t_)
                        self.__remove_node(child)

        # reset cache because we've changed the tree topology
        self.__cache = {}

    def get_dissimilarity_map(self) -> pd.DataFrame:
        """Gets the dissimilarity map."""
        if self.__dissimilarity_map is not None:
            return self.__dissimilarity_map.copy()
        else:
            return None

    def set_dissimilarity_map(self, dissimilarity_map: pd.DataFrame) -> None:
        """Sets the dissimilarity map variable in this object.

        Args:
            dissimilarity_map: Dissimilarity map relating all N x N distances
                between leaves.

        """
        character_matrix = self.character_matrix
        if character_matrix is not None:
            if (
                character_matrix.shape[0] != dissimilarity_map.shape[0]
                or collections.Counter(character_matrix.index)
                != collections.Counter(dissimilarity_map.index)
                or collections.Counter(character_matrix.index)
                != collections.Counter(dissimilarity_map.columns)
            ):
                warnings.warn(
                    "The samples in the existing character matrix and specified dissimilarity map do not agree.",
                    CassiopeiaTreeWarning,
                    stacklevel=2,
                )

        self.__dissimilarity_map = dissimilarity_map.copy()

    def get_dissimilarity(self, node: str) -> dict[str, float]:
        """Get the dissimilarity of a single leaf to all other leaves.

        Args:
            node: Name of the leaf node to get the dissimilarity to/from

        Raises:
            CassiopeiaTreeError if the dissimilarity map is not defined or the
                provided node is not in the dissimilarity map
        """
        if self.__dissimilarity_map is None:
            raise CassiopeiaTreeError("No dissimilarity map is defined.")
        if node not in self.__dissimilarity_map.index:
            raise CassiopeiaTreeError(f"`{node}` is not in the dissimilarity map.")
        return self.__dissimilarity_map.loc[node].to_dict()

    def set_dissimilarity(self, node: str, dissimilarity: dict[str, float]) -> None:
        """Set the dissimilarity of a single leaf.

        This function may be used only when a dissimilarity map already exists.

        Args:
            node: Name of the leaf node to set the dissimilarity to/from
            dissimilarity: Dictionary containing other leaf names as keys and
                dissimilarities as values

        Raises:
            CassiopeiaTreeError if the dissimilarity map does not exist, if
                the ``dissimilarity`` dictionary does not contain all other leaves
        """
        if self.__dissimilarity_map is None:
            raise CassiopeiaTreeError(
                "This function may only be used with an existing dissimilarity map. "
                "Use `compute_dissimilarity_map` or `set_dissimilarity_map` to set "
                "the dissimilarity map first."
            )
        if set(dissimilarity.keys()) - {node} != set(self.__dissimilarity_map.index) - {node}:
            raise CassiopeiaTreeError(
                "Provided dissimilarity dictionary does not provide dissimilarities to all other leaves."
            )

        self.__dissimilarity_map.loc[node] = 0
        self.__dissimilarity_map[node] = 0
        for other, dissim in dissimilarity.items():
            self.__dissimilarity_map.loc[node, other] = dissim
            self.__dissimilarity_map.loc[other, node] = dissim

    def compute_dissimilarity_map(
        self,
        dissimilarity_function: Callable[
            [np.array, np.array, int, dict[int, dict[int, float]]], float
        ]
        | None = None,
        prior_transformation: str = "negative_log",
        layer: str | None = None,
        threads: int = 1,
    ) -> None:
        """Computes a dissimilarity map.

        Given the dissimilarity function passed in, the pairwise dissimilarities
        will be computed over the samples in the character matrix. Populates
        the dissimilarity_map attribute in the object.

        If any of the leaves have ambiguous character states (detected by checking
        if any of the states are lists), then
        :func:`cassiopeia.solver.dissimilarity_functions.cluster_dissimilarity`
        is called instead, with the provided ``dissimilarity_function`` as
        the first argument.

        Args:
            dissimilarity_function: A function that will take in two character
                vectors and priors and produce a dissimilarity.
            prior_transformation: A function defining a transformation on the
                priors in forming weights. Supports the following
                transformations:
                    "negative_log": Transforms each probability by the negative
                        log
                    "inverse": Transforms each probability p by taking 1/p
                    "square_root_inverse": Transforms each probability by the
                        the square root of 1/p
            layer: Character matrix layer to use. If not specified, use the
                default :attr:`character_matrix`.
            threads: Number of threads to use for dissimilarity map computation.
        """
        if layer is not None:
            character_matrix = self.layers[layer]
        else:
            character_matrix = self.character_matrix

        if character_matrix is None:
            raise CassiopeiaTreeError("No character matrix is detected in this tree.")

        weights = None
        if self.priors:
            weights = solver_utilities.transform_priors(self.priors, prior_transformation)

        # Only compute dissimilarities between *unique* states to save runtime!
        cell_to_state = character_matrix.astype(str).apply("|".join, axis=1)
        state_to_cells = character_matrix.index.groupby(cell_to_state)
        dedup_character_matrix = character_matrix.drop_duplicates()

        N = dedup_character_matrix.shape[0]
        dissimilarity_map = utilities.compute_dissimilarity_map(
            dedup_character_matrix.to_numpy(),
            N,
            dissimilarity_function,
            weights,
            self.missing_state_indicator,
            threads=threads,
        )

        dissimilarity_map = scipy.spatial.distance.squareform(dissimilarity_map)

        # Expand deduplicated dissimilarity map back to all cells
        full_dissimilarity_map = np.pad(dissimilarity_map, (0, character_matrix.shape[0] - N))
        dissimilarity_cells = list(dedup_character_matrix.index)
        j = N
        for i, dedup_cell in enumerate(dedup_character_matrix.index):
            for cell in state_to_cells[cell_to_state[dedup_cell]]:
                if dedup_cell == cell:
                    continue
                dissimilarities = full_dissimilarity_map[i, :]
                full_dissimilarity_map[j, :] = dissimilarities
                full_dissimilarity_map[:, j] = dissimilarities
                dissimilarity_cells.append(cell)
                j += 1

        dissimilarity_map = pd.DataFrame(
            full_dissimilarity_map,
            index=dissimilarity_cells,
            columns=dissimilarity_cells,
        )
        self.set_dissimilarity_map(dissimilarity_map)

    def set_attribute(self, node: str, attribute_name: str, value: Any) -> None:
        """Sets an attribute in the tree.

        Args:
            node: Node name
            attribute_name: Name for the new attribute
            value: Value for the attribute.

        Raises:
            CassiopeiaTreeError if the tree has not been initialized.
        """
        self.__check_network_initialized()

        self.__network.nodes[node][attribute_name] = value

    def get_attribute(self, node: str, attribute_name: str) -> Any:
        """Retrieves the value of an attribute for a node.

        Args:
            node: Node name
            attribute_name: Name of the attribute.

        Returns:
            The value of the attribute for that node.

        Raises:
            CassiopeiaTreeError if the attribute has not been set for this node.
        """
        self.__check_network_initialized()

        try:
            return self.__network.nodes[node][attribute_name]
        except KeyError as error:
            raise CassiopeiaTreeError(
                f"Attribute {attribute_name} not detected for this node."
            ) from error

    def filter_nodes(self, condition: Callable[[str], bool]) -> list[str]:
        """Return nodes that satisfy a provided predicate.

        Parameters
        ---
        condition
            Function applied to each node identifier to determine membership.

        Returns:
        list[str] - Nodes for which the condition evaluates to ``True``.
        """
        self.__check_network_initialized()

        _filter = []
        for n in self.depth_first_traverse_nodes():
            if condition(n):
                _filter.append(n)

        return _filter

    def find_lcas_of_pairs(
        self,
        pairs: Iterator[tuple[str, str]] | list[tuple[str, str]] | None = None,
    ) -> Iterator[tuple[tuple[str, str], str]]:
        """Finds LCAs of all (provided) pairs.

        Args:
            pairs: Pairs of nodes for which to find LCAs. If not provided, LCAs of
                all pairs are computed. Defaults to None.

        Returns:
            A generator of ((u, v), LCA) tuples.

        Raises:
            CassiopeiaTreeError if the tree has not been initialized.
        """
        self.__check_network_initialized()
        return nx.tree_all_pairs_lowest_common_ancestor(self.__network, root=self.root, pairs=pairs)

    def find_lca(self, *nodes: str) -> str:
        """Finds the LCA of all provided nodes.

        Internally, this function calls :func:`get_all_ancestors` for each provided
        node and finds the deepest node that is shared.

        Args:
            *nodes: Nodes for which to find the LCA. At least two must be provided.

        Returns:
            The LCA node

        Raises:
            CassiopeiaTreeError if less than two nodes were provided or if the
            tree has not been initialized.
        """
        self.__check_network_initialized()
        nodes = set(nodes)
        if len(nodes) < 2:
            raise CassiopeiaTreeError("At least two distinct nodes must be provided")

        # Reversing get_all_ancestors gives a list of nodes from the root to each
        # node. Since get_all_ancestors doesn't include the node itself, we add
        # those manually.
        all_ancestors = [
            reversed(self.get_all_ancestors(node, include_node=True)) for node in nodes
        ]
        last_ancestor = self.root
        for ancestors in zip(*all_ancestors, strict=False):
            if len(set(ancestors)) > 1:
                break
            last_ancestor = ancestors[0]
        return last_ancestor

    def get_distance(self, node1: str, node2: str) -> float:
        """Compute the branch distance between two nodes.

        Internally, this function converts the tree to an undirected graph and
        finds the shortest path by calling :func:`nx.shortest_path_length`.

        Args:
            node1: First node
            node2: Second node

        Returns:
            The branch distance between the two nodes

        Raises:
            CassiopeiaTreeError if the tree has not been initialized.
        """
        self.__check_network_initialized()

        key = frozenset({node1, node2})
        if "distances" not in self.__cache or key not in self.__cache["distances"]:
            distances = self.__cache.setdefault("distances", {})
            undirected_network = self.__network.to_undirected()
            distances[key] = nx.shortest_path_length(
                undirected_network, source=node1, target=node2, weight="length"
            )

        return self.__cache["distances"][key]

    def get_distances(self, node: str, leaves_only: bool = False) -> dict[str, float]:
        """Compute the distances between the given node and every other node.

        Internally, this function converts the tree to an undirected graph and
        finds the shortest path by calling :func:`nx.shortest_path_length`.

        Args:
            node: Node from which to compute distance to all other nodes
            leaves_only: Calculate distances to leaves only and not internal nodes.
                Defaults to False.

        Returns:
            Dictionary of distances

        Raises:
            CassiopeiaTreeError if the tree has not been initialized.
        """
        self.__check_network_initialized()

        if "distances" not in self.__cache or any(
            frozenset({node, _node}) not in self.__cache["distances"] for _node in self.nodes
        ):
            distances = self.__cache.setdefault("distances", {})
            undirected_network = self.__network.to_undirected()
            for other_node, distance in nx.shortest_path_length(
                undirected_network, source=node, weight="length"
            ).items():
                distances[frozenset({node, other_node})] = distance

        return {
            _node: self.__cache["distances"][frozenset({node, _node})]
            for _node in (self.leaves if leaves_only else self.nodes)
        }

    def scale_to_unit_length(self) -> None:
        """Scales the tree to have unit length.

        The longest path from root to leaf will have length 1 after the
        scaling.

        Raises:
            CassiopeiaTreeError if the tree has not been initialized.
        """
        self.__check_network_initialized()

        current_times = self.get_times().values()
        max_time = max(current_times)
        min_time = min(current_times)
        new_times = {}
        for node in self.nodes:
            new_times[node] = (self.get_time(node) - min_time) / (max_time - min_time)
        self.set_times(new_times)

    def get_unmutated_characters_along_edge(self, parent: str, child: str) -> list[int]:
        """List of unmutated characters along edge.

        A character is considered to not mutate if it goes from the zero state
        to the zero state.

        Raises:
            CassiopeiaTreeError if the edge does not exist or if the tree is
                not initialized.
        """
        self.__check_network_initialized()

        if child not in self.children(parent):
            raise CassiopeiaTreeError("Edge does not exist.")

        parent_states = self.get_character_states(parent)
        child_states = self.get_character_states(child)
        res = []
        for i, (parent_state, child_state) in enumerate(
            zip(parent_states, child_states, strict=False)
        ):
            if parent_state == 0 and child_state == 0:
                res.append(i)
        return res

    def copy(self) -> "CassiopeiaTree":
        """Full copy of CassiopeiaTree."""
        return copy.deepcopy(self)

    def impute_deducible_missing_states(self):
        """Impute deducible missing states.

        If a state is missing in a node but present in its parent,
        then it can be imputed as the parent's state.
        We perform all these imputations.

        Raises:
            CassiopeiaTreeError if the character vectors do not all have
            the same length, or if the tree has not been initialized.
        """
        self.__check_network_initialized()
        for parent, child in self.depth_first_traverse_edges():
            parent_states = self.get_character_states(parent)
            child_states = self.get_character_states(child)
            if not len(parent_states) == len(child_states):
                raise CassiopeiaTreeError(
                    "Parent and child node have different length character states."
                )
            new_child_states = []
            for _i, (parent_state, child_state) in enumerate(
                zip(parent_states, child_states, strict=False)
            ):
                if (
                    parent_state != 0
                    and parent_state != self.missing_state_indicator
                    and child_state == self.missing_state_indicator
                ):
                    new_child_states.append(parent_state)
                else:
                    new_child_states.append(child_state)
            self.set_character_states(child, new_child_states)
