from typing import List, Optional, Tuple

import networkx as nx


class Tree:
    r"""
    A phylogenetic tree for holding data from lineages and lineage tracing
    experiments.

    (Currently implemented as a light wrapper over networkx.DiGraph)

    Args:
        tree: The networkx.DiGraph from which to create the tree.
    """

    def __init__(self, tree: nx.DiGraph):
        self.tree = tree

    def root(self) -> int:
        tree = self.tree
        root = [n for n in tree if tree.in_degree(n) == 0][0]
        return root

    def leaves(self) -> List[int]:
        tree = self.tree
        leaves = [
            n
            for n in tree
            if tree.out_degree(n) == 0 and tree.in_degree(n) == 1
        ]
        return leaves

    def internal_nodes(self) -> List[int]:
        tree = self.tree
        return [n for n in tree if n != self.root() and n not in self.leaves()]

    def non_root_nodes(self) -> List[int]:
        return self.leaves() + self.internal_nodes()

    def nodes(self):
        tree = self.tree
        return list(tree.nodes())

    def num_characters(self) -> int:
        return len(self.tree.nodes[self.root()]["characters"])

    def get_state(self, node: int) -> str:
        tree = self.tree
        return tree.nodes[node]["characters"]

    def set_state(self, node: int, state: str) -> None:
        tree = self.tree
        tree.nodes[node]["characters"] = state

    def set_states(self, node_state_list: List[Tuple[int, str]]) -> None:
        for (node, state) in node_state_list:
            self.set_state(node, state)

    def get_age(self, node: int) -> float:
        tree = self.tree
        return tree.nodes[node]["age"]

    def set_age(self, node: int, age: float) -> None:
        tree = self.tree
        tree.nodes[node]["age"] = age

    def edges(self) -> List[Tuple[int, int]]:
        """List of (parent, child) tuples"""
        tree = self.tree
        return list(tree.edges)

    def get_edge_length(self, parent: int, child: int) -> float:
        tree = self.tree
        assert parent in tree
        assert child in tree[parent]
        return tree.edges[parent, child]["length"]

    def set_edge_length(self, parent: int, child: int, length: float) -> None:
        tree = self.tree
        assert parent in tree
        assert child in tree[parent]
        tree.edges[parent, child]["length"] = length

    def set_edge_lengths(
        self, parent_child_and_length_list: List[Tuple[int, int, float]]
    ) -> None:
        for (parent, child, length) in parent_child_and_length_list:
            self.set_edge_length(parent, child, length)

    def children(self, node: int) -> List[int]:
        tree = self.tree
        return list(tree.adj[node])

    def to_newick_tree_format(
        self,
        print_node_names: bool = True,
        print_internal_nodes: bool = False,
        append_state_to_node_name: bool = False,
        print_pct_of_mutated_characters_along_edge: bool = False,
        add_N_to_node_id: bool = False,
        fmt_branch_lengths: str = "%s",
    ) -> str:
        r"""
        Converts tree into Newick tree format.

        Args:
            print_internal_nodes: If True, prints the names of internal
            nodes too.
            print_pct_of_mutated_characters_along_edge: Self-explanatory
            TODO
        """
        leaves = self.leaves()

        def format_node(v: int):
            node_id_prefix = "" if not add_N_to_node_id else "N"
            node_id = "" if not print_node_names else str(v)
            node_suffix = (
                ""
                if not append_state_to_node_name
                else "_" + str(self.get_state(v))
            )
            return node_id_prefix + node_id + node_suffix

        def subtree_newick_representation(v: int) -> str:
            if len(self.children(v)) == 0:
                return format_node(v)
            subtrees_newick = []
            for child in self.children(v):
                edge_length = self.get_edge_length(v, child)
                if child in leaves:
                    subtree_newick = subtree_newick_representation(child)
                else:
                    subtree_newick = (
                        "(" + subtree_newick_representation(child) + ")"
                    )
                    if print_internal_nodes:
                        subtree_newick += format_node(child)
                # Add edge length
                subtree_newick = (
                    subtree_newick + ":" + (fmt_branch_lengths % edge_length)
                )
                if print_pct_of_mutated_characters_along_edge:
                    # Also add number of mutations
                    number_of_unmutated_characters_in_parent = self.get_state(
                        v
                    ).count("0")
                    pct_of_mutated_characters_along_edge = (
                        self.number_of_mutations_along_edge(v, child)
                        / (number_of_unmutated_characters_in_parent + 1e-100)
                    )
                    subtree_newick = (
                        subtree_newick + "[&&NHX:muts="
                        f"{self._fmt(pct_of_mutated_characters_along_edge)}]"
                    )
                subtrees_newick.append(subtree_newick)
            newick = ",".join(subtrees_newick)
            return newick

        root = self.root()
        res = "(" + subtree_newick_representation(root) + ")"
        if print_internal_nodes:
            res += format_node(root)
        res += ");"
        return res

    def _fmt(self, x: float):
        return "%.2f" % x

    def reconstruct_ancestral_states(self):
        r"""
        Reconstructs ancestral states with maximum parsimony.
        """
        root = self.root()

        def dfs(v: int) -> None:
            children = self.children(v)
            n_children = len(children)
            if n_children == 0:
                return
            for child in children:
                dfs(child)
            children_states = [self.get_state(child) for child in children]
            n_characters = len(children_states[0])
            state = ""
            for character_id in range(n_characters):
                states_for_this_character = set(
                    [
                        children_states[i][character_id]
                        for i in range(n_children)
                    ]
                )
                if len(states_for_this_character) == 1:
                    state += states_for_this_character.pop()
                else:
                    state += "0"
            self.set_state(v, state)
            if v == root:
                # Reset state to all zeros!
                self.set_state(v, "0" * n_characters)

        dfs(root)

    def copy_branch_lengths(self, tree_other):
        r"""
        Copies the branch lengths of tree_other onto self
        """
        assert self.nodes() == tree_other.nodes()
        assert self.edges() == tree_other.edges()

        for node in self.nodes():
            new_age = tree_other.get_age(node)
            self.set_age(node, age=new_age)

        for (parent, child) in self.edges():
            new_edge_length = tree_other.get_age(parent) - tree_other.get_age(
                child
            )
            self.set_edge_length(parent, child, length=new_edge_length)

    def print_edges(self):
        for (parent, child) in self.edges():
            print(
                f"{parent}[{self.get_state(parent)}] -> "
                f"{child}[{self.get_state(child)}]: "
                f"{self.get_edge_length(parent, child)}"
            )

    def num_cuts(self, v: int) -> int:
        # TODO: Hardcoded '0'...
        res = self.num_characters() - self.get_state(v).count("0")
        return res

    def parent(self, v: int) -> int:
        if v == self.root():
            raise ValueError("Asked for parent of root node!")
        incident_edges_at_v = [edge for edge in self.edges() if edge[1] == v]
        assert len(incident_edges_at_v) == 1
        return incident_edges_at_v[0][0]

    def set_edge_lengths_from_node_ages(self) -> None:
        r"""
        Sets the edge lengths to match the node ages.
        """
        for (parent, child) in self.edges():
            self.set_edge_length(
                parent, child, self.get_age(parent) - self.get_age(child)
            )

    def length(self) -> float:
        r"""
        Total length of the tree
        """
        res = 0
        for (parent, child) in self.edges():
            res += self.get_edge_length(parent, child)
        return res

    def num_ancestors(self, node: int) -> int:
        r"""
        Number of ancestors of a node. Terribly inefficient implementation.
        """
        res = 0
        root = self.root()
        while node != root:
            node = self.parent(node)
            res += 1
        return res

    def number_of_mutations_along_edge(self, parent, child):
        return self.get_state(parent).count("0") - self.get_state(child).count(
            "0"
        )

    def number_of_nonmutations_along_edge(self, parent, child):
        return self.get_state(child).count("0")

    def num_uncut(self, v):
        return self.get_state(v).count("0")

    def num_cut(self, v):
        return self.get_state(v).count("1")

    def depth(self) -> int:
        r"""
        Depth of the tree.
        E.g. the tree 0 -> 1 has depth 1.
        """

        def dfs(v):
            res = 0
            for child in self.children(v):
                res = max(res, dfs(child) + 1)
            return res

        res = dfs(self.root())
        return res

    def scale(self, factor: float):
        r"""
        The branch lengths of the tree are all scaled by this factor
        """
        for node in self.nodes():
            self.set_age(node, factor * self.get_age(node))
        self.set_edge_lengths_from_node_ages()
