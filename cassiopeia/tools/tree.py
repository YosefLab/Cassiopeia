import networkx as nx
from typing import List, Tuple


class Tree():
    r"""
    networkx.Digraph wrapper to isolate networkx dependency and add custom tree
    methods.
    """
    def __init__(self, T: nx.DiGraph):
        self.T = T

    def root(self) -> int:
        T = self.T
        root = [n for n in T if T.in_degree(n) == 0][0]
        return root

    def leaves(self) -> List[int]:
        T = self.T
        leaves = [n for n in T if T.out_degree(n) == 0 and T.in_degree(n) == 1]
        return leaves

    def internal_nodes(self) -> List[int]:
        T = self.T
        return [n for n in T if n != self.root() and n not in self.leaves()]

    def non_root_nodes(self) -> List[int]:
        return self.leaves() + self.internal_nodes()

    def nodes(self):
        T = self.T
        return list(T.nodes())

    def num_characters(self) -> int:
        return len(self.T.nodes[self.root()]["characters"])

    def get_state(self, node: int) -> str:
        T = self.T
        return T.nodes[node]["characters"]

    def set_state(self, node: int, state: str) -> None:
        T = self.T
        T.nodes[node]["characters"] = state

    def set_states(self, node_state_list: List[Tuple[int, str]]) -> None:
        for (node, state) in node_state_list:
            self.set_state(node, state)

    def get_age(self, node: int) -> float:
        T = self.T
        return T.nodes[node]["age"]

    def set_age(self, node: int, age: float) -> None:
        T = self.T
        T.nodes[node]["age"] = age

    def edges(self) -> List[Tuple[int, int]]:
        """List of (parent, child) tuples"""
        T = self.T
        return list(T.edges)

    def get_edge_length(self, parent: int, child: int) -> float:
        T = self.T
        assert parent in T
        assert child in T[parent]
        return T.edges[parent, child]["length"]

    def set_edge_length(self, parent: int, child: int, length: float) -> None:
        T = self.T
        assert parent in T
        assert child in T[parent]
        T.edges[parent, child]["length"] = length

    def set_edge_lengths(
            self,
            parent_child_and_length_list: List[Tuple[int, int, float]]) -> None:
        for (parent, child, length) in parent_child_and_length_list:
            self.set_edge_length(parent, child, length)

    def children(self, node: int) -> List[int]:
        T = self.T
        return list(T.adj[node])

    def to_newick_tree_format(
        self,
        print_node_names: bool = True,
        print_internal_nodes: bool = False,
        append_state_to_node_name: bool = False,
        print_pct_of_mutated_characters_along_edge: bool = False,
        add_N_to_node_id: bool = False
    ) -> str:
        r"""
        Converts tree into Newick tree format for viewing in e.g. ITOL.
        Arguments:
            print_internal_nodes: If True, prints the names of internal
            nodes too.
            print_pct_of_mutated_characters_along_edge: Self-explanatory
        """
        leaves = self.leaves()

        def format_node(v: int):
            node_id_prefix = '' if not add_N_to_node_id else 'N'
            node_id = '' if not print_node_names else str(v)
            node_suffix =\
                '' if not append_state_to_node_name\
                else '_' + str(self.get_state(v))
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
                    subtree_newick =\
                        '(' + subtree_newick_representation(child) + ')'
                    if print_internal_nodes:
                        subtree_newick += format_node(child)
                # Add edge length
                subtree_newick = subtree_newick + ':' + str(edge_length)
                if print_pct_of_mutated_characters_along_edge:
                    # Also add number of mutations
                    number_of_unmutated_characters_in_parent =\
                        self.get_state(v).count('0')
                    number_of_mutations_along_edge =\
                        self.get_state(v).count('0')\
                        - self.get_state(child).count('0')
                    pct_of_mutated_characters_along_edge =\
                        number_of_mutations_along_edge /\
                        (number_of_unmutated_characters_in_parent + 1e-100)
                    subtree_newick = subtree_newick +\
                        "[&&NHX:muts="\
                        f"{self._fmt(pct_of_mutated_characters_along_edge)}]"
                subtrees_newick.append(subtree_newick)
            newick = ','.join(subtrees_newick)
            return newick

        root = self.root()
        res = '(' + subtree_newick_representation(root) + ')'
        if print_internal_nodes:
            res += format_node(root)
        res += ');'
        return res

    def _fmt(self, x: float):
        return '%.2f' % x

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
            state = ''
            for character_id in range(n_characters):
                states_for_this_character =\
                    set([children_states[i][character_id]
                         for i in range(n_children)])
                if len(states_for_this_character) == 1:
                    state += states_for_this_character.pop()
                else:
                    state += '0'
            self.set_state(v, state)
            if v == root:
                # Reset state to all zeros!
                self.set_state(v, '0' * n_characters)
        dfs(root)

    def copy_branch_lengths(self, T_other):
        r"""
        Copies the branch lengths of T_other onto self
        """
        assert(self.nodes() == T_other.nodes())
        assert(self.edges() == T_other.edges())

        for node in self.nodes():
            new_age = T_other.get_age(node)
            self.set_age(node, age=new_age)

        for (parent, child) in self.edges():
            new_edge_length =\
                T_other.get_age(parent) - T_other.get_age(child)
            self.set_edge_length(
                parent,
                child,
                length=new_edge_length)

    def print_edges(self):
        for (parent, child) in self.edges():
            print(f"{parent}[{self.get_state(parent)}] -> "
                  f"{child}[{self.get_state(child)}]: "
                  f"{self.get_edge_length(parent, child)}")
