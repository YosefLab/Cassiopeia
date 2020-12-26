from ete3 import Tree as ETEtree
from itolapi import Itol, ItolExport
import networkx as nx

from typing import List, Optional, Tuple


def upload_to_itol_and_export_figure(
    newick_tree: str,
    apiKey: str,
    projectName: str,
    tree_name_in_iTOL: str,
    figure_file: str,
    plot_grid_line_scale: Optional[float] = None,
    horizontal_scale_factor: Optional[int] = None,
    bootstrap_display: bool = True,
    verbose: bool = True
):
    r"""
    TODO: Can use 'metadata_source' to select what data to show!!!
    For all export parameters see: https://itol.embl.de/help.cgi#export
    :param tree_name_in_iTOL: The name of the uploaded tree in iTOL
    :param figure_file: Name of the file where the tree figure will be exported to.
    :param plot_grid_line_scale: If provided, the distance between lines on the grid.
    :param verbose: Verbosity
    """
    # Write out newick tree to file
    tree_to_plot_file = "tree_to_plot.tree"
    with open(tree_to_plot_file, "w") as file:
        file.write(newick_tree)

    # Upload newick tree
    itol_uploader = Itol()
    itol_uploader.add_file("tree_to_plot.tree")
    itol_uploader.params["treeName"] = tree_name_in_iTOL
    itol_uploader.params["APIkey"] = apiKey
    itol_uploader.params["projectName"] = projectName
    good_upload = itol_uploader.upload()
    if not good_upload:
        print("There was an error:" + itol_uploader.comm.upload_output)
    if verbose:
        print("iTOL output: " + str(itol_uploader.comm.upload_output))
        print("Tree Web Page URL: " + itol_uploader.get_webpage())
        print("Warnings: " + str(itol_uploader.comm.warnings))
    tree_id = itol_uploader.comm.tree_id

    # Export tree. See https://itol.embl.de/help.cgi#export for all parameters
    itol_exporter = ItolExport()
    itol_exporter.set_export_param_value("tree", tree_id)
    itol_exporter.set_export_param_value(
        "format", figure_file.split(".")[-1]
    )  # ['png', 'svg', 'eps', 'ps', 'pdf', 'nexus', 'newick']
    itol_exporter.set_export_param_value("display_mode", 1)  # rectangular tree
    itol_exporter.set_export_param_value("label_display", 1)  # Possible values: 0 or 1 (0=hide labels, 1=show labels)
    if plot_grid_line_scale is not None:
        itol_exporter.set_export_param_value("internal_scale", 1)
        itol_exporter.set_export_param_value("internalScale1", plot_grid_line_scale)
        itol_exporter.set_export_param_value("internalScale2", plot_grid_line_scale)
        itol_exporter.set_export_param_value("internalScale1Dashed", 1)
        itol_exporter.set_export_param_value("internalScale2Dashed", 1)
    else:
        itol_exporter.set_export_param_value("tree_scale", 0)
    if horizontal_scale_factor is not None:
        itol_exporter.set_export_param_value(
            "horizontal_scale_factor", horizontal_scale_factor
        )  # doesnt actually scale the artboard
    if bootstrap_display:
        itol_exporter.set_export_param_value("bootstrap_display", 1)
        itol_exporter.set_export_param_value("bootstrap_type", 2)
        itol_exporter.set_export_param_value("bootstrap_label_size", 18)
    # itol_exporter.set_export_param_value("bootstrap_label_position", 20)
    # itol_exporter.set_export_param_value("bootstrap_symbol_position", 20)
    # itol_exporter.set_export_param_value("bootstrap_label_sci", 1)
    # itol_exporter.set_export_param_value("bootstrap_slider_min", -1)
    # itol_exporter.set_export_param_value("bootstrap_symbol_position", 0)
    # itol_exporter.set_export_param_value("branchlength_display", 1)
    # itol_exporter.set_export_param_value("branchlength_label_rounding", 1)
    # itol_exporter.set_export_param_value("branchlength_label_age", 1)
    # itol_exporter.set_export_param_value("internalScale1Label", 1)
    # itol_exporter.set_export_param_value("newick_format", "ID")
    # itol_exporter.set_export_param_value("internalScale1Label", 1)
    itol_exporter.set_export_param_value("leaf_sorting", 1)  # Possible values: 1 or 2 (1=normal sorting, 2=no sorting)
    print(f"Exporting tree to {figure_file}")
    itol_exporter.export(figure_file)

    # Cleanup
    # os.remove("tree_to_plot.tree")


def create_networkx_DiGraph_from_newick_file(file_path: str) -> nx.DiGraph:
    def newick_to_network(
        newick_filepath,
        f=1
    ):
        """
        Given a file path to a newick file, convert to a directed graph.

        :param newick_filepath:
            File path to a newick text file
        :param f:
            Parameter to be passed to Ete3 while reading in the newick file. (Default 1)
        :return: a networkx file of the tree
        """

        G = nx.DiGraph()  # the new graph
        tree = ETEtree(newick_filepath, format=f)

        # Create dict from ete3 node to cassiopeia.Node
        # NOTE(sprillo): Instead of mapping to a Cassiopeia node, we'll map to a string (just the node name)
        e2cass = {}
        edge_lengths = {}
        internal_node_id = 0
        for n in tree.traverse("postorder"):
            node_name = ''
            if n.name == '':
                # print(f"Node without name, is internal.")
                node_name = 'state-node-' + str(internal_node_id)
                internal_node_id += 1
            else:
                node_name = n.name
            e2cass[n] = node_name
            G.add_node(node_name)
            edge_lengths[node_name] = n._dist

        for p in tree.traverse("postorder"):
            pn = e2cass[p]
            for c in p.children:
                cn = e2cass[c]
                G.add_edge(pn, cn)
                G.edges[pn, cn]["length"] = edge_lengths[cn]
        return G

    T = newick_to_network(file_path)
    return T


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
        return len(self.T.nodes[0]["characters"])

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
