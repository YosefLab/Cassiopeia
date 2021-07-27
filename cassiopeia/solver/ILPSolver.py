"""
Implements the Cassiopeia-ILP solver, as described in Jones et al, Genome Biol
2020. Briefly, this algorithm infers the maximum parsimony tree by solving for
a Steiner Tree over an inferred potential graph of potential intermediate
evolutionary states.
"""
import datetime
import logging
import time
from typing import Dict, List, Optional

import gurobipy
import hashlib
import itertools
import networkx as nx
import numpy as np
import pandas as pd

from cassiopeia.data import CassiopeiaTree
from cassiopeia.data import utilities as data_utilities
from cassiopeia.mixins import ILPSolverError, is_ambiguous_state, logger
from cassiopeia.solver import (
    CassiopeiaSolver,
    dissimilarity_functions,
    ilp_solver_utilities,
    solver_utilities,
)


class ILPSolver(CassiopeiaSolver.CassiopeiaSolver):
    """
    ILPSolver is a subclass of CassiopeiaSolver and implements the
    Cassiopeia-ILP algorithm described in Jones et al, Genome Biol 2020. The
    solver proceeds by constructing a tree over a network of possible
    evolutionary states known as the potential graph. The procedure for
    constructing this tree is done by solving for a Steiner Tree with an
    integer linear programming (ILP) optimization approach.

    Args:
        convergence_time_limit: Amount of time allotted to the ILP for
        convergence. Ignored if set to 0.
        convergence_iteration_limit: Number of iterations allowed for ILP
            convergence. Ignored if set to 0.
        maximum_potential_graph_layer_size: Maximum size allowed for an
            iteration of the potential graph inference procedure. If this is
            exceeded, we return the previous iteration's graph or abort
            altogether.
        maximum_potential_graph_lca_distance: Maximum height of LCA to add to the
            potential graph. If this parameter is not provided or the specified
            value is 0, the maximum distance between any pair of samples is used
            as the maximum lca height.
        weighted: Weight edges on the potential graph by the negative log
            likelihood of the mutations.
        seed: Random seed to use during ILP optimization.
        mip_gap: Objective gap for mixed integer linear programming problem.
        logfile: A file to log output to. This will contain information around
            the potential graph inference procedure as well as the Steiner Tree
            optimization.
        prior_transformation: Function to use when transforming priors into
            weights. Supports the following transformations:
                "negative_log": Transforms each probability by the negative
                    log (default)
                "inverse": Transforms each probability p by taking 1/p
                "square_root_inverse": Transforms each probability by the
                    the square root of 1/p
    """

    def __init__(
        self,
        convergence_time_limit: int = 12600,
        convergence_iteration_limit: int = 0,
        maximum_potential_graph_layer_size: int = 10000,
        maximum_potential_graph_lca_distance: Optional[int] = None,
        weighted: bool = False,
        seed: Optional[int] = None,
        mip_gap: float = 0.01,
        prior_transformation: str = "negative_log",
    ):

        super().__init__(prior_transformation)
        self.convergence_time_limit = convergence_time_limit
        self.convergence_iteration_limit = convergence_iteration_limit
        self.maximum_potential_graph_layer_size = (
            maximum_potential_graph_layer_size
        )
        self.maximum_potential_graph_lca_distance = (
            maximum_potential_graph_lca_distance
        )
        self.weighted = weighted
        self.seed = seed
        self.mip_gap = mip_gap

    @logger.namespaced("ILPSolver")
    def solve(
        self,
        cassiopeia_tree: CassiopeiaTree,
        logfile: str = "stdout.log",
        layer: Optional[str] = None,
    ):
        """Infers a tree with Cassiopeia-ILP.

        Solves a tree using the Cassiopeia-ILP algorithm and populates a tree
        in the provided CassiopeiaTree.

        Args:
            cassiopeia_tree: Input CassiopeiaTree
            logfile: Location to write standard out.
            layer: Layer storing the character matrix for solving. If None, the
                default character matrix is used in the CassiopeiaTree.
        """

        if self.weighted and not cassiopeia_tree.priors:
            raise ILPSolverError(
                "Specify prior probabilities in the CassiopeiaTree for weighted"
                " analysis."
            )

        # setup logfile config
        handler = logging.FileHandler(logfile)
        handler.setLevel(logging.INFO)
        logger.addHandler(handler)
        logger.info("Solving tree with the following parameters.")
        logger.info(f"Convergence time limit: {self.convergence_time_limit}")
        logger.info(
            f"Convergence iteration limit: {self.convergence_iteration_limit}"
        )
        logger.info(
            f"Max potential graph layer size: {self.maximum_potential_graph_layer_size}"
        )
        logger.info(
            f"Max potential graph lca distance: {self.maximum_potential_graph_lca_distance}"
        )
        logger.info(f"MIP gap: {self.mip_gap}")

        if layer:
            character_matrix = cassiopeia_tree.layers[layer].copy()
        else:
            character_matrix = cassiopeia_tree.character_matrix.copy()
        if any(
            is_ambiguous_state(state)
            for state in character_matrix.values.flatten()
        ):
            raise ILPSolverError("Solver does not support ambiguous states.")

        unique_character_matrix = character_matrix.drop_duplicates()

        weights = None
        if cassiopeia_tree.priors:
            weights = solver_utilities.transform_priors(
                cassiopeia_tree.priors, self.prior_transformation
            )

        # find the root of the tree & generate process ID
        root = tuple(
            data_utilities.get_lca_characters(
                unique_character_matrix.values.tolist(),
                cassiopeia_tree.missing_state_indicator,
            )
        )
        pid = hashlib.md5(
            "|".join([str(r) for r in root]).encode("utf-8")
        ).hexdigest()

        targets = [tuple(t) for t in unique_character_matrix.values.tolist()]

        if unique_character_matrix.shape[0] == 1:
            optimal_solution = nx.DiGraph()
            optimal_solution.add_node(root)
            optimal_solution = self.__append_sample_names(
                optimal_solution, character_matrix
            )
            cassiopeia_tree.populate_tree(optimal_solution, layer=layer)
            return

        # determine diameter of the dataset by evaluating maximum distance to
        # the root from each sample
        if (self.maximum_potential_graph_lca_distance is not None) and (
            self.maximum_potential_graph_lca_distance > 0
        ):
            max_lca_distance = self.maximum_potential_graph_lca_distance

        else:
            max_lca_distance = 0
            lca_distances = [
                dissimilarity_functions.hamming_distance(
                    root,
                    np.array(u),
                    ignore_missing_state=True,
                    missing_state_indicator=cassiopeia_tree.missing_state_indicator,
                )
                for u in targets
            ]

            for (i, j) in itertools.combinations(range(len(lca_distances)), 2):
                max_lca_distance = max(
                    max_lca_distance, lca_distances[i] + lca_distances[j] + 1
                )

        # infer the potential graph
        potential_graph = self.infer_potential_graph(
            unique_character_matrix,
            pid,
            max_lca_distance,
            weights,
            cassiopeia_tree.missing_state_indicator,
        )

        # generate Steiner Tree ILP model
        nodes = list(potential_graph.nodes())
        encoder = dict(zip(nodes, list(range(len(nodes)))))
        decoder = dict((v, k) for k, v in encoder.items())

        _potential_graph = nx.relabel_nodes(potential_graph, encoder)
        _targets = list(map(lambda x: encoder[x], targets))
        _root = encoder[root]

        model, edge_variables = self.generate_steiner_model(
            _potential_graph, _root, _targets
        )

        # solve the ILP problem and return a set of proposed solutions
        proposed_solutions = self.solve_steiner_instance(
            model, edge_variables, _potential_graph, pid, logfile
        )

        # select best model and post process the solution
        optimal_solution = proposed_solutions[0]
        optimal_solution = nx.relabel_nodes(optimal_solution, decoder)
        optimal_solution = self.post_process_steiner_solution(
            optimal_solution, root, targets, pid
        )

        optimal_solution = self.__append_sample_names(
            optimal_solution, character_matrix
        )

        cassiopeia_tree.populate_tree(optimal_solution, layer=layer)
        logger.removeHandler(handler)

    def infer_potential_graph(
        self,
        character_matrix: pd.DataFrame,
        pid: int,
        lca_height: int,
        weights: Optional[Dict[int, Dict[int, str]]] = None,
        missing_state_indicator: int = -1,
    ) -> nx.DiGraph:
        """Infers a potential graph for the observed states.

        Using the set of samples in the character matrix for this solver,
        this procedure creates a network which contains potential ancestors, or
        evolutionary intermediates.

        This procedure invokes
        `ilp_solver_utilities.infer_potential_graph_cython` which returns the
        edges of the potential graph in character string format
        (e.g., "1|2|3|..."). The procedure here decodes these strings, creates
        a Networkx directed graph, and adds edges to the graph. These weights
        are added to the edges of the graph using priors, if they are specified
        in the CassiopeiaTree, or the number of mutations along an edge.

        Args:
            character_matrix: Character matrix
            root: Specified root node, represented as a list of character states
            pid: Process ID for future reference
            lca_height: Maximum lca height to consider for connecting nodes to
                an LCA
            weights: Weights for character-state pairs, derived from the priors
                if these are available.
            missing_state_indicator: Indicator for missing data.

        Returns:
            A potential graph represented by a directed graph.
        """

        potential_graph_edges = (
            ilp_solver_utilities.infer_potential_graph_cython(
                character_matrix.values.astype(str),
                pid,
                lca_height,
                self.maximum_potential_graph_layer_size,
                missing_state_indicator,
            )
        )

        # the potential graph edges returned are strings in the form
        # "state1|state2|...", so we "decode" them here
        decoded_edges = []
        for e1, e2 in potential_graph_edges:
            e1 = np.array(
                e1.replace("-", str(missing_state_indicator)).split("|")
            ).astype(int)
            e2 = np.array(
                e2.replace("-", str(missing_state_indicator)).split("|")
            ).astype(int)
            decoded_edges.append((tuple(e1), tuple(e2)))

        potential_graph = nx.DiGraph()
        potential_graph.add_edges_from(decoded_edges)

        return self.add_edge_weights(
            potential_graph, weights, missing_state_indicator
        )

    def add_edge_weights(
        self,
        potential_graph: nx.DiGraph(),
        weights: Optional[Dict[int, Dict[int, str]]] = None,
        missing_state_indicator: int = -1,
    ) -> nx.DiGraph:
        """Annotates edges with the weight.

        Given a graph where nodes are iterable entities containing character
        states, annotated edges with respect to the number of mutations. If a
        prior dictionary is passed into the constructor, the log likelihood
        of the mutations is added instead. These values are stored in the
        `weight` attribute of the networkx graph.

        Args:
            potential_graph: Potential graph
            weights: Weights to use when comparing states between characters
            missing_state_indicator: Variable to indicate missing state
                information.

        Returns:
            The potential graph with edge weights added, stored in the `weight`
                attribute.
        """

        weighted_graph = potential_graph.copy()
        for u, v in weighted_graph.edges():
            weighted_graph[u][v][
                "weight"
            ] = dissimilarity_functions.weighted_hamming_distance(
                list(u), list(v), missing_state_indicator, weights
            )

        return weighted_graph

    def generate_steiner_model(
        self,
        potential_graph: nx.DiGraph,
        root: List[int],
        targets: List[List[int]],
    ):
        """Generates a Gurobi instance for Steiner Tree inference.

        Given a potential graph, a root to treat as the source, and a list of
        targets, create a Gurobi mixed integer linear programming instance for
        computing the Steiner Tree.

        Args:
            potential_graph: Potential graph representing the evolutionary
                space on which to solve for the Steiner Tree.
            root: A node in the graph to treat as the source.
            targets: A list of nodes in the tree that serve as targets for the
                Steiner Tree procedure.

        Returns:
            A Gurobipy Model instance and the edge variables involved.
        """
        source_flow = {v: 0 for v in potential_graph.nodes()}

        if root not in potential_graph.nodes:
            raise ILPSolverError("Root node not in potential graph.")
        for t in targets:
            if t not in potential_graph.nodes:
                raise ILPSolverError("Target node not in potential graph.")

        # remove source from targets if it exists there
        targets = [t for t in targets if t != root]

        source_flow[root] = len(targets)
        for target in targets:
            source_flow[target] = -1

        model = gurobipy.Model("steiner")

        ############# add variables #############

        # add flow for edges
        edge_variables = {}
        for u, v in potential_graph.edges():
            edge_variables[u, v] = model.addVar(
                vtype=gurobipy.GRB.INTEGER,
                lb=0,
                ub=len(targets),
                name=f"edge_{u}_{v}",
            )

        # add edge-usage indicator variable
        edge_variables_binary = {}
        for u, v in potential_graph.edges():
            edge_variables_binary[u, v] = model.addVar(
                vtype=gurobipy.GRB.BINARY, name=f"edge_binary_{u}_{v}"
            )

        model.update()

        ############# add constraints #############

        # check if edge is used
        for u, v in potential_graph.edges():
            model.addConstr(
                edge_variables_binary[u, v]
                >= (edge_variables[u, v] / len(targets))
            )

        # flow conservation constraints
        for v in potential_graph.nodes():
            model.addConstr(
                (
                    gurobipy.quicksum(
                        edge_variables[u, v]
                        for u in potential_graph.predecessors(v)
                    )
                    + source_flow[v]
                )
                == (
                    gurobipy.quicksum(
                        edge_variables[v, w]
                        for w in potential_graph.successors(v)
                    )
                )
            )

        ############ add objective #############

        objective_expression = gurobipy.quicksum(
            edge_variables_binary[u, v] * potential_graph[u][v]["weight"]
            for u, v in potential_graph.edges()
        )
        model.setObjective(objective_expression, gurobipy.GRB.MINIMIZE)

        return model, edge_variables

    def solve_steiner_instance(
        self,
        model: gurobipy.Model,
        edge_variables: gurobipy.Var,
        potential_graph: nx.DiGraph,
        pid: int,
        logfile: str,
    ) -> List[nx.DiGraph]:
        """Solves for a Steiner Tree from the Gurobi instance.

        This function works with a model that has been specified via Gurobi,
        and will solve the model using the stopping criteria that the user
        has specified in this class instance.

        Args:
            model: A Gurobi model corresponding to the Steiner Tree problem.
                This should be created with `generate_steiner_model`.
            edge_variables: Edge variables that were created during model
                generation. These are Gurobi variables that indicate whether
                two nodes are connected to one another in the Potential Graph;
                we use these variables to recreate a tree at the end from the
                Gurobi solution.
            potential_graph: Potential Graph that was used as input to the
                Steiner Tree problem.
            pid: Process ID
            logfile: Location to store standard out.

        Returns:
            A list of solutions
        """

        model.params.LogToConsole = 0

        # Adding constant parameters
        model.params.THREADS = 1
        model.params.Presolve = 2
        model.params.MIPFocus = 1
        model.params.Cuts = 1
        model.params.Method = 4

        # Add user-defined parameters
        model.params.MIPGAP = self.mip_gap
        model.params.LogFile = logfile

        if self.seed is not None:
            model.params.Seed = self.seed

        if self.convergence_iteration_limit > 0:
            model.params.IterationLimit = self.convergence_iteration_limit

        if self.convergence_time_limit > 0:
            model.params.TimeLimit = self.convergence_time_limit

        start_time = time.time()

        model.optimize()

        # recover subgraphs
        solutions = []
        for i in range(model.SolCount):
            model.params.SolutionNumber = i
            subgraph = nx.DiGraph()
            value_for_edge = model.getAttr("xn", edge_variables)
            for u, v in potential_graph.edges():
                if value_for_edge[u, v] > 0:
                    subgraph.add_edge(
                        u, v, weight=potential_graph[u][v]["weight"]
                    )
            solutions.append(subgraph)

        end_time = time.time()

        execution_delta = datetime.timedelta(seconds=(end_time - start_time))
        days = execution_delta.days
        hours = execution_delta.seconds // 3600
        minutes = execution_delta.seconds // 60
        seconds = execution_delta.seconds % 60

        logger.info(
            f"(Process {pid}) Steiner tree solving tool {days} days, "
            f"{hours} hours, {minutes} minutes, and {seconds} seconds."
        )
        if model.status != gurobipy.GRB.status.OPTIMAL:
            logger.info(
                f"(Process {pid}) Warning: Steiner tree solving did "
                "not result in an optimal model."
            )

        return solutions

    def post_process_steiner_solution(
        self,
        solution: nx.DiGraph,
        root: List[int],
        targets: List[List[int]],
        pid: int,
    ) -> nx.DiGraph:
        """Post-processes the returned graph from Gurobi.

        This procedure post-processes the proposed Steiner Tree from Gurobi
        by enforcing that no self-loops occur and that every node at most one
        parent.

        Args:
            solution: The Gurobi solution
            root: The root node
            targets: A list of targets
            pid: Process id

        Returns:
            A cleaned up networkx solution
        """

        processed_solution = solution.copy()
        for edge in nx.selfloop_edges(processed_solution):
            processed_solution.remove_edge(edge[0], edge[1])

        non_tree_nodes = [
            n
            for n in processed_solution.nodes()
            if processed_solution.in_degree(n) > 1
        ]
        for node in non_tree_nodes:
            parents = processed_solution.predecessors(node)
            parents = sorted(
                parents,
                key=lambda k: processed_solution[k][node]["weight"],
                reverse=True,
            )

            if len(parents) == 2 and (
                parents[1] in nx.ancestors(processed_solution, parents[0])
                or (parents[0] in nx.ancestors(processed_solution, parents[1]))
            ):
                if parents[1] in nx.ancestors(processed_solution, parents[0]):
                    processed_solution.remove_edge(parents[1], node)
                else:
                    processed_solution.remove_edge(parents[0], node)

            else:
                for parent in parents[1:]:
                    processed_solution.remove_edge(parent, node)

        # remove spurious roots
        spurious_roots = [
            n
            for n in processed_solution
            if processed_solution.in_degree(n) == 0
        ]
        while len(spurious_roots) > 1:
            for r in spurious_roots:
                if r != root:
                    processed_solution.remove_node(r)
            spurious_roots = [
                n
                for n in processed_solution
                if processed_solution.in_degree(n) == 0
            ]

        return processed_solution

    def __append_sample_names(
        self, solution: nx.DiGraph, character_matrix: pd.DataFrame
    ) -> nx.DiGraph:
        """Append sample names to character states in tree.

        Given a tree where every node corresponds to a set of character states,
        append sample names at the deepest node that has its character
        state. Sometimes character states can exist in two separate parts of
        the tree (especially when using the Hybrid algorithm where parts of
        the tree are built independently), so we make sure we only add a
        particular sample once to the tree.

        Args:
            solution: A Steiner Tree solution that we wish to add sample
                names to.
            character_matrix: Character matrix

        Returns:
            A solution with extra leaves corresponding to sample names.
        """

        root = [n for n in solution if solution.in_degree(n) == 0][0]

        sample_lookup = character_matrix.apply(
            lambda x: tuple(x.values), axis=1
        )

        states_added = []
        for node in nx.dfs_postorder_nodes(solution, source=root):

            # append nodes with this character state at the deepest place
            # possible
            if node in states_added:
                continue

            samples = sample_lookup[sample_lookup == node].index
            if len(samples) > 0:
                solution.add_edges_from([(node, sample) for sample in samples])
                states_added.append(node)

        return solution
