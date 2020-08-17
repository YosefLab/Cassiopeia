from __future__ import print_function
import sys

import concurrent.futures
import random
import functools
import multiprocessing
import networkx as nx
import numpy as np
import pandas as pd
import traceback
import hashlib
from collections import defaultdict

from tqdm import tqdm

from cassiopeia.TreeSolver.lineage_solver.greedy_solver import root_finder, greedy_build
from cassiopeia.TreeSolver.lineage_solver.greedy_solver_heritable import greedy_build_heritable
from cassiopeia.TreeSolver.lineage_solver.ILP_solver import (
    generate_mSteiner_model,
    solve_steiner_instance,
)
from cassiopeia.TreeSolver.lineage_solver.solver_utils import (
    build_potential_graph_from_base_graph,
    get_edge_length,
)
from cassiopeia.TreeSolver.Cassiopeia_Tree import Cassiopeia_Tree
from cassiopeia.TreeSolver.Node import Node
from cassiopeia.TreeSolver.utilities import find_neighbors


def solve_lineage_instance(
    _target_nodes,
    prior_probabilities=None,
    method="hybrid",
    threads=8,
    hybrid_cell_cutoff=200,
    hybrid_lca_cutoff=None,
    time_limit=1800,
    max_neighborhood_size=10000,
    seed=None,
    num_iter=-1,
    weighted_ilp=False,
    fuzzy=False,
    probabilistic=False,
    plot_diagnostics=True,
    maximum_alt_solutions=100,
    greedy_minimum_allele_rep=1.0,
    n_neighbors=10,
    missing_data_mode="lookahead",
    lookahead_depth=3,
    split_on_heritable=False
):
    """
	Aggregated lineage solving method, which given a set of target nodes, will find the maximum parsimony tree
	accounting the given target nodes

	:param target_nodes:
		A list of target nodes, where each node is in the form 'Ch1|Ch2|....|Chn'
	:param prior_probabilities:
		A nested dictionary containing prior probabilities for [character][state] mappings
		where characters are in the form of integers, and states are in the form of strings,
		and values are the probability of mutation from the '0' state.
	:param method:
		The method used for solving the problem ['ilp, 'hybrid', 'greedy']
			- ilp: Attempts to solve the problem based on steiner tree on the potential graph
				   (Recommended for instances with several hundred samples at most)
			- greedy: Runs a greedy algorithm to find the maximum parsimony tree based on choosing the most occurring split in a
				   top down fasion (Algorithm scales to any number of samples)
			- hybrid: Runs the greedy algorithm until there are less than hybrid_subset_cutoff samples left in each leaf of the
				   tree, and then returns a series of small instance ilp is then run on these smaller instances, and the
				   resulting graph is created by merging the smaller instances with the greedy top-down tree
	:param threads:
		The number of threads to use in parallel for the hybrid algorithm
	:param hybrid_subset_cutoff:
		The maximum number of nodes allowed before the greedy algorithm terminates for a given leaf node
	:return:
		A reconstructed subgraph representing the nodes
	"""

    if method == "hybrid":
        assert (
            hybrid_cell_cutoff is None or hybrid_lca_cutoff is None
        ), "You can only use one type of cutoff in Hybrid"

    target_nodes = [n.get_character_string() + "_" + n.name for n in _target_nodes]

    node_name_dict = dict(
        zip(
            [n.split("_")[0] for n in target_nodes],
            [n + "_target" for n in target_nodes],
        )
    )

    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    # clip identifier for now, but make sure to add later
    target_nodes = [n.split("_")[0] for n in target_nodes]

    # target_nodes = list(set(target_nodes))
    master_root = root_finder(target_nodes, split_on_heritable)
    if method == "ilp":

        subgraphs, r, pid, graph_sizes = find_good_gurobi_subgraph(
            master_root,
            target_nodes,
            node_name_dict,
            prior_probabilities,
            time_limit,
            1,
            max_neighborhood_size,
            seed=seed,
            num_iter=num_iter,
            weighted=weighted_ilp,
            n_neighbors=n_neighbors,
            split_on_heritable = split_on_heritable
        )

        subgraph = subgraphs[0]

        rdict = {}
        target_seen = []

        for n in subgraph:
            spl = n.split("_")
            nn = Node("state-node", spl[0].split("|"), is_target=False)

            if len(spl) == 2:
                if "target" in n and nn.char_string not in target_seen:
                    nn.is_target = True

            if len(spl) > 2:
                if "target" in n and nn.char_string not in target_seen:
                    nn.is_target = True
                nn.pid = spl[-1]

            if nn.is_target:
                target_seen.append(nn.char_string)

            rdict[n] = nn

        state_tree = nx.relabel_nodes(subgraph, rdict)

        return (
            Cassiopeia_Tree(
                method="ilp", network=state_tree, name="Cassiopeia_state_tree"
            ),
            graph_sizes,
        )

    if method == "hybrid":

        neighbors, distances = None, None
        if missing_data_mode == "knn":
            print("Computing neighbors for imputing missing values...")
            neighbors, distances = find_neighbors(target_nodes, n_neighbors=n_neighbors, split_on_heritable = split_on_heritable)

        network, target_sets = greedy_build(
            target_nodes,
            neighbors,
            distances,
            priors=prior_probabilities,
            cell_cutoff=hybrid_cell_cutoff,
            lca_cutoff=hybrid_lca_cutoff,
            fuzzy=fuzzy,
            probabilistic=probabilistic,
            minimum_allele_rep=greedy_minimum_allele_rep,
            missing_data_mode=missing_data_mode,
            lookahead_depth=lookahead_depth,
            split_on_heritable=split_on_heritable
        )

        print(
            "Using "
            + str(min(multiprocessing.cpu_count(), threads))
            + " threads, "
            + str(multiprocessing.cpu_count())
            + " available.",
            flush=True,
        )
        executor = concurrent.futures.ProcessPoolExecutor(
            min(multiprocessing.cpu_count(), threads)
        )
        print("Sending off Target Sets: " + str(len(target_sets)), flush=True)

        # just in case you've hit a target node during the greedy reconstruction, append name at this stage
        # so the composition step doesn't get confused when trying to join to the root.
        network = nx.relabel_nodes(network, node_name_dict)

        futures = [
            executor.submit(
                find_good_gurobi_subgraph,
                root,
                targets,
                node_name_dict,
                prior_probabilities,
                time_limit,
                1,
                max_neighborhood_size,
                seed,
                num_iter,
                weighted_ilp,
                n_neighbors,
                split_on_heritable = split_on_heritable
            )
            for root, targets in target_sets
        ]

        concurrent.futures.wait(futures)

        base_network = network.copy()
        base_rdict = {}
        for n in base_network:
            spl = n.split("_")
            nn = Node("state-node", spl[0].split("|"), is_target=False)
            if len(spl) > 1:
                nn.pid = spl[1]
            if spl[0] in node_name_dict:
                nn.is_target = True

            base_rdict[n] = nn

        base_network = nx.relabel_nodes(base_network, base_rdict)

        num_solutions = 1  # keep track of number of possible solutions
        potential_graph_sizes = []
        all_res = []
        alt_solutions = {}

        for future in futures:
            results, r, pid, graph_sizes = future.result()
            potential_graph_sizes.append(graph_sizes)

            subproblem_solutions = []
            for res in results:
                new_names = {}
                for n in res:
                    if res.in_degree(n) == 0 or n == r:
                        new_names[n] = n
                    else:
                        new_names[n] = n + "_" + str(pid)
                res = nx.relabel_nodes(res, new_names)
                subproblem_solutions.append(res)

            num_solutions *= len(subproblem_solutions)
            all_res.append(subproblem_solutions)

            rt = [
                n
                for n in subproblem_solutions[0]
                if subproblem_solutions[0].in_degree(n) == 0
            ][0]
            alt_solutions[base_rdict[rt]] = subproblem_solutions

            network = nx.compose(network, subproblem_solutions[0])

        rdict = {}
        target_seen = []

        for n in network:
            spl = n.split("_")
            nn = Node("state-node", spl[0].split("|"), is_target=False)

            if len(spl) == 2:
                if "target" in n and nn.char_string not in target_seen:
                    nn.is_target = True

            if len(spl) > 2:
                if "target" in n and nn.char_string not in target_seen:
                    nn.is_target = True
                nn.pid = spl[-1]

            if nn.is_target:
                target_seen.append(nn.char_string)

            rdict[n] = nn

        state_tree = nx.relabel_nodes(network, rdict)

        # create alternative solutions
        pbar = tqdm(
            total=len(alt_solutions.keys()), desc="Enumerating alternative solutions"
        )
        for r in alt_solutions.keys():
            soln_list = []

            # get original target char strings
            # sub_targets = [n.char_string for n in state_tree.successors(r) if n.is_target]
            for res in alt_solutions[r]:

                rdict = {}
                for n in res:
                    spl = n.split("_")
                    nn = Node("state-node", spl[0].split("|"), is_target=False)

                    if len(spl) > 2:
                        nn.pid = spl[-1]

                    rdict[n] = nn

                res = nx.relabel_nodes(res, rdict)
                soln_list.append(res)

            alt_solutions[r] = soln_list

            pbar.update(1)  # update progress bar

        # iterate through all possible solutions
        # alt_solutions = []

        # if num_solutions > 1:

        # 	num_considered_solutions = 0
        # 	sol_identifiers = []  # keep track of solutions already sampled

        # 	# we'll sample maximum_alt_solutions from the set of possible solutions
        # 	pbar = tqdm(
        # 		total=maximum_alt_solutions, desc="Enumerating alternative solutions"
        # 	)
        # 	while num_considered_solutions < min(num_solutions, maximum_alt_solutions):

        # 		current_sol = []
        # 		for res_list in all_res:
        # 			current_sol.append(np.random.choice(len(res_list)))

        # 		if tuple(current_sol) not in sol_identifiers:

        # 			new_network = base_network.copy()
        # 			for i in range(len(current_sol)):
        # 				res_list = all_res[i]
        # 				net = res_list[current_sol[i]]
        # 				new_network = nx.compose(new_network, net)

        # 			rdict = {}
        # 			target_seen = []
        # 			for n in new_network:
        # 				spl = n.split("_")
        # 				nn = Node("state-node", spl[0].split("|"), is_target=False)

        # 				if len(spl) == 2:
        # 					if "target" in n and n not in target_seen:
        # 						nn.is_target = True

        # 				if len(spl) > 2:
        # 					if 'target' in n and n not in target_seen:
        # 						nn.is_target = True
        # 					nn.pid = spl[-1]

        # 				if nn.is_target:
        # 					target_seen.append(nn.char_string)

        # 				rdict[n] = nn

        # 			new_network = nx.relabel_nodes(new_network, rdict)

        # 			alt_solutions.append(new_network)

        # 			sol_identifiers.append(tuple(current_sol))
        # 			num_considered_solutions += 1

        # 			pbar.update(1)  # update progress bar

        return (
            Cassiopeia_Tree(
                method="hybrid",
                network=state_tree,
                name="Cassiopeia_state_tree",
                alternative_solutions=alt_solutions,
                base_network=base_network,
            ),
            potential_graph_sizes,
        )

    if method == "greedy":

        neighbors, distances = None, None
        if missing_data_mode == "knn":
            print("Computing neighbors for imputing missing values...")
            neighbors, distances = find_neighbors(target_nodes, n_neighbors=n_neighbors, split_on_heritable = split_on_heritable)

        print("lineage_solver")
        print(split_on_heritable)
        graph = greedy_build(
            target_nodes,
            neighbors,
            distances,
            priors=prior_probabilities,
            cell_cutoff=-1,
            lca_cutoff=None,
            fuzzy=fuzzy,
            probabilistic=probabilistic,
            minimum_allele_rep=greedy_minimum_allele_rep,
            missing_data_mode=missing_data_mode,
            lookahead_depth=lookahead_depth,
            split_on_heritable=split_on_heritable
        )[0]

        rdict = {}
        for n in graph:
            spl = n.split("_")
            nn = Node("state-node", spl[0].split("|"), is_target=False)
            if len(spl) > 1:
                nn.pid = spl[1]
            if spl[0] in node_name_dict and len(spl) == 1:
                nn.is_target = True
            rdict[n] = nn

        state_tree = nx.relabel_nodes(graph, rdict)

        return (
            Cassiopeia_Tree(
                method="greedy", network=state_tree, name="Cassiopeia_state_tree"
            ),
            None,
        )

    if method == "greedy_heritable":

        neighbors, distances = None, None
        if missing_data_mode == "knn":
            print("Computing neighbors for imputing missing values...")
            neighbors, distances = find_neighbors(target_nodes, n_neighbors=n_neighbors)

        graph = greedy_build_heritable(
            target_nodes,
            neighbors,
            distances,
            priors=prior_probabilities,
            cell_cutoff=-1,
            lca_cutoff=None,
            fuzzy=fuzzy,
            probabilistic=probabilistic,
            minimum_allele_rep=greedy_minimum_allele_rep,
            missing_data_mode=missing_data_mode,
            lookahead_depth=lookahead_depth
        )[0]

        rdict = {}
        for n in graph:
            spl = n.split("_")
            nn = Node("state-node", spl[0].split("|"), is_target=False)
            if len(spl) > 1:
                nn.pid = spl[1]
            if spl[0] in node_name_dict and len(spl) == 1:
                nn.is_target = True
            rdict[n] = nn

        state_tree = nx.relabel_nodes(graph, rdict)

        return (
            Cassiopeia_Tree(
                method="greedy_heritable", network=state_tree, name="Cassiopeia_state_tree"
            ),
            None,
        )

    else:
        raise Exception(
            "Please specify one of the following methods: ilp, hybrid, greedy"
        )


def reraise_with_stack(func):
    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            traceback_str = traceback.format_exc(e)
            raise StandardError(
                "Error occurred. Original traceback " "is\n%s\n" % traceback_str
            )

    return wrapped


@reraise_with_stack
def prune_unique_alleles(root, targets):

    # "prune" unique alleles
    cp = pd.DataFrame(np.array([t.split("|") for t in targets]))

    # find unique indels
    counts = cp.apply(lambda x: np.unique(x, return_counts=True), axis=0).values
    unique_alleles = list(
        map(
            lambda x: x[0][np.where(x[1] == 1)]
            if len(np.where(x[1] == 1)) > 0
            else None,
            counts,
        )
    )

    #  mask over a unique character in the character matrix
    for uniq, col in zip(unique_alleles, cp.columns):
        if len(uniq) == 0:
            continue
        filt = list(map(lambda x: x in uniq and x != "-", cp[col].values))
        cp.loc[filt, col] = "0"

    targets_pruned = list(cp.apply(lambda x: "|".join(x.values), axis=1).values)

    # prune root appropriately
    lroot = root.split("|")
    for i, uniq in zip(range(len(unique_alleles)), unique_alleles):
        if lroot[i] != "-" and lroot[i] in uniq:
            lroot[i] = "0"
    proot = "|".join(lroot)

    # create mapping storing where each leaf is connected in the pruned target set
    pruned_to_orig = defaultdict(list)
    for i in range(cp.shape[0]):
        pruned = "|".join(cp.iloc[i, :].values)
        if pruned != targets[i]:
            pruned_to_orig[pruned].append(targets[i])

    targets_pruned = list(set(targets_pruned))

    return proot, targets_pruned, pruned_to_orig


@reraise_with_stack
def post_process_ILP(
    subgraph, root, pruned_to_orig, proot, targets, node_name_dict, pid, split_on_heritable = False
):

    # add back in de-pruned leaves
    for k, v in pruned_to_orig.items():
        for n in v:
            subgraph.add_edge(k, n, weight=get_edge_length(k, n, split_on_heritable = split_on_heritable))

    if proot != root:
        # add edge from real root to pruned root
        subgraph.add_edge(root, proot, weight=get_edge_length(root, proot, split_on_heritable = split_on_heritable))

        if subgraph.has_edge(proot, root):
            subgraph.remove_edge(proot, root)

    clean_ilp_network(subgraph)

    # remove spurious roots left in the solution
    subgraph_roots = [n for n in subgraph if subgraph.in_degree(n) == 0]
    while len(subgraph_roots) > 1:
        for r in subgraph_roots:
            if r != root:
                subgraph.remove_node(r)
        subgraph_roots = [n for n in subgraph if subgraph.in_degree(n) == 0]

    node_name_dict_cleaned = {}
    for n in node_name_dict.keys():
        if n in targets:
            node_name_dict_cleaned[n] = node_name_dict[n]

    if root in node_name_dict:
        node_name_dict_cleaned[root] = node_name_dict[root]

    subgraph = nx.relabel_nodes(subgraph, node_name_dict_cleaned)

    return subgraph


@reraise_with_stack
def find_good_gurobi_subgraph(
    root,
    targets,
    node_name_dict,
    prior_probabilities,
    time_limit,
    num_threads,
    max_neighborhood_size,
    seed=None,
    num_iter=-1,
    weighted=False,
    n_neighbors=10,
    split_on_heritable = False
):
    """
	Sub-Function used for multi-threading in hybrid method

	:param root:
		Sub-root of the subgraph that is attempted to be reconstructed
	:param targets:
		List of sub-targets for a given subroot where each node is in the form 'Ch1|Ch2|....|Chn'
	:param prior_probabilities:
		A nested dictionary containing prior probabilities for [character][state] mappings
		where characters are in the form of integers, and states are in the form of strings,
		and values are the probability of mutation from the '0' state.
	:param time_limit:
		Length of time allowed for ILP convergence.
	:param num_threads:
		Number of threads to be used during ILP solving.
	:param max_neighborhood_size:
		Maximum size of potential graph allowed.
	:return:
		Optimal ilp subgraph for a given subset of nodes in the time limit allowed.
	"""

    if weighted:
        assert prior_probabilities is not None

    pid = hashlib.md5(root.encode("utf-8")).hexdigest()

    print(
        "Started new thread for: "
        + str(root)
        + " (num targets = "
        + str(len(targets))
        + ") , pid = "
        + str(pid),
        flush=True,
    )

    if len(set(targets)) == 1:
        graph = nx.DiGraph()
        graph.add_node(node_name_dict[root])
        return [graph], root, pid, {}

    proot, targets_pruned, pruned_to_orig = prune_unique_alleles(root, targets)

    lca = root_finder(targets_pruned, split_on_heritable = split_on_heritable)

    distances = [get_edge_length(lca, t, split_on_heritable = split_on_heritable) for t in targets_pruned]
    widths = [0]
    for i in range(len(distances)):
        for j in range(i, len(distances)):
            if i != j:
                widths.append(distances[i] + distances[j] + 1)

    max_lca = max(widths)

    (
        potential_network_priors,
        lca_dist,
        graph_sizes,
    ) = build_potential_graph_from_base_graph(
        targets_pruned,
        proot,
        priors=prior_probabilities,
        max_neighborhood_size=max_neighborhood_size,
        pid=pid,
        weighted=weighted,
        lca_dist=max_lca,
        split_on_heritable = split_on_heritable
    )

    # network was too large to compute, so just run greedy on it
    if potential_network_priors is None:
        neighbors, distances = find_neighbors(targets, n_neighbors=n_neighbors)
        subgraph = greedy_build(
            targets, neighbors, distances, priors=prior_probabilities, cell_cutoff=-1
        )[0]
        subgraph = nx.relabel_nodes(subgraph, node_name_dict)
        print("Max Neighborhood Exceeded", flush=True)
        return [subgraph], root, pid, graph_sizes

    print(
        "Potential Graph built with maximum LCA of "
        + str(lca_dist)
        + " (pid: "
        + str(pid)
        + "). Proceeding to solver."
    )

    for l in nx.selfloop_edges(potential_network_priors):
        potential_network_priors.remove_edge(l[0], l[1])

    nodes = list(potential_network_priors.nodes())
    encoder = dict(zip(nodes, list(range(len(nodes)))))
    decoder = dict((v, k) for k, v in encoder.items())

    assert len(encoder) == len(decoder)

    _potential_network = nx.relabel_nodes(potential_network_priors, encoder)
    _targets = map(lambda x: encoder[x], targets_pruned)

    model, edge_variables = generate_mSteiner_model(
        _potential_network, encoder[proot], _targets
    )
    subgraphs = solve_steiner_instance(
        model,
        _potential_network,
        edge_variables,
        MIPGap=0.01,
        detailed_output=False,
        time_limit=time_limit,
        num_threads=num_threads,
        seed=seed,
        num_iter=num_iter,
    )

    all_subgraphs = []
    for subgraph in subgraphs:

        subgraph = nx.relabel_nodes(subgraph, decoder)

        subgraph = subgraph = post_process_ILP(
            subgraph, root, pruned_to_orig, proot, targets, node_name_dict, pid, split_on_heritable = split_on_heritable
        )

        all_subgraphs.append(subgraph)

    r_name = root
    if root in node_name_dict:
        r_name = node_name_dict[root]

    return all_subgraphs, r_name, pid, graph_sizes


def clean_ilp_network(network):
    """
	Post-processes networks after an ILP run. At times the ILP will return Steiner Trees which are not necessarily 
	tres (specifically, a node may have more than one parent). To get around this we remove these spurious edges so
	that the trees returned are truly trees. CAUTION: this will modify the network in place. 

	:param network: 
		Network returned from an ILP run.
	:return:
		None. 

	"""

    for u, v in network.edges():
        if u == v:
            network.remove_edge(u, v)
    trouble_nodes = [node for node in network.nodes() if network.in_degree(node) > 1]
    for node in trouble_nodes:
        pred = network.predecessors(node)
        pred = sorted(pred, key=lambda k: network[k][node]["weight"], reverse=True)
        if len(pred) == 2 and (
            pred[1] in nx.ancestors(network, pred[0])
            or pred[0] in nx.ancestors(network, pred[1])
        ):
            print("CASE 1: X-Y->Z, X->Z")
            if pred[1] in nx.ancestors(network, pred[0]):
                network.remove_edge(pred[1], node)
            else:
                network.remove_edge(pred[0], node)
        else:
            print("CASE 2: R->X->Z, R->Y->Z")
            for anc_node in pred[1:]:
                network.remove_edge(anc_node, node)
